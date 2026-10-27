# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""The no-GPU gates of the fused DFlash spec graph's data-parallel build.

DP attention serves one-device replicas: the batch splits over a mesh whose
single axis is named ``dp``, the routed experts stay expert-parallel over
every device on the P2P backend, and the drafter holds full weights per
replica. Everything of that this slice changed that can be checked without
an accelerator is checked here: the signature's DP pair, context lengths
and EP tail; the staged DP2xEP2 trace's batch split, dispatch and combine;
the drafter's and the KV tree's full-head placement; the production config
build's per-replica checks; and the runtime half -- the batch processor's
DP packing -- beside the refusals. Mirrors the ModuleV3 base model's own
EP/DP construction gates.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import (
    CPU,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn import kernels
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.dflash_mimo_v2_modulev3 import DFlashMiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.memory_planner import (
    MiMoV2ModuleV3MemoryPlanner,
)
from max.pipelines.architectures.mimo_v2_modulev3.model import (
    MiMoV2ModuleV3Model,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.quant import (
    MXFP8_DISPATCH_QUANT_CONFIG,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.batch_processor import (
    UnifiedDflashMiMoV2ModuleV3BatchProcessor,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.memory_planner import (
    UnifiedDflashMiMoV2ModuleV3MemoryPlanner,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.model import (
    UnifiedDflashMiMoV2ModuleV3Model,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.model_config import (
    DRAFT,
    verify_widened_row_budget,
)
from max.pipelines.architectures.unified_dflash_mimo_v2_modulev3.unified_dflash_mimo_v2 import (
    DP_SIGNAL_BUFFER_INDEX,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.batch_processor import BatchProcessorRuntime
from max.pipelines.request.base import RequestID
from mimo_dflash_v3_fixtures import (
    BLOCK,
    DRAFTS,
    _FakeAttrConfig,
    build_module,
    build_spec,
    fused_weights,
    mask_embedding,
    target_kv_params,
)

N_DEVICES = 2
N_TAIL = 7
"""draft tokens, seed and the five sampling parameters; the bitmask triple
stays off, as the 1-device staging gate keeps it."""


def _ep_config(num_devices: int) -> EPConfig:
    """The P2P EP config the pipeline model builds at DP, tiny-sized.

    DP attention is the P2P dispatch's precondition (each token lives on
    exactly one replica), and the allreduce backend has no whole-batch
    allreduce left to complete its combine under DP.
    """
    return EPConfig(
        dispatch_dtype=DType.float8_e4m3fn,
        dispatch_quant_config=MXFP8_DISPATCH_QUANT_CONFIG,
        combine_dtype=DType.bfloat16,
        hidden_size=4096,
        top_k=8,
        n_experts=16,
        max_tokens_per_rank=512,
        n_gpus_per_node=num_devices,
        n_nodes=1,
        use_allreduce=False,
    )


def _gpu_refs() -> list[DeviceRef]:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)
    return [DeviceRef.GPU(i) for i in range(N_DEVICES)]


@pytest.fixture(scope="module")
def dp_built() -> Any:
    """The fused module built at DP2xEP2, its spec and KV tree, never run."""
    refs = _gpu_refs()
    # The trace never runs, so it needs no GPU; the pin makes the kernel
    # wrappers trace the SM100 shape paths the B200 box will compile.
    kernels._is_sm10x_gpu = lambda: True
    spec, _config, _draft, kv_tree = build_spec(
        refs,
        data_parallel_degree=N_DEVICES,
        ep_config=_ep_config(N_DEVICES),
    )
    module = build_module(
        spec,
        kv_tree,
        structured_output=False,
        ep_batch_manager=EPBatchManager(_ep_config(N_DEVICES)),
    )
    return module, spec, kv_tree


def test_the_production_config_serves_a_whole_dp_mesh() -> None:
    """The production ``from_huggingface_config`` builds the DP form over
    the whole mesh: its tensor-parallel divisibility checks hold over one
    replica's devices, where the weights shard -- every one-device replica
    holds them unsharded -- where the whole-mesh check a real
    ``--data-parallel-degree 8`` load used to die on ``4 % 8``, and still
    refuses 8 devices without the degree."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(8)
    refs = [DeviceRef.GPU(i) for i in range(8)]
    kv = target_kv_params(refs, num_draft_tokens=BLOCK, data_parallel_degree=8)
    config = MiMoV2Config.from_huggingface_config(
        _FakeAttrConfig(),
        devices=refs,
        kv_params=kv,
        max_seq_len=512,
        data_parallel_degree=8,
    )
    assert config.devices == refs
    with pytest.raises(
        ValueError,
        match="8 devices do not divide the 4 sliding_attention qkv_proj chunks",
    ):
        MiMoV2Config.from_huggingface_config(
            _FakeAttrConfig(),
            devices=refs,
            kv_params=kv,
            max_seq_len=512,
        )


# --------------------------------------------------------------------------- #
# The signature the DP form declares
# --------------------------------------------------------------------------- #


def test_the_dp_signature_declares_the_distributed_group(dp_built: Any) -> None:
    """The DP pair sits inside the batch inputs, the context lengths and
    the EP tail follow the KV tree, and no signal buffers are declared --
    the trace inserts them at its own slot, so the declared signature
    carries none."""
    module, spec, kv_tree = dp_built
    types = module.input_types()
    assert types[0] == TensorType(
        DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU(0)
    )
    assert types[1] == TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=DeviceRef.GPU(0)
    )
    # The CPU host mirror of the row offsets, beside them.
    assert types[2] == TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=DeviceRef.CPU()
    )
    assert types[3] == TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    # Per-replica boundaries: one entry per replica plus the total.
    assert types[4] == TensorType(
        DType.int64, shape=[N_DEVICES + 1], device=DeviceRef.CPU()
    )
    # The tail's EP count is the batch manager's own, pinned as the base
    # graph's EP gate pins it: the atomic counters plus three pointer
    # tables per buffer group.
    n_ep = len(EPBatchManager(_ep_config(N_DEVICES)).input_types())
    assert n_ep == 2 * N_DEVICES + 3 * 2
    # The declared count is exactly the batch side, the KV tree, one
    # context length per device, the EP tail and the spec tail -- no
    # signal buffers of its own anywhere.
    n_kv = len(kv_tree.flattened_kv_inputs())
    assert len(types) == 5 + n_kv + N_DEVICES + n_ep + N_TAIL
    # Past the KV tree: one per-device context length, then the EP tail.
    tail_start = len(types) - N_TAIL
    for i in range(tail_start - n_ep - N_DEVICES, tail_start - n_ep):
        assert types[i] == TensorType(
            DType.int32, shape=[1], device=DeviceRef.CPU()
        ), f"slot {i}"
    # The spec-decode tail: draft tokens, then seed and the parameters.
    assert types[tail_start] == TensorType(
        DType.int64,
        shape=["batch_size", "num_steps"],
        device=DeviceRef.GPU(0),
    )
    # The spec's degree is the mesh's, and the drafter's matches it.
    assert spec.target.data_parallel_degree == N_DEVICES
    assert spec.draft.data_parallel_degree == N_DEVICES


def test_the_tp_signature_declares_no_distributed_group() -> None:
    """At one replica the signature is exactly the TP export's: no host
    offsets, no batch splits, no context lengths and no EP tail."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    spec, _config, _draft, kv_tree = build_spec([DeviceRef.GPU()])
    module = build_module(spec, kv_tree, structured_output=False)
    types = module.input_types()
    assert len(types) == 35
    # The single-device graph declares its own signal buffer at the
    # engine's slot; the batch inputs ahead of it are the TP three.
    assert isinstance(types[3], BufferType)
    assert types[:3] == (
        TensorType(
            DType.int64, shape=["total_seq_len"], device=DeviceRef.GPU(0)
        ),
        TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.GPU(0),
        ),
        TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        ),
    )


# --------------------------------------------------------------------------- #
# The whole graph at DP2xEP2, staged on virtual B200s
# --------------------------------------------------------------------------- #


def test_the_dp2_graph_stages_with_the_batch_split(dp_built: Any) -> None:
    """The DP2xEP2 graph stages end to end: the trace's signal buffers sit
    past the batch-split pair at the engine's slot, the split carves each
    replica's rows out of the replicated batch, and the MoE keeps its
    global expert dispatch over every device."""
    module, spec, _kv_tree = dp_built
    staged, _registry = module.stage(
        *module.input_types(),
        weights=fused_weights(spec.target, spec.draft),
        signal_buffer_index=DP_SIGNAL_BUFFER_INDEX,
    )
    inputs = staged.graph.inputs
    # The trace inserts the mesh's signal buffers between the batch-split
    # pair and the KV tree, where the engine's block adapter binds them.
    for i in range(N_DEVICES):
        signal = inputs[DP_SIGNAL_BUFFER_INDEX + i].type
        assert isinstance(signal, BufferType)
        assert signal.dtype == DType.uint8
        assert signal.device == DeviceRef.GPU(i)

    text = str(staged)
    # The split carves each replica's rows from the replicated batch: the
    # slice dims are named after the split, as the base graph's are.
    assert "_input_split_0" in text and "_input_split_1" in text
    # The MoE keeps its global expert dispatch over every device.
    assert text.count("ep.dispatch") >= N_DEVICES
    assert text.count("ep.combine") >= N_DEVICES
    assert text.count("mo.call @mimo_v2_moe") == 3


def test_the_drafter_holds_full_heads_per_replica(dp_built: Any) -> None:
    """At DP2 the drafter keeps every head local -- all its query and KV
    heads per replica, as the target's attention does -- where the TP
    build splits them over the devices."""
    module, _spec, _kv_tree = dp_built
    for layer in module.draft.layers:
        attention = layer.self_attn
        assert attention.data_parallel
        assert attention.local_heads == attention.num_heads
        assert attention.local_key_value_heads == attention.num_key_value_heads


def test_the_kv_tree_holds_every_kv_head_per_replica(dp_built: Any) -> None:
    """Under DP every replica's cache holds the whole KV head set -- the
    axis the TP build shards and the signature comparisons skip on a KV
    block, so a head-sharding regression under DP lands here, not there."""
    _module, _spec, kv_tree = dp_built
    target = kv_tree.children["target"]
    assert isinstance(target, MultiKVCacheParams)
    for name, leaf in target.children.items():
        assert isinstance(leaf, MHAKVCacheParams)
        assert leaf.tensor_parallel_degree == 1, name
        assert leaf.n_kv_heads_per_device == leaf.n_kv_heads, name
    draft = kv_tree.children[DRAFT]
    assert isinstance(draft, MHAKVCacheParams)
    assert draft.n_kv_heads_per_device == draft.n_kv_heads
    # The TP build over the same devices shards the same heads: the
    # contrast that makes the whole-head assert mean something.
    _s, _c, _d, tp_tree = build_spec(
        [DeviceRef.GPU(i) for i in range(N_DEVICES)]
    )
    tp_target = tp_tree.children["target"]
    assert isinstance(tp_target, MultiKVCacheParams)
    for name, leaf in tp_target.children.items():
        assert isinstance(leaf, MHAKVCacheParams)
        assert leaf.n_kv_heads_per_device == leaf.n_kv_heads // N_DEVICES, name
    tp_draft = tp_tree.children[DRAFT]
    assert isinstance(tp_draft, MHAKVCacheParams)
    assert tp_draft.n_kv_heads_per_device == tp_draft.n_kv_heads // N_DEVICES


def test_the_dp_module_loads_the_adapted_checkpoint(dp_built: Any) -> None:
    """The fused module's parameters and the adapted ``target.``/``draft.``
    checkpoint stay equal as sets: DP replicates weights rather than
    renaming them, but the strict load still has to pin the contract."""
    module, spec, _kv_tree = dp_built
    names = {name for name, _ in module.parameters}
    weights = fused_weights(spec.target, spec.draft)
    assert names == set(weights), (
        f"module lacks {sorted(set(weights) - names)[:8]}; no parameter "
        f"matches {sorted(names - set(weights))[:8]}"
    )


# --------------------------------------------------------------------------- #
# The runtime half: the batch processor's DP packing
# --------------------------------------------------------------------------- #


def _context(prompt: list[int]) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=1000,
        tokens=TokenBuffer(np.asarray(prompt, dtype=np.int64)),
    )


def _dp_processor(dp: int) -> UnifiedDflashMiMoV2ModuleV3BatchProcessor:
    """The batch processor on real CPU devices, for runtime input packing."""
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(
            Any, SimpleNamespace(needs_bitmask_constraints=False)
        ),
        # One device per replica, as the runtime serves them.
        devices=[CPU() for _ in range(dp)],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=4,
        max_seq_len=256,
        max_batch_input_tokens=256,
        data_parallel_degree=dp,
        signal_buffers=[Buffer.zeros([1], DType.uint8) for _ in range(dp)],
    )
    return UnifiedDflashMiMoV2ModuleV3BatchProcessor(
        cast(Any, SimpleNamespace()), runtime
    )


def _fill_sampling_tail(inputs: Any) -> None:
    """Sets the spec-decode tail the scheduler supplies after packing."""
    device = CPU()
    inputs.draft_tokens = Buffer.from_numpy(
        np.zeros([1, 0], dtype=np.int64)
    ).to(device)
    inputs.temperature = Buffer.from_numpy(np.zeros([1], np.float32)).to(device)
    inputs.top_k = Buffer.from_numpy(np.ones([1], np.int64)).to(device)
    inputs.max_k = Buffer.from_numpy(np.ones([1], np.int64))
    inputs.top_p = Buffer.from_numpy(np.ones([1], np.float32)).to(device)
    inputs.min_top_p = Buffer.from_numpy(np.ones([1], np.float32))


def test_dp_initial_inputs_carry_the_split_tensors() -> None:
    """``prepare_initial_token_inputs`` concatenates every replica's
    requests into one device-0 ragged batch and fills the CPU
    ``data_parallel_splits`` boundaries and host row offsets beside it --
    the buffers order mirroring the declared signature slot for slot, an
    empty replica included."""
    processor = _dp_processor(2)
    inputs = processor.prepare_initial_token_inputs(
        [[_context([1, 2, 3]), _context([4])], [_context([5, 6])]],
        kv_cache_inputs=cast(Any, ()),
    )
    # [0, requests on replica 0, requests on replica 1]: two replicas,
    # each owning whole requests.
    assert inputs.data_parallel_splits is not None
    np.testing.assert_array_equal(
        inputs.data_parallel_splits.to_numpy(), [0, 2, 3]
    )
    # The host row offsets mirror the concatenated batch's offsets.
    assert inputs.host_input_row_offsets is not None
    np.testing.assert_array_equal(
        inputs.host_input_row_offsets.to_numpy(), [0, 3, 4, 6]
    )
    np.testing.assert_array_equal(inputs.tokens.to_numpy(), [1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 3, 4, 6]
    )
    # One preallocated per-device context length, as the M3 model's are.
    assert len(inputs.batch_context_lengths) == 2

    # An empty replica contributes a zero-width boundary -- splits
    # [0, 2, 2] -- with the host mirror carrying the concatenated batch's
    # own offsets, so the in-graph split hands the replica a 0-token shard.
    empty = _dp_processor(2).prepare_initial_token_inputs(
        [[_context([1, 2, 3]), _context([4])], []],
        kv_cache_inputs=cast(Any, ()),
    )
    assert empty.data_parallel_splits is not None
    np.testing.assert_array_equal(
        empty.data_parallel_splits.to_numpy(), [0, 2, 2]
    )
    assert empty.host_input_row_offsets is not None
    np.testing.assert_array_equal(
        empty.host_input_row_offsets.to_numpy(), [0, 3, 4]
    )
    np.testing.assert_array_equal(empty.input_row_offsets.to_numpy(), [0, 3, 4])

    _fill_sampling_tail(inputs)
    buffers = inputs.buffers
    assert buffers[0] is inputs.tokens
    assert buffers[1] is inputs.input_row_offsets
    assert buffers[2] is inputs.host_input_row_offsets
    assert buffers[3] is inputs.return_n_logits
    assert buffers[4] is inputs.data_parallel_splits
    assert buffers[5:7] == tuple(inputs.signal_buffers)
    assert buffers[7:9] == tuple(inputs.batch_context_lengths)
    assert buffers[9] is inputs.draft_tokens

    # Without DP the packing leaves the pair and the context lengths
    # unset, keeping the TP export's ABI.
    plain = _dp_processor(1).prepare_initial_token_inputs(
        [[_context([9])]], kv_cache_inputs=cast(Any, ())
    )
    assert plain.data_parallel_splits is None
    assert plain.host_input_row_offsets is None
    assert plain.batch_context_lengths == []


def test_dp_replica_count_mismatch_is_refused() -> None:
    """A DP2 runtime packing one replica batch is a wiring error, not a
    silent shard."""
    with pytest.raises(ValueError, match="expects 2 replica batches"):
        _dp_processor(2).prepare_initial_token_inputs(
            [[_context([1])]], kv_cache_inputs=cast(Any, ())
        )


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_dp_without_ep_is_refused() -> None:
    """DP attention without an EP batch manager is a wiring error, not a
    silent tensor-parallel build -- the base target's own rule, reached
    through the fused spec target."""
    spec, _config, _draft, kv_tree = build_spec(
        _gpu_refs(), data_parallel_degree=N_DEVICES
    )
    with pytest.raises(ValueError, match="no EP batch manager"):
        build_module(spec, kv_tree, structured_output=False)


def test_a_dp_tp_mesh_is_refused() -> None:
    """A dp x tp mesh -- 2 replicas over 4 devices -- fits no one-device
    replica per device: the target's own guard refuses the module build,
    the same rule the pipeline model refuses the load with."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(4)
    refs = [DeviceRef.GPU(i) for i in range(4)]
    spec, _config, _draft, kv_tree = build_spec(
        refs, data_parallel_degree=2, ep_config=_ep_config(4)
    )
    with pytest.raises(NotImplementedError, match="must equal the device"):
        build_module(spec, kv_tree, structured_output=False)


def test_a_degree_above_the_device_count_is_refused() -> None:
    """A drafter degree with no device to hold each replica is refused by
    the drafter's own guard; the target's covers the rest of the mesh."""
    _refs = _gpu_refs()
    _spec, _config, draft, _kv_tree = build_spec(
        _refs,
        data_parallel_degree=N_DEVICES,
        ep_config=_ep_config(N_DEVICES),
    )
    mismatched = replace(draft, data_parallel_degree=N_DEVICES + 1)
    with pytest.raises(NotImplementedError, match="must equal the device"):
        DFlashMiMoV2(mismatched, mask_embedding=mask_embedding())


def test_the_drafter_reads_its_degree_from_the_spec() -> None:
    """The module builds the drafter straight from the spec -- the pipeline
    model sets the degree beside the draft config -- so a spec whose
    drafter degree mismatches trips the drafter's own guard instead of
    being silently repaired at the build."""
    _refs = _gpu_refs()
    spec, _config, _draft, kv_tree = build_spec(
        _refs,
        data_parallel_degree=N_DEVICES,
        ep_config=_ep_config(N_DEVICES),
    )
    mismatched = replace(
        spec, draft=replace(spec.draft, data_parallel_degree=N_DEVICES + 1)
    )
    with pytest.raises(NotImplementedError, match="must equal the device"):
        build_module(
            mismatched,
            kv_tree,
            structured_output=False,
            ep_batch_manager=EPBatchManager(_ep_config(N_DEVICES)),
        )


def test_the_model_refuses_a_dp_tp_mesh_before_loading() -> None:
    """The pipeline model's ``__init__`` refuses an unservable DP mesh up
    front -- the base target's own degree check, reached in seconds where
    the load used to hit it after the whole checkpoint and drafter -- so
    nothing past the validation is ever touched."""
    model = cast(
        UnifiedDflashMiMoV2ModuleV3Model,
        object.__new__(UnifiedDflashMiMoV2ModuleV3Model),
    )
    pipeline_config = cast(
        Any,
        SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=2),
            runtime=SimpleNamespace(ep_size=4),
            speculative=SimpleNamespace(draft_width=DRAFTS),
        ),
    )
    with pytest.raises(ValueError, match="must equal the device count"):
        UnifiedDflashMiMoV2ModuleV3Model.__init__(
            model,
            pipeline_config,
            cast(Any, None),
            cast(Any, [None] * 4),
            cast(Any, None),
            cast(Any, None),
            memory_plan=cast(Any, None),
        )
    # DP without EP over every device is refused the same way, up front.
    without_ep = cast(
        Any,
        SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=2),
            runtime=SimpleNamespace(ep_size=1),
            speculative=SimpleNamespace(draft_width=DRAFTS),
        ),
    )
    with pytest.raises(ValueError, match="requires expert parallelism"):
        UnifiedDflashMiMoV2ModuleV3Model.__init__(
            model,
            without_ep,
            cast(Any, None),
            cast(Any, [None, None]),
            cast(Any, None),
            cast(Any, None),
            memory_plan=cast(Any, None),
        )


# --------------------------------------------------------------------------- #
# The EP dispatch buffers' verify-row budget
# --------------------------------------------------------------------------- #


def _spec_pipeline_config(max_batch_size: int | None) -> Any:
    """A pipeline-config stand-in carrying the knobs the row budget reads."""
    return SimpleNamespace(
        model=SimpleNamespace(data_parallel_degree=N_DEVICES),
        runtime=SimpleNamespace(
            ep_size=N_DEVICES,
            ep_use_allreduce=False,
            enable_chunked_prefill=True,
            max_batch_input_tokens=256,
            max_batch_size=max_batch_size,
        ),
        speculative=SimpleNamespace(draft_width=DRAFTS),
    )


def _spec_model(pipeline_config: Any) -> UnifiedDflashMiMoV2ModuleV3Model:
    """The pipeline model shell the EP config build reads, no load."""
    model = cast(
        UnifiedDflashMiMoV2ModuleV3Model,
        object.__new__(UnifiedDflashMiMoV2ModuleV3Model),
    )
    model.pipeline_config = pipeline_config
    return model


def test_the_ep_config_counts_the_verify_rows() -> None:
    """The DP export's dispatch buffers hold a full decode step's merged
    verify rows -- the scheduler's active-token budget plus every
    request's drafts -- where the base target's hold the budget alone,
    and nothing else about the config differs."""
    _refs = _gpu_refs()
    _spec, config, _draft, _kv_tree = build_spec(
        _refs, data_parallel_degree=N_DEVICES
    )
    pipeline_config = _spec_pipeline_config(max_batch_size=32)
    built = _spec_model(pipeline_config)._build_ep_config(
        N_DEVICES, N_DEVICES, config
    )
    base_model = cast(MiMoV2ModuleV3Model, object.__new__(MiMoV2ModuleV3Model))
    base_model.pipeline_config = pipeline_config
    base = base_model._build_ep_config(N_DEVICES, N_DEVICES, config)
    # 256 active rows + 32 requests x 7 draft rows; the P2P divide is a
    # no-op at DP, where tp_size is 1.
    assert built == replace(base, max_tokens_per_rank=256 + 32 * DRAFTS)


def test_the_ep_config_refuses_an_unbounded_verify_batch() -> None:
    """Without ``max_batch_size`` nothing bounds the requests a verify
    step carries, so the draft rows are unbounded too -- refused at
    build, not overrun at runtime."""
    _refs = _gpu_refs()
    _spec, config, _draft, _kv_tree = build_spec(
        _refs, data_parallel_degree=N_DEVICES
    )
    model = _spec_model(_spec_pipeline_config(max_batch_size=None))
    with pytest.raises(ValueError, match="needs max_batch_size"):
        model._build_ep_config(N_DEVICES, N_DEVICES, config)


def test_the_spec_planner_reserves_the_widened_budget() -> None:
    """The spec arch's planner follows the same widened budget the model
    sizes the dispatch buffers with, where the base planner's follows the
    scheduler's active-token budget alone."""
    _spec, config, _draft, _kv_tree = build_spec([DeviceRef.GPU(0)])
    pipeline_config = _spec_pipeline_config(max_batch_size=32)
    assert MiMoV2ModuleV3MemoryPlanner(config).ep_dispatch_row_budget(
        pipeline_config
    ) == (pipeline_config.runtime.max_batch_input_tokens)
    assert UnifiedDflashMiMoV2ModuleV3MemoryPlanner(
        config
    ).ep_dispatch_row_budget(pipeline_config) == (
        verify_widened_row_budget(pipeline_config)
    )
