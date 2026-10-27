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
"""The no-GPU gates of the ModuleV3 MiMo-V2 DFlash pairing's data-parallel
build.

DP attention serves one-device replicas: the batch splits over a mesh whose
single axis is named ``dp``, the routed experts stay expert-parallel over
every device on the P2P backend, and the drafter holds full weights per
replica. What can be checked without an accelerator is checked here: the
signature's DP pair, context lengths and EP tail; the staged DP2xEP2
trace's batch split, dispatch and combine; the drafter's and the KV tree's
full-head placement; the production config build's per-replica checks; the
packed inputs' slots; the EP buffers' verify-row budget; and the refusals.
Mirrors the ModuleV3 base model's own EP/DP construction gates.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

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
from max.nn.legacy import kernels
from max.nn.legacy.comm.ep import EPBatchManager, EPConfig
from max.nn.legacy.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.legacy.transformer import ReturnHiddenStates
from max.pipelines.architectures.dflash_mimo_v2_modulev3 import DFlashMiMoV2
from max.pipelines.architectures.mimo_v2_modulev3.memory_planner import (
    ep_dispatch_row_budget,
)
from max.pipelines.architectures.mimo_v2_modulev3.model import (
    MiMoV2ModuleV3Model,
    _ep_config,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.quant import (
    MXFP8_DISPATCH_QUANT_CONFIG,
)
from max.pipelines.lib.interfaces.pipeline_model import ModelLoadContext
from max.pipelines.lib.pipeline_variants.spec_decode_pipeline_model import (
    _Inputs,
)
from mimo_dflash_v3_fixtures import (
    BLOCK,
    DRAFTS,
    _FakeAttrConfig,
    build_module,
    build_parts,
    fused_weights,
    mask_embedding,
    target_kv_params,
)

N_DEVICES = 2
N_TAIL = 7
"""draft tokens, seed and the five sampling parameters; the bitmask triple
stays off, as the 1-device staging gate keeps it."""
DP_SIGNAL_BUFFER_INDEX = 5
"""Where the engine binds the signal buffers under data parallelism: past
the batch inputs and the batch-split pair the distributed signature
declares among them."""


def _p2p_ep_config(num_devices: int) -> EPConfig:
    """The P2P EP config the pairing builds at DP, tiny-sized.

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


def _gpu_refs(n_devices: int = N_DEVICES) -> list[DeviceRef]:
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    return [DeviceRef.GPU(i) for i in range(n_devices)]


@pytest.fixture(scope="module")
def dp_built() -> Any:
    """The fused module built at DP2xEP2, its configs and KV tree, never
    run."""
    refs = _gpu_refs()
    # The trace never runs, so it needs no GPU; the pin makes the kernel
    # wrappers trace the SM100 shape paths the B200 box will compile.
    kernels._is_sm10x_gpu = lambda: True
    config, draft, kv_tree = build_parts(
        refs,
        data_parallel_degree=N_DEVICES,
        ep_config=_p2p_ep_config(N_DEVICES),
    )
    module = build_module(
        config,
        draft,
        kv_tree,
        structured_output=False,
        ep_batch_manager=EPBatchManager(_p2p_ep_config(N_DEVICES)),
    )
    return module, config, draft, kv_tree


def test_the_production_config_serves_a_whole_dp_mesh() -> None:
    """The production ``from_huggingface_config`` builds the DP form over
    the whole mesh: its tensor-parallel divisibility checks hold over one
    replica's devices, where the weights shard -- every one-device replica
    holds them unsharded -- where the whole-mesh check a real
    ``--data-parallel-degree 8`` load used to die on ``4 % 8``, and still
    refuses 8 devices without the degree."""
    refs = _gpu_refs(8)
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
    the trace inserts them at the engine's slot past the pair."""
    module, config, draft, kv_tree = dp_built
    call = module.driver_call()
    types = call.input_types
    assert call.driver.input_spec.distributed
    assert call.driver.input_spec.signal_buffer_index == DP_SIGNAL_BUFFER_INDEX
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
    n_ep = len(EPBatchManager(_p2p_ep_config(N_DEVICES)).input_types())
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
    # The target's degree is the mesh's, and the drafter's matches it.
    assert config.data_parallel_degree == N_DEVICES
    assert draft.data_parallel_degree == N_DEVICES


def test_the_tp_signature_declares_no_distributed_group() -> None:
    """At one replica the signature is exactly the TP export's: no host
    offsets, no batch splits, no context lengths and no EP tail."""
    config, draft, kv_tree = build_parts(_gpu_refs(1))
    module = build_module(config, draft, kv_tree, structured_output=False)
    call = module.driver_call()
    types = call.input_types
    assert not call.driver.input_spec.distributed
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
    module, config, draft, _kv_tree = dp_built
    call = module.driver_call()
    staged, _registry = module.stage(
        *call.input_types,
        weights=fused_weights(config, draft),
        signal_buffer_index=call.driver.input_spec.signal_buffer_index,
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
    module, _config, _draft, _kv_tree = dp_built
    for layer in module.draft.layers:
        attention = layer.self_attn
        assert attention.data_parallel
        assert attention.local_heads == attention.num_heads
        assert attention.local_key_value_heads == attention.num_key_value_heads


def test_the_kv_tree_holds_every_kv_head_per_replica(dp_built: Any) -> None:
    """Under DP every replica's cache holds the whole KV head set -- the
    axis the TP build shards and the signature comparisons skip on a KV
    block, so a head-sharding regression under DP lands here, not there."""
    _module, _config, _draft, kv_tree = dp_built
    target = kv_tree.children["target"]
    assert isinstance(target, MultiKVCacheParams)
    for name, leaf in target.children.items():
        assert isinstance(leaf, MHAKVCacheParams)
        assert leaf.tensor_parallel_degree == 1, name
        assert leaf.n_kv_heads_per_device == leaf.n_kv_heads, name
    draft = kv_tree.children["draft"]
    assert isinstance(draft, MHAKVCacheParams)
    assert draft.n_kv_heads_per_device == draft.n_kv_heads
    # The TP build over the same devices shards the same heads: the
    # contrast that makes the whole-head assert mean something.
    _c, _d, tp_tree = build_parts([DeviceRef.GPU(i) for i in range(N_DEVICES)])
    tp_target = tp_tree.children["target"]
    assert isinstance(tp_target, MultiKVCacheParams)
    for name, leaf in tp_target.children.items():
        assert isinstance(leaf, MHAKVCacheParams)
        assert leaf.n_kv_heads_per_device == leaf.n_kv_heads // N_DEVICES, name
    tp_draft = tp_tree.children["draft"]
    assert isinstance(tp_draft, MHAKVCacheParams)
    assert tp_draft.n_kv_heads_per_device == tp_draft.n_kv_heads // N_DEVICES


def test_the_dp_module_loads_the_adapted_checkpoint(dp_built: Any) -> None:
    """The fused module's parameters and the adapted ``target.``/``draft.``
    checkpoint stay equal as sets: DP replicates weights rather than
    renaming them, but the strict load still has to pin the contract."""
    module, config, draft, _kv_tree = dp_built
    names = {name for name, _ in module.parameters}
    weights = fused_weights(config, draft)
    assert names == set(weights), (
        f"module lacks {sorted(set(weights) - names)[:8]}; no parameter "
        f"matches {sorted(names - set(weights))[:8]}"
    )


def test_the_packed_inputs_fill_the_dp_slots(dp_built: Any) -> None:
    """The pairing's packed inputs mirror the DP signature slot for slot:
    the host offsets and the splits among the batch inputs, the signal
    buffers inserted at the engine's slot past them, then the KV tree, the
    context lengths and the EP tail."""
    module, _config, _draft, _kv_tree = dp_built
    spec = module.driver_call().driver.input_spec

    def buffer() -> Buffer:
        return Buffer.zeros([1], DType.int64, device=CPU())

    inputs = _Inputs(
        tokens=buffer(),
        input_row_offsets=buffer(),
        return_n_logits=buffer(),
        signal_buffers=[buffer() for _ in range(N_DEVICES)],
        host_input_row_offsets=buffer(),
        data_parallel_splits=buffer(),
        batch_context_lengths=[buffer() for _ in range(N_DEVICES)],
        ep_inputs=(buffer(),),
        draft_tokens=buffer(),
        seed=buffer(),
        temperature=buffer(),
        top_k=buffer(),
        max_k=buffer(),
        top_p=buffer(),
        min_top_p=buffer(),
        spec=spec,
    )
    buffers = inputs.buffers
    assert buffers[:5] == (
        inputs.tokens,
        inputs.input_row_offsets,
        inputs.host_input_row_offsets,
        inputs.return_n_logits,
        inputs.data_parallel_splits,
    )
    end = DP_SIGNAL_BUFFER_INDEX + N_DEVICES
    assert buffers[DP_SIGNAL_BUFFER_INDEX:end] == tuple(inputs.signal_buffers)
    assert buffers[end : end + N_DEVICES] == tuple(inputs.batch_context_lengths)
    assert buffers[end + N_DEVICES] is inputs.ep_inputs[0]
    assert buffers[end + N_DEVICES + 1] is inputs.draft_tokens


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


def test_dp_without_ep_is_refused() -> None:
    """DP attention without an EP batch manager is a wiring error, not a
    silent tensor-parallel build -- the base target's own rule, reached
    through the spec target."""
    config, draft, kv_tree = build_parts(
        _gpu_refs(), data_parallel_degree=N_DEVICES
    )
    with pytest.raises(ValueError, match="no EP batch manager"):
        build_module(config, draft, kv_tree, structured_output=False)


def test_a_dp_tp_mesh_is_refused() -> None:
    """A dp x tp mesh -- 2 replicas over 4 devices -- fits no one-device
    replica per device: the target's own guard refuses the module build,
    the same rule the target's build refuses the load with."""
    config, draft, kv_tree = build_parts(
        _gpu_refs(4), data_parallel_degree=2, ep_config=_p2p_ep_config(4)
    )
    with pytest.raises(NotImplementedError, match="must equal the device"):
        build_module(config, draft, kv_tree, structured_output=False)


def test_a_degree_above_the_device_count_is_refused() -> None:
    """A drafter degree with no device to hold each replica is refused by
    the drafter's own guard; the target's covers the rest of the mesh."""
    _config, draft, _kv_tree = build_parts(
        _gpu_refs(),
        data_parallel_degree=N_DEVICES,
        ep_config=_p2p_ep_config(N_DEVICES),
    )
    mismatched = replace(draft, data_parallel_degree=N_DEVICES + 1)
    with pytest.raises(NotImplementedError, match="must equal the device"):
        DFlashMiMoV2(mismatched, mask_embedding=mask_embedding())


def _load_context(model: object, runtime: object) -> ModelLoadContext:
    """A load context carrying only what the build reads before its
    checks."""
    return cast(
        ModelLoadContext,
        SimpleNamespace(
            pipeline_config=SimpleNamespace(model=model, runtime=runtime),
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
        ),
    )


def test_the_build_refuses_an_unservable_mesh_before_loading() -> None:
    """The target's build refuses a dp x tp mesh, DP without EP over every
    device, and EP without DP before it reads the checkpoint."""
    four = SimpleNamespace(data_parallel_degree=2, device_specs=[None] * 4)
    with pytest.raises(ValueError, match="must equal the device count"):
        MiMoV2ModuleV3Model.build_module(
            _load_context(four, SimpleNamespace(ep_size=4))
        )
    two = SimpleNamespace(data_parallel_degree=2, device_specs=[None] * 2)
    with pytest.raises(ValueError, match="requires expert parallelism"):
        MiMoV2ModuleV3Model.build_module(
            _load_context(two, SimpleNamespace(ep_size=1))
        )
    # The TP spec signature has no slot for the EP buffers.
    tp = SimpleNamespace(data_parallel_degree=1, device_specs=[None] * 2)
    with pytest.raises(NotImplementedError, match="data-parallel attention"):
        MiMoV2ModuleV3Model.build_module(
            _load_context(tp, SimpleNamespace(ep_size=2))
        )


# --------------------------------------------------------------------------- #
# The EP dispatch buffers' verify-row budget
# --------------------------------------------------------------------------- #


def _pipeline_config(
    max_batch_size: int | None, *, speculative: bool = True
) -> Any:
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
        speculative=(
            SimpleNamespace(draft_width=DRAFTS) if speculative else None
        ),
    )


def test_the_ep_config_counts_the_verify_rows() -> None:
    """A speculative build's dispatch buffers hold a full decode step's
    merged verify rows -- the scheduler's active-token budget plus every
    request's drafts -- where the base target's hold the budget alone,
    and nothing else about the config differs. The planner reserves the
    same budget the build sizes the buffers with."""
    config, _draft, _kv_tree = build_parts(
        _gpu_refs(), data_parallel_degree=N_DEVICES
    )
    speculating = _pipeline_config(max_batch_size=32)
    base = _pipeline_config(max_batch_size=32, speculative=False)
    built = _ep_config(speculating, N_DEVICES, N_DEVICES, config)
    # 256 active rows + 32 requests x 7 draft rows; the P2P divide is a
    # no-op at DP, where tp_size is 1.
    assert built == replace(
        _ep_config(base, N_DEVICES, N_DEVICES, config),
        max_tokens_per_rank=256 + 32 * DRAFTS,
    )
    assert ep_dispatch_row_budget(speculating) == 256 + 32 * DRAFTS
    assert ep_dispatch_row_budget(base) == 256


def test_the_ep_config_refuses_an_unbounded_verify_batch() -> None:
    """Without ``max_batch_size`` nothing bounds the requests a verify
    step carries, so the draft rows are unbounded too -- refused at
    build, not overrun at runtime."""
    config, _draft, _kv_tree = build_parts(
        _gpu_refs(), data_parallel_degree=N_DEVICES
    )
    with pytest.raises(ValueError, match="needs max_batch_size"):
        _ep_config(
            _pipeline_config(max_batch_size=None), N_DEVICES, N_DEVICES, config
        )
