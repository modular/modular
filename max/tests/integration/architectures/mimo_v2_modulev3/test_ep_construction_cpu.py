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
"""The no-GPU gates of MiMo-V2's expert-parallel ModuleV3 path.

The GPU tests stay the numerics gates; everything they depend on that can be
checked without an accelerator is checked here: the parallelism-mode
selection, the EP MoE's whole-expert weight placement under GLOBAL expert
indices (per-shard reindexing would silently load the wrong experts), the
P2P refusal, the DP-attention settings and input declarations, and the
staged trace of the whole EP2 graph on virtual B200s -- the same no-GPU
verification lane the TP harness runs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
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
from max.experimental import functional as F
from max.experimental.sharding import DeviceMesh
from max.experimental.tensor import (
    Tensor,
    default_device,
    default_dtype,
)
from max.graph import DeviceRef, TensorType
from max.graph.weights import SafetensorWeights, WeightData
from max.nn import kernels
from max.nn.comm.ep import EPBatchManager, EPConfig
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import (
    MiMoV2ExpertParallelMoE,
)
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import (
    MiMoV2,
    ParallelismMode,
    select_parallelism_mode,
)
from max.pipelines.architectures.mimo_v2_modulev3.model import (
    MiMoV2Inputs,
    MiMoV2ModuleV3BatchProcessor,
    MiMoV2ModuleV3Model,
    _validate_data_parallel_attention,
)
from max.pipelines.architectures.mimo_v2_modulev3.model_config import (
    MiMoV2Config,
)
from max.pipelines.architectures.mimo_v2_modulev3.quant import (
    MXFP8_DISPATCH_QUANT_CONFIG,
)
from max.pipelines.architectures.mimo_v2_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.context import TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.batch_processor import (
    BatchProcessorRuntime,
    modulev3_ragged_kv_symbolic_inputs,
)
from max.pipelines.request.base import RequestID
from mimo_v3_runner import kv_params
from tiny_fixture import (
    EXPERTS,
    HIDDEN,
    MOE_DIM,
    TOP_K,
    tensors,
    write,
)
from transformers.configuration_utils import PretrainedConfig

N_DEVICES = 2


def _ep_config(num_devices: int, use_allreduce: bool = True) -> EPConfig:
    """The allreduce EP config the pipeline model builds, tiny-sized."""
    return EPConfig(
        dispatch_dtype=DType.float8_e4m3fn,
        dispatch_quant_config=MXFP8_DISPATCH_QUANT_CONFIG,
        combine_dtype=DType.bfloat16,
        hidden_size=HIDDEN,
        top_k=TOP_K,
        n_experts=EXPERTS,
        max_tokens_per_rank=512,
        n_gpus_per_node=num_devices,
        n_nodes=1,
        use_allreduce=use_allreduce,
    )


def _mxfp4_experts_config() -> QuantConfig:
    """The routed experts' MXFP4 config, as ``parse_quant_scheme`` builds."""
    return QuantConfig.mxfp4(mlp_quantized_layers={1, 2, 3})


# --------------------------------------------------------------------------- #
# Mode selection
# --------------------------------------------------------------------------- #


def _dp_runtime_batch_processor(
    dp: int,
) -> MiMoV2ModuleV3BatchProcessor:
    """The batch processor on a real CPU device, for runtime input packing."""
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(Any, SimpleNamespace()),
        devices=[CPU()],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=4,
        max_seq_len=256,
        max_batch_input_tokens=256,
        data_parallel_degree=dp,
    )
    return MiMoV2ModuleV3BatchProcessor(
        cast(MiMoV2Config, SimpleNamespace()), runtime
    )


def _context(prompt: Sequence[int]) -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=1000,
        tokens=TokenBuffer(np.asarray(prompt, dtype=np.int64)),
    )


def test_dp_initial_inputs_carry_the_split_tensors() -> None:
    """The runtime produces the DP tensors the forward splits by.

    `prepare_initial_token_inputs` concatenates every replica's requests
    into one device-0 ragged batch and fills the CPU `data_parallel_splits`
    boundaries and host row offsets beside it -- without them the graph's
    KV leaves would land in the DP slots (the runtime half the staging
    tests cannot see).
    """
    processor = _dp_runtime_batch_processor(2)
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
    # One device-0 batch over both replicas, in the declared buffer order.
    np.testing.assert_array_equal(inputs.tokens.to_numpy(), [1, 2, 3, 4, 5, 6])
    assert isinstance(inputs.input_row_offsets, Buffer)
    np.testing.assert_array_equal(
        inputs.input_row_offsets.to_numpy(), [0, 3, 4, 6]
    )
    buffers = inputs.buffers
    assert buffers[3] is inputs.data_parallel_splits
    assert buffers[4] is inputs.host_input_row_offsets
    # Without DP the inherited single-replica path leaves the pair unset.
    plain = _dp_runtime_batch_processor(1).prepare_initial_token_inputs(
        [[_context([9])]], kv_cache_inputs=cast(Any, ())
    )
    assert plain.data_parallel_splits is None
    assert plain.host_input_row_offsets is None


def test_select_parallelism_mode() -> None:
    ep = _ep_config(N_DEVICES)
    assert select_parallelism_mode(1, 1, None) is ParallelismMode.TP_TP
    assert select_parallelism_mode(2, 1, None) is ParallelismMode.TP_TP
    assert select_parallelism_mode(2, 1, ep) is ParallelismMode.TP_EP
    assert select_parallelism_mode(8, 1, ep) is ParallelismMode.TP_EP
    assert select_parallelism_mode(8, 4, ep) is ParallelismMode.DP_EP
    # The selector is a pure mapping; the pipeline model refuses DP without
    # EP (see test_dp_attention_refusals) before the module is built.
    assert select_parallelism_mode(2, 2, None) is ParallelismMode.DP_EP


# --------------------------------------------------------------------------- #
# DP-attention settings and input declarations
# --------------------------------------------------------------------------- #


def test_dp_attention_refusals() -> None:
    """The pipeline refuses DP settings the module build cannot serve.

    DP attention serves one-device replicas only, so the degree must equal
    the device count -- the same constraint `MiMoV2.__init__` enforces,
    checked before the EP config and its buffers exist.
    """
    # DP without EP: attention has nothing to shard its batch with.
    with pytest.raises(ValueError, match="requires expert parallelism"):
        _validate_data_parallel_attention(dp_degree=2, ep_size=1, n_devices=2)
    # dp < n is a dp x tp mesh the module build refuses.
    with pytest.raises(ValueError, match="must equal the device count"):
        _validate_data_parallel_attention(dp_degree=2, ep_size=4, n_devices=4)
    # dp > n cannot fit a replica per device.
    with pytest.raises(ValueError, match="must equal the device count"):
        _validate_data_parallel_attention(dp_degree=4, ep_size=2, n_devices=2)
    # One device per replica over every device passes; EP runs without DP.
    _validate_data_parallel_attention(dp_degree=2, ep_size=2, n_devices=2)
    _validate_data_parallel_attention(dp_degree=1, ep_size=1, n_devices=2)


def _dp_batch_processor(dp: int) -> MiMoV2ModuleV3BatchProcessor:
    """The MiMo batch processor at the given DP degree.

    Only the runtime is read before batching starts, so the config is a
    placeholder.
    """
    runtime = BatchProcessorRuntime(
        pipeline_config=cast(Any, SimpleNamespace()),
        devices=[],
        return_logits=ReturnLogits.LAST_TOKEN,
        max_batch_size=1,
        max_seq_len=256,
        max_batch_input_tokens=256,
        data_parallel_degree=dp,
    )
    return MiMoV2ModuleV3BatchProcessor(
        cast(MiMoV2Config, SimpleNamespace()), runtime
    )


def test_dp_inputs_declared_between_batch_and_kv(checkpoint: Path) -> None:
    """The DP batch-split tensors sit right after the batch inputs.

    MiniMax-M2's placement: after the per-batch tensors, before the KV
    leaves (and before the EP tail that follows them). Both are
    host-resident; the splits are sized dp+1 and the host row offsets
    mirror the device row offsets' shape.
    """
    hf = _hf(checkpoint)
    refs = [DeviceRef.GPU(i) for i in range(N_DEVICES)]
    kv = kv_params(hf, refs, 128)
    base = modulev3_ragged_kv_symbolic_inputs(kv_params=kv, device_refs=refs)
    declared = _dp_batch_processor(N_DEVICES).get_symbolic_inputs(
        kv_params=kv, device_refs=refs
    )
    assert declared[3] == TensorType(
        DType.int64, shape=[N_DEVICES + 1], device=DeviceRef.CPU()
    )
    assert declared[4] == TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=DeviceRef.CPU()
    )
    assert declared[:3] == base[:3]
    assert declared[5:] == base[3:]
    # Without DP the tree is exactly the shared ModuleV3 one.
    assert (
        _dp_batch_processor(1).get_symbolic_inputs(
            kv_params=kv, device_refs=refs
        )
        == base
    )


def test_dp_inputs_occupy_the_runtime_slots() -> None:
    """``MiMoV2Inputs.buffers`` keeps the declared slot order.

    The runtime ABI mirrors the symbolic tree: tokens, the logit count, the
    row offsets, then the DP pair, then the KV leaves and the EP tail.
    """
    tokens = Buffer.from_numpy(np.array([1], dtype=np.int64)).to(CPU())
    row_offsets = Buffer.from_numpy(np.array([0, 1], dtype=np.uint32)).to(CPU())
    return_n_logits = Buffer.from_numpy(np.array([1], dtype=np.int64))
    splits = Buffer.from_numpy(np.array([0, 1, 1], dtype=np.int64))
    inputs = MiMoV2Inputs(
        tokens=tokens,
        input_row_offsets=row_offsets,
        return_n_logits=return_n_logits,
        data_parallel_splits=splits,
        host_input_row_offsets=row_offsets,
    )
    buffers = inputs.buffers
    assert len(buffers) == 5
    assert buffers[0] is tokens
    assert buffers[1] is return_n_logits
    assert buffers[2] is row_offsets
    assert buffers[3] is splits
    assert buffers[4] is row_offsets


# --------------------------------------------------------------------------- #
# The EP MoE's weight placement
# --------------------------------------------------------------------------- #


def _cpu_mesh() -> DeviceMesh:
    """A CPU mesh standing in for the expert-parallel devices."""
    return DeviceMesh((CPU(),) * N_DEVICES, (N_DEVICES,), ("tp",))


def _ep_moe() -> MiMoV2ExpertParallelMoE:
    """The EP MoE on the tiny fixture's shapes; build it under ``F.lazy``.

    The experts' E8M0 scale tensors compile eagerly on a host backend that
    refuses the dtype, so every construction in this file is lazy.
    """
    return MiMoV2ExpertParallelMoE(
        hidden_dim=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_dim=MOE_DIM,
        norm_topk_prob=True,
        quant_config=_mxfp4_experts_config(),
        ep_batch_manager=EPBatchManager(_ep_config(N_DEVICES)),
    )


def test_ep_moe_keeps_global_expert_indices() -> None:
    """The module declares one expert per GLOBAL index, on every layer.

    MAX stacks the experts in-graph; a per-shard reindexing would silently
    load one device's experts into another's slots.
    """
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _ep_moe()
    names = {name for name, _ in layer.parameters}
    expert_weights = {
        name
        for name in names
        if name.startswith("experts.") and name.endswith(".gate_proj.weight")
    }
    assert expert_weights == {
        f"experts.{i}.gate_proj.weight" for i in range(EXPERTS)
    }


def test_ep_moe_holds_whole_experts_per_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each device's stack covers its contiguous experts at full width.

    Contrast the tensor-parallel stacks, whose axis-1 shard splits every
    expert's channels; the EP stacks keep the whole ``moe_dim`` and cut the
    expert axis instead.
    """
    # The interleave picks its scale layout from the host's architecture.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _ep_moe()
        stacked = layer._stacked_weights(_cpu_mesh())
    local = EXPERTS // N_DEVICES
    assert len(stacked) == N_DEVICES
    for gate_up, gate_up_scales, down, down_scales in stacked:
        assert list(gate_up.shape) == [local, 2 * MOE_DIM, HIDDEN // 2]
        assert list(down.shape) == [local, HIDDEN, MOE_DIM // 2]
        # The interleaved weight scales keep the same leading dims.
        assert gate_up_scales.shape[0] == local
        assert down_scales.shape[0] == local


def test_ep_moe_refuses_the_p2p_backend() -> None:
    """MiMo's replicated residual rules the P2P dispatch out (for now)."""
    # The zero-row placeholder below needs a device context, which the
    # virtual-device API provides without a GPU.
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(1)
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = MiMoV2ExpertParallelMoE(
            hidden_dim=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=TOP_K,
            moe_dim=MOE_DIM,
            norm_topk_prob=True,
            quant_config=_mxfp4_experts_config(),
            ep_batch_manager=EPBatchManager(
                _ep_config(N_DEVICES, use_allreduce=False)
            ),
        )
    with (
        pytest.raises(NotImplementedError, match="allreduce"),
        F.lazy(),
        default_device(DeviceRef.GPU().to_device()),
        default_dtype(DType.bfloat16),
    ):
        # The P2P refusal fires before x is read, so a zero-row lazy
        # placeholder satisfies the signature without ignoring its type.
        layer.forward(Tensor.zeros([]), None)


# --------------------------------------------------------------------------- #
# The whole graph at EP2, staged on virtual B200s
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny fixture checkpoint written to disk, as the GPU tests do."""
    path = tmp_path_factory.mktemp("mimo_v3_ep_cpu")
    write(path, tensors())
    return path


def _hf(checkpoint: Path) -> PretrainedConfig:
    return PretrainedConfig(
        **json.loads((checkpoint / "config.json").read_text())
    )


def _ep_config_model(
    checkpoint: Path, *, use_allreduce: bool = True
) -> MiMoV2Config:
    """The tiny fixture's config, with the EP config attached."""
    hf = _hf(checkpoint)
    refs = [DeviceRef.GPU(i) for i in range(N_DEVICES)]
    config = MiMoV2Config.from_huggingface_config(
        hf,
        devices=refs,
        kv_params=kv_params(hf, refs, 128),
        max_seq_len=256,
    )
    config.ep_config = _ep_config(N_DEVICES, use_allreduce)
    return config


def test_no_ep_init_leaves_the_extra_input_tail_defaulted() -> None:
    """The no-EP path keeps the compile input tail's default.

    ``graph_specs`` reads ``_modulev3_extra_input_types`` on every load,
    and the plain (no-EP) build is the default serve path: the override
    must leave the base's empty-tail default in place when it returns
    early, or every non-EP load fails with an AttributeError.
    """
    model = cast(MiMoV2ModuleV3Model, object.__new__(MiMoV2ModuleV3Model))
    model.pipeline_config = cast(
        Any,
        SimpleNamespace(
            model=SimpleNamespace(data_parallel_degree=1),
            runtime=SimpleNamespace(ep_size=1),
        ),
    )
    model.devices = []
    model._init_distributed_runtime(cast(Any, None))
    assert model._modulev3_extra_input_types == []


def test_mimo_v2_ep2_stages_with_the_ep_tail(
    monkeypatch: pytest.MonkeyPatch, checkpoint: Path
) -> None:
    """The EP2 graph stages end to end: mode selection, whole-expert MoE
    layers, the EP dispatch/combine ops in the shared MoE subgraph, and the
    EP input tail after the KV leaves."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)
    # The trace never runs, so it needs no GPU; the pin makes the kernel
    # wrappers trace the SM100 shape paths the B200 box will compile.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)

    config = _ep_config_model(checkpoint)
    assert config.ep_config is not None
    ep_config = config.ep_config
    ep_batch_manager = EPBatchManager(ep_config)
    devices = [ref.to_device() for ref in config.devices]
    mesh = DeviceMesh(tuple(devices), (N_DEVICES,), ("tp",))
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    adapted: dict[str, WeightData] = convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=_hf(checkpoint)
    )

    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config, ep_batch_manager)
        inputs = [
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=config.kv_params,
                device_refs=config.devices,
            ),
            *ep_batch_manager.input_types(),
        ]
        assert model.mode is ParallelismMode.TP_EP
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = model.stage(*inputs, weights=adapted)
        text = str(staged)

    # The EP tail is declared: the atomic counters plus the three pointer
    # tables per buffer group.
    assert len(ep_batch_manager.input_types()) == 2 * N_DEVICES + 3 * 2
    # The shared MoE subgraph still exists and now carries the EP
    # dispatch/combine; the ops are per-device inside the one body.
    assert text.count("mo.graph @mimo_v2_moe") == 1
    assert text.count("mo.call @mimo_v2_moe") == len(config.moe_layers)
    assert text.count("ep.dispatch") >= N_DEVICES
    assert text.count("ep.combine") >= N_DEVICES
    graphs = set(re.findall(r"mo\.graph @([A-Za-z0-9_]+)", text))
    assert graphs - {"MiMoV2"} == {"mimo_v2_moe"}, graphs


def test_mimo_v2_dp2_stages_with_the_batch_split(
    monkeypatch: pytest.MonkeyPatch, checkpoint: Path
) -> None:
    """The DP2 graph stages end to end: the batch-split tensors are read
    between the batch inputs and the KV tree, the split ops carve each
    replica's rows out of the replicated batch, and the MoE stays
    expert-parallel over every device."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)

    # DP attention shards the batch per replica, which is the P2P
    # dispatch's precondition; the allreduce backend has no whole-batch
    # allreduce to complete its combine under DP.
    config = _ep_config_model(checkpoint, use_allreduce=False)
    assert config.ep_config is not None
    config.data_parallel_degree = N_DEVICES
    ep_batch_manager = EPBatchManager(config.ep_config)
    devices = [ref.to_device() for ref in config.devices]
    # The pipeline names the one mesh axis "dp" under data parallelism, so
    # the split placement keys on it.
    mesh = DeviceMesh(tuple(devices), (N_DEVICES,), ("dp",))
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    adapted: dict[str, WeightData] = convert_safetensor_state_dict(
        dict(weights.items()), huggingface_config=_hf(checkpoint)
    )

    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2(config, ep_batch_manager)
        assert model.mode is ParallelismMode.DP_EP
        inputs = [
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=config.kv_params,
                device_refs=config.devices,
            ),
            *ep_batch_manager.input_types(),
        ]
        # The DP batch-split tensors, where the batch processor declares
        # them: after the batch inputs, before the KV leaves.
        dp_inputs = _dp_batch_processor(N_DEVICES).get_symbolic_inputs(
            kv_params=config.kv_params, device_refs=config.devices
        )[3:5]
        inputs = [*inputs[:3], *dp_inputs, *inputs[3:]]
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = model.stage(*inputs, weights=adapted)
        text = str(staged)

    # The split carves each replica's rows from the replicated batch: the
    # slice dims are named after the replica, and the per-device dispatch
    # consumes them.
    assert "_input_split_0" in text and "_input_split_1" in text
    # The MoE keeps its global expert dispatch.
    assert text.count("ep.dispatch") >= N_DEVICES
    assert text.count("ep.combine") >= N_DEVICES
    assert text.count("mo.call @mimo_v2_moe") == len(config.moe_layers)


def test_mimo_v2_refuses_dp_and_orphan_ep_managers(
    checkpoint: Path,
) -> None:
    """DP attention without an EP batch manager is a wiring error, not a
    silent tensor-parallel build, and an EP batch manager without an EP
    config is the same."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)

    config = _ep_config_model(checkpoint)
    config.data_parallel_degree = N_DEVICES
    mesh = DeviceMesh(
        tuple(ref.to_device() for ref in config.devices), (N_DEVICES,), ("tp",)
    )
    with (
        pytest.raises(ValueError, match="no EP batch manager"),
        F.lazy(),
        default_device(mesh),
        default_dtype(config.dtype),
    ):
        MiMoV2(config, None)

    config.data_parallel_degree = 1
    config.ep_config = None
    with (
        pytest.raises(ValueError, match="batch manager"),
        F.lazy(),
        default_device(mesh),
        default_dtype(config.dtype),
    ):
        MiMoV2(config, EPBatchManager(_ep_config(N_DEVICES)))


@pytest.mark.parametrize(
    ("sliding", "full"),
    [(True, False), (False, True), (True, True)],
    ids=["sliding", "full", "both"],
)
def test_mimo_v2_refuses_fused_qkv_under_dp_attention(
    checkpoint: Path, sliding: bool, full: bool
) -> None:
    """Fused QKV preparation has no parity check for DP replicas, which hold
    every chunk, so the build refuses it rather than serve it unchecked."""
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)

    config = _ep_config_model(checkpoint, use_allreduce=False)
    assert config.ep_config is not None
    config.data_parallel_degree = N_DEVICES
    config.fused_sliding_qkv = sliding
    config.fused_full_qkv = full
    mesh = DeviceMesh(
        tuple(ref.to_device() for ref in config.devices), (N_DEVICES,), ("dp",)
    )
    with (
        pytest.raises(NotImplementedError, match="data-parallel attention"),
        F.lazy(),
        default_device(mesh),
        default_dtype(config.dtype),
    ):
        MiMoV2(config, EPBatchManager(config.ep_config))

    config.data_parallel_degree = 1
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        assert MiMoV2(config, EPBatchManager(config.ep_config)).mode is (
            ParallelismMode.TP_EP
        )
