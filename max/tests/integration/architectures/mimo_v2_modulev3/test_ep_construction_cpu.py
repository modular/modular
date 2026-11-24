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
from pathlib import Path

import numpy as np
import pytest
from expert_stacks import (
    PROJECTIONS,
    check_staged_expert_stacks,
    gate_up,
    load_random_experts,
    realize,
)
from max import functional as F
from max import tree
from max.driver import (
    CPU,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import SafetensorWeights, WeightData
from max.nn import _graph_ops as kernels
from max.nn.ep.init import payload_types
from max.nn.ep_config import EPConfig
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.mimo_v2_modulev3.layers.moe import (
    MiMoV2ExpertParallelMoE,
)
from max.pipelines.architectures.mimo_v2_modulev3.mimo_v2 import (
    MiMoV2Decoder,
    ParallelismMode,
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
from max.pipelines.lib.module_registration import (
    _DecoderGraph,
    decoder_input_types,
)
from max.tensor import (
    Tensor,
    default_device,
    default_dtype,
)
from max.tensor.sharding import DeviceMesh
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
# The EP MoE's weight placement
# --------------------------------------------------------------------------- #


def _cpu_mesh(n_devices: int = N_DEVICES) -> DeviceMesh:
    """A CPU mesh standing in for the expert-parallel devices."""
    return DeviceMesh((CPU(),) * n_devices, (n_devices,), ("tp",))


def _ep_moe(n_devices: int = N_DEVICES) -> MiMoV2ExpertParallelMoE:
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
        ep_config=_ep_config(n_devices),
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


@pytest.mark.parametrize("n_devices", [2, 4])
def test_ep_moe_holds_whole_experts_per_device(
    monkeypatch: pytest.MonkeyPatch, n_devices: int
) -> None:
    """Device ``d`` holds its contiguous block of whole experts, byte for
    byte.

    Contrast the tensor-parallel stacks, whose axis-1 shard splits every
    expert's channels; the EP stacks keep the whole ``moe_dim`` and cut the
    expert axis instead.
    """
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _ep_moe(n_devices)
    ref = load_random_experts(layer, monkeypatch)
    got = realize(layer, _cpu_mesh(n_devices))
    local = EXPERTS // n_devices
    assert len(got) == n_devices
    for d, (gate_up_w, gate_up_scales, down, down_scales) in enumerate(got):
        block = np.s_[d * local : (d + 1) * local]
        (gate, gate_s), (up, up_s), (down_ref, down_s) = (
            ref[name] for name in PROJECTIONS
        )
        np.testing.assert_array_equal(
            gate_up_w, gate_up(gate[block], up[block])
        )
        np.testing.assert_array_equal(
            gate_up_scales, gate_up(gate_s[block], up_s[block])
        )
        np.testing.assert_array_equal(down, down_ref[block])
        np.testing.assert_array_equal(down_scales, down_s[block])


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
            ep_config=_ep_config(N_DEVICES, use_allreduce=False),
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
    ep_inputs = tree.leaves(payload_types(ep_config))
    devices = [ref.to_device() for ref in config.devices]
    mesh = DeviceMesh(tuple(devices), (N_DEVICES,), ("tp",))
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    adapted: dict[str, WeightData] = convert_safetensor_state_dict(
        dict(weights.items())
    )

    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2Decoder(config)
        inputs = decoder_input_types(
            config.kv_params,
            config.devices[0],
            ep=payload_types(ep_config),
            input_layout=MiMoV2Decoder.input_layout,
        )
        assert model.mode is ParallelismMode.TP_EP
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = _DecoderGraph(model, ep_mesh=mesh).stage(
            inputs,
            weights=adapted,
            name=MiMoV2Decoder.graph_name,
        )
        text = str(staged)

    # The EP tail is declared: the atomic counters plus the three pointer
    # tables per buffer group.
    assert len(ep_inputs) == 2 * N_DEVICES + 3 * 2
    # The shared MoE subgraph still exists and now carries the EP
    # dispatch/combine; the ops are per-device inside the one body.
    assert text.count("mo.graph @mimo_v2_moe") == 1
    assert text.count("mo.call @mimo_v2_moe") == len(config.moe_layers)
    assert text.count("ep.dispatch") >= N_DEVICES
    assert text.count("ep.combine") >= N_DEVICES
    graphs = set(re.findall(r"mo\.graph @([A-Za-z0-9_]+)", text))
    assert graphs - {"MiMoV2"} == {"mimo_v2_moe"}, graphs
    check_staged_expert_stacks(text, N_DEVICES, expert_parallel=True)


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
    ep_config = config.ep_config
    devices = [ref.to_device() for ref in config.devices]
    # The pipeline names the one mesh axis "dp" under data parallelism, so
    # the split placement keys on it.
    mesh = DeviceMesh(tuple(devices), (N_DEVICES,), ("dp",))
    weights = SafetensorWeights(sorted(checkpoint.glob("*.safetensors")))
    adapted: dict[str, WeightData] = convert_safetensor_state_dict(
        dict(weights.items())
    )

    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        model = MiMoV2Decoder(config)
        assert model.mode is ParallelismMode.DP_EP
        # The registered decoder's tree: the DP batch-split tensors
        # between the batch inputs and the KV leaves, the EP tail last.
        inputs = decoder_input_types(
            config.kv_params,
            config.devices[0],
            data_parallel_degree=N_DEVICES,
            ep=payload_types(ep_config),
            input_layout=MiMoV2Decoder.input_layout,
        )
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        staged, _registry = _DecoderGraph(model, ep_mesh=mesh).stage(
            inputs,
            weights=adapted,
            name=MiMoV2Decoder.graph_name,
        )
        text = str(staged)

    # The split carves each replica's rows from the replicated batch: the
    # slice dims are named after the replica, and the per-device dispatch
    # consumes them.
    assert "_input_split_0" in text and "_input_split_1" in text
    # The MoE keeps its global expert dispatch.
    assert text.count("ep.dispatch") >= N_DEVICES
    assert text.count("ep.combine") >= N_DEVICES
    assert text.count("mo.call @mimo_v2_moe") == len(config.moe_layers)


def test_mimo_v2_refuses_dp_without_an_ep_config(
    checkpoint: Path,
) -> None:
    """DP attention without an EP config is a wiring error, not a silent
    tensor-parallel build: the registered decoder reads its EP config from
    ``config.ep_config`` alone, so a DP build with no EP config reaches the
    module with none and is refused there.

    The reverse wiring -- an EP config without an EP mode -- has no caller
    since the shims were deleted: the registered decoder derives both from
    the config, so no construction can pair the two.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(N_DEVICES)

    config = _ep_config_model(checkpoint)
    config.data_parallel_degree = N_DEVICES
    config.ep_config = None
    mesh = DeviceMesh(
        tuple(ref.to_device() for ref in config.devices), (N_DEVICES,), ("tp",)
    )
    with (
        pytest.raises(ValueError, match="no EP config"),
        F.lazy(),
        default_device(mesh),
        default_dtype(config.dtype),
    ):
        MiMoV2Decoder(config)


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
        MiMoV2Decoder(config)

    config.data_parallel_degree = 1
    with F.lazy(), default_device(mesh), default_dtype(config.dtype):
        assert MiMoV2Decoder(config).mode is (ParallelismMode.TP_EP)
