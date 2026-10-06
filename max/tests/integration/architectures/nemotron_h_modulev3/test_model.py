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
"""Nemotron-H's Mamba state on the KV cache, and its weight loading."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import replace
from unittest.mock import Mock

import pytest
from _tiny_config import TINY, TINY_LAYERS, WIDE, hf_config, model_config
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.functional import collective_ops
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    Replicated,
    Sharded,
    auto_reshard,
)
from max.experimental.tensor import Tensor, default_device, default_dtype
from max.graph import DeviceRef
from max.pipelines.architectures.nemotron_h_modulev3.layers.attention import (
    NemotronHAttention,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.mamba2 import (
    NemotronHMamba2Mixer,
)
from max.pipelines.architectures.nemotron_h_modulev3.layers.moe import (
    NemotronHMoE,
)
from max.pipelines.architectures.nemotron_h_modulev3.memory_planner import (
    NemotronHMemoryPlanner,
)
from max.pipelines.architectures.nemotron_h_modulev3.model_config import (
    LayerKind,
    NemotronHConfig,
)
from max.pipelines.architectures.nemotron_h_modulev3.nemotron_h import (
    NemotronH,
)
from max.pipelines.architectures.nemotron_h_modulev3.quantization import (
    ModuleFormat,
    NemotronHQuantScheme,
)
from max.pipelines.kv_cache.paged_kv_cache.jenga_block_pool import (
    plan_jenga_geometry,
)
from max.pipelines.lib.interfaces.batch_processor import (
    modulev3_ragged_kv_symbolic_inputs,
)

MIB = 1024**2


def _model(config: NemotronHConfig) -> NemotronH:
    with F.lazy(), default_dtype(DType.bfloat16):
        model = NemotronH(config)
        model.to(CPU())
    return model


@pytest.mark.parametrize("n_devices", [1, 2])
def test_lightning_state_tiles_one_huge_page_exactly(n_devices: int) -> None:
    """23 Mamba layers is prime, so the huge page is 414 MiB and unpadded.

    Jenga addresses a state's layers as consecutive rows of its page, which
    holds only while no page is padded. Each of two devices holds half of
    every page, so its huge page is half as large.
    """
    # Nemotron-3.5-Lightning's layer schedule and mixer shapes.
    layers = [
        {"M": "mamba", "E": "moe", "*": "attention"}[c]
        for c in "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    ]
    config = model_config(
        layers,
        dict(
            hidden_size=2688,
            num_attention_heads=32,
            num_key_value_heads=2,
            head_dim=128,
            mamba_num_heads=64,
            mamba_head_dim=64,
            n_groups=8,
            ssm_state_size=128,
            conv_kernel=4,
        ),
        n_devices=n_devices,
    )
    leaves = config.kv_params.leaves()
    page_bytes = {key: leaf.bytes_per_page for key, leaf in leaves.items()}
    assert page_bytes == {
        "attn.full_group": 768 * 1024 // n_devices,
        "mamba/conv": 23 * 6144 * 3 * 2 // n_devices,
        "mamba/ssm": 23 * 64 * 64 * 128 * 4 // n_devices,
    }

    geometry = plan_jenga_geometry(
        100 * 1024 * MIB,
        page_bytes,
        {key: leaf.row_bytes for key, leaf in leaves.items()},
    )

    assert geometry.huge_page_bytes == 414 * MIB // n_devices
    assert geometry.ratios == {
        "attn.full_group": 552,
        "mamba/conv": 512,
        "mamba/ssm": 9,
    }
    assert geometry.padded_sizes == page_bytes


def test_each_mamba_layer_reads_its_own_state_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharing a row would load, run and return fluent nonsense.

    The Mamba layers share one subgraph and slice their pool rows inside it,
    so each call must pass its own layer index.
    """
    config = model_config(TINY_LAYERS, TINY)
    model = _model(config)
    layers: list[int] = []
    original: Callable[..., Tensor] = F.constant

    def record(
        value: object, dtype: DType, *args: object, **kwargs: object
    ) -> Tensor:
        if dtype == DType.int64 and isinstance(value, int):
            layers.append(value)
        return original(value, dtype, *args, **kwargs)

    monkeypatch.setattr(F, "constant", record)
    model.trace(
        *modulev3_ragged_kv_symbolic_inputs(
            kv_params=config.kv_params, device_refs=config.devices
        )
    )

    # One index per Mamba layer, in order.
    assert layers == [0, 1, 2]


def _model_on_devices(config: NemotronHConfig) -> NemotronH:
    n = len(config.devices)
    mesh = DeviceMesh((CPU(),) * n, (n,), ("tp",))
    with F.lazy(), default_dtype(DType.bfloat16), default_device(mesh):
        return NemotronH(config)


def _trace_on_devices(
    monkeypatch: pytest.MonkeyPatch,
    n: int,
    dims: dict[str, int] = TINY,
    fp8_mamba: bool = False,
) -> tuple[list[DType], int, list[str]]:
    """Traces the tiny model at TP=``n`` on a CPU mesh.

    Only the allreduce of partial sums may reshard, so any other collective
    fails the trace.

    Returns:
        The dtypes broadcast to the other devices, the allreduce count, and
        the reports of each peer copy onto the mesh.
    """
    config = model_config(TINY_LAYERS, dims, n_devices=n)
    if fp8_mamba:
        config.quant_scheme = NemotronHQuantScheme(
            quantized={
                f"{mixer}.{proj}": ModuleFormat.FP8_STATIC_TENSOR
                for mixer in config.mixers(LayerKind.MAMBA)
                for proj in ("in_proj", "out_proj")
            }
        )
        config.fp8_mamba_projections = True
    broadcast: list[DType] = []
    allreduces: list[Tensor] = []
    allreduce_sum: Callable[..., Tensor] = collective_ops.allreduce_sum

    def record_broadcast(t: Tensor, mesh: DeviceMesh) -> Tensor:
        broadcast.append(t.dtype)
        # A CPU mesh has no signal buffers to run the collective with.
        return t.to(mesh)

    def record_allreduce(t: Tensor, *args: object, **kwargs: object) -> Tensor:
        allreduces.append(t)
        return allreduce_sum(t, *args, **kwargs)

    model = _model_on_devices(config)
    monkeypatch.setattr(F, "distributed_broadcast", record_broadcast)
    monkeypatch.setattr(collective_ops, "allreduce_sum", record_allreduce)
    # "raise" would refuse the allreduce too, and "silent" hides a peer
    # copy, which is a move but not a placement transition.
    with (
        warnings.catch_warnings(record=True) as moves,
        auto_reshard({(Partial, Replicated)}, mode="warn"),
    ):
        warnings.simplefilter("always")
        model.trace(
            *modulev3_ragged_kv_symbolic_inputs(
                kv_params=config.kv_params, device_refs=config.devices
            )
        )
    peer_copies = [
        str(m.message) for m in moves if "cross_mesh_transfer" in str(m.message)
    ]
    return broadcast, len(allreduces), peer_copies


@pytest.mark.parametrize("n", [2, 4, 8])
def test_inputs_reach_the_other_devices_by_collective(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    """Device graph capture records each device's stream on its own.

    A peer copy of the inputs makes one device's stream wait on another's,
    which invalidates the capture, so the tokens and row offsets reach the
    other devices through the broadcast collective. Past two devices, so do
    the MoE layer's expert ids and weights, which device 0 picks.
    """
    broadcast, _, _ = _trace_on_devices(monkeypatch, n, WIDE)
    routing = [DType.int32, DType.float32] if n > 2 else []
    assert broadcast == [DType.int64, DType.uint32, *routing]


@pytest.mark.parametrize("fp8_mamba", [False, True])
@pytest.mark.parametrize("n", [2, 4, 8])
def test_each_sharded_mixer_reduces_once(
    monkeypatch: pytest.MonkeyPatch, n: int, fp8_mamba: bool
) -> None:
    """Attention, the MoE and the Mamba mixer each end in one allreduce,
    and nothing else moves activations between devices.

    The routed experts and the shared expert both return partial sums, which
    the residual add reduces together. The three Mamba layers share one
    subgraph, so it is traced once. The FP8 Mamba projections keep the
    placements of the BF16 ones.
    """
    _, allreduces, peer_copies = _trace_on_devices(
        monkeypatch, n, WIDE, fp8_mamba
    )
    assert allreduces == 3
    assert peer_copies == []


def test_each_device_holds_a_whole_kv_head() -> None:
    """With more devices than KV heads, each head repeats per device."""
    config = model_config(TINY_LAYERS, WIDE, n_devices=4)
    attention = _model_on_devices(config).backbone.layers[3].mixer
    assert isinstance(attention, NemotronHAttention)
    for proj in (attention.qkv_proj.k_proj, attention.qkv_proj.v_proj):
        # Four heads of eight channels: one per device.
        assert tuple(int(d) for d in proj.weight.shape) == (32, 32)
        assert proj.weight.mapping.placements == (Sharded(0),)


def test_moe_shards_by_expert() -> None:
    """Each device holds half the routed experts and half the shared expert.

    Splitting each expert's channels instead would leave the W4A4 down
    projection 58 block scales wide, which its interleaved layout cannot
    hold.
    """
    model = _model_on_devices(model_config(TINY_LAYERS, TINY, n_devices=2))
    moe = model.backbone.layers[1].mixer
    assert isinstance(moe, NemotronHMoE)

    def placements(t: Tensor) -> tuple[object, ...]:
        return t.mapping.placements

    assert placements(moe.up_weight) == (Sharded(0),)
    assert placements(moe.down_weight) == (Sharded(0),)
    assert placements(moe.shared_experts.up_proj.weight) == (Sharded(0),)
    assert placements(moe.shared_experts.down_proj.weight) == (Sharded(1),)
    # Every device routes every token.
    assert placements(moe.gate.weight) == (Replicated(),)


def test_mamba_shards_by_head() -> None:
    """Each device runs half the heads and the groups they read."""
    model = _model_on_devices(model_config(TINY_LAYERS, TINY, n_devices=2))
    mamba = model.backbone.layers[0].mixer
    assert isinstance(mamba, NemotronHMamba2Mixer)
    for param in (
        mamba.in_proj.weight,
        mamba.conv1d.weight,
        mamba.conv1d.bias,
        mamba.A_log,
        mamba.D,
        mamba.dt_bias,
        mamba.norm.weight,
    ):
        assert param.mapping.placements == (Sharded(0),)
    assert mamba.out_proj.weight.mapping.placements == (Sharded(1),)


def test_fp8_mamba_projections_keep_their_checkpoint_tensors() -> None:
    """Only a mixer with both projections in FP8 runs them in FP8."""
    config = model_config(TINY_LAYERS, TINY)
    fp8 = ModuleFormat.FP8_STATIC_TENSOR
    config.quant_scheme = NemotronHQuantScheme(
        quantized={
            "backbone.layers.0.mixer.in_proj": fp8,
            "backbone.layers.0.mixer.out_proj": fp8,
            "backbone.layers.2.mixer.in_proj": fp8,
        }
    )
    config.fp8_mamba_projections = True
    assert config.fp8_mamba_mixers() == {"backbone.layers.0.mixer"}

    weights = dict(_model(config).parameters)
    for proj in ("in_proj", "out_proj"):
        prefix = f"backbone.layers.0.mixer.{proj}"
        assert weights[f"{prefix}.weight"].dtype == DType.float8_e4m3fn
        for scale in ("weight_scale", "input_scale"):
            assert weights[f"{prefix}.{scale}"].device == CPU()
    assert (
        weights["backbone.layers.2.mixer.in_proj.weight"].dtype
        == DType.bfloat16
    )
    assert "backbone.layers.2.mixer.in_proj.input_scale" not in weights


def test_loading_is_strict() -> None:
    """A missing weight or a stray scale fails the load by name."""
    config = model_config(TINY_LAYERS, TINY)
    model = _model(config)
    weights = dict(model.parameters)
    names = set(weights)

    missing = "backbone.layers.1.mixer.down_weight"
    del weights[missing]
    with pytest.raises(KeyError, match=missing):
        model.compile(weights=weights)
    # A scale beside a BF16 weight.
    stray = "backbone.layers.0.mixer.in_proj"
    with pytest.raises(ValueError, match=stray):
        config.quant_scheme.check_weights(names | {f"{stray}.weight_scale"})


def test_w4a4_selects_the_moe_mixers_with_nvfp4_experts() -> None:
    config = model_config(TINY_LAYERS, TINY)
    nvfp4 = {
        f"backbone.layers.1.mixer.experts.{e}.{proj}": (
            ModuleFormat.NVFP4_WEIGHT_ONLY
        )
        for e in range(config.num_experts)
        for proj in ("up_proj", "down_proj")
    }
    config = replace(
        config, w4a4_experts=True, quant_scheme=NemotronHQuantScheme(nvfp4)
    )
    assert config.w4a4_mixers() == {"backbone.layers.1.mixer"}


@pytest.mark.parametrize(
    "field, value, n_devices, sharded",
    [
        ("num_attention_heads", 3, 2, "3 attention heads"),
        ("n_routed_experts", 5, 2, "5 routed experts"),
        ("moe_shared_expert_intermediate_size", 25, 2, "25 shared expert"),
        ("mamba_num_heads", 3, 2, "3 Mamba heads"),
        ("n_groups", 3, 2, "3 Mamba groups"),
        # Three devices neither divide two KV heads nor are a multiple.
        ("num_key_value_heads", 2, 3, "2 KV heads"),
        ("num_key_value_heads", 3, 2, "3 KV heads"),
    ],
)
def test_sharded_dims_must_divide_across_devices(
    field: str, value: int, n_devices: int, sharded: str
) -> None:
    """Each device runs an equal share of every sharded dimension, and a
    whole number of KV heads."""
    config = model_config(TINY_LAYERS, TINY)
    hf = hf_config(TINY_LAYERS, TINY)
    setattr(hf, field, value)
    with pytest.raises(ValueError, match=sharded):
        NemotronHConfig.from_huggingface(
            hf,
            kv_params=config.kv_params,
            devices=[DeviceRef.GPU(i) for i in range(n_devices)],
            max_seq_len=256,
        )


def _bytes_on_devices(model: NemotronH, n: int) -> int:
    """Sums the parameter bytes across devices, from their placements."""
    return sum(
        math.prod(int(d) for d in param.shape)
        * param.dtype.size_in_bytes
        * (n if all(isinstance(p, Replicated) for p in param.placements) else 1)
        for _, param in model.parameters
    )


@pytest.mark.parametrize("n", [2, 4, 8])
def test_weights_are_planned_as_the_modules_place_them(n: int) -> None:
    """Sharded weights are counted once, replicated ones per device, and a
    repeated KV head once per device that holds it."""
    pipeline = Mock()
    pipeline.model.weights_size.return_value = 1000
    one_config = model_config(TINY_LAYERS, WIDE)
    config = model_config(TINY_LAYERS, WIDE, n_devices=n)

    one = NemotronHMemoryPlanner(one_config).estimate_weights_size(pipeline)
    planned = NemotronHMemoryPlanner(config).estimate_weights_size(pipeline)

    placed = _bytes_on_devices(_model_on_devices(config), n)
    assert planned - one == placed - _bytes_on_devices(_model(one_config), 1)
    assert planned > one
