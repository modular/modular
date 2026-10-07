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
from _tiny_config import (
    NVFP4_DIMS,
    TINY,
    TINY_LAYERS,
    WIDE,
    dense_nvfp4_scheme,
    hf_config,
    lightning_scheme,
    model_config,
)
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
from max.nn import kernels
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
    NemotronHConfig,
    moe_channels_per_device,
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


Scheme = Callable[[NemotronHConfig], NemotronHQuantScheme]


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
    scheme: Scheme | None = None,
) -> tuple[list[DType], int, list[str]]:
    """Traces the tiny model at TP=``n`` on a CPU mesh.

    Only the allreduce of partial sums may reshard, so any other collective
    fails the trace.

    Args:
        monkeypatch: Replaces the collectives with recorders.
        n: The number of devices.
        dims: The tiny model's dimensions.
        scheme: Makes the model's quantization from its config.

    Returns:
        The dtypes broadcast to the other devices, the allreduce count, and
        the reports of each peer copy onto the mesh.
    """
    config = model_config(TINY_LAYERS, dims, n_devices=n)
    if scheme is not None:
        config.quant_scheme = scheme(config)
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


@pytest.mark.parametrize("n", [2, 4, 8])
def test_each_sharded_mixer_reduces_once(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    """Attention, the MoE and the Mamba mixer each end in one allreduce,
    and nothing else moves activations between devices.

    The routed experts and the shared expert both return partial sums, which
    the residual add reduces together. The three Mamba layers share one
    subgraph, so it is traced once.
    """
    _, allreduces, peer_copies = _trace_on_devices(monkeypatch, n, WIDE)
    assert allreduces == 3
    assert peer_copies == []


@pytest.mark.parametrize("scheme", [lightning_scheme, dense_nvfp4_scheme])
def test_quantized_mixers_reduce_once(
    monkeypatch: pytest.MonkeyPatch, scheme: Scheme
) -> None:
    """The FP8 Mamba projections and the NVFP4 experts and LM head keep the
    placements of the BF16 layers, whether the shared expert runs as routed
    experts or as its own linear layers."""
    # NVFP4 runs only on SM100, and the quantize op picks its scale layout
    # from the host's GPU. The trace never runs, so it needs no GPU.
    monkeypatch.setattr(kernels, "_is_sm10x_gpu", lambda: True)
    _, allreduces, peer_copies = _trace_on_devices(
        monkeypatch, 2, NVFP4_DIMS, scheme
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


def _with_w4a4_experts(config: NemotronHConfig) -> NemotronHConfig:
    """Returns ``config`` with the first MoE mixer's experts in W4A4."""
    nvfp4 = {
        f"backbone.layers.1.mixer.experts.{e}.{proj}": (
            ModuleFormat.NVFP4_WEIGHT_ONLY
        )
        for e in range(config.num_experts)
        for proj in ("up_proj", "down_proj")
    }
    return replace(config, quant_scheme=NemotronHQuantScheme(nvfp4))


@pytest.mark.parametrize("w4a4", [False, True])
def test_moe_splits_each_experts_channels(w4a4: bool) -> None:
    """Each device holds its share of every routed expert's channels,
    padded to 64, and half the shared expert.

    The up projection splits its rows and the down projection its columns.
    The W4A4 block scales stay on the host in the same padded blocks, for
    the graph to interleave each device's block at init.
    """
    # Hidden size 64 gives the W4A4 up projection whole scale atoms.
    config = model_config(TINY_LAYERS, TINY | dict(hidden_size=64), n_devices=2)
    if w4a4:
        config = _with_w4a4_experts(config)
    moe = _model_on_devices(config).backbone.layers[1].mixer
    assert isinstance(moe, NemotronHMoE)

    def layout(t: Tensor) -> tuple[tuple[int, ...], tuple[object, ...]]:
        return tuple(int(d) for d in t.shape), t.mapping.placements

    # Four experts of 16 channels: 8 per device, padded to 64.
    if w4a4:
        assert layout(moe.up_weight) == ((4, 128, 32), (Sharded(1),))
        assert layout(moe.down_weight) == ((4, 64, 64), (Sharded(2),))
        assert tuple(int(d) for d in moe.up_block_scale.shape) == (4, 128, 4)
        assert tuple(int(d) for d in moe.down_block_scale.shape) == (4, 64, 8)
        for scales in (moe.up_block_scale, moe.down_block_scale):
            assert scales.device == CPU()
        for scale in (moe.up_scale, moe.down_scale):
            assert layout(scale) == ((4,), (Replicated(),))
    else:
        assert layout(moe.up_weight) == ((4, 128, 64), (Sharded(1),))
        assert layout(moe.down_weight) == ((4, 64, 128), (Sharded(2),))
    assert moe.shared_experts is not None
    assert moe.shared_experts.up_proj.weight.placements == (Sharded(0),)
    assert moe.shared_experts.down_proj.weight.placements == (Sharded(1),)
    # Every device routes every token.
    assert moe.gate.weight.placements == (Replicated(),)


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


def test_quantized_modules_keep_their_checkpoint_tensors() -> None:
    """Each quantized dense module is built in its stored format, whatever
    the format of its neighbors."""
    config = model_config(TINY_LAYERS, TINY)
    config.quant_scheme = NemotronHQuantScheme(
        quantized={
            "backbone.layers.0.mixer.in_proj": ModuleFormat.FP8_STATIC_TENSOR,
            "backbone.layers.0.mixer.out_proj": ModuleFormat.FP8_STATIC_TENSOR,
            "backbone.layers.2.mixer.in_proj": ModuleFormat.FP8_STATIC_TENSOR,
        }
    )

    weights = dict(_model(config).parameters)
    for module in (
        "backbone.layers.0.mixer.in_proj",
        "backbone.layers.0.mixer.out_proj",
        "backbone.layers.2.mixer.in_proj",
    ):
        assert weights[f"{module}.weight"].dtype == DType.float8_e4m3fn
        for scale in ("weight_scale", "input_scale"):
            assert weights[f"{module}.{scale}"].device == CPU()
    out_proj = "backbone.layers.2.mixer.out_proj"
    assert weights[f"{out_proj}.weight"].dtype == DType.bfloat16
    assert f"{out_proj}.input_scale" not in weights


def test_nvfp4_linears_are_placed_like_bf16_ones() -> None:
    config = model_config(TINY_LAYERS, NVFP4_DIMS, n_devices=2)
    config.quant_scheme = dense_nvfp4_scheme(config)
    weights = dict(_model_on_devices(config).parameters)
    shared = "backbone.layers.1.mixer.shared_experts"
    for module, placement in (
        (f"{shared}.up_proj", Sharded(0)),
        (f"{shared}.down_proj", Sharded(1)),
        ("lm_head", Replicated()),
    ):
        in_dim, out_dim = config.linear_shape(module)
        weight = weights[f"{module}.weight"]
        assert weight.dtype == DType.uint8
        assert [int(d) for d in weight.shape] == [out_dim, in_dim // 2]
        assert weight.placements == (placement,)
        block = weights[f"{module}.weight_scale"]
        assert block.dtype == DType.float8_e4m3fn
        assert block.placements == (
            Replicated() if placement == Replicated() else Sharded(0),
        )
        assert weights[f"{module}.weight_scale_2"].dtype == DType.float32


@pytest.mark.parametrize(
    "quantized, error",
    [
        # An attention projection is built in BF16 only.
        (
            {"backbone.layers.3.mixer.q_proj": ModuleFormat.FP8_STATIC_TENSOR},
            "q_proj",
        ),
        # Half of a mixer's routed experts in NVFP4.
        (
            {
                f"backbone.layers.1.mixer.experts.0.{proj}": (
                    ModuleFormat.NVFP4_WEIGHT_ONLY
                )
                for proj in ("up_proj", "down_proj")
            },
            "routed experts",
        ),
        # FP8 routed experts.
        (
            {
                f"backbone.layers.1.mixer.experts.{e}.{proj}": (
                    ModuleFormat.FP8_STATIC_TENSOR
                )
                for e in range(4)
                for proj in ("up_proj", "down_proj")
            },
            "routed experts",
        ),
    ],
)
def test_unsupported_quantized_modules_are_refused(
    quantized: dict[str, ModuleFormat], error: str
) -> None:
    config = model_config(TINY_LAYERS, TINY)
    config.quant_scheme = NemotronHQuantScheme(quantized)
    with pytest.raises(NotImplementedError, match=error):
        config.check_quantized_modules()


def test_nvfp4_row_parallel_input_splits_into_whole_scale_blocks() -> None:
    """256 shared-expert channels split into whole 64-column blocks on two
    devices, but not on eight."""
    for n, ok in ((2, True), (8, False)):
        config = model_config(TINY_LAYERS, NVFP4_DIMS, n_devices=n)
        config.quant_scheme = dense_nvfp4_scheme(config)
        if ok:
            config.check_quantized_modules()
            continue
        with pytest.raises(
            ValueError,
            match=r"shared_experts\.down_proj' of this checkpoint on 1, 2 or 4 "
            r"devices, not 8",
        ):
            config.check_quantized_modules()


@pytest.mark.parametrize("n", [1, 2])
def test_nvfp4_shared_expert_runs_as_routed_experts(n: int) -> None:
    """The shared expert's two expert-wide slices stack after the routed
    experts, so their channels split across devices with the routed
    experts'. The block scales load on the host, where the graph interleaves
    them at init."""
    config = model_config(TINY_LAYERS, NVFP4_DIMS, n_devices=n)
    config.quant_scheme = lightning_scheme(config)
    mixer = "backbone.layers.1.mixer"
    assert config.shared_expert_slices(mixer) == 2
    config.check_quantized_modules()

    model = _model_on_devices(config) if n > 1 else _model(config)
    moe = model.backbone.layers[1].mixer
    assert isinstance(moe, NemotronHMoE)
    assert moe.shared_experts is None
    weights = dict(model.parameters)
    assert not any(".shared_experts." in name for name in weights)
    stacked = config.num_experts + 2
    for proj, axis in (("up", 1), ("down", 2)):
        for suffix in ("weight", "block_scale", "scale"):
            param = weights[f"{mixer}.{proj}_{suffix}"]
            assert int(param.shape[0]) == stacked
            if suffix == "block_scale":
                assert param.device == CPU()
            elif n > 1 and suffix == "weight":
                assert param.placements == (Sharded(axis),)


def test_a_shared_expert_runs_alone_unless_it_slices_evenly() -> None:
    mixer = "backbone.layers.1.mixer"
    # 320 channels are not a whole number of 128-channel routed experts.
    dims = NVFP4_DIMS | {"moe_shared_expert_intermediate_size": 320}
    config = model_config(TINY_LAYERS, dims, n_devices=2)
    config.quant_scheme = lightning_scheme(config)
    assert config.shared_expert_slices(mixer) == 0
    # BF16 routed experts run the BF16 grouped matmul.
    config = model_config(TINY_LAYERS, NVFP4_DIMS)
    config.quant_scheme = dense_nvfp4_scheme(config)
    assert config.shared_expert_slices(mixer) == 0


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
    config = _with_w4a4_experts(model_config(TINY_LAYERS, TINY))
    assert config.w4a4_mixers() == {"backbone.layers.1.mixer"}


@pytest.mark.parametrize(
    "field, value, n_devices, sharded",
    [
        ("num_attention_heads", 3, 2, "3 attention heads"),
        ("moe_intermediate_size", 15, 2, "15 routed expert channels"),
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


@pytest.mark.parametrize(
    "n, channels", [(1, 1856), (2, 960), (4, 512), (8, 256)]
)
def test_each_device_pads_its_expert_channels(n: int, channels: int) -> None:
    """Lightning's 1856 channels split into shares padded to 64."""
    assert moe_channels_per_device(1856, n) == channels


@pytest.mark.parametrize(
    "n, split_by_block", [(1, True), (2, True), (4, True), (8, False)]
)
def test_nvfp4_experts_split_between_blocks(
    n: int, split_by_block: bool
) -> None:
    """At eight devices, Lightning's 232 channels per device end mid block,
    so NVFP4 routed experts are refused rather than dequantized."""
    config = model_config(
        TINY_LAYERS, WIDE | dict(moe_intermediate_size=1856), n_devices=n
    )
    assert config.nvfp4_experts_split_by_block is split_by_block
    config.check_quantized_modules()
    config = _with_w4a4_experts(config)
    if split_by_block:
        config.check_quantized_modules()
        return
    with pytest.raises(ValueError, match="not whole 16-channel blocks"):
        config.check_quantized_modules()


def _on_every_device(param: Tensor, n: int) -> bool:
    """Returns whether each of ``n`` devices holds a whole copy of ``param``.

    The FP8 scales are pinned to the host, outside the model's mesh.
    """
    return param.mesh.num_devices == n and all(
        isinstance(p, Replicated) for p in param.placements
    )


def _interleaved_bytes(name: str, param: Tensor, n: int) -> int:
    """Returns the device bytes of host W4A4 block scales once interleaved.

    Each device interleaves its block, the rows of ``up`` or the columns of
    ``down``, padding rows to whole granules of 128.
    """
    experts, rows, cols = (int(d) for d in param.shape)
    if name.endswith(".up_block_scale"):
        rows //= n
    else:
        cols //= n
    return n * experts * math.ceil(rows / 128) * 128 * cols


def _bytes_on_devices(model: NemotronH, n: int) -> int:
    """Sums the parameter bytes across devices, from their placements.

    The W4A4 block scales count as the devices hold them, interleaved.
    """
    total = 0
    for name, param in model.parameters:
        if name.endswith("_block_scale"):
            total += _interleaved_bytes(name, param, n)
            continue
        total += (
            math.prod(int(d) for d in param.shape)
            * param.dtype.size_in_bytes
            * (n if _on_every_device(param, n) else 1)
        )
    return total


W4A4_DIMS = WIDE | dict(hidden_size=64, moe_intermediate_size=128)


def _scheme_with_w4a4_experts(config: NemotronHConfig) -> NemotronHQuantScheme:
    return _with_w4a4_experts(config).quant_scheme


@pytest.mark.parametrize(
    "n, dims, scheme",
    [
        (2, WIDE, None),
        (4, WIDE, None),
        (8, WIDE, None),
        (2, NVFP4_DIMS, lightning_scheme),
        (2, NVFP4_DIMS, dense_nvfp4_scheme),
        # Whole W4A4 scale atoms in both projections, and whole NVFP4 blocks
        # on each of eight devices.
        (2, W4A4_DIMS, _scheme_with_w4a4_experts),
        (4, W4A4_DIMS, _scheme_with_w4a4_experts),
        (8, W4A4_DIMS, _scheme_with_w4a4_experts),
    ],
)
def test_weights_are_planned_as_the_modules_place_them(
    n: int, dims: dict[str, int], scheme: Scheme | None
) -> None:
    """Sharded weights are counted once, replicated ones per device, and a
    repeated KV head once per device that holds it. NVFP4 block scales are
    counted with each device's padding."""
    pipeline = Mock()
    pipeline.model.weights_size.return_value = 1000
    one_config = model_config(TINY_LAYERS, dims)
    config = model_config(TINY_LAYERS, dims, n_devices=n)
    if scheme is not None:
        one_config.quant_scheme = scheme(one_config)
        config.quant_scheme = scheme(config)

    one = NemotronHMemoryPlanner(one_config).estimate_weights_size(pipeline)
    planned = NemotronHMemoryPlanner(config).estimate_weights_size(pipeline)

    placed = _bytes_on_devices(_model_on_devices(config), n)
    assert planned - one == placed - _bytes_on_devices(_model(one_config), 1)
    assert planned > one
