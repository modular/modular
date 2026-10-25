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
"""Tests for the clipped (calibration-bounded) Gemma 4 vision linears.

A checkpoint with ``use_clipped_linears`` ships four bounds beside every
vision projection and its reference clamps with them.  With the flag off the
modules must declare no bounds at all.  With it on the attention, the MLP and
their shards must clamp like the reference, at bounds the activations exceed,
so a silently unclamped layer cannot pass.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from conftest import (
    VISION_EMBED_HIDDEN_SIZE,
    VISION_NUM_HEADS,
    VISION_RMS_NORM_EPS,
    TorchGemma4MLP,
    TorchGemma4VisionAttention,
)
from conftest import (
    Gemma4ClippableLinear as RefClippableLinear,
)
from max.driver import Accelerator, Buffer, DLPackArray
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorType,
    TensorValue,
    ops,
)
from max.nn import Linear
from max.pipelines.architectures.gemma4.vision_model.attention import (
    Gemma4VisionAttention,
)
from max.pipelines.architectures.gemma4.vision_model.clipping import (
    CLAMP_BOUND_NAMES,
    Gemma4ClippableLinear,
    Gemma4VisionMLP,
)
from max.pipelines.architectures.gemma4.vision_model.encoding import (
    Gemma4VisionEncoderLayer,
)

VISION_HEAD_DIM = VISION_EMBED_HIDDEN_SIZE // VISION_NUM_HEADS
VISION_INTERMEDIATE_SIZE = 4 * VISION_EMBED_HIDDEN_SIZE

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

ATOL = 8 * torch.finfo(TORCH_DTYPE).eps
RTOL = 2e-2

# The inputs are scaled past the input bounds so the clamps bind.
INPUT_SCALE = 3.0

# The checkpoint's name for the activation, and the name MAX maps it to.
HF_HIDDEN_ACTIVATION = "gelu_pytorch_tanh"
MAX_HIDDEN_ACTIVATION = "gelu_tanh"

ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# Per-projection bounds, ordered (input_min, input_max, output_min,
# output_max). Each quadruple is asymmetric so a swapped min and max shows up,
# and distinct per projection so one projection clamping with a neighbour's
# bounds shows up. Every value is a multiple of 1/16, i.e. exact in bfloat16.
PROJECTION_BOUNDS = {
    "q_proj": (-2.0, 0.5, -0.25, 0.75),
    "k_proj": (-1.5, 0.25, -0.375, 0.625),
    "v_proj": (-2.5, 0.75, -0.5, 0.875),
    "o_proj": (-1.0, 0.4375, -0.1875, 0.5625),
    "gate_proj": (-1.75, 0.625, -0.3125, 0.8125),
    "up_proj": (-2.25, 0.375, -0.4375, 0.6875),
    "down_proj": (-1.25, 0.875, -0.5625, 0.9375),
}


def _make_max_vision_config(
    use_clipped_linears: bool, device: DeviceRef | None = None
) -> SimpleNamespace:
    """Minimal stub accepted by the MAX vision modules' constructors."""
    vision_cfg = SimpleNamespace(
        hidden_size=VISION_EMBED_HIDDEN_SIZE,
        num_attention_heads=VISION_NUM_HEADS,
        num_key_value_heads=VISION_NUM_HEADS,
        head_dim=VISION_HEAD_DIM,
        rms_norm_eps=VISION_RMS_NORM_EPS,
        attention_bias=False,
        hidden_activation=MAX_HIDDEN_ACTIVATION,
        intermediate_size=VISION_INTERMEDIATE_SIZE,
        use_clipped_linears=use_clipped_linears,
    )
    return SimpleNamespace(
        devices=[DeviceRef.GPU() if device is None else device],
        unquantized_dtype=MAX_DTYPE,
        vision_config=vision_cfg,
    )


def _max_attention(
    use_clipping: bool, device: DeviceRef | None = None
) -> Gemma4VisionAttention:
    return Gemma4VisionAttention(
        _make_max_vision_config(use_clipping, device),  # type: ignore[arg-type]
        layer_idx=0,
    )


def _max_mlp(
    use_clipping: bool, device: DeviceRef | None = None
) -> Gemma4VisionMLP:
    return Gemma4VisionMLP(
        dtype=MAX_DTYPE,
        quantization_encoding=None,
        hidden_dim=VISION_EMBED_HIDDEN_SIZE,
        feed_forward_length=VISION_INTERMEDIATE_SIZE,
        devices=[DeviceRef.GPU() if device is None else device],
        activation_function=MAX_HIDDEN_ACTIVATION,
        use_clipping=use_clipping,
    )


def _clipped(use_clipping: bool) -> SimpleNamespace:
    """Config the vendored reference ``Gemma4ClippableLinear`` reads."""
    return SimpleNamespace(use_clipped_linears=use_clipping)


def _bounds(projection: str) -> dict[str, torch.Tensor]:
    """That projection's four calibration scalars, in checkpoint layout."""
    input_min, input_max, output_min, output_max = PROJECTION_BOUNDS[projection]
    return {
        "input_min": torch.tensor(input_min, dtype=TORCH_DTYPE),
        "input_max": torch.tensor(input_max, dtype=TORCH_DTYPE),
        "output_min": torch.tensor(output_min, dtype=TORCH_DTYPE),
        "output_max": torch.tensor(output_max, dtype=TORCH_DTYPE),
    }


def _without_bounds(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in weights.items()
        if name.rsplit(".", 1)[-1] not in CLAMP_BOUND_NAMES
    }


def _set_bounds(module: torch.nn.Module, device: torch.device) -> None:
    """Pins each clipped projection's own four scalars on the reference.

    The reference registers them as persistent buffers, so the state-dict
    load carries them in; this re-pins them on the device the module runs
    on, which is the value the clamp reads.
    """
    for proj in ATTENTION_PROJECTIONS + MLP_PROJECTIONS:
        child = getattr(module, proj, None)
        if child is None:
            continue
        for bound_name, bound in _bounds(proj).items():
            setattr(child, bound_name, bound.to(device))


def _random_weight(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def _projection_weights(
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for proj, shape in shapes.items():
        weights[f"{proj}.weight"] = _random_weight(shape)
        for bound_name, bound in _bounds(proj).items():
            weights[f"{proj}.{bound_name}"] = bound
    return weights


def _attention_weights() -> dict[str, torch.Tensor]:
    nh, hd, hs = VISION_NUM_HEADS, VISION_HEAD_DIM, VISION_EMBED_HIDDEN_SIZE
    proj_out = nh * hd
    weights = _projection_weights(
        {
            "q_proj": (proj_out, hs),
            "k_proj": (proj_out, hs),
            "v_proj": (proj_out, hs),
            "o_proj": (hs, proj_out),
        }
    )
    weights["q_norm.weight"] = _random_weight((hd,))
    weights["k_norm.weight"] = _random_weight((hd,))
    return weights


def _mlp_weights() -> dict[str, torch.Tensor]:
    hs, ff = VISION_EMBED_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE
    return _projection_weights(
        {
            "gate_proj": (ff, hs),
            "up_proj": (ff, hs),
            "down_proj": (hs, ff),
        }
    )


class _ClippedRefAttention(TorchGemma4VisionAttention):
    """The reference attention with every projection clipped."""

    def __init__(
        self, hidden_size: int, num_heads: int, head_dim: int, eps: float
    ) -> None:
        super().__init__(hidden_size, num_heads, head_dim, eps)
        proj_out = num_heads * head_dim
        self.q_proj = RefClippableLinear(
            _clipped(True),
            hidden_size,
            proj_out,
            bias=False,
            dtype=TORCH_DTYPE,
        )
        self.k_proj = RefClippableLinear(
            _clipped(True),
            hidden_size,
            proj_out,
            bias=False,
            dtype=TORCH_DTYPE,
        )
        self.v_proj = RefClippableLinear(
            _clipped(True),
            hidden_size,
            proj_out,
            bias=False,
            dtype=TORCH_DTYPE,
        )
        self.o_proj = RefClippableLinear(
            _clipped(True),
            proj_out,
            hidden_size,
            bias=False,
            dtype=TORCH_DTYPE,
        )


class _ClippedRefMLP(TorchGemma4MLP):
    """The reference MLP with every projection clipped."""

    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_activation: str
    ) -> None:
        super().__init__(hidden_size, intermediate_size, hidden_activation)
        self.gate_proj = RefClippableLinear(
            _clipped(True),
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=TORCH_DTYPE,
        )
        self.up_proj = RefClippableLinear(
            _clipped(True),
            hidden_size,
            intermediate_size,
            bias=False,
            dtype=TORCH_DTYPE,
        )
        self.down_proj = RefClippableLinear(
            _clipped(True),
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=TORCH_DTYPE,
        )


# ---------------------------------------------------------------------------
# Weight names and wiring
# ---------------------------------------------------------------------------


def test_projection_bounds_are_distinct_and_asymmetric() -> None:
    """The fixture has to separate projections, and min from max.

    Symmetric bounds would pass a swapped ``input_min``/``input_max``, and one
    quadruple shared by every projection would pass a crossed pair.
    """
    for proj, (in_min, in_max, out_min, out_max) in PROJECTION_BOUNDS.items():
        assert in_min != -in_max, proj
        assert out_min != -out_max, proj
    quadruples = list(PROJECTION_BOUNDS.values())
    assert len(set(quadruples)) == len(quadruples)


def test_clamp_weights_are_declared_only_when_clipping() -> None:
    """The flag alone decides whether the four bounds exist."""
    clipped_attn = _max_attention(True)
    declared = set(clipped_attn.raw_state_dict())
    assert declared - set(_max_attention(False).raw_state_dict()) == {
        f"{proj}.{bound_name}"
        for proj in ATTENTION_PROJECTIONS
        for bound_name in CLAMP_BOUND_NAMES
    }

    clipped_mlp = _max_mlp(True)
    assert set(clipped_mlp.raw_state_dict()) - set(
        _max_mlp(False).raw_state_dict()
    ) == {
        f"{proj}.{bound_name}"
        for proj in MLP_PROJECTIONS
        for bound_name in CLAMP_BOUND_NAMES
    }


def test_encoder_layer_builds_the_clipped_mlp() -> None:
    """The vision config's flag is what selects the clipping MLP.

    Nothing else covers the encoder layer's subclass selection, so a regression
    there would put the stock fused MLP back on a clipped checkpoint.
    """
    layer = Gemma4VisionEncoderLayer(
        _make_max_vision_config(True),  # type: ignore[arg-type]
        layer_idx=0,
    )
    assert isinstance(layer.mlp, Gemma4VisionMLP)
    assert layer.mlp.use_clipping
    assert set(layer.raw_state_dict()) >= {
        f"{prefix}.{proj}.{bound_name}"
        for prefix, projections in (
            ("self_attn", ATTENTION_PROJECTIONS),
            ("mlp", MLP_PROJECTIONS),
        )
        for proj in projections
        for bound_name in CLAMP_BOUND_NAMES
    }

    plain = Gemma4VisionEncoderLayer(
        _make_max_vision_config(False),  # type: ignore[arg-type]
        layer_idx=0,
    )
    assert not isinstance(plain.mlp, Gemma4VisionMLP)
    assert not [
        name
        for name in plain.raw_state_dict()
        if name.rsplit(".", 1)[-1] in CLAMP_BOUND_NAMES
    ]


def test_each_projection_keeps_its_own_bounds() -> None:
    """Every projection loads its own quadruple, not a neighbour's."""
    attn = _max_attention(True)
    attn.load_state_dict(_attention_weights())
    mlp = _max_mlp(True)
    mlp.load_state_dict(_mlp_weights())

    for module, projections in (
        (attn, ATTENTION_PROJECTIONS),
        (mlp, MLP_PROJECTIONS),
    ):
        loaded = module.state_dict()
        for proj in projections:
            expected = _bounds(proj)
            for bound_name in CLAMP_BOUND_NAMES:
                torch.testing.assert_close(
                    torch.from_dlpack(loaded[f"{proj}.{bound_name}"]).float(),
                    expected[bound_name].float(),
                    rtol=0,
                    atol=0,
                )


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------


def test_clipping_with_a_plain_linear_class_is_refused() -> None:
    """An explicit non-clamping ``linear_cls`` would clamp nothing.

    Fusion is off whenever clipping is on, so a plain ``Linear`` here would
    quietly drop every clamp instead of falling back to the fused path.
    """
    with pytest.raises(ValueError, match="does not clamp"):
        Gemma4VisionMLP(
            dtype=MAX_DTYPE,
            quantization_encoding=None,
            hidden_dim=VISION_EMBED_HIDDEN_SIZE,
            feed_forward_length=VISION_INTERMEDIATE_SIZE,
            devices=[DeviceRef.GPU()],
            linear_cls=Linear,
            use_clipping=True,
        )


def test_clipped_linear_refuses_weight_sharding() -> None:
    """The strategy setter refuses what the output clamp cannot survive."""
    clipped = Gemma4ClippableLinear(
        in_dim=8,
        out_dim=8,
        dtype=MAX_DTYPE,
        device=DeviceRef.CPU(),
        use_clipping=True,
    )
    with pytest.raises(ValueError, match="replicate"):
        clipped.sharding_strategy = ShardingStrategy.columnwise(2)
    clipped.sharding_strategy = ShardingStrategy.replicate(2)

    # The shard-time check is a backstop for a strategy set on the weight.
    clipped.weight.sharding_strategy = ShardingStrategy.columnwise(2)
    with pytest.raises(ValueError, match="replicate"):
        clipped.shard([DeviceRef.CPU()] * 2)

    plain = Gemma4ClippableLinear(
        in_dim=8,
        out_dim=8,
        dtype=MAX_DTYPE,
        device=DeviceRef.CPU(),
    )
    plain.sharding_strategy = ShardingStrategy.columnwise(2)
    assert [type(shard) for shard in plain.shard([DeviceRef.CPU()] * 2)] == [
        Gemma4ClippableLinear,
        Gemma4ClippableLinear,
    ]


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_devices", [1, 2])
def test_clipped_attention_shards_are_clipped(num_devices: int) -> None:
    """A replicate shard is a clipped attention carrying every bound."""
    attn = _max_attention(True, DeviceRef.CPU())
    attn.sharding_strategy = ShardingStrategy.replicate(num_devices)
    shards = attn.shard([DeviceRef.CPU()] * num_devices)
    assert len(shards) == num_devices

    for shard in shards:
        assert isinstance(shard, Gemma4VisionAttention)
        for proj in ATTENTION_PROJECTIONS:
            shard_proj = getattr(shard, proj)
            assert isinstance(shard_proj, Gemma4ClippableLinear)
            for bound_name in CLAMP_BOUND_NAMES:
                assert getattr(shard_proj, bound_name) is getattr(
                    getattr(attn, proj), bound_name
                )


@pytest.mark.parametrize("num_devices", [1, 2])
def test_clipped_mlp_shards_are_clipped(num_devices: int) -> None:
    """A replicate shard is a clipped MLP, fused path still off."""
    mlp = _max_mlp(True, DeviceRef.CPU())
    mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
    shards = mlp.shard([DeviceRef.CPU()] * num_devices)
    assert len(shards) == num_devices

    for shard in shards:
        assert isinstance(shard, Gemma4VisionMLP)
        assert shard.use_clipping
        # A stock MLP shard would fuse the gate and up matmuls here.
        assert not shard._can_used_fused_mlp()
        for proj in MLP_PROJECTIONS:
            shard_proj = getattr(shard, proj)
            assert isinstance(shard_proj, Gemma4ClippableLinear)
            for bound_name in CLAMP_BOUND_NAMES:
                assert getattr(shard_proj, bound_name) is getattr(
                    getattr(mlp, proj), bound_name
                )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _run_max_attention(
    attn: Gemma4VisionAttention,
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
    weights: Mapping[str, DLPackArray],
) -> torch.Tensor:
    """Compiles the vision attention graph and executes it once."""
    num_patches = hidden_states.shape[0]
    device = Accelerator(0)
    session = InferenceSession(devices=[device])

    with Graph(
        "gemma4_vision_clipped_attn_test",
        input_types=[
            TensorType(
                MAX_DTYPE,
                ("num_patches", VISION_EMBED_HIDDEN_SIZE),
                device=DeviceRef.GPU(),
            ),
            TensorType(
                MAX_DTYPE,
                ("num_patches", VISION_HEAD_DIM // 2, 2),
                device=DeviceRef.GPU(),
            ),
            TensorType(DType.uint32, ("seqlens_size",), device=DeviceRef.GPU()),
        ],
    ) as graph:
        hs_in, freqs_in, cu_seqlens_in = graph.inputs
        assert isinstance(hs_in, TensorValue)
        assert isinstance(freqs_in, TensorValue)
        assert isinstance(cu_seqlens_in, TensorValue)
        max_seq_len = ops.constant(
            num_patches, DType.uint32, device=DeviceRef.CPU()
        )
        graph.output(attn(hs_in, freqs_in, cu_seqlens_in, max_seq_len))

    cu_seqlens = torch.tensor([0, num_patches], dtype=torch.uint32)
    compiled = session.load(graph, weights_registry=weights)
    (result,) = compiled.execute(
        Buffer.from_dlpack(hidden_states).to(device),
        Buffer.from_dlpack(freqs_cis).to(device),
        Buffer.from_dlpack(cu_seqlens).to(device),
    )
    assert isinstance(result, Buffer)
    return torch.from_dlpack(result).cpu()


def _run_max_mlp(
    mlp: Gemma4VisionMLP,
    x: torch.Tensor,
    weights: Mapping[str, DLPackArray],
) -> torch.Tensor:
    """Compiles the vision MLP graph and executes it once."""
    device = Accelerator(0)
    session = InferenceSession(devices=[device])

    with Graph(
        "gemma4_vision_clipped_mlp_test",
        input_types=[
            TensorType(
                MAX_DTYPE,
                ("num_tokens", VISION_EMBED_HIDDEN_SIZE),
                device=DeviceRef.GPU(),
            )
        ],
    ) as graph:
        (x_in,) = graph.inputs
        assert isinstance(x_in, TensorValue)
        graph.output(mlp(x_in))

    compiled = session.load(graph, weights_registry=weights)
    (result,) = compiled.execute(Buffer.from_dlpack(x).to(device))
    assert isinstance(result, Buffer)
    return torch.from_dlpack(result).cpu()


@torch.no_grad()
def test_clipped_attention_matches_reference() -> None:
    """Attention with clips equals the clipped reference, and the clips bind."""
    torch.manual_seed(11)
    weights = _attention_weights()
    num_patches = 16

    hidden_states = (
        torch.randn(num_patches, VISION_EMBED_HIDDEN_SIZE, dtype=TORCH_DTYPE)
        * INPUT_SCALE
    )
    freqs_cis, position_embeddings = _vision_freqs_cis(num_patches)

    ref = _ClippedRefAttention(
        VISION_EMBED_HIDDEN_SIZE,
        VISION_NUM_HEADS,
        VISION_HEAD_DIM,
        VISION_RMS_NORM_EPS,
    )
    ref.load_state_dict(weights)
    ref = ref.cuda().to(TORCH_DTYPE).eval()
    _set_bounds(ref, torch.device("cuda"))
    ref_out = (
        ref(hidden_states.cuda(), _cuda(position_embeddings)).cpu().float()
    )

    # The same reference without the clamps, to prove the bounds are binding
    # rather than a no-op on this input.
    unclipped = TorchGemma4VisionAttention(
        VISION_EMBED_HIDDEN_SIZE,
        VISION_NUM_HEADS,
        VISION_HEAD_DIM,
        VISION_RMS_NORM_EPS,
    )
    unclipped.load_state_dict(_without_bounds(weights))
    unclipped = unclipped.cuda().to(TORCH_DTYPE).eval()
    unclipped_out = (
        unclipped(hidden_states.cuda(), _cuda(position_embeddings))
        .cpu()
        .float()
    )
    assert not torch.allclose(ref_out, unclipped_out, rtol=RTOL, atol=ATOL)

    max_attn = _max_attention(True)
    max_attn.load_state_dict(weights)
    max_out = _run_max_attention(
        max_attn, hidden_states, freqs_cis, max_attn.state_dict()
    ).float()

    torch.testing.assert_close(ref_out, max_out, rtol=RTOL, atol=ATOL)


@torch.no_grad()
def test_clipped_mlp_matches_reference() -> None:
    """The vision MLP clamps each projection instead of fusing them."""
    torch.manual_seed(13)
    weights = _mlp_weights()
    num_tokens = 8

    x = (
        torch.randn(num_tokens, VISION_EMBED_HIDDEN_SIZE, dtype=TORCH_DTYPE)
        * INPUT_SCALE
    )

    ref = _ClippedRefMLP(
        VISION_EMBED_HIDDEN_SIZE,
        VISION_INTERMEDIATE_SIZE,
        HF_HIDDEN_ACTIVATION,
    )
    ref.load_state_dict(weights)
    ref = ref.cuda().to(TORCH_DTYPE).eval()
    _set_bounds(ref, torch.device("cuda"))
    ref_out = ref(x.cuda()).cpu().float()

    unclipped = TorchGemma4MLP(
        VISION_EMBED_HIDDEN_SIZE,
        VISION_INTERMEDIATE_SIZE,
        HF_HIDDEN_ACTIVATION,
    )
    unclipped.load_state_dict(_without_bounds(weights))
    unclipped = unclipped.cuda().to(TORCH_DTYPE).eval()
    unclipped_out = unclipped(x.cuda()).cpu().float()
    assert not torch.allclose(ref_out, unclipped_out, rtol=RTOL, atol=ATOL)

    max_mlp = _max_mlp(True)
    max_mlp.load_state_dict(weights)
    max_out = _run_max_mlp(max_mlp, x, max_mlp.state_dict()).float()

    torch.testing.assert_close(ref_out, max_out, rtol=RTOL, atol=ATOL)


@torch.no_grad()
def test_clipped_attention_shard_matches_unsharded() -> None:
    """A replicate shard computes what the unsharded attention computes.

    The two runs build separate module trees: a weight caches its graph value,
    so one weight cannot feed two graphs.
    """
    torch.manual_seed(17)
    num_patches = 16
    weights = _attention_weights()
    hidden_states = (
        torch.randn(num_patches, VISION_EMBED_HIDDEN_SIZE, dtype=TORCH_DTYPE)
        * INPUT_SCALE
    )
    freqs_cis, _ = _vision_freqs_cis(num_patches)

    attn = _max_attention(True)
    attn.load_state_dict(weights)
    expected = _run_max_attention(
        attn, hidden_states, freqs_cis, attn.state_dict()
    )

    sharded_attn = _max_attention(True)
    sharded_attn.load_state_dict(weights)
    sharded_attn.sharding_strategy = ShardingStrategy.replicate(1)
    (shard,) = sharded_attn.shard([DeviceRef.GPU()])
    got = _run_max_attention(
        shard, hidden_states, freqs_cis, sharded_attn.state_dict()
    )

    torch.testing.assert_close(expected, got, rtol=RTOL, atol=ATOL)


@torch.no_grad()
def test_clipped_mlp_shard_matches_unsharded() -> None:
    """A replicate shard computes what the unsharded MLP computes."""
    torch.manual_seed(19)
    weights = _mlp_weights()
    x = (
        torch.randn(8, VISION_EMBED_HIDDEN_SIZE, dtype=TORCH_DTYPE)
        * INPUT_SCALE
    )

    mlp = _max_mlp(True)
    mlp.load_state_dict(weights)
    expected = _run_max_mlp(mlp, x, mlp.state_dict())

    sharded_mlp = _max_mlp(True)
    sharded_mlp.load_state_dict(weights)
    sharded_mlp.sharding_strategy = ShardingStrategy.replicate(1)
    (shard,) = sharded_mlp.shard([DeviceRef.GPU()])
    got = _run_max_mlp(shard, x, sharded_mlp.state_dict())

    torch.testing.assert_close(expected, got, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cuda(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    return cos.cuda(), sin.cuda()


def _vision_freqs_cis(
    num_patches: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Builds MAX-format ``freqs_cis`` and the HF-format ``(cos, sin)`` pair."""
    head_dim = VISION_HEAD_DIM
    freqs_per_dim = head_dim // 4
    angles_d0 = torch.rand(num_patches, freqs_per_dim) * 2 * math.pi
    angles_d1 = torch.rand(num_patches, freqs_per_dim) * 2 * math.pi

    cos_d0 = torch.cos(angles_d0).to(TORCH_DTYPE)
    sin_d0 = torch.sin(angles_d0).to(TORCH_DTYPE)
    cos_d1 = torch.cos(angles_d1).to(TORCH_DTYPE)
    sin_d1 = torch.sin(angles_d1).to(TORCH_DTYPE)

    freqs_cis = torch.stack(
        [
            torch.cat([cos_d0, cos_d1], dim=-1),
            torch.cat([sin_d0, sin_d1], dim=-1),
        ],
        dim=-1,
    )
    cos_hf = torch.cat([cos_d0, cos_d0, cos_d1, cos_d1], dim=-1)
    sin_hf = torch.cat([sin_d0, sin_d0, sin_d1, sin_d1], dim=-1)
    return freqs_cis, (cos_hf, sin_hf)
