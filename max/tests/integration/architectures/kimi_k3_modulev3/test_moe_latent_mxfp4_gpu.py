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
"""Numerical verification for :class:`KimiK3LatentMoE`'s AMD W4A8 MXFP4 path.

Builds the same layer twice: once MXFP4-packed (the real serving path),
once dense BF16 from the exact values those packed bytes decode to. Both
run the class's own unmodified ``forward``, so only the MXFP4 dispatch
branch is under test -- exactly the kind of positional-argument or
per-expert-bound bug a shape-only test would miss (see
``kimi_k3/layers/moe_latent.py``'s ``_host_usage_stats`` docstring for the
real incident this guards against).

``num_experts=64``: a smaller count isn't kernel-valid (see
``test_moe_latent.py``), and the uneven token-to-expert split it produces
is what that incident needed to surface.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Accelerator, Buffer, Device, accelerator_api
from max.dtype import DType
from max.nn.legacy.quant_config import QuantConfig
from max.pipelines.architectures.kimi_k3_modulev3.layers.moe_latent import (
    KimiK3LatentMoE,
)
from max.tensor import Tensor, default_device

_HIDDEN = 64
_LATENT = 128
_MOE_INTERMEDIATE = 128
_NUM_SHARED_EXPERTS = 2
_NUM_EXPERTS = 64
_TOP_K = 4
_SEQ_LEN = 24
_EPS = 1e-5
_BETA = 1.3
_LINEAR_BETA = 0.9
_SCALE = 1.7

_SF_VECTOR_SIZE = 32

_E2M1_VALUES = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float64,
)


def _decode_mxfp4(
    packed: npt.NDArray[np.uint8], scale: npt.NDArray[np.uint8]
) -> np.ndarray:
    """Dequantizes packed E2M1 codes against E8M0 scales to fp64.

    ``packed``'s last axis is ``K // 2`` bytes (two 4-bit codes each, low
    nibble first); ``scale``'s last axis is ``K // 32`` E8M0 exponents, one
    per unpacked block.
    """
    lo = _E2M1_VALUES[packed & 0xF]
    hi = _E2M1_VALUES[packed >> 4]
    values = np.empty(
        packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.float64
    )
    values[..., 0::2] = lo
    values[..., 1::2] = hi
    expanded_scale = np.exp2(scale.astype(np.float64) - 127).repeat(
        _SF_VECTOR_SIZE, axis=-1
    )
    return values * expanded_scale


def _packed_tensor(
    array: npt.NDArray[np.uint8], dtype: DType, device: Device
) -> Tensor:
    """Wraps raw ``uint8`` bytes as a Tensor of ``dtype`` without converting
    values -- a bit relabel, not a numeric cast (``Tensor(array,
    dtype=...)`` rejects a dtype that disagrees with the array's own)."""
    buf = (
        Buffer.from_dlpack(array)
        .to(device)
        .view(dtype=dtype, shape=tuple(int(d) for d in array.shape))
    )
    return Tensor(storage=buf)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _random_weights(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shared_intermediate = _MOE_INTERMEDIATE * _NUM_SHARED_EXPERTS

    def arr(*shape: int, scale: float = 1.0) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    return {
        "gate_score_weight": arr(_NUM_EXPERTS, _HIDDEN, scale=0.1),
        "e_score_correction_bias": arr(_NUM_EXPERTS, scale=0.1),
        "routed_expert_down_proj_weight": arr(
            _LATENT, _HIDDEN, scale=1 / np.sqrt(_HIDDEN)
        ),
        "routed_expert_up_proj_weight": arr(
            _HIDDEN, _LATENT, scale=1 / np.sqrt(_LATENT)
        ),
        "routed_expert_norm_weight": np.ones(_LATENT, dtype=np.float32),
        "shared_gate_up_weight": arr(
            2 * shared_intermediate, _HIDDEN, scale=1 / np.sqrt(_HIDDEN)
        ),
        "shared_down_weight": arr(
            _HIDDEN, shared_intermediate, scale=1 / np.sqrt(shared_intermediate)
        ),
    }


def _encode_mxfp4(
    values: np.ndarray, sf_vector_size: int = _SF_VECTOR_SIZE
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Quantizes realistic fp64 weights to packed E2M1 codes and E8M0 scales.

    Per block: picks the power-of-two scale that puts the block's peak at
    E2M1's top code (6), then rounds to the nearest of the 8 magnitudes.
    Using realistic (not random) weights is what lets the test below
    measure this format's real quantization error, rather than the much
    larger error uncorrelated random bytes would produce.
    """
    *lead, k = values.shape
    blocks = values.reshape(*lead, k // sf_vector_size, sf_vector_size)
    peak = np.maximum(np.max(np.abs(blocks), axis=-1, keepdims=True), 1e-30)
    # Highest E2M1 magnitude is 6; an exponent one step higher would clip it.
    exponent = np.clip(np.floor(np.log2(peak / 6.0)), -127, 127)
    scale_bytes = (exponent + 127).astype(np.uint8)
    normalized = blocks / np.exp2(exponent)
    magnitudes = np.abs(normalized)
    code_idx = np.argmin(
        np.abs(magnitudes[..., None] - _E2M1_VALUES[:8]), axis=-1
    )
    nibble = code_idx.astype(np.uint8) | np.where(
        normalized < 0, 0x8, 0x0
    ).astype(np.uint8)
    nibble = nibble.reshape(*lead, k // 2, 2)
    packed_bytes = (nibble[..., 0] | (nibble[..., 1] << 4)).astype(np.uint8)
    return packed_bytes, scale_bytes.reshape(*lead, k // sf_vector_size)


def _random_packed_experts(
    seed: int,
) -> tuple[dict[str, npt.NDArray[np.uint8]], dict[str, np.ndarray]]:
    """Packed gate/up/down experts quantized from realistic Gaussian weights,
    and their fp64-decoded equivalents (round-trips through the SAME encode
    path :func:`_encode_mxfp4` uses, so ``decoded`` is exactly what the real
    checkpoint's packed bytes represent, not an independent approximation)."""
    rng = np.random.default_rng(seed)

    def gaussian(*shape: int, scale: float) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float64) * scale

    raw = {
        "gate_up": gaussian(
            _NUM_EXPERTS,
            2 * _MOE_INTERMEDIATE,
            _LATENT,
            scale=1 / np.sqrt(_LATENT),
        ),
        "down": gaussian(
            _NUM_EXPERTS,
            _LATENT,
            _MOE_INTERMEDIATE,
            scale=1 / np.sqrt(_MOE_INTERMEDIATE),
        ),
    }
    packed: dict[str, npt.NDArray[np.uint8]] = {}
    scale: dict[str, npt.NDArray[np.uint8]] = {}
    for key, values in raw.items():
        packed[key], scale[key] = _encode_mxfp4(values)
    decoded = {
        "experts_gate_up_proj": _decode_mxfp4(
            packed["gate_up"], scale["gate_up"]
        ),
        "experts_down_proj": _decode_mxfp4(packed["down"], scale["down"]),
    }
    return {**packed, **{f"{k}_scale": v for k, v in scale.items()}}, decoded


def _build(
    weights: dict[str, np.ndarray],
    *,
    mxfp4: dict[str, npt.NDArray[np.uint8]] | None = None,
    decoded: dict[str, np.ndarray] | None = None,
    device: Device,
) -> KimiK3LatentMoE:
    layer = KimiK3LatentMoE(
        hidden_size=_HIDDEN,
        latent_size=_LATENT,
        moe_intermediate_size=_MOE_INTERMEDIATE,
        shared_intermediate_size=_MOE_INTERMEDIATE * _NUM_SHARED_EXPERTS,
        num_experts=_NUM_EXPERTS,
        num_experts_per_token=_TOP_K,
        beta=_BETA,
        linear_beta=_LINEAR_BETA,
        rms_norm_eps=_EPS,
        routed_scaling_factor=_SCALE,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        dtype=DType.bfloat16,
        quant_config=(
            QuantConfig.mxfp4(mlp_quantized_layers=[0]) if mxfp4 else None
        ),
        amd_w4a8=True if mxfp4 else None,
    )
    layer.gate.gate_score.weight = Tensor(
        weights["gate_score_weight"], dtype=DType.float32
    )
    layer.gate.e_score_correction_bias = Tensor(
        weights["e_score_correction_bias"], dtype=DType.float32
    )

    def bf16(array: np.ndarray) -> Tensor:
        return Tensor(array.astype(np.float32), dtype=DType.float32).cast(
            DType.bfloat16
        )

    layer.routed_expert_down_proj.weight = bf16(
        weights["routed_expert_down_proj_weight"]
    )
    layer.routed_expert_up_proj.weight = bf16(
        weights["routed_expert_up_proj_weight"]
    )
    layer.routed_expert_norm.weight = bf16(weights["routed_expert_norm_weight"])
    layer.shared_experts.gate_up_proj.weight = bf16(
        weights["shared_gate_up_weight"]
    )
    layer.shared_experts.down_proj.weight = bf16(weights["shared_down_weight"])
    if mxfp4 is not None:
        layer.experts_gate_up_proj = _packed_tensor(
            mxfp4["gate_up"], DType.uint8, device
        )
        layer.experts_gate_up_proj_scale = _packed_tensor(
            mxfp4["gate_up_scale"], DType.float8_e8m0fnu, device
        )
        layer.experts_down_proj = _packed_tensor(
            mxfp4["down"], DType.uint8, device
        )
        layer.experts_down_proj_scale = _packed_tensor(
            mxfp4["down_scale"], DType.float8_e8m0fnu, device
        )
    else:
        assert decoded is not None
        layer.experts_gate_up_proj = bf16(decoded["experts_gate_up_proj"])
        layer.experts_down_proj = bf16(decoded["experts_down_proj"])
    return layer


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="MXFP4 W4A8 AMD grouped matmul is CDNA4-only",
)
@pytest.mark.compile_on_gpu(
    "mo.moe.single.group.router is staged as an undeclared custom op the interpreter can't bind"
)
def test_amd_w4a8_matches_dense_from_decoded_weights() -> None:
    """The MXFP4 dispatch must match a dense run of what it is supposed to
    represent, not just produce finite numbers.

    The only numerically lossy step between the two runs is activation
    quantization to MXFP8 (the weights are bit-exact decodes of the packed
    ones, not independently sampled), so the tolerance is sized from that
    alone -- a wrong per-expert row bound, a swapped GEMM operand, or a
    misrouted scale tensor would each produce an error far outside it.
    """
    rng = np.random.default_rng(3)
    x_np = rng.standard_normal((_SEQ_LEN, _HIDDEN)).astype(np.float32)
    weights = _random_weights(11)
    mxfp4, decoded = _random_packed_experts(13)

    device = Accelerator()
    with default_device(device):
        x_bf16 = Tensor(x_np, dtype=DType.float32).cast(DType.bfloat16)

        dense_layer = _build(weights, decoded=decoded, device=device)
        dense_out = (
            dense_layer(x_bf16)
            .cast(DType.float32)
            .to_numpy()
            .astype(np.float64)
        )

        mxfp4_layer = _build(weights, mxfp4=mxfp4, device=device)
        mxfp4_out = (
            mxfp4_layer(x_bf16)
            .cast(DType.float32)
            .to_numpy()
            .astype(np.float64)
        )

    assert mxfp4_out.shape == (_SEQ_LEN, _HIDDEN)
    assert np.all(np.isfinite(mxfp4_out))

    # A real misrouted tensor decorrelates the two outputs almost entirely:
    # reversing `experts_down_proj`'s expert axis (shape-compatible, so a
    # shape-only test would miss it -- pairs each expert's gate/up output
    # with a different expert's down projection) measures 0.035 on this
    # fixture's seed. The two legitimate sources of difference here (gate/up
    # and down activations each re-quantized to MXFP8, independently per
    # GEMM) stay far above that: measured at 0.9988 on the same seed.
    correlation = np.corrcoef(mxfp4_out.flatten(), dense_out.flatten())[0, 1]
    assert correlation > 0.95, (
        f"correlation {correlation} is too low for activation-quantization"
        " noise alone -- likely a real misrouting bug, not just MXFP8 error"
    )

    # Elementwise bound, sized from the same measurement rather than a priori
    # theory: two independent MXFP8 round-trips (one per GEMM) plus a
    # nonlinear SiTU in between and a top-4 weighted combine measure a 4.9%
    # relative noise floor here (diff stddev / dense stddev, both ~1.0), with
    # a handful of near-zero elements dominated by the additive floor rather
    # than the relative term; this leaves roughly 3x headroom on both.
    error = np.abs(mxfp4_out - dense_out)
    tolerance = 0.15 * np.abs(dense_out) + 0.2
    assert np.all(error <= tolerance), (
        f"max error {error.max()} exceeds tolerance at"
        f" {np.unravel_index(np.argmax(error - tolerance), error.shape)}"
    )


@pytest.mark.skipif(
    accelerator_api() != "hip",
    reason="MXFP4 W4A8 AMD grouped matmul is CDNA4-only",
)
def test_amd_w4a8_disabled_without_opt_in() -> None:
    """``amd_w4a8=None`` (the serving default) stays off the env var's
    absence, even with an MXFP4 ``quant_config`` and CDNA4 hardware --
    opt-in is explicit, matching V2's own gate."""
    device = Accelerator()
    with default_device(device):
        layer = KimiK3LatentMoE(
            hidden_size=_HIDDEN,
            latent_size=_LATENT,
            moe_intermediate_size=_MOE_INTERMEDIATE,
            shared_intermediate_size=_MOE_INTERMEDIATE * _NUM_SHARED_EXPERTS,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_TOP_K,
            beta=_BETA,
            linear_beta=_LINEAR_BETA,
            rms_norm_eps=_EPS,
            routed_scaling_factor=_SCALE,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
            dtype=DType.bfloat16,
            quant_config=QuantConfig.mxfp4(mlp_quantized_layers=[0]),
        )
    assert layer.mxfp4
    assert not layer.amd_tp_w4a8
