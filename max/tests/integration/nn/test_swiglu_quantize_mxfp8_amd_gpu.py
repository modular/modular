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
"""Tests the AMD CDNA4 (MI355X) branch of ``swiglu_quantize_dynamic_block_scaled``.

The Mojo kernel test
(``max/kernels/test/gpu/quantization/test_swiglu_quantize_mxfp8_amd.mojo``)
checks the fused kernel's bytes against the unfused chain and an fp64
oracle. This test checks the Python side: ``GatedActivation("situ")``
validation, the wrapper's AMD-only-MXFP8 format rules, its rank-2 output
types, and its end-to-end bytes on MI355X against an independent NumPy
oracle.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_api, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy.activation import GatedActivation
from max.nn.legacy.kernels import swiglu_quantize_dynamic_block_scaled

SF_VECTOR_SIZE = 32
BETA = 4.0
LINEAR_BETA = 25.0


def _needs_amd() -> None:
    if accelerator_count() == 0 or accelerator_api() != "hip":
        pytest.skip("needs an AMD (HIP) GPU")


def test_gated_activation_situ_validation() -> None:
    """``situ`` takes alpha (beta, non-zero) and limit (linear_beta, 0 =
    pass the up half through)."""
    situ = GatedActivation("situ", alpha=BETA, limit=LINEAR_BETA)
    assert situ.kernel_parameters() == {
        "activation": "situ",
        "swiglu_alpha": BETA,
        "swiglu_limit": LINEAR_BETA,
    }
    assert GatedActivation("situ", alpha=BETA).limit == 0.0

    with pytest.raises(ValueError, match="situ needs a non-zero alpha"):
        GatedActivation("situ")
    with pytest.raises(ValueError, match="situ needs a non-zero alpha"):
        GatedActivation("situ", limit=LINEAR_BETA)
    with pytest.raises(ValueError, match="limit must be non-negative"):
        GatedActivation("situ", alpha=BETA, limit=-1.0)
    with pytest.raises(ValueError, match="replaces the fused activation"):
        GatedActivation("situ", alpha=BETA, gated_activation_fn=lambda x, n: x)


def _decode_e4m3(codes: np.ndarray) -> np.ndarray:
    """Decodes finite E4M3 bytes to their float32 values."""
    bits = np.asarray(codes, dtype=np.uint8)
    exponent = ((bits >> 3) & 15).astype(np.int32)
    fraction = (bits & 7).astype(np.float32)
    magnitude = np.where(
        exponent == 0,
        fraction * np.float32(2.0**-9),
        np.ldexp(1.0 + fraction / 8.0, exponent - 7),
    ).astype(np.float32)
    return np.where((bits & 128) != 0, -magnitude, magnitude)


def _situ_bf16(gate_up: np.ndarray) -> np.ndarray:
    """SiTU in fp64, rounded to bf16 as the kernel does."""
    n = gate_up.shape[1] // 2
    gate = gate_up[:, :n].astype(np.float64)
    up = gate_up[:, n:].astype(np.float64)
    situ = BETA * np.tanh(gate / BETA) * (1.0 / (1.0 + np.exp(-gate)))
    up = LINEAR_BETA * np.tanh(up / LINEAR_BETA)
    activated = (situ * up).astype(np.float32)
    return (
        torch.from_numpy(activated).to(torch.bfloat16).to(torch.float32).numpy()
    )


def _quantize_mxfp8_np(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MXFP8-quantizes bf16-rounded values, mirroring the kernel's math.

    The E8M0 scale is the fp32 `group_max * fp32(1/448)` with the exponent
    rounded up (fp32 -> E8M0 casts ceil, never to nearest); the data is the
    value scaled by the exact reciprocal, then rounded to nearest even in
    E4M3 code space.
    """
    m, n = values.shape
    blocks_per_row = n // SF_VECTOR_SIZE
    blocks = values.reshape(m, blocks_per_row, SF_VECTOR_SIZE)
    maxima = np.abs(blocks).max(axis=-1)
    ratio = maxima * np.float32(1.0 / 448.0)
    mantissa, exponent = np.frexp(ratio.astype(np.float64))
    power = exponent - (mantissa == 0.5)
    power = np.clip(power, -127, 127)
    # A ratio at E8M0's subnormal floor has no finite reciprocal on the
    # device; such blocks store zero data and a zero scale.
    zero = maxima == 0
    dead = (~zero) & ((ratio == 0) | (power == -127))
    scale_bytes = np.where(zero | dead, 0, power + 127).astype(np.uint8)
    multiplier = np.ldexp(np.ones_like(maxima), -power)
    multiplier = np.where(zero | dead, 0.0, multiplier).astype(np.float32)
    normalized = (blocks * multiplier[..., None]).reshape(values.shape)

    # E4M3 bytes 0..126 in code order hold ascending magnitudes; 127 is NaN.
    e4m3_codes = np.arange(127, dtype=np.uint8)
    magnitudes = np.abs(_decode_e4m3(e4m3_codes))
    sign = np.signbit(normalized)
    magnitude = np.minimum(np.abs(normalized), 448.0)
    upper = np.minimum(np.searchsorted(magnitudes, magnitude), 126)
    lower = np.maximum(upper - 1, 0)
    low_error = magnitude - magnitudes[lower]
    high_error = magnitudes[upper] - magnitude
    pick_upper = (high_error < low_error) | (
        (high_error == low_error) & (upper % 2 == 0)
    )
    data = np.where(pick_upper, upper, lower).astype(np.uint8)
    data |= np.where(sign, 128, 0).astype(np.uint8)
    data.reshape(m, blocks_per_row, SF_VECTOR_SIZE)[dead] = 0
    return data, scale_bytes


def _signed_e4m3_code(data: np.ndarray) -> np.ndarray:
    """E4M3 bytes as sign-magnitude codes: adjacent same-sign codes differ
    by 1 and +0/-0 compare equal."""
    magnitude = (data & 0x7F).astype(np.int32)
    return np.where((data & 0x80) != 0, -magnitude, magnitude)


def _gate_up_np(m: int, n: int, seed: int) -> np.ndarray:
    """Signed BF16 `[gate | up]` input driving every branch: gate values
    that saturate tanh and drive sigmoid to 0/1, up values past the soft
    saturation, an all-zero block, and a one-outlier block."""
    rng = np.random.default_rng(seed)
    gate_up = (rng.standard_normal((m, 2 * n)) * 2).astype(np.float32)
    gate_up[:, 0:n:13] = 40.0
    gate_up[:, 6:n:13] = -40.0
    gate_up[:, n::17] = 100.0
    gate_up[:, n + 8 :: 17] = -100.0
    gate_up[0, :SF_VECTOR_SIZE] = 0.0
    gate_up[0, n : n + SF_VECTOR_SIZE] = 0.0
    gate_up[1, SF_VECTOR_SIZE : 2 * SF_VECTOR_SIZE] = 2.0**-10
    gate_up[1, 2 * SF_VECTOR_SIZE - 1] = 2.0**10
    return (
        torch.from_numpy(gate_up).to(torch.bfloat16).to(torch.float32).numpy()
    )


def _run_wrapper(
    m: int, n: int, gate_up: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Runs the fused wrapper in a MAX graph; returns (data, scales) bytes."""
    _needs_amd()
    device = Accelerator()
    device_ref = DeviceRef(device.label, device.id)
    session = InferenceSession(devices=[device])
    activation = GatedActivation("situ", alpha=BETA, limit=LINEAR_BETA)
    with Graph(
        "swiglu_quantize_mxfp8_amd",
        input_types=[
            TensorType(DType.bfloat16, shape=(m, 2 * n), device=device_ref)
        ],
    ) as graph:
        gate_up_v = graph.inputs[0].tensor
        q, scales = swiglu_quantize_dynamic_block_scaled(
            gate_up_v,
            activation,
            sf_vector_size=SF_VECTOR_SIZE,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        assert [int(d) for d in q.shape] == [m, n]
        assert [int(d) for d in scales.shape] == [m, n // SF_VECTOR_SIZE]
        assert scales.rank == 2
        graph.output(q, scales)
    model = session.load(graph)

    gate_up_t = torch.from_numpy(gate_up).to(torch.bfloat16)
    (q_buf, scales_buf) = model.execute(
        Buffer.from_dlpack(gate_up_t).to(device)
    )
    return (
        q_buf.view(DType.uint8).to_numpy(),
        scales_buf.view(DType.uint8).to_numpy(),
    )


@pytest.mark.parametrize(
    ("m", "n"),
    [(64, 384), (8, 3072)],  # TP8 decode rows and the TP1 width
)
def test_swiglu_quantize_mxfp8_amd_matches_numpy_oracle(m: int, n: int) -> None:
    """The wrapper's bytes against the independent NumPy oracle, with the
    bounded tolerances an fp64-vs-fp32 transcendental gap needs."""
    gate_up = _gate_up_np(m, n, seed=m)
    got_data, got_scales = _run_wrapper(m, n, gate_up)
    situ_bf16 = _situ_bf16(gate_up)
    want_data, want_scales = _quantize_mxfp8_np(situ_bf16)

    num_blocks = got_scales.size
    scale_mismatches = np.count_nonzero(got_scales != want_scales)
    if scale_mismatches:
        steps = np.abs(got_scales.astype(int) - want_scales.astype(int))
        assert steps.max() == 1, f"scale off by {steps.max()} E8M0 steps"
        assert scale_mismatches * 1000 <= num_blocks, (
            f"{scale_mismatches} of {num_blocks} scales mismatch (over 0.1%)"
        )
    same_scale = got_scales == want_scales
    blocks_per_row = n // SF_VECTOR_SIZE
    code_delta = np.abs(
        _signed_e4m3_code(got_data) - _signed_e4m3_code(want_data)
    ).reshape(m, blocks_per_row, SF_VECTOR_SIZE)
    byte_mismatches = np.count_nonzero((code_delta > 1) & same_scale[..., None])
    assert byte_mismatches == 0, (
        f"{byte_mismatches} data bytes are more than one E4M3 code step"
        " from the oracle"
    )

    scale_values = np.ldexp(
        np.ones((m, blocks_per_row, 1), dtype=np.float32),
        got_scales.astype(np.int32)[..., None] - 127,
    )
    dequant = (
        _decode_e4m3(got_data).reshape(m, blocks_per_row, SF_VECTOR_SIZE)
        * scale_values
    )
    ref = situ_bf16.reshape(m, blocks_per_row, SF_VECTOR_SIZE)
    allowed = np.maximum(
        scale_values * np.float32(2.0**-8), np.abs(ref) * np.float32(0.25)
    )
    worst = float(np.max(np.abs(dequant - ref) / allowed))
    assert worst <= 1.0, "dequantized value off by more than two code steps"
    print(
        f"M={m} N={n}: scale mismatches {scale_mismatches}/{num_blocks},"
        f" byte mismatches >1 step {byte_mismatches}, dequant worst {worst}"
    )


def test_swiglu_quantize_amd_rejects_non_mxfp8_formats() -> None:
    """On AMD the op runs MXFP8 only, with no per-tensor scale factor."""
    _needs_amd()
    device_ref = DeviceRef.GPU()
    with Graph(
        "swiglu_quantize_amd_format_rules",
        input_types=[
            TensorType(DType.bfloat16, shape=(64, 768), device=device_ref)
        ],
    ) as graph:
        gate_up = graph.inputs[0].tensor
        activation = GatedActivation("situ", alpha=BETA, limit=LINEAR_BETA)
        with pytest.raises(ValueError, match="MXFP8 only"):
            swiglu_quantize_dynamic_block_scaled(
                gate_up,
                activation,
                sf_vector_size=SF_VECTOR_SIZE,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.uint8,
            )
        with pytest.raises(ValueError, match="sf_vector_size must be 32"):
            swiglu_quantize_dynamic_block_scaled(
                gate_up,
                activation,
                sf_vector_size=16,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
        with pytest.raises(ValueError, match="float8_e8m0fnu for MXFP8"):
            swiglu_quantize_dynamic_block_scaled(
                gate_up,
                activation,
                sf_vector_size=SF_VECTOR_SIZE,
                scales_type=DType.float8_e4m3fn,
                out_type=DType.float8_e4m3fn,
            )
        with pytest.raises(ValueError, match=r"tensor_sf must be 1\.0"):
            swiglu_quantize_dynamic_block_scaled(
                gate_up,
                activation,
                tensor_sf=0.5,
                sf_vector_size=SF_VECTOR_SIZE,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
        graph.output(gate_up)
