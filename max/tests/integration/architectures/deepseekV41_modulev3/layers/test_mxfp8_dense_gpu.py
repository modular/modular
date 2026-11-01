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
"""GPU execution tests for the V4.1 MXFP8 dense projection.

These run the binding itself -- ``MXFP8Linear.forward``'s fused
quantize and cluster split-K matmul, and ``forward_rope_tail``'s RoPE
epilogue -- on a real device, against an independent oracle that
dequantizes both operands with the QAT recipe the composed chain it
replaced implemented (per-32-block amax floored at 1e-4, UE8M0 ceil
scale, E4M3 codes) and matmuls in float32.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.layers.mxfp8_linear import (
    MXFP8Linear,
)
from max.tensor import Tensor

AMAX_FLOOR = 1e-4
FP8_MAX = 448.0
_SF = 32


def _round_bf16(x: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


def _act_dequant(x: np.ndarray) -> np.ndarray:
    """The QAT activation quantize-dequantize, numpy + one torch fp8 cast.

    Per 32-wide block: ``amax = max(|x|, 1e-4)``, the smallest power of
    two ``scale`` with ``scale >= amax / 448``, codes clamped to +-448
    and cast to E4M3.
    """
    rows, width = x.shape
    grouped = x.astype(np.float32).reshape(rows, width // _SF, _SF)
    amax = np.maximum(np.abs(grouped).max(axis=-1, keepdims=True), AMAX_FLOOR)
    unrounded = amax / FP8_MAX
    scale = np.exp2(np.ceil(np.log2(unrounded)))
    # Exact ceil: bump while the scale undershoots, drop while it overshoots.
    scale = np.where(scale < unrounded, scale * 2, scale)
    scale = np.where(scale / 2 >= unrounded, scale / 2, scale)
    clamped = np.clip(grouped / scale, -FP8_MAX, FP8_MAX)
    codes = (
        torch.from_dlpack(np.ascontiguousarray(clamped))
        .to(torch.float8_e4m3fn)
        .float()
        .numpy()
    )
    return (codes * scale).reshape(rows, width)


def _weight_codes(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """E4M3 code values, as the exact float32 numbers they decode to."""
    raw = torch.from_numpy(rng.standard_normal((rows, cols)).astype(np.float32))
    return raw.to(torch.float8_e4m3fn).float().numpy()


def _scale_values(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    """UE8M0 scale values: random powers of two with exponents in [-8, 4]."""
    exponents = rng.integers(-8, 5, size=(rows, cols))
    return np.power(2.0, exponents).astype(np.float32)


def _weight_dequant(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Expands the per-32x32-block scales and dequantizes the weight."""
    out_rows = codes.shape[0]
    expanded = np.repeat(scale, _SF, axis=0)[:out_rows]
    expanded = np.repeat(expanded, _SF, axis=1)
    return (codes * expanded).astype(np.float32)


def _fp8(t: np.ndarray) -> Tensor:
    """The array's exact values as an E4M3 device tensor."""
    return Tensor.from_dlpack(t).to(Accelerator()).cast(DType.float8_e4m3fn)


def _e8m0(t: np.ndarray) -> Tensor:
    """Powers of two as a UE8M0 device tensor."""
    return Tensor.from_dlpack(t).to(Accelerator()).cast(DType.float8_e8m0fnu)


def _bf16(t: np.ndarray) -> Tensor:
    return Tensor.from_dlpack(t).to(Accelerator()).cast(DType.bfloat16)


def _out(t: Tensor) -> np.ndarray:
    return t.cast(DType.float32).to_numpy()


def _check(out: np.ndarray, want: np.ndarray, atol: float = 2e-2) -> None:
    abs_err = np.abs(out - want)
    rel_err = abs_err / np.maximum(np.abs(want), 1e-2)
    print(
        f"[receipt] max abs err {abs_err.max():.4g}, "
        f"max rel err {rel_err.max():.4g}"
    )
    np.testing.assert_allclose(out, want, rtol=2e-2, atol=atol)


@pytest.mark.parametrize(
    ("in_dim", "out_dim", "tokens"),
    [
        # A partial last 128-row tile of the weight scales.
        (512, 160, 17),
        # The wqa_wkv shape at a decode batch, where split-K applies.
        (4096, 1536, 12),
    ],
)
def test_dense_matches_dequantized_reference(
    in_dim: int, out_dim: int, tokens: int
) -> None:
    """``MXFP8Linear.forward`` against the dequantized oracle."""
    rng = np.random.default_rng(21)
    x = (rng.standard_normal((tokens, in_dim)) * 2.0).astype(np.float32)
    codes = _weight_codes(out_dim, in_dim, rng)
    scale = _scale_values((out_dim + 31) // 32, in_dim // 32, rng)

    linear = MXFP8Linear(in_dim, out_dim)
    linear.weight = _fp8(codes)
    linear.scale = _e8m0(scale)
    out = _out(linear(_bf16(x)))

    a = _act_dequant(_round_bf16(x))
    w = _weight_dequant(codes, scale)
    _check(out, a.astype(np.float32) @ w.T)


def _rope_tail(heads: np.ndarray, cos_sin: np.ndarray) -> np.ndarray:
    """Rotates each head's trailing ``2 * cos_sin.shape[1]`` dims in float32."""
    tokens, n_heads, head_dim = heads.shape
    rope_dim = cos_sin.shape[1] * 2
    pairs = heads[..., head_dim - rope_dim :].reshape(
        tokens, n_heads, rope_dim // 2, 2
    )
    cos = cos_sin[:, None, :, 0]
    sin = cos_sin[:, None, :, 1]
    rotated = np.stack(
        [
            pairs[..., 0] * cos - pairs[..., 1] * sin,
            pairs[..., 0] * sin + pairs[..., 1] * cos,
        ],
        axis=-1,
    ).reshape(tokens, n_heads, rope_dim)
    return np.concatenate([heads[..., : head_dim - rope_dim], rotated], axis=-1)


def test_rope_tail_matches_matmul_then_rotation() -> None:
    """``forward_rope_tail`` equals the dense projection, then the rotation.

    The oracle is the composed path: the bf16-rounded product, rotated in
    float32 and rounded again. Rotated pairs cancel, so the absolute
    tolerance is two bf16 ulps of the largest product, not a relative one.
    """
    rng = np.random.default_rng(34)
    in_dim, heads, head_dim, rope_dim, tokens = 1024, 4, 512, 64, 12
    out_dim = heads * head_dim
    x = (rng.standard_normal((tokens, in_dim)) * 2.0).astype(np.float32)
    codes = _weight_codes(out_dim, in_dim, rng)
    scale = _scale_values(out_dim // 32, in_dim // 32, rng)
    angles = rng.uniform(-3.0, 3.0, (tokens, rope_dim // 2))
    cos_sin = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(
        np.float32
    )

    linear = MXFP8Linear(in_dim, out_dim)
    linear.weight = _fp8(codes)
    linear.scale = _e8m0(scale)
    cos_sin_gpu = Tensor.from_dlpack(cos_sin).to(Accelerator())
    out = _out(linear.forward_rope_tail(_bf16(x), cos_sin_gpu, head_dim))

    a = _act_dequant(_round_bf16(x))
    product = _round_bf16(a @ _weight_dequant(codes, scale).T).reshape(
        tokens, heads, head_dim
    )
    want = _round_bf16(_rope_tail(product, cos_sin)).reshape(tokens, out_dim)
    _check(out, want, atol=2.0**-7 * float(np.abs(product).max()))


def test_swiglu_down_matches_composed_swiglu_then_dense() -> None:
    """``forward_swiglu`` equals the clamped SwiGLU, rounded to bf16, then the
    dense projection -- the shared expert's ``w2`` path before the fusion.

    Inputs reach past the clamp limit, so both clamps are live.
    """
    rng = np.random.default_rng(55)
    tokens, inter, hidden, limit = 12, 512, 1024, 10.0
    gate_up = _round_bf16(
        (rng.standard_normal((tokens, 2 * inter)) * 8.0).astype(np.float32)
    )
    codes = _weight_codes(hidden, inter, rng)
    scale = _scale_values(hidden // 32, inter // 32, rng)

    linear = MXFP8Linear(inter, hidden)
    linear.weight = _fp8(codes)
    linear.scale = _e8m0(scale)
    out = _out(linear.forward_swiglu(_bf16(gate_up), limit))

    gate = np.minimum(gate_up[:, :inter], limit)
    up = np.clip(gate_up[:, inter:], -limit, limit)
    activated = _round_bf16(gate / (1.0 + np.exp(-gate)) * up)
    _check(out, _act_dequant(activated) @ _weight_dequant(codes, scale).T)
