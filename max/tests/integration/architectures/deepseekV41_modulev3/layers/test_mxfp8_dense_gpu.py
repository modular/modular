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

These run the binding itself -- ``MXFP8Linear.forward``'s
activation-quantize -> scale-interleave -> ``dynamic_block_scaled_matmul``
chain -- on a real device, against an independent oracle that
dequantizes both operands with the QAT recipe (per-32-block amax
floored at 1e-4, UE8M0 ceil scale, E4M3 codes) and matmuls in float32.
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV41_modulev3.layers.mxfp8_linear import (
    MXFP8Linear,
)

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


def test_dense_matches_dequantized_reference() -> None:
    """``MXFP8Linear.forward`` on a partial final scale block (padding trim).

    in_dim 512, out_dim 160: the checkpoint's final 32-row scale block is
    partial, so the path pads the weight, runs the block-scaled matmul and
    trims back -- all on device.
    """
    rng = np.random.default_rng(21)
    in_dim, out_dim, tokens = 512, 160, 17
    x = (rng.standard_normal((tokens, in_dim)) * 2.0).astype(np.float32)
    codes = _weight_codes(out_dim, in_dim, rng)
    scale = _scale_values((out_dim + 31) // 32, in_dim // 32, rng)

    linear = MXFP8Linear(in_dim, out_dim)
    linear.weight = _fp8(codes)
    linear.scale = _e8m0(scale)
    out = _out(linear(_bf16(x)))

    a = _act_dequant(_round_bf16(x))
    w = _weight_dequant(codes, scale)
    want = a.astype(np.float32) @ w.T
    np.testing.assert_allclose(out, want, rtol=2e-2, atol=2e-2)
