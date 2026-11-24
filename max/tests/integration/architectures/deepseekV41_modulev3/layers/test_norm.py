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
"""CPU reference tests for the V4.1 reference-order RMSNorm.

The oracle recomputes the HF order in NumPy with an emulated bf16:
normalize in fp32, multiply by the (promoted) bf16 weight *before*
casting back -- the order ``ops.rms_norm(..., multiply_before_cast=True)``
pins, and the one rounding it differs from the Llama-style
normalize-cast-scale order.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.dtype import DType
from max.pipelines.architectures.deepseekV41_modulev3.layers.norm import (
    EPS,
    RMSNorm,
)
from max.tensor import Tensor

_HIDDEN = 512


def _to_bf16(x: np.ndarray) -> np.ndarray:
    """Rounds fp32 to bf16 (nearest-even) and back to fp32."""
    bits = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return (rounded << 16).view(np.float32)


def _oracle(x_bf: np.ndarray, w_bf: np.ndarray, eps: float) -> np.ndarray:
    """``bf16(w * (x * rsqrt(mean(x^2) + eps)))``, all in fp32."""
    x32 = x_bf.astype(np.float32)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    normalized = x32 * (1.0 / np.sqrt(var + eps))
    out = w_bf.astype(np.float32) * normalized
    return _to_bf16(out)


def _bf16_tensor(x: np.ndarray) -> Tensor:
    return F.cast(Tensor.from_dlpack(x), DType.bfloat16)


def _make_norm(weight: np.ndarray) -> RMSNorm:
    norm = RMSNorm(_HIDDEN)
    norm.weight = _bf16_tensor(weight)
    return norm


def _run(norm: RMSNorm, x_bf: Tensor) -> np.ndarray:
    out = norm(x_bf)
    assert out.dtype == DType.bfloat16
    return F.cast(out, DType.float32).to_numpy()


def test_reference_order_production_width() -> None:
    """512-wide rows: weight multiplies the fp32 normalization result."""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((9, _HIDDEN)) * 2.0).astype(np.float32)
    w = (rng.standard_normal(_HIDDEN) * 0.5).astype(np.float32)
    x_bf, w_bf = _to_bf16(x), _to_bf16(w)
    norm = _make_norm(w_bf)
    out = _run(norm, _bf16_tensor(x_bf))
    np.testing.assert_array_equal(out, _oracle(x_bf, w_bf, EPS))


def test_all_zero_row_is_zero_not_nan() -> None:
    """eps inside the rsqrt keeps an all-zero row at exact zeros."""
    rng = np.random.default_rng(4)
    x = (rng.standard_normal((3, _HIDDEN)) * 2.0).astype(np.float32)
    x[1] = 0.0
    w = np.ones(_HIDDEN, dtype=np.float32)
    norm = _make_norm(_to_bf16(w))
    out = _run(norm, _bf16_tensor(_to_bf16(x)))
    np.testing.assert_array_equal(out, _oracle(_to_bf16(x), _to_bf16(w), EPS))
    np.testing.assert_array_equal(out[1], 0.0)


def test_multiply_before_cast_order_is_observable() -> None:
    """The pinned order differs from normalize-cast-then-scale.

    With a weight far from 1, ``bf16(w * x_normalized)`` and
    ``w * bf16(x_normalized)`` disagree on some elements; the test pins
    the former (the reference) by exact oracle equality, and asserts the
    two orders actually differ so the property is not vacuous.
    """
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((16, _HIDDEN)) * 3.0).astype(np.float32)
    w = np.full(_HIDDEN, 1.03125, dtype=np.float32)  # exactly on the bf16 grid
    x_bf = _to_bf16(x)
    norm = _make_norm(_to_bf16(w))
    out = _run(norm, _bf16_tensor(x_bf))
    np.testing.assert_array_equal(out, _oracle(x_bf, _to_bf16(w), EPS))
    # The Llama order would round the normalization to bf16 first.
    x32 = x_bf.astype(np.float32)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    llama = _to_bf16(w) * _to_bf16(x32 * (1.0 / np.sqrt(var + EPS)))
    assert not np.array_equal(out, llama)


def test_dim_property() -> None:
    """The norm's ``dim`` is the constructor's width, not the input's."""
    assert RMSNorm(_HIDDEN).dim == _HIDDEN
