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
"""Tests for vision utility functions."""

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.lib.bfloat16_utils import float32_to_bfloat16_as_uint16


def uint16_to_float32(
    bf16_bits: npt.NDArray[np.uint16],
) -> npt.NDArray[np.float32]:
    """Convert bfloat16 (stored as uint16) back to float32 for verification."""
    uint32_vals = bf16_bits.astype(np.uint32) << 16
    return uint32_vals.view(np.float32)


def test_exact_roundtrip() -> None:
    """Powers of 2, simple fractions, and zero should round-trip exactly."""
    values = np.array(
        [0.0, -0.0, 1.0, 2.0, -1.0, -2.0, 0.5, -0.5, 4.0, 8.0],
        dtype=np.float32,
    )

    bf16_bits = float32_to_bfloat16_as_uint16(values)
    reconstructed = uint16_to_float32(bf16_bits)

    np.testing.assert_array_equal(reconstructed, values)


def test_infinity() -> None:
    """Infinity should round-trip exactly."""
    values = np.array([np.inf, -np.inf], dtype=np.float32)

    bf16_bits = float32_to_bfloat16_as_uint16(values)
    reconstructed = uint16_to_float32(bf16_bits)

    np.testing.assert_array_equal(reconstructed, values)


def test_nan() -> None:
    """NaN should remain NaN after conversion."""
    values = np.array([np.nan], dtype=np.float32)

    bf16_bits = float32_to_bfloat16_as_uint16(values)
    reconstructed = uint16_to_float32(bf16_bits)

    assert np.isnan(reconstructed[0])


def test_precision_within_tolerance() -> None:
    """Inexact values should be within bfloat16's max rounding error.

    BFloat16 has 7 mantissa bits, so consecutive values differ by 2^-7.
    Maximum rounding error = half the step = 2^-8 ≈ 0.4%.
    We use 1% tolerance to be conservative.
    """
    values = np.array(
        [1.001, 3.14159, 2.71828, 0.123456, 1e10, 1e-10, -1e10, -1e-10],
        dtype=np.float32,
    )

    bf16_bits = float32_to_bfloat16_as_uint16(values)
    reconstructed = uint16_to_float32(bf16_bits)

    np.testing.assert_allclose(reconstructed, values, rtol=0.01)


def test_rounding_direction() -> None:
    """Verify rounding goes to nearest value, not truncation."""
    # The next bfloat16 value after 1.0 is 1.0078125 (1 + 1/128)
    # Midpoint is 1.00390625

    just_below_mid = np.array([1.003], dtype=np.float32)
    just_above_mid = np.array([1.005], dtype=np.float32)

    bits_below = float32_to_bfloat16_as_uint16(just_below_mid)
    bits_above = float32_to_bfloat16_as_uint16(just_above_mid)

    reconstructed_below = uint16_to_float32(bits_below)
    reconstructed_above = uint16_to_float32(bits_above)

    assert reconstructed_below[0] == 1.0
    assert reconstructed_above[0] == 1.0078125  # 1 + 1/128


def test_output_contract() -> None:
    """Output should preserve shape and have dtype uint16."""
    for shape in [(10,), (3, 4), (2, 3, 4)]:
        values = np.ones(shape, dtype=np.float32)
        bf16_bits = float32_to_bfloat16_as_uint16(values)
        assert bf16_bits.shape == shape
        assert bf16_bits.dtype == np.uint16


def test_rejects_non_float32_input() -> None:
    """Should reject non-float32 input."""
    values = np.array([1.0], dtype=np.float64)

    with pytest.raises(AssertionError):
        float32_to_bfloat16_as_uint16(values)  # type: ignore[arg-type]
