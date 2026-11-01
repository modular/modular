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
"""Tests max.driver's bfloat16 conversions against ml_dtypes."""

import ml_dtypes
import numpy as np
import pytest
from max.driver import (
    CPU,
    bfloat16_as_uint16_to_float32,
    float32_array_to_buffer,
    float32_to_bfloat16_as_uint16,
)
from max.dtype import DType

_SPECIAL_FLOAT32_BITS = [
    0x00000000,  # +0
    0x80000000,  # -0
    0x00000001,  # smallest denormal
    0x007FFFFF,  # largest denormal
    0x3F808000,  # tie, rounds down to even
    0x3F818000,  # tie, rounds up to even
    0x7F7FFFFF,  # largest finite, rounds to +inf
    0x7F800000,  # +inf
    0xFF800000,  # -inf
    0x7F800001,  # signaling NaN, smallest payload
    0xFF800001,
    0x7FC00000,  # canonical quiet NaN
    0x7FFFFFFF,  # quiet NaN, all payload bits set
    0xFFFFFFFF,
]


def test_widening_matches_ml_dtypes_for_every_bfloat16() -> None:
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)

    ours = bfloat16_as_uint16_to_float32(bits)
    reference = bits.view(ml_dtypes.bfloat16).astype(np.float32)

    np.testing.assert_array_equal(
        ours.view(np.uint32), reference.view(np.uint32)
    )


def test_narrowing_matches_ml_dtypes() -> None:
    rng = np.random.default_rng(0)
    bits = np.concatenate(
        [
            rng.integers(0, 1 << 32, size=1 << 20, dtype=np.uint64).astype(
                np.uint32
            ),
            np.array(_SPECIAL_FLOAT32_BITS, dtype=np.uint32),
        ]
    )
    values = bits.view(np.float32)

    ours = float32_to_bfloat16_as_uint16(values)
    reference = values.astype(ml_dtypes.bfloat16).view(np.uint16)

    np.testing.assert_array_equal(ours, reference)


@pytest.mark.parametrize("shape", [(), (10,), (3, 4), (2, 3, 4)])
def test_conversions_preserve_shape(shape: tuple[int, ...]) -> None:
    values = np.ones(shape, dtype=np.float32)

    bits = float32_to_bfloat16_as_uint16(values)
    assert bits.shape == shape
    assert bits.dtype == np.uint16

    widened = bfloat16_as_uint16_to_float32(bits)
    assert widened.shape == shape
    assert widened.dtype == np.float32


def test_narrowing_rejects_non_float32_input() -> None:
    with pytest.raises(AssertionError):
        float32_to_bfloat16_as_uint16(np.array([1.0], dtype=np.float64))


@pytest.mark.parametrize(
    "dtype", [DType.bfloat16, DType.float32, DType.float16]
)
def test_float32_array_to_buffer(dtype: DType) -> None:
    values = np.random.default_rng(0).standard_normal((3, 5), np.float32)

    buffer = float32_array_to_buffer(values, dtype=dtype, device=CPU())

    assert buffer.dtype == dtype
    assert tuple(buffer.shape) == values.shape
    if dtype == DType.bfloat16:
        np.testing.assert_array_equal(
            buffer.view(DType.uint16).to_numpy(),
            values.astype(ml_dtypes.bfloat16).view(np.uint16),
        )
    else:
        np.testing.assert_array_equal(
            buffer.to_numpy(), values.astype(dtype.to_numpy())
        )
