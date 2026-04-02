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
"""End-to-end tests for MO interpreter with Mojo ops.

These tests verify that the Mojo op implementations produce correct results
by comparing against numpy reference implementations.
"""

from collections.abc import Sequence

import numpy as np
import pytest
from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random as max_random
from max.experimental import realization_context as rc
from max.experimental.realization_context import set_seed
from max.experimental.tensor import Tensor, realization_context

# DTypes to test for elementwise operations
# Note: bfloat16 is excluded since NumPy doesn't support it natively
FLOAT_DTYPES = [DType.float32, DType.float64]
INT_DTYPES = [DType.int8, DType.int16, DType.int32, DType.int64]
UINT_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]
SIGNED_DTYPES = FLOAT_DTYPES + INT_DTYPES
ELEMENTWISE_DTYPES = SIGNED_DTYPES + UINT_DTYPES
# DTypes to test for matmul operations (float and integer)
MATMUL_DTYPES = FLOAT_DTYPES + INT_DTYPES


class TestBinaryElementwiseOps:
    """Tests for binary elementwise Mojo ops."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_add(self, dtype: DType) -> None:
        """Test add op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(12, 24, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_sub(self, dtype: DType) -> None:
        """Test sub op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, 24, dtype=np_dtype).reshape(shape)
        b_np = np.arange(12, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a - b

        expected = np.subtract(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_mul(self, dtype: DType) -> None:
        """Test mul op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.arange(2, 14, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = np.multiply(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_div(self, dtype: DType) -> None:
        """Test div op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.arange(1, 13, dtype=np_dtype).reshape(shape) + 0.5

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a / b

        expected = np.divide(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_pow(self, dtype: DType) -> None:
        """Test pow op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        # Use positive base values to avoid NaN from fractional exponents
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.full(shape, 2.0, dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = np.power(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_max(self, dtype: DType) -> None:
        """Test elementwise max op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(11, -1, -1, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = F.max(a, b)

        expected = np.maximum(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_min(self, dtype: DType) -> None:
        """Test elementwise min op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(11, -1, -1, dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = F.min(a, b)

        expected = np.minimum(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_clamp(self, dtype: DType) -> None:
        """Test elementwise clamp op matches numpy."""
        shape = [3]
        np_dtype = dtype.to_numpy()
        a_np = np.array([12, 12, 12], dtype=np_dtype).reshape(shape)
        lower_bound_np = np.array([0, 2, 12], dtype=np_dtype).reshape(shape)
        upper_bound_np = np.array([13, 3, 15], dtype=np_dtype).reshape(shape)

        a = Tensor.from_dlpack(a_np)
        lower_bound = Tensor.from_dlpack(lower_bound_np)
        upper_bound = Tensor.from_dlpack(upper_bound_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = F.clamp(a, lower_bound, upper_bound)

        expected = np.clip(a_np, lower_bound_np, upper_bound_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)


class TestBinaryComparisonOps:
    """Tests for binary comparison Mojo ops (output is bool)."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_equal(self, dtype: DType) -> None:
        """Test equal op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        b_np = np.array([1, 5, 3, 6], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a == b

        result_np = np.from_dlpack(c)
        expected = np.equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_not_equal(self, dtype: DType) -> None:
        """Test not_equal op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        b_np = np.array([1, 5, 3, 6], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a != b

        result_np = np.from_dlpack(c)
        expected = np.not_equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_greater(self, dtype: DType) -> None:
        """Test greater op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 5, 3, 6], dtype=np_dtype)
        b_np = np.array([2, 3, 3, 4], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a > b

        result_np = np.from_dlpack(c)
        expected = np.greater(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_greater_equal(self, dtype: DType) -> None:
        """Test greater_equal op returns bool and matches numpy."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 5, 3, 6], dtype=np_dtype)
        b_np = np.array([2, 3, 3, 4], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a >= b

        result_np = np.from_dlpack(c)
        expected = np.greater_equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_


class TestUnaryElementwiseOps:
    """Tests for unary elementwise Mojo ops."""

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_negative(self, dtype: DType) -> None:
        """Test negative op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(-6, 6, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = -x

        expected = np.negative(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_abs(self, dtype: DType) -> None:
        """Test abs op matches numpy for signed types."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(-6, 6, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = abs(x)

        expected = np.abs(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", UINT_DTYPES)
    def test_abs_unsigned(self, dtype: DType) -> None:
        """Test abs op matches numpy for unsigned types."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        # Use non-negative values for unsigned types
        x_np = np.arange(0, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = abs(x)

        expected = np.abs(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_exp(self, dtype: DType) -> None:
        """Test exp op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-2, 2, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.exp(x)

        expected = np.exp(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_log(self, dtype: DType) -> None:
        """Test log op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(0.1, 10, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.log(x)

        expected = np.log(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sqrt(self, dtype: DType) -> None:
        """Test sqrt op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(0, 10, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.sqrt(x)

        expected = np.sqrt(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_tanh(self, dtype: DType) -> None:
        """Test tanh op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-3, 3, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tanh(x)

        expected = np.tanh(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_relu(self, dtype: DType) -> None:
        """Test relu op matches numpy maximum(x, 0)."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-3, 3, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.relu(x)

        expected = np.maximum(x_np, 0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sin(self, dtype: DType) -> None:
        """Test sin op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-np.pi, np.pi, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.sin(x)

        expected = np.sin(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cos(self, dtype: DType) -> None:
        """Test cos op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-np.pi, np.pi, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cos(x)

        expected = np.cos(x_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(y), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_floor(self, dtype: DType) -> None:
        """Test floor op matches numpy."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-2.5, 2.5, 12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.floor(x)

        expected = np.floor(x_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestUnaryMixedOps:
    """Tests for unary mixed-dtype Mojo ops (cast, is_nan, is_inf)."""

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (DType.float32, DType.int32),
            (DType.float64, DType.float32),
            (DType.int32, DType.float64),
            (DType.int32, DType.float32),
            (DType.float32, DType.float64),
            (DType.int8, DType.int32),
            (DType.uint8, DType.float32),
            (DType.float32, DType.int64),
            (DType.int64, DType.float32),
        ],
    )
    def test_cast(self, in_dtype: DType, out_dtype: DType) -> None:
        """Test cast op converts dtype correctly."""
        in_np_dtype = in_dtype.to_numpy()
        out_np_dtype = out_dtype.to_numpy()
        x_np = np.arange(12, dtype=in_np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(out_np_dtype)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == out_np_dtype

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_is_nan(self, dtype: DType) -> None:
        """Test is_nan op detects NaN values."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1.0, np.nan, 3.0, np.nan, np.inf, 0.0], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.is_nan(x)

        result_np = np.from_dlpack(y)
        expected = np.isnan(x_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_is_inf(self, dtype: DType) -> None:
        """Test is_inf op detects Inf values."""
        np_dtype = dtype.to_numpy()
        x_np = np.array(
            [1.0, np.inf, -np.inf, np.nan, 0.0, 42.0], dtype=np_dtype
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.is_inf(x)

        result_np = np.from_dlpack(y)
        expected = np.isinf(x_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    def test_cast_identity(self) -> None:
        """Test cast to same dtype is identity."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.float32)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)

    def test_cast_float_to_int_truncation(self) -> None:
        """Test cast from float to int truncates toward zero."""
        x_np = np.array([1.7, -2.3, 3.9, -4.1], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.int32)

        expected = x_np.astype(np.int32)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_is_nan_all_normal(self) -> None:
        """Test is_nan returns all False for normal values."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.is_nan(x)

        np.testing.assert_array_equal(
            np.from_dlpack(y), np.array([False, False, False, False])
        )

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            # Signed integer narrowing: values exceed target range
            (DType.int32, DType.int8),
            (DType.int64, DType.int16),
            (DType.int32, DType.int16),
            (DType.int64, DType.int8),
        ],
    )
    def test_cast_signed_integer_overflow(
        self, in_dtype: DType, out_dtype: DType
    ) -> None:
        """Test cast with signed integer values that overflow the target type."""
        in_np_dtype = in_dtype.to_numpy()
        out_np_dtype = out_dtype.to_numpy()
        # Values that exceed target range (e.g., 200 overflows int8 [-128,127])
        x_np = np.array(
            [200, -200, 1000, -1000, 0, 127, 128], dtype=in_np_dtype
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(out_np_dtype)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == out_np_dtype

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (DType.uint32, DType.int8),
            (DType.uint16, DType.int8),
            (DType.uint32, DType.uint8),
        ],
    )
    def test_cast_unsigned_integer_overflow(
        self, in_dtype: DType, out_dtype: DType
    ) -> None:
        """Test cast with unsigned integer values that overflow the target."""
        in_np_dtype = in_dtype.to_numpy()
        out_np_dtype = out_dtype.to_numpy()
        # Positive values that exceed target range
        x_np = np.array(
            [200, 300, 1000, 65535, 0, 127, 128, 255, 256],
            dtype=in_np_dtype,
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(out_np_dtype)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == out_np_dtype

    def test_cast_float64_to_float32_precision_loss(self) -> None:
        """Test cast from float64 to float32 loses precision."""
        # Use values that have more precision than float32 can represent.
        # Avoid subnormal float32 values (< ~1.18e-38) which may be
        # flushed to zero by SIMD worker threads with FTZ enabled.
        x_np = np.array(
            [1.0000000000000002, 1.23456789012345678, 1e-30, 1e38],
            dtype=np.float64,
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.float32)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(np.float32)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.float32

    def test_cast_float_to_int_narrowing(self) -> None:
        """Test cast from float to narrow int with truncation and wrapping."""
        # Use float32→int32 with fractional values to test truncation
        x_np = np.array(
            [1e9, -1e9, 1.5e9, -1.5e9, 0.0, 1.0, -1.0], dtype=np.float32
        )

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.int32)

        result_np = np.from_dlpack(y)
        expected = x_np.astype(np.int32)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.int32


class TestBooleanLogicOps:
    """Tests for boolean logic Mojo ops."""

    def test_and(self) -> None:
        """Test logical and op."""
        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a & b

        expected = np.logical_and(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    def test_or(self) -> None:
        """Test logical or op."""
        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a | b

        expected = np.logical_or(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    def test_xor(self) -> None:
        """Test logical xor op."""
        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a ^ b

        expected = np.logical_xor(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    def test_not(self) -> None:
        """Test logical not op."""
        x_np = np.array([True, False, True, False], dtype=np.bool_)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = ~x

        expected = np.logical_not(x_np)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestChainedOperations:
    """Tests for chained operations using Mojo ops."""

    def test_chained_arithmetic(self) -> None:
        """Test chained add/sub/mul operations."""
        shape = [3, 4]
        x_np = np.arange(12, dtype=np.float32).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            # (x + 1) * 2 - 3
            one = Tensor.from_dlpack(np.ones(shape, dtype=np.float32))
            two = Tensor.from_dlpack(np.full(shape, 2.0, dtype=np.float32))
            three = Tensor.from_dlpack(np.full(shape, 3.0, dtype=np.float32))
            result = (x + one) * two - three

        expected = (x_np + 1) * 2 - 3
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    def test_comparison_with_arithmetic(self) -> None:
        """Test combining comparisons with arithmetic operations."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            # Compare (x * 2) > 5
            two = Tensor.from_dlpack(np.full([4], 2.0, dtype=np.float32))
            five = Tensor.from_dlpack(np.full([4], 5.0, dtype=np.float32))
            result = (x * two) > five

        result_np = np.from_dlpack(result)
        expected = (x_np * 2) > 5
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_


class TestBasicOpExecution:
    """Tests for basic op execution through the interpreter."""

    def test_add_two_constants(self) -> None:
        """Test adding two constants."""
        a = Tensor.from_dlpack(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        b = Tensor.from_dlpack(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        )
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_mul_two_constants(self) -> None:
        """Test multiplying two constants."""
        a = Tensor.from_dlpack(np.array([2.0, 3.0, 4.0], dtype=np.float32))
        b = Tensor.from_dlpack(np.array([5.0, 6.0, 7.0], dtype=np.float32))
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = np.array([10.0, 18.0, 28.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_unary_operations(self) -> None:
        """Test unary operations like exp, sqrt, tanh."""
        x = Tensor.from_dlpack(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.exp(x)

        expected = np.exp(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(
            np.from_dlpack(result), expected, decimal=5
        )


class TestDataPassthrough:
    """Tests for data passthrough via the interpreter."""

    def test_passthrough_basic(self) -> None:
        """Test that data passes through correctly via interpreter."""
        input_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        zeros_np = np.zeros((3, 4), dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        z = Tensor.from_dlpack(zeros_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = x + z

        np.testing.assert_array_almost_equal(np.from_dlpack(result), input_np)

    def test_passthrough_multiple_dtypes(self) -> None:
        """Test data passes through correctly with different dtypes."""
        for np_dtype in [np.float32, np.float64, np.int32, np.int64]:
            input_np = np.array([1, 2, 3, 4], dtype=np_dtype)
            zeros_np = np.zeros([4], dtype=np_dtype)

            x = Tensor.from_dlpack(input_np)
            z = Tensor.from_dlpack(zeros_np)
            with (
                rc.EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                result = x + z

            np.testing.assert_array_equal(np.from_dlpack(result), input_np)

    def test_passthrough_preserves_shape(self) -> None:
        """Test that operations preserve tensor shape."""
        for shape in [[4], [2, 3], [2, 3, 4], [1, 2, 3, 4]]:
            size = 1
            for dim in shape:
                size *= dim
            input_np = np.arange(size, dtype=np.float32).reshape(shape)
            zeros_np = np.zeros(shape, dtype=np.float32)

            x = Tensor.from_dlpack(input_np)
            z = Tensor.from_dlpack(zeros_np)
            with (
                rc.EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                result = x + z

            result_np = np.from_dlpack(result)
            assert result_np.shape == tuple(shape)
            np.testing.assert_array_almost_equal(result_np, input_np)


class TestShapeOps:
    """Tests for shape operations (rebind, broadcast_to) in the interpreter."""

    def test_broadcast_to_static_shape(self) -> None:
        """Test that broadcast_to correctly broadcasts to a static target shape."""
        input_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3])

        expected = np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_broadcast_to_higher_rank(self) -> None:
        """Test broadcasting to a higher rank tensor."""
        input_np = np.array([5.0], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        expected = np.full((2, 3, 4), 5.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_broadcast_to_2d_to_3d(self) -> None:
        """Test broadcasting a 2D tensor to 3D."""
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[3, 2, 2])

        expected = np.broadcast_to(input_np, (3, 2, 2))
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_broadcast_then_add(self) -> None:
        """Test broadcasting followed by element-wise operation."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_np = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )

        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            x_broadcast = x.broadcast_to(shape=[2, 3])
            z = x_broadcast + y

        expected = np.array(
            [[11.0, 22.0, 33.0], [41.0, 52.0, 63.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(np.from_dlpack(z), expected)


class TestInterpreterVsCompiled:
    """Tests comparing interpreter results to compiled execution."""

    def test_interpreter_matches_compiled_add(self) -> None:
        """Test that interpreter add matches compiled add."""
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        # Execute via interpreter
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            interp_result = a + b

        # Execute via compiled path
        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = a + b

        # Results should match
        np.testing.assert_array_almost_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )

    def test_interpreter_matches_compiled_mul(self) -> None:
        """Test that interpreter mul matches compiled mul."""
        a_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([5.0, 6.0, 7.0], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            interp_result = a * b

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = a * b

        np.testing.assert_array_almost_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )

    def test_interpreter_matches_compiled_chained(self) -> None:
        """Test that interpreter matches compiled for chained operations."""
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        two_np = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        one_np = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        two = Tensor.from_dlpack(two_np)
        one = Tensor.from_dlpack(one_np)

        # x * 2 + 1
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            interp_result = x * two + one

        with (
            rc.EagerRealizationContext() as ctx,
            realization_context(ctx),
        ):
            compiled_result = x * two + one

        np.testing.assert_array_almost_equal(
            np.from_dlpack(interp_result), np.from_dlpack(compiled_result)
        )


class TestStaticBroadcastToOp:
    """Tests for StaticBroadcastTo using the Tensor API with MO interpreter."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_broadcast_1d_to_2d(self, dtype: DType) -> None:
        """Test broadcasting 1D tensor to 2D."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_1d_to_3d(self, dtype: DType) -> None:
        """Test broadcasting 1D tensor to 3D."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3, 4))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_2d_to_3d(self, dtype: DType) -> None:
        """Test broadcasting 2D tensor to 3D."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[3, 2, 2])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (3, 2, 2))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_size1_dim(self, dtype: DType) -> None:
        """Test broadcasting with size-1 dimension."""
        np_dtype = dtype.to_numpy()
        # Shape [1, 3] -> [4, 3]
        input_np = np.array([[1.0, 2.0, 3.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[4, 3])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (4, 3))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_multiple_size1_dims(self, dtype: DType) -> None:
        """Test broadcasting with multiple size-1 dimensions."""
        np_dtype = dtype.to_numpy()
        # Shape [1, 3, 1] -> [2, 3, 4]
        input_np = np.array([[[1.0], [2.0], [3.0]]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3, 4))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_scalar_like(self, dtype: DType) -> None:
        """Test broadcasting a scalar-like tensor [1] to higher dimensions."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([42.0], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 4])

        result = np.from_dlpack(y)
        expected = np.full((2, 3, 4), 42.0, dtype=np_dtype)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_same_shape(self, dtype: DType) -> None:
        """Test broadcasting when shapes are already compatible (no-op)."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 2])

        result = np.from_dlpack(y)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_broadcast_to_4d(self, dtype: DType) -> None:
        """Test broadcasting to 4D tensor."""
        np_dtype = dtype.to_numpy()
        input_np = np.array([[1.0, 2.0]], dtype=np_dtype)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=[2, 3, 1, 2])

        result = np.from_dlpack(y)
        expected = np.broadcast_to(input_np, (2, 3, 1, 2))
        np.testing.assert_array_equal(result, expected)

    def test_broadcast_integer_types(self) -> None:
        """Test broadcasting with integer types."""
        for dtype in INT_DTYPES:
            np_dtype = dtype.to_numpy()
            input_np = np.array([1, 2, 3], dtype=np_dtype)

            x = Tensor.from_dlpack(input_np)
            with (
                rc.EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                y = x.broadcast_to(shape=[2, 3])

            result = np.from_dlpack(y)
            expected = np.broadcast_to(input_np, (2, 3))
            np.testing.assert_array_equal(result, expected)

    def test_broadcast_preserves_values(self) -> None:
        """Test that broadcast preserves exact values during chained operations."""
        input_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ones_np = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        x = Tensor.from_dlpack(input_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            # Broadcast then add
            x_broadcast = x.broadcast_to(shape=[2, 3])
            ones = Tensor.from_dlpack(ones_np)
            y = x_broadcast + ones

        result = np.from_dlpack(y)
        expected = np.array(
            [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(result, expected)


class TestMatmulOp:
    """Tests for matmul Mojo op."""

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_matmul_basic(self, dtype: DType) -> None:
        """Test basic 2D matmul matches numpy."""
        np_dtype = dtype.to_numpy()
        # Use small values to avoid overflow for integer types
        a_np = np.arange(12, dtype=np_dtype).reshape(3, 4) % 10
        b_np = np.arange(20, dtype=np_dtype).reshape(4, 5) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_matmul_square(self, dtype: DType) -> None:
        """Test square matrix matmul."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(16, dtype=np_dtype).reshape(4, 4) % 5
        b_np = np.arange(16, dtype=np_dtype).reshape(4, 4) % 5

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matmul_float_precision(self, dtype: DType) -> None:
        """Test matmul with random floats for precision."""
        np_dtype = dtype.to_numpy()
        np.random.seed(42)
        a_np = np.random.randn(8, 16).astype(np_dtype)
        b_np = np.random.randn(16, 8).astype(np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(c), expected, decimal=5
        )

    def test_matmul_vector(self) -> None:
        """Test matmul with vector-like shapes."""
        a_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)


class TestBatchMatmulOp:
    """Tests for batch matmul Mojo op."""

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_batch_matmul_3d(self, dtype: DType) -> None:
        """Test 3D batch matmul: (2, 3, 4) @ (2, 4, 5)."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(24, dtype=np_dtype).reshape(2, 3, 4) % 10
        b_np = np.arange(40, dtype=np_dtype).reshape(2, 4, 5) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_batch_matmul_4d(self, dtype: DType) -> None:
        """Test 4D batch matmul: (2, 3, 4, 5) @ (2, 3, 5, 6)."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(120, dtype=np_dtype).reshape(2, 3, 4, 5) % 10
        b_np = np.arange(180, dtype=np_dtype).reshape(2, 3, 5, 6) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_batch_matmul_float_precision(self, dtype: DType) -> None:
        """Test batch matmul with random floats for precision."""
        np_dtype = dtype.to_numpy()
        np.random.seed(42)
        a_np = np.random.randn(2, 8, 16).astype(np_dtype)
        b_np = np.random.randn(2, 16, 8).astype(np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(
            np.from_dlpack(c), expected, decimal=5
        )

    def test_batch_matmul_single_batch(self) -> None:
        """Test batch matmul with single batch dimension."""
        a_np = np.arange(12, dtype=np.float32).reshape(1, 3, 4) % 10
        b_np = np.arange(20, dtype=np.float32).reshape(1, 4, 5) % 10

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a @ b

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(np.from_dlpack(c), expected)


class TestRangeOp:
    """Tests for range Mojo op via Tensor.arange with typed tensor inputs."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_basic(self, dtype: DType) -> None:
        """Test basic range op matches numpy arange."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(10, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(1, dtype=np_dtype))

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=10,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0, 10, 1, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_with_step(self, dtype: DType) -> None:
        """Test range op with custom step size."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(10, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(2, dtype=np_dtype))

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=5,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0, 10, 2, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_float_step(self, dtype: DType) -> None:
        """Test range op with float step size."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0.0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(1.0, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(0.25, dtype=np_dtype))

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=4,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0.0, 1.0, 0.25, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_negative_step(self, dtype: DType) -> None:
        """Test range op with negative step."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(5, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(-1, dtype=np_dtype))

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=5,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(5, 0, -1, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_range_int(self, dtype: DType) -> None:
        """Test range op with integer dtypes."""
        np_dtype = dtype.to_numpy()
        start_t = Tensor.from_dlpack(np.array(0, dtype=np_dtype))
        stop_t = Tensor.from_dlpack(np.array(10, dtype=np_dtype))
        step_t = Tensor.from_dlpack(np.array(1, dtype=np_dtype))

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=10,
                dtype=dtype,
                device=CPU(),
            )

        expected = np.arange(0, 10, 1, dtype=np_dtype)
        np.testing.assert_array_equal(np.from_dlpack(t), expected)

    def test_range_nonzero_start(self) -> None:
        """Test range op with nonzero start value."""
        start_t = Tensor.from_dlpack(np.array(5, dtype=np.float32))
        stop_t = Tensor.from_dlpack(np.array(15, dtype=np.float32))
        step_t = Tensor.from_dlpack(np.array(2, dtype=np.float32))

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(
                start_t,
                stop_t,
                step_t,
                out_dim=5,
                dtype=DType.float32,
                device=CPU(),
            )

        expected = np.arange(5, 15, 2, dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)


class TestReduceOps:
    """Tests for reduction Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_reduce_max_last_axis(self, dtype: DType) -> None:
        """Test reduce_max on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_first_axis(self, dtype: DType) -> None:
        """Test reduce_max on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=0)

        expected = np.max(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_middle_axis(self, dtype: DType) -> None:
        """Test reduce_max on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=1)

        expected = np.max(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_2d(self, dtype: DType) -> None:
        """Test reduce_max on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- ReduceMin tests ---

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_reduce_min_last_axis(self, dtype: DType) -> None:
        """Test reduce_min on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_min_first_axis(self, dtype: DType) -> None:
        """Test reduce_min on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=0)

        expected = np.min(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_min_middle_axis(self, dtype: DType) -> None:
        """Test reduce_min on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=1)

        expected = np.min(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_min_2d(self, dtype: DType) -> None:
        """Test reduce_min on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        expected = np.min(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- ReduceAdd (sum) tests ---

    @pytest.mark.parametrize(
        "dtype",
        FLOAT_DTYPES + [DType.int32, DType.int64],
    )
    def test_reduce_sum_last_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        expected = np.sum(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_sum_first_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=0)

        expected = np.sum(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_sum_middle_axis(self, dtype: DType) -> None:
        """Test reduce_sum on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=1)

        expected = np.sum(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_sum_2d(self, dtype: DType) -> None:
        """Test reduce_sum on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        expected = np.sum(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    # --- Mean tests ---

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_last_axis(self, dtype: DType) -> None:
        """Test mean on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        expected = np.mean(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_first_axis(self, dtype: DType) -> None:
        """Test mean on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=0)

        expected = np.mean(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_middle_axis(self, dtype: DType) -> None:
        """Test mean on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=1)

        expected = np.mean(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_mean_2d(self, dtype: DType) -> None:
        """Test mean on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        expected = np.mean(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    # --- ReduceMul (prod) tests ---

    @pytest.mark.parametrize(
        "dtype",
        FLOAT_DTYPES + [DType.int32, DType.int64],
    )
    def test_reduce_mul_last_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the last axis matches numpy."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        # Use small values to avoid overflow
        x_np = np.arange(1, 61, dtype=np_dtype).reshape(shape) * 0.1 + 1
        x_np = x_np.astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        expected = np.prod(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_mul_first_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the first axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = (rng.standard_normal((3, 4, 5)) * 0.5 + 1).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=0)

        expected = np.prod(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_mul_middle_axis(self, dtype: DType) -> None:
        """Test reduce_mul on a middle axis."""
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = (rng.standard_normal((2, 3, 4)) * 0.5 + 1).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=1)

        expected = np.prod(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_mul_2d(self, dtype: DType) -> None:
        """Test reduce_mul on 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(1, 21, dtype=np_dtype).reshape(shape) * 0.1 + 1
        x_np = x_np.astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        expected = np.prod(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


def _numpy_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax reference implementation."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x_shifted)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _numpy_logsoftmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable logsoftmax reference implementation."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    return x_shifted - np.log(
        np.sum(np.exp(x_shifted), axis=axis, keepdims=True)
    )


class TestSoftmaxOps:
    """Tests for softmax and logsoftmax Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_softmax_last_axis_3d(self, dtype: DType) -> None:
        """Test softmax on the last axis of a 3D tensor."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.softmax(x, axis=-1)

        expected = _numpy_softmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_softmax_2d(self, dtype: DType) -> None:
        """Test softmax on a 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.softmax(x, axis=-1)

        expected = _numpy_softmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_logsoftmax_last_axis_3d(self, dtype: DType) -> None:
        """Test logsoftmax on the last axis of a 3D tensor."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.logsoftmax(x, axis=-1)

        expected = _numpy_logsoftmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_logsoftmax_2d(self, dtype: DType) -> None:
        """Test logsoftmax on a 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.logsoftmax(x, axis=-1)

        expected = _numpy_logsoftmax(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestBroadcastBinaryOps:
    """Tests for implicit broadcasting in binary ops on CPU.

    These tests exercise the ShapeOfOp -> BroadcastShapeOp -> BroadcastToOp
    chain that gets generated when binary elementwise ops have operands with
    different shapes.
    """

    def test_add_broadcast_1d_2d(self) -> None:
        """Test add with broadcasting: [3] + [2,3] -> [2,3]."""
        a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b_np = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32
        )

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_mul_broadcast_scalar_like(self) -> None:
        """Test mul with broadcasting: [1] * [3,4] -> [3,4]."""
        a_np = np.array([2.0], dtype=np.float32)
        b_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = np.multiply(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    def test_sub_broadcast_different_ranks(self) -> None:
        """Test sub with broadcasting: [4] - [2,3,4] -> [2,3,4]."""
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a - b

        expected = np.subtract(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_add_broadcast_size1_dim(self, dtype: DType) -> None:
        """Test add with broadcasting: [1,4] + [3,4] -> [3,4]."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(4, dtype=np_dtype).reshape(1, 4)
        b_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(c), expected)


class TestRandomNormalOp:
    """Tests for random normal op via max.random.gaussian with interpreter."""

    def test_random_normal_shape_and_dtype(self) -> None:
        """Test that random normal produces correct shape and dtype."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.gaussian(
                (3, 4), dtype=DType.float32, device=CPU()
            )

        result_np = np.from_dlpack(result)
        assert result_np.shape == (3, 4)
        assert result_np.dtype == np.float32

    def test_random_normal_deterministic(self) -> None:
        """Test that same seed produces identical results."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        np.testing.assert_array_equal(
            np.from_dlpack(result1), np.from_dlpack(result2)
        )

    def test_random_normal_statistics(self) -> None:
        """Test that random normal has approximately correct mean and std."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.gaussian(
                (1000, 1000),
                mean=5.0,
                std=2.0,
                dtype=DType.float32,
                device=CPU(),
            )

        result_np = np.from_dlpack(result)
        # With 1M samples, statistics should be quite close
        np.testing.assert_allclose(result_np.mean(), 5.0, atol=0.1)
        np.testing.assert_allclose(result_np.std(), 2.0, atol=0.1)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float64])
    def test_random_normal_dtypes(self, dtype: DType) -> None:
        """Test random normal with different float dtypes."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.gaussian((10, 10), dtype=dtype, device=CPU())

        result_np = np.from_dlpack(result)
        assert result_np.shape == (10, 10)
        assert result_np.dtype == dtype.to_numpy()


class TestRandomUniformOp:
    """Tests for random uniform op via max.random.uniform with interpreter."""

    def test_random_uniform_shape_and_dtype(self) -> None:
        """Test that random uniform produces correct shape and dtype."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.uniform(
                (3, 4), dtype=DType.float32, device=CPU()
            )

        result_np = np.from_dlpack(result)
        assert result_np.shape == (3, 4)
        assert result_np.dtype == np.float32

    def test_random_uniform_deterministic(self) -> None:
        """Test that same seed produces identical results."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=CPU()
            )

        np.testing.assert_array_equal(
            np.from_dlpack(result1), np.from_dlpack(result2)
        )

    def test_random_uniform_statistics(self) -> None:
        """Test that random uniform has approximately correct statistics."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.uniform(
                (1000, 1000),
                range=(2.0, 5.0),
                dtype=DType.float32,
                device=CPU(),
            )

        result_np = np.from_dlpack(result)
        # With 1M samples, statistics should be quite close
        np.testing.assert_allclose(result_np.mean(), 3.5, atol=0.1)
        assert result_np.min() >= 2.0
        assert result_np.max() <= 5.0

    @pytest.mark.parametrize("dtype", [DType.float32, DType.float64])
    def test_random_uniform_dtypes(self, dtype: DType) -> None:
        """Test random uniform with different float dtypes."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.uniform((10, 10), dtype=dtype, device=CPU())

        result_np = np.from_dlpack(result)
        assert result_np.shape == (10, 10)
        assert result_np.dtype == dtype.to_numpy()


class TestShapeChangeOps:
    """Tests for shape change operations (squeeze, unsqueeze, reshape variants).

    These test the reshape semantics that SqueezeShapeOp, UnsqueezeShapeOp,
    AddSingletonDimOp, SplitDimOp, and MergeDimOp implement. Since these ops
    are emitted by MLIR lowering passes rather than the Python API directly,
    we test through the Tensor API methods that produce equivalent reshapes.
    """

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_squeeze_single_dim(self, dtype: DType) -> None:
        """Test squeeze removes a size-1 dimension."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 1, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=1)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=1)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_squeeze_first_dim(self, dtype: DType) -> None:
        """Test squeeze on the first dimension."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(1, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=0)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_squeeze_last_dim(self, dtype: DType) -> None:
        """Test squeeze on the last dimension."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=-1)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=-1)
        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_unsqueeze_beginning(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension at the beginning."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=0)
        assert result.shape == (1, 3, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_unsqueeze_middle(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension in the middle."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=1)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=1)
        assert result.shape == (3, 1, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_unsqueeze_end(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension at the end."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=-1)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=-1)
        assert result.shape == (3, 4, 1)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reshape_split_dim(self, dtype: DType) -> None:
        """Test reshape that splits a dimension (equivalent to SplitDimOp).

        E.g., [12, 3] -> [3, 4, 3] splits dimension 0 into (3, 4).
        """
        np_dtype = dtype.to_numpy()
        x_np = np.arange(36, dtype=np_dtype).reshape(12, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 4, 3])

        result = np.from_dlpack(y)
        expected = x_np.reshape(3, 4, 3)
        assert result.shape == (3, 4, 3)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reshape_merge_dims(self, dtype: DType) -> None:
        """Test reshape that merges adjacent dimensions (equivalent to MergeDimOp).

        E.g., [2, 3, 4] -> [6, 4] merges dimensions 0 and 1.
        """
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(2, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([6, 4])

        result = np.from_dlpack(y)
        expected = x_np.reshape(6, 4)
        assert result.shape == (6, 4)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reshape_add_singleton(self, dtype: DType) -> None:
        """Test reshape that adds a singleton dimension (equiv to AddSingletonDimOp).

        E.g., [3, 4] -> [3, 1, 4] adds a dimension of size 1.
        """
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 1, 4])

        result = np.from_dlpack(y)
        expected = x_np.reshape(3, 1, 4)
        assert result.shape == (3, 1, 4)
        np.testing.assert_array_equal(result, expected)

    def test_squeeze_then_unsqueeze_roundtrip(self) -> None:
        """Test that squeeze then unsqueeze returns to original shape."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 1, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            squeezed = x.squeeze(axis=1)
            unsqueezed = squeezed.unsqueeze(axis=1)

        result = np.from_dlpack(unsqueezed)
        assert result.shape == (3, 1, 4)
        np.testing.assert_array_equal(result, x_np)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_squeeze_integer_types(self, dtype: DType) -> None:
        """Test squeeze with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(6, dtype=np_dtype).reshape(1, 2, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.squeeze(x_np, axis=0)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_unsqueeze_integer_types(self, dtype: DType) -> None:
        """Test unsqueeze with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(6, dtype=np_dtype).reshape(2, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=0)

        result = np.from_dlpack(y)
        expected = np.expand_dims(x_np, axis=0)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result, expected)


class TestSelectOp:
    """Tests for select (where) op via F.where with interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_select_basic(self, dtype: DType) -> None:
        """Test basic select op matches numpy.where."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array(
            [True, False, True, False, True, False], dtype=np.bool_
        )
        x_np = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np_dtype)
        y_np = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np_dtype)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_select_2d(self, dtype: DType) -> None:
        """Test select op with 2D tensors."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array(
            [[True, False, True], [False, True, False]], dtype=np.bool_
        )
        x_np = np.arange(1, 7, dtype=np_dtype).reshape(2, 3)
        y_np = np.arange(10, 70, 10, dtype=np_dtype).reshape(2, 3)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_select_int(self, dtype: DType) -> None:
        """Test select op with integer dtypes."""
        np_dtype = dtype.to_numpy()
        cond_np = np.array([True, False, True, False], dtype=np.bool_)
        x_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        y_np = np.array([10, 20, 30, 40], dtype=np_dtype)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = np.where(cond_np, x_np, y_np)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_select_all_true(self) -> None:
        """Test select with all-true condition returns x."""
        cond_np = np.ones(4, dtype=np.bool_)
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        np.testing.assert_array_equal(np.from_dlpack(result), x_np)

    def test_select_all_false(self) -> None:
        """Test select with all-false condition returns y."""
        cond_np = np.zeros(4, dtype=np.bool_)
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        y_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        cond = Tensor.from_dlpack(cond_np)
        x = Tensor.from_dlpack(x_np)
        y = Tensor.from_dlpack(y_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        np.testing.assert_array_equal(np.from_dlpack(result), y_np)


class TestConcatOp:
    """Tests for concat op via F.concat with interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concat_axis0(self, dtype: DType) -> None:
        """Test concat along axis 0."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concat_axis1(self, dtype: DType) -> None:
        """Test concat along axis 1."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np_dtype)
        b_np = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=1)

        expected = np.concatenate([a_np, b_np], axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_concat_negative_axis(self, dtype: DType) -> None:
        """Test concat along negative axis (-1 = last dim)."""
        np_dtype = dtype.to_numpy()
        a_np = np.arange(6, dtype=np_dtype).reshape(2, 3)
        b_np = np.arange(4, dtype=np_dtype).reshape(2, 2)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=-1)

        expected = np.concatenate([a_np, b_np], axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_concat_int_dtypes(self, dtype: DType) -> None:
        """Test concat with integer dtypes."""
        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3], dtype=np_dtype)
        b_np = np.array([4, 5, 6], dtype=np_dtype)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_concat_multiple_tensors(self) -> None:
        """Test concat with more than two tensors."""
        a_np = np.array([[1.0, 2.0]], dtype=np.float32)
        b_np = np.array([[3.0, 4.0]], dtype=np.float32)
        c_np = np.array([[5.0, 6.0]], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        c = Tensor.from_dlpack(c_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b, c], axis=0)

        expected = np.concatenate([a_np, b_np, c_np], axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)

    def test_concat_single_tensor(self) -> None:
        """Test concat with a single tensor is a no-op."""
        a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        a = Tensor.from_dlpack(a_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a], axis=0)

        np.testing.assert_array_almost_equal(np.from_dlpack(result), a_np)

    def test_concat_3d(self) -> None:
        """Test concat with 3D tensors."""
        a_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        b_np = np.arange(24, 48, dtype=np.float32).reshape(2, 3, 4)

        a = Tensor.from_dlpack(a_np)
        b = Tensor.from_dlpack(b_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = np.concatenate([a_np, b_np], axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(result), expected)


def _numpy_layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Numerically stable layer normalization reference implementation."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


class TestLayerNormOps:
    """Tests for layer_norm Mojo op."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_layer_norm_2d(self, dtype: DType) -> None:
        """Test layer_norm on a 2D tensor."""
        shape = [4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)
        gamma_np = rng.standard_normal(shape[-1]).astype(np_dtype)
        beta_np = rng.standard_normal(shape[-1]).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_layer_norm_3d(self, dtype: DType) -> None:
        """Test layer_norm on a 3D tensor."""
        shape = [3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)
        gamma_np = rng.standard_normal(shape[-1]).astype(np_dtype)
        beta_np = rng.standard_normal(shape[-1]).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_layer_norm_4d(self, dtype: DType) -> None:
        """Test layer_norm on a 4D tensor."""
        shape = [2, 3, 4, 5]
        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np_dtype)
        gamma_np = rng.standard_normal(shape[-1]).astype(np_dtype)
        beta_np = rng.standard_normal(shape[-1]).astype(np_dtype)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_layer_norm_large_feature_dim(self) -> None:
        """Test layer_norm with a large feature dimension."""
        shape = [8, 128]
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np.float32)
        gamma_np = rng.standard_normal(shape[-1]).astype(np.float32)
        beta_np = rng.standard_normal(shape[-1]).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_layer_norm_single_element_feature(self) -> None:
        """Test layer_norm with a single-element feature dimension."""
        shape = [4, 1]
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal(shape).astype(np.float32)
        gamma_np = rng.standard_normal(shape[-1]).astype(np.float32)
        beta_np = rng.standard_normal(shape[-1]).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        gamma = Tensor.from_dlpack(gamma_np)
        beta = Tensor.from_dlpack(beta_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.layer_norm(x, gamma, beta, epsilon=1e-5)

        expected = _numpy_layer_norm(x_np, gamma_np, beta_np, eps=1e-5)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestSliceOp:
    """Tests for slice operations."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_1d(self, dtype: DType) -> None:
        """Test basic 1D slice."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(10, dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(2, 7)])

        expected = x_np[2:7]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_slice_2d(self, dtype: DType) -> None:
        """Test 2D slice across both dimensions."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3)])

        expected = x_np[0:2, 1:3]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_with_step(self) -> None:
        """Test slice with step > 1."""
        x_np = np.arange(10, dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 10, 2)])

        expected = x_np[0:10:2]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_3d(self) -> None:
        """Test slice with 3D tensor."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 2), slice(1, 3), slice(0, 2)])

        expected = x_np[0:2, 1:3, 0:2]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_slice_int_dtypes(self, dtype: DType) -> None:
        """Test slice with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(1, 3), slice(0, 2)])

        expected = x_np[1:3, 0:2]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_single_element(self) -> None:
        """Test slice extracting a single element along each dim."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(1, 2), slice(2, 3)])

        expected = x_np[1:2, 2:3]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)

    def test_slice_full_dim(self) -> None:
        """Test slice that takes the full range of a dimension."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.slice_tensor(x, [slice(0, 3), slice(0, 4)])

        expected = x_np[0:3, 0:4]
        np.testing.assert_array_equal(np.from_dlpack(result), expected)


class TestCumsumOps:
    """Tests for cumsum Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_1d(self, dtype: DType) -> None:
        """Test cumsum on a 1D tensor."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0)

        expected = np.cumsum(x_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_2d_last_axis(self, dtype: DType) -> None:
        """Test cumsum on the last axis of a 2D tensor."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=-1)

        expected = np.cumsum(x_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_2d_first_axis(self, dtype: DType) -> None:
        """Test cumsum on the first axis of a 2D tensor."""
        shape = [3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0)

        expected = np.cumsum(x_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_3d_middle_axis(self, dtype: DType) -> None:
        """Test cumsum on the middle axis of a 3D tensor."""
        shape = [2, 3, 4]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=1)

        expected = np.cumsum(x_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_4d(self, dtype: DType) -> None:
        """Test cumsum on a 4D tensor along axis 2."""
        shape = [2, 3, 4, 5]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(120, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=2)

        expected = np.cumsum(x_np, axis=2)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_5d(self, dtype: DType) -> None:
        """Test cumsum on a 5D tensor along axis 3."""
        shape = [2, 3, 2, 4, 2]
        np_dtype = dtype.to_numpy()
        x_np = np.arange(96, dtype=np_dtype).reshape(shape)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=3)

        expected = np.cumsum(x_np, axis=3)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_exclusive(self, dtype: DType) -> None:
        """Test cumsum with exclusive=True."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0, exclusive=True)

        # exclusive cumsum: [0, 1, 3]
        expected = np.array([0, 1, 3], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_reverse(self, dtype: DType) -> None:
        """Test cumsum with reverse=True."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0, reverse=True)

        # reverse cumsum: [6, 5, 3]
        expected = np.array([6, 5, 3], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cumsum_exclusive_reverse(self, dtype: DType) -> None:
        """Test cumsum with both exclusive=True and reverse=True."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0, exclusive=True, reverse=True)

        # exclusive reverse cumsum: [5, 3, 0]
        expected = np.array([5, 3, 0], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
    def test_cumsum_integer(self, dtype: DType) -> None:
        """Test cumsum with integer dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([1, 2, 3, 4], dtype=np_dtype)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.cumsum(x, axis=0)

        expected = np.cumsum(x_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestGatherOp:
    """Tests for gather op via MO interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_gather_axis0(self, dtype: DType) -> None:
        """Test gather along axis 0 on a 2D tensor."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([2, 0], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_gather_axis1(self, dtype: DType) -> None:
        """Test gather along axis 1 on a 2D tensor."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([3, 1, 0], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=1)

        expected = np.take(x_np, idx_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_negative_axis(self) -> None:
        """Test gather with negative axis."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([0, 2], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=-1)

        expected = np.take(x_np, idx_np, axis=-1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_3d(self) -> None:
        """Test gather on a 3D tensor along axis 1."""
        x_np = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        idx_np = np.array([1, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=1)

        expected = np.take(x_np, idx_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_3d_axis2(self) -> None:
        """Test gather on a 3D tensor along last axis."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        idx_np = np.array([0, 3], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=2)

        expected = np.take(x_np, idx_np, axis=2)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_multidim_indices(self) -> None:
        """Test gather with 2D indices tensor."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        idx_np = np.array([[0, 2], [1, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_int32_indices(self) -> None:
        """Test gather with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([2, 0], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_gather_integer_data(self, dtype: DType) -> None:
        """Test gather with integer data dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([1, 0, 2], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather(x, idx, axis=0)

        expected = np.take(x_np, idx_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestGatherNdOp:
    """Tests for gather_nd op via MO interpreter."""

    def test_gather_nd_basic(self) -> None:
        """Test basic 2D gather_nd."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([[0, 1], [2, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # idx has shape (2, 2), index_depth=2 indexes fully into (3,4)
        # output shape: idx[:-1] + input[2:] = (2,) + () = (2,)
        expected = np.array([x_np[0, 1], x_np[2, 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_3d(self) -> None:
        """Test gather_nd on a 3D input with partial indexing."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        # index_depth=1: index into first dim only, slicing remaining
        idx_np = np.array([[0], [1]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: idx[:-1] + input[1:] = (2,) + (3,4) = (2,3,4)
        expected = x_np[idx_np.flatten()]
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_batch_dims(self) -> None:
        """Test gather_nd with batch_dims > 0."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        # batch_dims=1, index_depth=1: per-batch index into dim 1
        idx_np = np.array([[1], [0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx, batch_dims=1)

        # output shape: input[:1] + idx[1:-1] + input[1+1:] = (2,) + () + (4,) = (2,4)
        expected = np.array([x_np[0, 1], x_np[1, 0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_single_index(self) -> None:
        """Test gather_nd where index_depth=1 (single dimension indexing)."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        idx_np = np.array([[0], [2], [3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: idx[:-1] + input[1:] = (3,) + (5,) = (3,5)
        expected = x_np[[0, 2, 3]]
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_full_index(self) -> None:
        """Test gather_nd where index_depth = input rank (full indexing)."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        idx_np = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        # output shape: idx[:-1] + input[3:] = (2,) + () = (2,)
        expected = np.array([x_np[0, 1, 2], x_np[1, 2, 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_gather_nd_dtype(self, dtype: DType) -> None:
        """Test gather_nd with various float dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        idx_np = np.array([[1, 2], [0, 0]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        expected = np.array([x_np[1, 2], x_np[0, 0]], dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_gather_nd_int32_indices(self) -> None:
        """Test gather_nd with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        idx_np = np.array([[2, 1], [0, 3]], dtype=np.int32)

        x = Tensor.from_dlpack(x_np)
        idx = Tensor.from_dlpack(idx_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.gather_nd(x, idx)

        expected = np.array([x_np[2, 1], x_np[0, 3]], dtype=np.float32)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)


class TestArgMaxMinOp:
    """Tests for ArgMax and ArgMin interpreter ops.

    Parameterized on op (argmax/argmin) and axis to avoid duplication.
    """

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d_axes(self, op_name: str, axis: int, dtype: DType) -> None:
        """Test argmax/argmin on a 2D tensor along each axis."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[1, 5, 3], [4, 2, 6]], dtype=np_dtype)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=axis)

        expected = np_op(x_np, axis=axis, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_3d_middle_axis(self, op_name: str) -> None:
        """Test on a 3D tensor along the middle axis."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np.float32)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=1)

        expected = np_op(x_np, axis=1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize(
        "op_name,tie_data",
        [
            ("argmax", [5.0, 5.0, 3.0, 5.0]),
            ("argmin", [1.0, 3.0, 1.0, 5.0]),
        ],
    )
    def test_ties_lowest_index(
        self, op_name: str, tie_data: list[float]
    ) -> None:
        """Test that ties return the lowest index."""
        x_np = np.array(tie_data, dtype=np.float32)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=0)

        result = np.from_dlpack(y)
        assert result.item() == 0, (
            f"Expected index 0 for tie, got {result.item()}"
        )

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, op_name: str, dtype: DType) -> None:
        """Test with integer input dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[10, 3, 7], [1, 8, 4]], dtype=np_dtype)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=1)

        expected = np_op(x_np, axis=1, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_1d(self, op_name: str) -> None:
        """Test on a 1D tensor."""
        x_np = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], dtype=np.float32)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=0)

        expected = np_op(x_np, axis=0, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("op_name", ["argmax", "argmin"])
    def test_4d(self, op_name: str) -> None:
        """Test on a 4D tensor along axis 2."""
        rng = np.random.default_rng(99)
        x_np = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)
        np_op = getattr(np, op_name)
        f_op = getattr(F, op_name)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = f_op(x, axis=2)

        expected = np_op(x_np, axis=2, keepdims=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestSplitOp:
    """Tests for split op via MO interpreter."""

    @staticmethod
    def _assert_split_equal(
        results: Sequence[object],
        expected: list[np.ndarray],
    ) -> None:
        assert len(results) == len(expected)
        for result, exp in zip(results, expected, strict=True):
            assert isinstance(result, Tensor)
            np.testing.assert_array_equal(np.from_dlpack(result), exp)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("axis", [0, 1])
    def test_2d_axes(self, dtype: DType, axis: int) -> None:
        """Test split on a 2D tensor along each axis."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(24, dtype=np_dtype).reshape(6, 4)
        split_sizes = [2, 4] if axis == 0 else [1, 3]

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, split_sizes, axis=axis)

        indices = np.cumsum(split_sizes[:-1])
        expected = np.split(x_np, indices, axis=axis)
        self._assert_split_equal(results, expected)

    def test_3d_middle_axis(self) -> None:
        """Test split on a 3D tensor along the middle axis."""
        x_np = np.arange(60, dtype=np.float32).reshape(3, 4, 5)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2, 1], axis=1)

        expected = np.split(x_np, [1, 3], axis=1)
        self._assert_split_equal(results, expected)

    def test_negative_axis(self) -> None:
        """Test split with a negative axis value."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 3], axis=-1)

        expected = np.split(x_np, [1], axis=-1)
        self._assert_split_equal(results, expected)

    def test_three_way_split(self) -> None:
        """Test splitting into three uneven parts."""
        x_np = np.arange(30, dtype=np.float32).reshape(5, 6)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2, 3], axis=1)

        expected = np.split(x_np, [1, 3], axis=1)
        self._assert_split_equal(results, expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test split with integer input dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [1, 2], axis=0)

        expected = np.split(x_np, [1], axis=0)
        self._assert_split_equal(results, expected)

    def test_4d(self) -> None:
        """Test split on a 4D tensor."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 4, 3)).astype(np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, [2, 1, 3], axis=1)

        expected = np.split(x_np, [2, 3], axis=1)
        self._assert_split_equal(results, expected)

    def test_equal_split(self) -> None:
        """Test splitting into equal-size chunks via int split_size."""
        x_np = np.arange(12, dtype=np.float32).reshape(4, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            results = F.split(x, 2, axis=0)

        expected = np.split(x_np, 2, axis=0)
        self._assert_split_equal(results, expected)


class TestScatterOp:
    """Tests for scatter op via MO interpreter (CPU-only, MO_HostOnly)."""

    @staticmethod
    def _scatter_ref(
        x: np.ndarray, updates: np.ndarray, indices: np.ndarray, axis: int
    ) -> np.ndarray:
        """Numpy reference: copy x, then put_along_axis."""
        out = x.copy()
        np.put_along_axis(out, indices, updates, axis=axis)
        return out

    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_2d(self, axis: int, dtype: DType) -> None:
        """Test scatter on a 2D tensor along axis 0 and 1."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        if axis == 0:
            updates_np = np.array(
                [[90, 91, 92, 93], [94, 95, 96, 97]], dtype=np_dtype
            )
            indices_np = np.array([[2, 1, 0, 2], [0, 2, 1, 0]], dtype=np.int64)
        else:
            updates_np = np.array(
                [[90, 91], [92, 93], [94, 95]], dtype=np_dtype
            )
            indices_np = np.array([[3, 0], [2, 1], [0, 3]], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=axis)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_negative_axis(self) -> None:
        """Test scatter with negative axis (-1 == last axis)."""
        x_np = np.zeros((3, 4), dtype=np.float32)
        updates_np = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        )
        indices_np = np.array(
            [[0, 3, 1, 2], [2, 1, 3, 0], [3, 0, 2, 1]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=-1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=-1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_3d_middle_axis(self) -> None:
        """Test scatter on a 3D tensor along axis 1."""
        x_np = np.zeros((2, 4, 3), dtype=np.float32)
        updates_np = np.ones((2, 2, 3), dtype=np.float32) * 7.0
        indices_np = np.array(
            [[[0, 0, 0], [3, 3, 3]], [[1, 1, 1], [2, 2, 2]]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    def test_4d(self) -> None:
        """Test scatter on a 4D tensor along axis 2."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 5, 4)).astype(np.float32)
        indices_np = rng.integers(0, 5, size=(2, 3, 2, 4)).astype(np.int64)
        updates_np = np.ones((2, 3, 2, 4), dtype=np.float32) * 99.0

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=2)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=2)
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test scatter with integer data dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dtype).reshape(3, 4)
        updates_np = np.array(
            [[50, 60, 70, 80], [90, 100, 110, 120], [10, 20, 30, 40]],
            dtype=np_dtype,
        )
        indices_np = np.array(
            [[1, 0, 3, 2], [3, 2, 0, 1], [0, 1, 2, 3]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_duplicate_indices(self) -> None:
        """Test scatter with duplicate indices (last write wins)."""
        x_np = np.zeros((4,), dtype=np.float32)
        updates_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        indices_np = np.array([1, 1, 1, 2], dtype=np.int64)

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=0)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_partial_update_along_axis(self) -> None:
        """Test scatter where updates are smaller than input along axis."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5)
        updates_np = np.array(
            [[-1.0, -2.0, -3.0, -4.0, -5.0], [-6.0, -7.0, -8.0, -9.0, -10.0]],
            dtype=np.float32,
        )
        indices_np = np.array(
            [[2, 0, 3, 1, 0], [0, 3, 1, 2, 3]], dtype=np.int64
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=0)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_int32_indices(self) -> None:
        """Test scatter with int32 index dtype."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        updates_np = np.array(
            [
                [99.0, 88.0, 77.0, 66.0],
                [55.0, 44.0, 33.0, 22.0],
                [11.0, 0.0, -1.0, -2.0],
            ],
            dtype=np.float32,
        )
        indices_np = np.array(
            [[2, 0, 3, 1], [1, 3, 0, 2], [0, 1, 2, 3]], dtype=np.int32
        )

        x = Tensor.from_dlpack(x_np)
        updates = Tensor.from_dlpack(updates_np)
        indices = Tensor.from_dlpack(indices_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.scatter(x, updates, indices, axis=1)

        expected = self._scatter_ref(x_np, updates_np, indices_np, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestConv2dOp:
    """Tests for the mo.conv (2D forward convolution) interpreter handler."""

    @staticmethod
    def _conv2d_ref(
        x: np.ndarray,
        filt: np.ndarray,
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int, int, int],
        groups: int,
    ) -> np.ndarray:
        """Pure-numpy 2D convolution reference (NHWC input, RSCF filter)."""
        n, in_h, in_w, _in_c = x.shape
        kh, kw, ic_pg, out_c = filt.shape
        sh, sw = stride
        dh, dw = dilation
        ph0, ph1, pw0, pw1 = padding

        oh = 1 + (in_h + ph0 + ph1 - (1 + dh * (kh - 1))) // sh
        ow = 1 + (in_w + pw0 + pw1 - (1 + dw * (kw - 1))) // sw
        oc_pg = out_c // groups

        out = np.zeros((n, oh, ow, out_c), dtype=x.dtype)
        for b in range(n):
            for g in range(groups):
                for ohi in range(oh):
                    for owi in range(ow):
                        for oci in range(oc_pg):
                            acc = np.float64(0)
                            for fi in range(kh):
                                ih = ohi * sh - ph0 + fi * dh
                                if ih < 0 or ih >= in_h:
                                    continue
                                for fj in range(kw):
                                    iw = owi * sw - pw0 + fj * dw
                                    if iw < 0 or iw >= in_w:
                                        continue
                                    for ic in range(ic_pg):
                                        acc += float(
                                            x[b, ih, iw, g * ic_pg + ic]
                                        ) * float(
                                            filt[fi, fj, ic, g * oc_pg + oci]
                                        )
                            out[b, ohi, owi, g * oc_pg + oci] = acc
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_3x3(self, dtype: DType) -> None:
        """Test basic 3x3 conv, stride 1, no padding."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(1 * 5 * 5 * 1, dtype=np_dt).reshape(1, 5, 5, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np_dt)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_stride_2(self) -> None:
        """Test 2x2 conv with stride 2."""
        x_np = np.arange(1 * 6 * 6 * 1, dtype=np.float32).reshape(1, 6, 6, 1)
        f_np = np.ones((2, 2, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, stride=(2, 2))

        expected = self._conv2d_ref(x_np, f_np, (2, 2), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_padding(self) -> None:
        """Test 3x3 conv with symmetric padding."""
        x_np = np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 4, 4, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)
        padding = (1, 1, 1, 1)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, padding=padding)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), padding, 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_dilation(self) -> None:
        """Test 3x3 conv with dilation 2."""
        x_np = np.arange(1 * 7 * 7 * 1, dtype=np.float32).reshape(1, 7, 7, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, dilation=(2, 2))

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (2, 2), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_groups(self) -> None:
        """Test grouped convolution (groups=2)."""
        x_np = np.arange(1 * 4 * 4 * 4, dtype=np.float32).reshape(1, 4, 4, 4)
        f_np = np.ones((3, 3, 2, 4), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f, groups=2)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 2)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_1x1_conv(self) -> None:
        """Test pointwise (1x1) convolution."""
        x_np = np.arange(1 * 3 * 3 * 2, dtype=np.float32).reshape(1, 3, 3, 2)
        f_np = np.array([[[[1, -1], [0, 1]]]], dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_non_square_kernel(self) -> None:
        """Test non-square (2, 3) kernel."""
        x_np = np.arange(1 * 5 * 6 * 1, dtype=np.float32).reshape(1, 5, 6, 1)
        f_np = np.ones((2, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d(x, f)

        expected = self._conv2d_ref(x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), 1)
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )


class TestConvTranspose2dOp:
    """Tests for the mo.conv_transpose (2D transposed conv) handler."""

    @staticmethod
    def _conv_transpose2d_ref(
        x: np.ndarray,
        filt: np.ndarray,
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int, int, int],
        output_padding: tuple[int, int],
    ) -> np.ndarray:
        """Pure-numpy 2D transposed conv reference (NHWC, RSCF filter).

        Filter layout for conv_transpose: [kH, kW, out_c, in_c].
        """
        n, in_h, in_w, in_c = x.shape
        kh, kw, out_c, filt_in_c = filt.shape
        assert filt_in_c == in_c
        sh, sw = stride
        dh, dw = dilation
        ph0, ph1, pw0, pw1 = padding
        oph, opw = output_padding

        oh = (in_h - 1) * sh - ph0 - ph1 + dh * (kh - 1) + 1 + oph
        ow = (in_w - 1) * sw - pw0 - pw1 + dw * (kw - 1) + 1 + opw

        out = np.zeros((n, oh, ow, out_c), dtype=x.dtype)
        for b in range(n):
            for ohi in range(oh):
                for owi in range(ow):
                    for oci in range(out_c):
                        acc = np.float64(0)
                        for fi in range(kh):
                            h_cand = ohi + ph0 - fi * dh
                            if h_cand < 0 or h_cand % sh != 0:
                                continue
                            ih = h_cand // sh
                            if ih >= in_h:
                                continue
                            for fj in range(kw):
                                w_cand = owi + pw0 - fj * dw
                                if w_cand < 0 or w_cand % sw != 0:
                                    continue
                                iw = w_cand // sw
                                if iw >= in_w:
                                    continue
                                for ic in range(in_c):
                                    acc += float(x[b, ih, iw, ic]) * float(
                                        filt[fi, fj, oci, ic]
                                    )
                        out[b, ohi, owi, oci] = acc
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_3x3(self, dtype: DType) -> None:
        """Test basic 3x3 conv_transpose, stride 1, no padding."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(1 * 3 * 3 * 1, dtype=np_dt).reshape(1, 3, 3, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np_dt)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d_transpose(x, f)

        expected = self._conv_transpose2d_ref(
            x_np, f_np, (1, 1), (1, 1), (0, 0, 0, 0), (0, 0)
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_stride_2(self) -> None:
        """Test conv_transpose with stride 2 (upsampling)."""
        x_np = np.arange(1 * 2 * 2 * 1, dtype=np.float32).reshape(1, 2, 2, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d_transpose(x, f, stride=(2, 2))

        expected = self._conv_transpose2d_ref(
            x_np, f_np, (2, 2), (1, 1), (0, 0, 0, 0), (0, 0)
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_padding(self) -> None:
        """Test conv_transpose with non-zero padding."""
        x_np = np.arange(1 * 4 * 4 * 1, dtype=np.float32).reshape(1, 4, 4, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)
        padding = (1, 1, 1, 1)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d_transpose(x, f, padding=padding)

        expected = self._conv_transpose2d_ref(
            x_np, f_np, (1, 1), (1, 1), padding, (0, 0)
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_dilation(self) -> None:
        """Test conv_transpose with dilation."""
        x_np = np.arange(1 * 3 * 3 * 1, dtype=np.float32).reshape(1, 3, 3, 1)
        f_np = np.ones((3, 3, 1, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        f = Tensor.from_dlpack(f_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.conv2d_transpose(x, f, dilation=(2, 2))

        expected = self._conv_transpose2d_ref(
            x_np, f_np, (1, 1), (2, 2), (0, 0, 0, 0), (0, 0)
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )


class TestMaxPoolOp:
    """Tests for max_pool2d op via MO interpreter (CPU+GPU)."""

    @staticmethod
    def _max_pool_ref(
        x_nhwc: np.ndarray,
        kernel: tuple[int, int],
        stride: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int, int, int],
        ceil_mode: bool = False,
    ) -> np.ndarray:
        """Pure-numpy NHWC max_pool2d reference implementation."""
        n, h, w, c = x_nhwc.shape
        kh, kw = kernel
        sh, sw = stride
        dh, dw = dilation
        ph_b, ph_a, pw_b, pw_a = padding

        def _out_dim(in_dim: int, k: int, s: int, d: int, pad: int) -> int:
            num = in_dim + pad - (d * (k - 1) + 1)
            if ceil_mode:
                return 1 + -(-num // s)
            return 1 + num // s

        oh = _out_dim(h, kh, sh, dh, ph_b + ph_a)
        ow = _out_dim(w, kw, sw, dw, pw_b + pw_a)

        out = np.full((n, oh, ow, c), -np.inf, dtype=x_nhwc.dtype)
        for bi in range(n):
            for ohi in range(oh):
                for owi in range(ow):
                    for ki in range(kh):
                        ih = ohi * sh - ph_b + ki * dh
                        if ih < 0 or ih >= h:
                            continue
                        for kj in range(kw):
                            iw = owi * sw - pw_b + kj * dw
                            if iw < 0 or iw >= w:
                                continue
                            out[bi, ohi, owi, :] = np.maximum(
                                out[bi, ohi, owi, :],
                                x_nhwc[bi, ih, iw, :],
                            )
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_2x2(self, dtype: DType) -> None:
        """Test 2x2 max pool, stride 1, no padding."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(16, dtype=np_dtype).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2))

        expected = self._max_pool_ref(
            x_np, (2, 2), (1, 1), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_stride_2(self) -> None:
        """Test 2x2 max pool with stride 2."""
        x_np = np.arange(36, dtype=np.float32).reshape(1, 6, 6, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        expected = self._max_pool_ref(
            x_np, (2, 2), (2, 2), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_padding(self) -> None:
        """Test 3x3 max pool with padding=1."""
        x_np = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), padding=1)

        expected = self._max_pool_ref(
            x_np, (3, 3), (1, 1), (1, 1), (1, 1, 1, 1)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_dilation(self) -> None:
        """Test 3x3 max pool with dilation=2."""
        x_np = np.arange(49, dtype=np.float32).reshape(1, 7, 7, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), dilation=2)

        expected = self._max_pool_ref(
            x_np, (3, 3), (1, 1), (2, 2), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_ceil_mode(self) -> None:
        """Test ceil_mode produces larger output than floor mode."""
        x_np = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y_floor = F.max_pool2d(
                x, kernel_size=(2, 2), stride=3, ceil_mode=False
            )
            y_ceil = F.max_pool2d(
                x, kernel_size=(2, 2), stride=3, ceil_mode=True
            )

        floor_shape = np.from_dlpack(y_floor).shape
        ceil_shape = np.from_dlpack(y_ceil).shape
        assert ceil_shape[1] >= floor_shape[1]
        assert ceil_shape[2] >= floor_shape[2]

        expected_floor = self._max_pool_ref(
            x_np, (2, 2), (3, 3), (1, 1), (0, 0, 0, 0), ceil_mode=False
        )
        expected_ceil = self._max_pool_ref(
            x_np, (2, 2), (3, 3), (1, 1), (0, 0, 0, 0), ceil_mode=True
        )
        np.testing.assert_array_equal(np.from_dlpack(y_floor), expected_floor)
        np.testing.assert_array_equal(np.from_dlpack(y_ceil), expected_ceil)

    def test_non_square_kernel(self) -> None:
        """Test max pool with non-square kernel (2, 3)."""
        x_np = np.arange(24, dtype=np.float32).reshape(1, 4, 6, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 3), stride=(1, 2))

        expected = self._max_pool_ref(
            x_np, (2, 3), (1, 2), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_multi_channel_batch(self) -> None:
        """Test max pool with multiple channels and batch size > 1."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 8, 8, 3)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(3, 3), stride=2, padding=1)

        expected = self._max_pool_ref(
            x_np, (3, 3), (2, 2), (1, 1), (1, 1, 1, 1)
        )
        np.testing.assert_array_almost_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test max pool with integer data dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.arange(16, dtype=np_dtype).reshape(1, 4, 4, 1)
        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        expected = self._max_pool_ref(
            x_np, (2, 2), (2, 2), (1, 1), (0, 0, 0, 0)
        )
        np.testing.assert_array_equal(np.from_dlpack(y), expected)


class TestTileOp:
    """Tests for the mo.tile interpreter handler."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_1d(self, dtype: DType) -> None:
        """Test tiling a 1-D tensor."""
        x_np = np.array([1, 2, 3], dtype=dtype.to_numpy())
        reps = (3,)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize(
        "reps", [(2, 3), (1, 4), (3, 1)], ids=["2x3", "1x4", "3x1"]
    )
    def test_2d(self, dtype: DType, reps: tuple[int, ...]) -> None:
        """Test tiling a 2-D tensor with various repeat patterns."""
        x_np = np.arange(6, dtype=dtype.to_numpy()).reshape(2, 3)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    def test_3d(self) -> None:
        """Test tiling a 3-D tensor along all axes."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        reps = (2, 3, 2)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    def test_repeat_one(self) -> None:
        """Test tile with all repeats = 1 (identity)."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        reps = (1, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)

    def test_large_repeat(self) -> None:
        """Test tile with a large repeat factor on one axis."""
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        reps = (1, 10)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))

    @pytest.mark.parametrize(
        "dtype",
        [DType.int32, DType.int64],
        ids=["int32", "int64"],
    )
    def test_integer_dtypes(self, dtype: DType) -> None:
        """Test tile with integer dtypes."""
        x_np = np.arange(6, dtype=dtype.to_numpy()).reshape(2, 3)
        reps = (2, 2)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.tile(x, reps)

        np.testing.assert_array_equal(np.from_dlpack(y), np.tile(x_np, reps))


class TestBandPartOp:
    """Tests for the mo.linalg.band_part interpreter handler."""

    @staticmethod
    def _band_part_ref(
        x: np.ndarray,
        num_lower: int,
        num_upper: int,
        exclude: bool = False,
    ) -> np.ndarray:
        """Pure-numpy band_part reference."""
        shape = x.shape
        M, N = shape[-2], shape[-1]
        m = np.arange(M)[:, None]
        n = np.arange(N)[None, :]
        lower_ok = (num_lower < 0) | ((m - n) <= num_lower)
        upper_ok = (num_upper < 0) | ((n - m) <= num_upper)
        in_band = lower_ok & upper_ok
        if exclude:
            in_band = ~in_band
        mask = np.broadcast_to(in_band, shape)
        return np.where(mask, x, np.zeros_like(x))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_lower_triangle(self, dtype: DType) -> None:
        """Test lower triangle: num_lower=None (-1), num_upper=0."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dt).reshape(3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=0)

        expected = self._band_part_ref(x_np, -1, 0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_upper_triangle(self) -> None:
        """Test upper triangle: num_lower=0, num_upper=None (-1)."""
        x_np = np.arange(12, dtype=np.float32).reshape(3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=0, num_upper=None)

        expected = self._band_part_ref(x_np, 0, -1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_diagonal(self) -> None:
        """Test diagonal only: num_lower=0, num_upper=0."""
        x_np = np.arange(9, dtype=np.float32).reshape(3, 3) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=0, num_upper=0)

        expected = self._band_part_ref(x_np, 0, 0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_band(self) -> None:
        """Test tridiagonal band: num_lower=1, num_upper=1."""
        x_np = np.arange(20, dtype=np.float32).reshape(4, 5) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=1, num_upper=1)

        expected = self._band_part_ref(x_np, 1, 1)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_exclude(self) -> None:
        """Test inverted mask: exclude=True zeroes the band."""
        x_np = np.arange(9, dtype=np.float32).reshape(3, 3) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=0, exclude=True)

        expected = self._band_part_ref(x_np, -1, 0, exclude=True)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    def test_batched(self) -> None:
        """Test batched input [B, M, N]."""
        x_np = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=1, num_upper=0)

        expected = self._band_part_ref(x_np, 1, 0)
        np.testing.assert_array_equal(np.from_dlpack(y), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_full_matrix(self, dtype: DType) -> None:
        """Test keeping entire matrix: num_lower=None, num_upper=None."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(12, dtype=np_dt).reshape(3, 4) + 1

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.band_part(x, num_lower=None, num_upper=None)

        np.testing.assert_array_equal(np.from_dlpack(y), x_np)


class TestAvgPool2dOp:
    """Tests for the mo.avg_pool / mo.avg_pool_ceil_mode_true interpreter
    handler."""

    @staticmethod
    def _avg_pool2d_ref(
        x: np.ndarray,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        dilation: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        ceil_mode: bool = False,
        count_boundary: bool = True,
    ) -> np.ndarray:
        """Pure-numpy avg_pool2d reference (NHWC layout)."""
        N, H, W, C = x.shape
        kH, kW = kernel_size
        sH, sW = stride
        dH, dW = dilation
        pH, pW = padding

        eff_kH = dH * (kH - 1) + 1
        eff_kW = dW * (kW - 1) + 1
        if ceil_mode:
            oH = int(np.ceil((H + 2 * pH - eff_kH + 1) / sH))
            oW = int(np.ceil((W + 2 * pW - eff_kW + 1) / sW))
        else:
            oH = (H + 2 * pH - eff_kH) // sH + 1
            oW = (W + 2 * pW - eff_kW) // sW + 1

        out = np.zeros((N, oH, oW, C), dtype=x.dtype)
        for n in range(N):
            for oh in range(oH):
                for ow in range(oW):
                    for c in range(C):
                        s = 0.0
                        cnt = 0
                        for fh in range(kH):
                            ih = oh * sH - pH + fh * dH
                            if ih < 0 or ih >= H:
                                if count_boundary:
                                    cnt += kW
                                continue
                            for fw in range(kW):
                                iw = ow * sW - pW + fw * dW
                                if iw < 0 or iw >= W:
                                    if count_boundary:
                                        cnt += 1
                                    continue
                                s += float(x[n, ih, iw, c])
                                cnt += 1
                        out[n, oh, ow, c] = s / cnt if cnt > 0 else 0.0
        return out

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_basic_2x2(self, dtype: DType) -> None:
        """Test 2x2 kernel, stride 1, no padding."""
        np_dt = dtype.to_numpy()
        x_np = np.arange(16, dtype=np_dt).reshape(1, 4, 4, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(2, 2))

        expected = self._avg_pool2d_ref(x_np, (2, 2))
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_stride_and_padding(self) -> None:
        """Test stride 2 with padding 1."""
        x_np = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(
                x, kernel_size=(3, 3), stride=2, padding=1, count_boundary=True
            )

        expected = self._avg_pool2d_ref(
            x_np, (3, 3), stride=(2, 2), padding=(1, 1), count_boundary=True
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_dilation(self) -> None:
        """Test dilated average pooling."""
        x_np = np.arange(36, dtype=np.float32).reshape(1, 6, 6, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(2, 2), dilation=2)

        expected = self._avg_pool2d_ref(x_np, (2, 2), dilation=(2, 2))
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_ceil_mode(self) -> None:
        """Test ceil mode output shape."""
        x_np = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(x, kernel_size=(3, 3), stride=2, ceil_mode=True)

        expected = self._avg_pool2d_ref(
            x_np, (3, 3), stride=(2, 2), ceil_mode=True
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )

    def test_count_boundary_false(self) -> None:
        """Test excluding padding from divisor."""
        x_np = np.ones((1, 3, 3, 1), dtype=np.float32)

        x = Tensor.from_dlpack(x_np)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.avg_pool2d(
                x,
                kernel_size=(3, 3),
                padding=1,
                count_boundary=False,
            )

        expected = self._avg_pool2d_ref(
            x_np, (3, 3), padding=(1, 1), count_boundary=False
        )
        np.testing.assert_allclose(
            np.from_dlpack(y), expected, rtol=1e-5, atol=1e-5
        )


class TestTopKOp:
    """Tests for TopK interpreter op (mo.top_k).

    Uses F.top_k which routes through ops.top_k -> rmo.top_k -> mo.top_k ->
    interpreter handler.  The reference implementation uses numpy stable
    argsort to match the kernel's selection-sort ordering.
    """

    @staticmethod
    def _top_k_ref(
        x_np: np.ndarray, k: int, axis: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy reference: top-k values and original indices, descending."""
        sorted_idx = np.argsort(
            -x_np.astype(np.float64), axis=axis, stable=True
        )
        idx = np.take(sorted_idx, np.arange(k), axis=axis)
        vals = np.take_along_axis(x_np, idx, axis=axis)
        return vals, idx

    @pytest.mark.parametrize("axis", [-1, 0])
    def test_basic_2d(self, axis: int) -> None:
        """Test top-2 on a 2D tensor along axis 0 and -1."""
        x_np = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=2, axis=axis)

        ref_vals, ref_idxs = self._top_k_ref(x_np, k=2, axis=axis)
        np.testing.assert_array_equal(np.from_dlpack(vals), ref_vals)
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)

    def test_3d_middle_axis(self) -> None:
        """Test top-3 on a 3D tensor along axis 1."""
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 6, 4)).astype(np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=3, axis=1)

        ref_vals, ref_idxs = self._top_k_ref(x_np, k=3, axis=1)
        np.testing.assert_allclose(
            np.from_dlpack(vals), ref_vals, rtol=1e-6, atol=0
        )
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)

    def test_k_equals_1(self) -> None:
        """k=1 must return the same element as argmax."""
        x_np = np.array([3.0, 7.0, 1.0, 5.0], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=1, axis=0)
            argmax_result = F.argmax(x, axis=0)

        np.testing.assert_array_equal(
            np.from_dlpack(idxs), np.from_dlpack(argmax_result)
        )
        np.testing.assert_allclose(np.from_dlpack(vals), np.array([7.0]))

    def test_k_equals_dim(self) -> None:
        """k equal to the axis size returns a full sorted permutation."""
        x_np = np.array([[4.0, 2.0, 1.0, 3.0]], dtype=np.float32)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=4, axis=1)

        np.testing.assert_array_equal(
            np.from_dlpack(vals), np.array([[4.0, 3.0, 2.0, 1.0]])
        )
        np.testing.assert_array_equal(
            np.from_dlpack(idxs), np.array([[0, 3, 1, 2]])
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.int32]
    )
    def test_dtypes(self, dtype: DType) -> None:
        """Test top-2 with numeric dtypes."""
        np_dtype = dtype.to_numpy()
        x_np = np.array([[5, 1, 8, 3], [9, 2, 7, 4]], dtype=np_dtype)
        x = Tensor.from_dlpack(x_np)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            vals, idxs = F.top_k(x, k=2, axis=1)

        ref_vals, ref_idxs = self._top_k_ref(x_np, k=2, axis=1)
        np.testing.assert_array_equal(np.from_dlpack(vals), ref_vals)
        np.testing.assert_array_equal(np.from_dlpack(idxs), ref_idxs)
