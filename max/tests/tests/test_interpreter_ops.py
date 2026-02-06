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

import numpy as np
import pytest
from max import _realization_context as rc
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.tensor import Tensor, realization_context

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


class TestRangeOp:
    """Tests for range Mojo op via Tensor.arange with interpreter."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_basic(self, dtype: DType) -> None:
        """Test basic range op matches numpy arange."""
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(10, dtype=dtype, device=CPU())

        np_dtype = dtype.to_numpy()
        expected = np.arange(0, 10, 1, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_with_step(self, dtype: DType) -> None:
        """Test range op with custom step size."""
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(0, 10, 2, dtype=dtype, device=CPU())

        np_dtype = dtype.to_numpy()
        expected = np.arange(0, 10, 2, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_float_step(self, dtype: DType) -> None:
        """Test range op with float step size."""
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(0.0, 1.0, 0.25, dtype=dtype, device=CPU())

        np_dtype = dtype.to_numpy()
        expected = np.arange(0.0, 1.0, 0.25, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_range_negative_step(self, dtype: DType) -> None:
        """Test range op with negative step."""
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(5, 0, -1, dtype=dtype, device=CPU())

        np_dtype = dtype.to_numpy()
        expected = np.arange(5, 0, -1, dtype=np_dtype)
        np.testing.assert_array_almost_equal(np.from_dlpack(t), expected)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_range_int(self, dtype: DType) -> None:
        """Test range op with integer dtypes."""
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(10, dtype=dtype, device=CPU())

        np_dtype = dtype.to_numpy()
        expected = np.arange(0, 10, 1, dtype=np_dtype)
        np.testing.assert_array_equal(np.from_dlpack(t), expected)

    def test_range_nonzero_start(self) -> None:
        """Test range op with nonzero start value."""
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(5, 15, 2, dtype=DType.float32, device=CPU())

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
