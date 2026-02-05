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
from max._interpreter import MOInterpreter
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine.api import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops

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
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("add_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.add(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(12, 24, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.add(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_sub(self, dtype: DType) -> None:
        """Test sub op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("sub_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.sub(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, 24, dtype=np_dtype).reshape(shape)
        b_np = np.arange(12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.subtract(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_mul(self, dtype: DType) -> None:
        """Test mul op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("mul_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.mul(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.arange(2, 14, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.multiply(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_div(self, dtype: DType) -> None:
        """Test div op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("div_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.div(a, b)  # type: ignore[arg-type]
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(1, 13, dtype=np_dtype).reshape(shape)
        b_np = np.arange(1, 13, dtype=np_dtype).reshape(shape) + 0.5

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.divide(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_max(self, dtype: DType) -> None:
        """Test elementwise max op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("max_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.max(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(11, -1, -1, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.maximum(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_min(self, dtype: DType) -> None:
        """Test elementwise min op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("min_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.min(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(12, dtype=np_dtype).reshape(shape)
        b_np = np.arange(11, -1, -1, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.minimum(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestBinaryComparisonOps:
    """Tests for binary comparison Mojo ops (output is bool)."""

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_equal(self, dtype: DType) -> None:
        """Test equal op returns bool and matches numpy."""
        device = CPU()
        shape = [4]
        input_type = TensorType(dtype, shape, device)

        with Graph("eq_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.equal(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        b_np = np.array([1, 5, 3, 6], dtype=np_dtype)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)
        result_np = result.to_numpy()

        expected = np.equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_not_equal(self, dtype: DType) -> None:
        """Test not_equal op returns bool and matches numpy."""
        device = CPU()
        shape = [4]
        input_type = TensorType(dtype, shape, device)

        with Graph("ne_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.not_equal(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 2, 3, 4], dtype=np_dtype)
        b_np = np.array([1, 5, 3, 6], dtype=np_dtype)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)
        result_np = result.to_numpy()

        expected = np.not_equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_greater(self, dtype: DType) -> None:
        """Test greater op returns bool and matches numpy."""
        device = CPU()
        shape = [4]
        input_type = TensorType(dtype, shape, device)

        with Graph("gt_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.greater(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 5, 3, 6], dtype=np_dtype)
        b_np = np.array([2, 3, 3, 4], dtype=np_dtype)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        result_np = result.to_numpy()
        expected = np.greater(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_

    @pytest.mark.parametrize("dtype", ELEMENTWISE_DTYPES)
    def test_greater_equal(self, dtype: DType) -> None:
        """Test greater_equal op returns bool and matches numpy."""
        device = CPU()
        shape = [4]
        input_type = TensorType(dtype, shape, device)

        with Graph("ge_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.greater_equal(a, b)
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.array([1, 5, 3, 6], dtype=np_dtype)
        b_np = np.array([2, 3, 3, 4], dtype=np_dtype)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)
        result_np = result.to_numpy()

        expected = np.greater_equal(a_np, b_np)
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_


class TestUnaryElementwiseOps:
    """Tests for unary elementwise Mojo ops."""

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_negative(self, dtype: DType) -> None:
        """Test negative op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("neg_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.negate(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.arange(-6, 6, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.negative(x_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", SIGNED_DTYPES)
    def test_abs(self, dtype: DType) -> None:
        """Test abs op matches numpy for signed types."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("abs_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.abs(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.arange(-6, 6, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.abs(x_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", UINT_DTYPES)
    def test_abs_unsigned(self, dtype: DType) -> None:
        """Test abs op matches numpy for unsigned types."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("abs_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.abs(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        # Use non-negative values for unsigned types
        x_np = np.arange(0, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.abs(x_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_exp(self, dtype: DType) -> None:
        """Test exp op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("exp_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.exp(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-2, 2, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.exp(x_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_log(self, dtype: DType) -> None:
        """Test log op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("log_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.log(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(0.1, 10, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.log(x_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sqrt(self, dtype: DType) -> None:
        """Test sqrt op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("sqrt_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.sqrt(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(0, 10, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.sqrt(x_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_tanh(self, dtype: DType) -> None:
        """Test tanh op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("tanh_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.tanh(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-3, 3, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.tanh(x_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_relu(self, dtype: DType) -> None:
        """Test relu op matches numpy maximum(x, 0)."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("relu_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.relu(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-3, 3, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.maximum(x_np, 0)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_sin(self, dtype: DType) -> None:
        """Test sin op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("sin_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.sin(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-np.pi, np.pi, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.sin(x_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_cos(self, dtype: DType) -> None:
        """Test cos op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("cos_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.cos(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-np.pi, np.pi, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.cos(x_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_floor(self, dtype: DType) -> None:
        """Test floor op matches numpy."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("floor_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.floor(x)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.linspace(-2.5, 2.5, 12, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.floor(x_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestBooleanLogicOps:
    """Tests for boolean logic Mojo ops."""

    def test_and(self) -> None:
        """Test logical and op."""
        device = CPU()
        shape = [4]
        input_type = TensorType(DType.bool, shape, device)

        with Graph("and_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.logical_and(a, b)
            graph.output(c)

        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.logical_and(a_np, b_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_or(self) -> None:
        """Test logical or op."""
        device = CPU()
        shape = [4]
        input_type = TensorType(DType.bool, shape, device)

        with Graph("or_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.logical_or(a, b)
            graph.output(c)

        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.logical_or(a_np, b_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_xor(self) -> None:
        """Test logical xor op."""
        device = CPU()
        shape = [4]
        input_type = TensorType(DType.bool, shape, device)

        with Graph("xor_test", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.logical_xor(a, b)
            graph.output(c)

        a_np = np.array([True, True, False, False], dtype=np.bool_)
        b_np = np.array([True, False, True, False], dtype=np.bool_)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.logical_xor(a_np, b_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_not(self) -> None:
        """Test logical not op."""
        device = CPU()
        shape = [4]
        input_type = TensorType(DType.bool, shape, device)

        with Graph("not_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.logical_not(x)
            graph.output(y)

        x_np = np.array([True, False, True, False], dtype=np.bool_)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.logical_not(x_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestChainedOperations:
    """Tests for chained operations using Mojo ops."""

    def test_chained_arithmetic(self) -> None:
        """Test chained add/sub/mul operations."""
        device = CPU()
        shape = [3, 4]
        input_type = TensorType(DType.float32, shape, device)

        with Graph("chain_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            # (x + 1) * 2 - 3
            one = ops.constant(np.ones(shape, dtype=np.float32), device=device)
            two = ops.constant(
                np.full(shape, 2.0, dtype=np.float32), device=device
            )
            three = ops.constant(
                np.full(shape, 3.0, dtype=np.float32), device=device
            )
            y = ops.add(x, one)
            z = ops.mul(y, two)
            result = ops.sub(z, three)
            graph.output(result)

        x_np = np.arange(12, dtype=np.float32).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = (x_np + 1) * 2 - 3
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_comparison_with_arithmetic(self) -> None:
        """Test combining comparisons with arithmetic operations."""
        device = CPU()
        shape = [4]
        input_type = TensorType(DType.float32, shape, device)

        with Graph("cmp_arith_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            # Compare (x * 2) > 5
            two = ops.constant(
                np.full(shape, 2.0, dtype=np.float32), device=device
            )
            five = ops.constant(
                np.full(shape, 5.0, dtype=np.float32), device=device
            )
            scaled = ops.mul(x, two)
            result = ops.greater(scaled, five)
            graph.output(result)

        x_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)
        result_np = result.to_numpy()

        expected = (x_np * 2) > 5
        np.testing.assert_array_equal(result_np, expected)
        assert result_np.dtype == np.bool_


class TestBasicOpExecution:
    """Tests for basic op execution through the interpreter."""

    def test_add_two_constants(self) -> None:
        """Test adding two constants."""
        with Graph("add_constants", input_types=[]) as graph:
            a = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=CPU()
            )
            b = ops.constant(
                [[5.0, 6.0], [7.0, 8.0]], dtype=DType.float32, device=CPU()
            )
            c = ops.add(a, b)
            graph.output(c)

        device = CPU()
        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array([[6.0, 8.0], [10.0, 12.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_mul_two_constants(self) -> None:
        """Test multiplying two constants."""
        with Graph("mul_constants", input_types=[]) as graph:
            a = ops.constant([2.0, 3.0, 4.0], dtype=DType.float32, device=CPU())
            b = ops.constant([5.0, 6.0, 7.0], dtype=DType.float32, device=CPU())
            c = ops.mul(a, b)
            graph.output(c)

        device = CPU()
        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.array([10.0, 18.0, 28.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_unary_operations(self) -> None:
        """Test unary operations like exp, sqrt, tanh."""
        with Graph("unary_ops", input_types=[]) as graph:
            x = ops.constant([0.0, 1.0, 2.0], dtype=DType.float32, device=CPU())
            exp_x = ops.exp(x)
            graph.output(exp_x)

        device = CPU()
        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)

        expected = np.exp(np.array([0.0, 1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )


class TestMutableLoadOp:
    """Tests for MutableLoadOp handling in the interpreter."""

    def test_buffer_load_basic(self) -> None:
        """Test that buffer_load correctly loads buffer contents."""
        device = CPU()
        buffer_type = BufferType(DType.float32, [3, 4], DeviceRef.CPU())

        with Graph("buffer_load_test", input_types=[buffer_type]) as graph:
            buffer = graph.inputs[0].buffer
            tensor = ops.buffer_load(buffer)
            graph.output(tensor)

        # Create input buffer
        input_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        input_buffer = Buffer.from_numpy(input_np)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [input_buffer])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        np.testing.assert_array_almost_equal(result.to_numpy(), input_np)

    def test_buffer_load_multiple_dtypes(self) -> None:
        """Test buffer_load with different dtypes."""
        device = CPU()

        for dtype, np_dtype in [
            (DType.float32, np.float32),
            (DType.float64, np.float64),
            (DType.int32, np.int32),
            (DType.int64, np.int64),
        ]:
            buffer_type = BufferType(dtype, [4], DeviceRef.CPU())

            with Graph(
                f"buffer_load_{dtype}", input_types=[buffer_type]
            ) as graph:
                buffer = graph.inputs[0].buffer
                tensor = ops.buffer_load(buffer)
                graph.output(tensor)

            input_np = np.array([1, 2, 3, 4], dtype=np_dtype)
            input_buffer = Buffer.from_numpy(input_np)

            interp = MOInterpreter()
            outputs = interp.execute(graph, [input_buffer])

            assert len(outputs) == 1
            result = outputs[0]
            assert isinstance(result, Buffer)
            np.testing.assert_array_equal(result.to_numpy(), input_np)

    def test_buffer_load_preserves_shape(self) -> None:
        """Test that buffer_load preserves tensor shape."""
        device = CPU()

        # Test various shapes
        for shape in [[4], [2, 3], [2, 3, 4], [1, 2, 3, 4]]:
            buffer_type = BufferType(DType.float32, shape, DeviceRef.CPU())

            with Graph(
                f"buffer_load_shape_{len(shape)}d", input_types=[buffer_type]
            ) as graph:
                buffer = graph.inputs[0].buffer
                tensor = ops.buffer_load(buffer)
                graph.output(tensor)

            size = 1
            for dim in shape:
                size *= dim
            input_np = np.arange(size, dtype=np.float32).reshape(shape)
            input_buffer = Buffer.from_numpy(input_np)

            interp = MOInterpreter()
            outputs = interp.execute(graph, [input_buffer])

            assert len(outputs) == 1
            result = outputs[0]
            assert isinstance(result, Buffer)
            assert result.shape == tuple(shape)
            np.testing.assert_array_almost_equal(result.to_numpy(), input_np)


class TestShapeOps:
    """Tests for shape operations (rebind, broadcast_to) in the interpreter."""

    def test_broadcast_to_static_shape(self) -> None:
        """Test that broadcast_to correctly broadcasts to a static target shape."""
        device = CPU()

        with Graph("broadcast_static", input_types=[]) as graph:
            # Create a 1D tensor and broadcast to 2D
            x = ops.constant(
                [1.0, 2.0, 3.0], dtype=DType.float32, device=device
            )
            # Broadcast [3] -> [2, 3]
            y = ops.broadcast_to(x, shape=[2, 3])
            graph.output(y)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        expected = np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_to_higher_rank(self) -> None:
        """Test broadcasting to a higher rank tensor."""
        device = CPU()

        with Graph("broadcast_higher_rank", input_types=[]) as graph:
            # Create a scalar-like 1D tensor and broadcast to 3D
            x = ops.constant([5.0], dtype=DType.float32, device=device)
            # Broadcast [1] -> [2, 3, 4]
            y = ops.broadcast_to(x, shape=[2, 3, 4])
            graph.output(y)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        expected = np.full((2, 3, 4), 5.0, dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_to_2d_to_3d(self) -> None:
        """Test broadcasting a 2D tensor to 3D."""
        device = CPU()

        with Graph("broadcast_2d_to_3d", input_types=[]) as graph:
            # Create a 2D tensor and broadcast to 3D
            x = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], dtype=DType.float32, device=device
            )
            # Broadcast [2, 2] -> [3, 2, 2]
            y = ops.broadcast_to(x, shape=[3, 2, 2])
            graph.output(y)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        input_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = np.broadcast_to(input_np, (3, 2, 2))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_broadcast_then_add(self) -> None:
        """Test broadcasting followed by element-wise operation."""
        device = CPU()

        with Graph("broadcast_then_add", input_types=[]) as graph:
            # Create tensors of different shapes
            x = ops.constant(
                [1.0, 2.0, 3.0], dtype=DType.float32, device=device
            )
            y = ops.constant(
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                dtype=DType.float32,
                device=device,
            )
            # Broadcast x from [3] to [2, 3] and add to y
            x_broadcast = ops.broadcast_to(x, shape=[2, 3])
            z = ops.add(x_broadcast, y)
            graph.output(z)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        expected = np.array(
            [[11.0, 22.0, 33.0], [41.0, 52.0, 63.0]], dtype=np.float32
        )
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestInterpreterVsCompiled:
    """Tests comparing interpreter results to compiled execution."""

    def test_interpreter_matches_compiled_add(self) -> None:
        """Test that interpreter add matches compiled add."""
        device = CPU()
        input_type = TensorType(DType.float32, [4], device)

        # Build graph
        with Graph("add_graph", input_types=[input_type, input_type]) as graph:
            a, b = graph.inputs
            c = ops.add(a, b)
            graph.output(c)

        # Test data
        a_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

        # Need to build fresh graphs since graphs can only be used once
        with Graph(
            "add_graph_interp", input_types=[input_type, input_type]
        ) as interp_graph:
            a, b = interp_graph.inputs
            c = ops.add(a, b)
            interp_graph.output(c)

        with Graph(
            "add_graph_compiled", input_types=[input_type, input_type]
        ) as compiled_graph:
            a, b = compiled_graph.inputs
            c = ops.add(a, b)
            compiled_graph.output(c)

        # Execute via interpreter
        interp = MOInterpreter()
        interp_result = interp.execute(
            interp_graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(interp_result, Buffer)

        # Execute via compiled path
        session = InferenceSession(devices=[device])
        model = session.load(compiled_graph)
        compiled_result = model(
            Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)
        )[0].to_numpy()

        # Results should match
        np.testing.assert_array_almost_equal(
            interp_result.to_numpy(), compiled_result
        )

    def test_interpreter_matches_compiled_mul(self) -> None:
        """Test that interpreter mul matches compiled mul."""
        device = CPU()
        input_type = TensorType(DType.float32, [3], device)

        with Graph(
            "mul_graph_interp", input_types=[input_type, input_type]
        ) as interp_graph:
            a, b = interp_graph.inputs
            c = ops.mul(a, b)
            interp_graph.output(c)

        with Graph(
            "mul_graph_compiled", input_types=[input_type, input_type]
        ) as compiled_graph:
            a, b = compiled_graph.inputs
            c = ops.mul(a, b)
            compiled_graph.output(c)

        a_np = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        b_np = np.array([5.0, 6.0, 7.0], dtype=np.float32)

        interp = MOInterpreter()
        interp_result = interp.execute(
            interp_graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(interp_result, Buffer)

        session = InferenceSession(devices=[device])
        model = session.load(compiled_graph)
        compiled_result = model(
            Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)
        )[0].to_numpy()

        np.testing.assert_array_almost_equal(
            interp_result.to_numpy(), compiled_result
        )

    def test_interpreter_matches_compiled_chained(self) -> None:
        """Test that interpreter matches compiled for chained operations."""
        device = CPU()
        input_type = TensorType(DType.float32, [3], device)

        with Graph("chain_interp", input_types=[input_type]) as interp_graph:
            x = interp_graph.inputs[0]
            # x * 2 + 1
            two = ops.constant(
                [2.0, 2.0, 2.0], dtype=DType.float32, device=device
            )
            y = ops.mul(x, two)
            one = ops.constant(
                [1.0, 1.0, 1.0], dtype=DType.float32, device=device
            )
            z = ops.add(y, one)
            interp_graph.output(z)

        with Graph(
            "chain_compiled", input_types=[input_type]
        ) as compiled_graph:
            x = compiled_graph.inputs[0]
            two = ops.constant(
                [2.0, 2.0, 2.0], dtype=DType.float32, device=device
            )
            y = ops.mul(x, two)
            one = ops.constant(
                [1.0, 1.0, 1.0], dtype=DType.float32, device=device
            )
            z = ops.add(y, one)
            compiled_graph.output(z)

        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        interp = MOInterpreter()
        interp_result = interp.execute(interp_graph, [Buffer.from_numpy(x_np)])[
            0
        ]
        assert isinstance(interp_result, Buffer)

        session = InferenceSession(devices=[device])
        model = session.load(compiled_graph)
        compiled_result = model(Buffer.from_numpy(x_np))[0].to_numpy()

        np.testing.assert_array_almost_equal(
            interp_result.to_numpy(), compiled_result
        )


class TestMatmulOp:
    """Tests for matmul Mojo op."""

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_matmul_basic(self, dtype: DType) -> None:
        """Test basic 2D matmul matches numpy."""
        device = CPU()
        lhs_shape = [3, 4]
        rhs_shape = [4, 5]
        lhs_type = TensorType(dtype, lhs_shape, device)
        rhs_type = TensorType(dtype, rhs_shape, device)

        with Graph("matmul_test", input_types=[lhs_type, rhs_type]) as graph:
            a, b = graph.inputs
            c = ops.matmul(a, b)  # type: ignore[arg-type]
            graph.output(c)

        np_dtype = dtype.to_numpy()
        # Use small values to avoid overflow for integer types
        a_np = np.arange(12, dtype=np_dtype).reshape(lhs_shape) % 10
        b_np = np.arange(20, dtype=np_dtype).reshape(rhs_shape) % 10

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", MATMUL_DTYPES)
    def test_matmul_square(self, dtype: DType) -> None:
        """Test square matrix matmul."""
        device = CPU()
        shape = [4, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph(
            "matmul_square_test", input_types=[input_type, input_type]
        ) as graph:
            a, b = graph.inputs
            c = ops.matmul(a, b)  # type: ignore[arg-type]
            graph.output(c)

        np_dtype = dtype.to_numpy()
        a_np = np.arange(16, dtype=np_dtype).reshape(shape) % 5
        b_np = np.arange(16, dtype=np_dtype).reshape(shape) % 5

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_matmul_float_precision(self, dtype: DType) -> None:
        """Test matmul with random floats for precision."""
        device = CPU()
        lhs_shape = [8, 16]
        rhs_shape = [16, 8]
        lhs_type = TensorType(dtype, lhs_shape, device)
        rhs_type = TensorType(dtype, rhs_shape, device)

        with Graph(
            "matmul_float_test", input_types=[lhs_type, rhs_type]
        ) as graph:
            a, b = graph.inputs
            c = ops.matmul(a, b)  # type: ignore[arg-type]
            graph.output(c)

        np_dtype = dtype.to_numpy()
        np.random.seed(42)
        a_np = np.random.randn(8, 16).astype(np_dtype)
        b_np = np.random.randn(16, 8).astype(np_dtype)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(
            result.to_numpy(), expected, decimal=5
        )

    def test_matmul_vector(self) -> None:
        """Test matmul with vector-like shapes."""
        device = CPU()
        dtype = DType.float32
        lhs_type = TensorType(dtype, [1, 4], device)
        rhs_type = TensorType(dtype, [4, 1], device)

        with Graph(
            "matmul_vec_test", input_types=[lhs_type, rhs_type]
        ) as graph:
            a, b = graph.inputs
            c = ops.matmul(a, b)  # type: ignore[arg-type]
            graph.output(c)

        a_np = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        b_np = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

        interp = MOInterpreter()
        result = interp.execute(
            graph, [Buffer.from_numpy(a_np), Buffer.from_numpy(b_np)]
        )[0]
        assert isinstance(result, Buffer)

        expected = np.matmul(a_np, b_np)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestReduceOps:
    """Tests for reduction Mojo ops."""

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
    def test_reduce_max_last_axis(self, dtype: DType) -> None:
        """Test reduce_max on the last axis matches numpy."""
        device = CPU()
        shape = [3, 4, 5]
        input_type = TensorType(dtype, shape, device)

        with Graph("reduce_max_test", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            # ops.max with single input uses reduce_max (not elementwise max)
            y = ops.max(x, axis=-1)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.arange(60, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_first_axis(self, dtype: DType) -> None:
        """Test reduce_max on the first axis."""
        device = CPU()
        shape = [3, 4, 5]
        input_type = TensorType(dtype, shape, device)

        with Graph("reduce_max_axis0", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.max(x, axis=0)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((3, 4, 5)).astype(np_dtype)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.max(x_np, axis=0, keepdims=True)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_middle_axis(self, dtype: DType) -> None:
        """Test reduce_max on a middle axis."""
        device = CPU()
        shape = [2, 3, 4]
        input_type = TensorType(dtype, shape, device)

        with Graph("reduce_max_axis1", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.max(x, axis=1)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        rng = np.random.default_rng(42)
        x_np = rng.standard_normal((2, 3, 4)).astype(np_dtype)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.max(x_np, axis=1, keepdims=True)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_reduce_max_2d(self, dtype: DType) -> None:
        """Test reduce_max on 2D tensor."""
        device = CPU()
        shape = [4, 5]
        input_type = TensorType(dtype, shape, device)

        with Graph("reduce_max_2d", input_types=[input_type]) as graph:
            x = graph.inputs[0]
            y = ops.max(x, axis=-1)
            graph.output(y)

        np_dtype = dtype.to_numpy()
        x_np = np.arange(20, dtype=np_dtype).reshape(shape)

        interp = MOInterpreter()
        result = interp.execute(graph, [Buffer.from_numpy(x_np)])[0]
        assert isinstance(result, Buffer)

        expected = np.max(x_np, axis=-1, keepdims=True)
        np.testing.assert_array_equal(result.to_numpy(), expected)
