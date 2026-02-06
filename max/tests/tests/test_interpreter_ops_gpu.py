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
"""GPU tests for MO interpreter operations."""

from typing import Any

import pytest
import torch
from max import _realization_context as rc
from max._interpreter import MOInterpreter
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.tensor import Tensor, realization_context

# Mapping from MAX DType to torch dtype
DTYPE_TO_TORCH = {
    DType.float32: torch.float32,
    DType.float16: torch.float16,
    DType.bfloat16: torch.bfloat16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
    DType.uint32: torch.uint32,
    DType.uint64: torch.uint64,
    DType.bool: torch.bool,
}


class TestConstantGPU:
    """Tests for GPU constant handling in the interpreter."""

    def test_constant_on_gpu(self) -> None:
        """Test that constants can be created directly on GPU."""
        gpu = Accelerator()

        with Graph("gpu_constant", input_types=[]) as graph:
            c = ops.constant([1.0, 2.0, 3.0], dtype=DType.float32, device=gpu)
            graph.output(c)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 1
        result = outputs[0]
        assert isinstance(result, Buffer)
        assert not result.device.is_host

        # Verify values using PyTorch
        torch_tensor = torch.from_dlpack(result)
        expected = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(torch_tensor, expected)

    def test_constant_gpu_2d(self) -> None:
        """Test 2D constant on GPU."""
        gpu = Accelerator()

        with Graph("gpu_constant_2d", input_types=[]) as graph:
            c = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=DType.float32,
                device=gpu,
            )
            graph.output(c)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        result = outputs[0]
        assert isinstance(result, Buffer)
        assert not result.device.is_host

        # Verify values using PyTorch
        torch_tensor = torch.from_dlpack(result)
        expected = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda"
        )
        torch.testing.assert_close(torch_tensor, expected)

    def test_constant_mixed_devices(self) -> None:
        """Test graph with constants on both CPU and GPU."""
        cpu = CPU()
        gpu = Accelerator()

        with Graph("mixed_constants", input_types=[]) as graph:
            cpu_const = ops.constant(
                [1.0, 2.0], dtype=DType.float32, device=cpu
            )
            gpu_const = ops.constant(
                [3.0, 4.0], dtype=DType.float32, device=gpu
            )
            graph.output(cpu_const, gpu_const)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        assert len(outputs) == 2
        assert isinstance(outputs[0], Buffer)
        assert isinstance(outputs[1], Buffer)
        assert outputs[0].device.is_host
        assert not outputs[1].device.is_host

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32, DType.int64])
    def test_constant_gpu_dtypes(self, dtype: DType) -> None:
        """Test GPU constants with various dtypes."""
        gpu = Accelerator()

        with Graph(f"gpu_constant_{dtype}", input_types=[]) as graph:
            c = ops.constant([1, 2, 3, 4], dtype=dtype, device=gpu)
            graph.output(c)

        interp = MOInterpreter()
        outputs = interp.execute(graph, [])

        result = outputs[0]
        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == dtype


class TestBinaryComparisonOpsGPU:
    """Tests for GPU binary comparison operations in the interpreter."""

    @pytest.mark.parametrize(
        "op_func,torch_func",
        [
            (ops.equal, torch.eq),
            (ops.not_equal, torch.ne),
            (ops.greater, torch.gt),
            (ops.greater_equal, torch.ge),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_comparison_ops_gpu(
        self, op_func: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test comparison ops on GPU with various dtypes."""
        gpu = Accelerator()
        shape = [4]
        torch_dtype = DTYPE_TO_TORCH[dtype]
        input_type = TensorType(dtype, shape, gpu)

        with Graph(
            f"gpu_comparison_op_{dtype}", input_types=[input_type, input_type]
        ) as graph:
            a, b = graph.inputs
            c = op_func(a, b)
            graph.output(c)

        # Use test data that exercises both equal and unequal cases
        # For greater/greater_equal, use values that test ordering
        if op_func in (ops.greater, ops.greater_equal):
            a_torch = torch.tensor(
                [1, 5, 3, 6], dtype=torch_dtype, device="cuda"
            )
            b_torch = torch.tensor(
                [2, 3, 3, 4], dtype=torch_dtype, device="cuda"
            )
        else:
            a_torch = torch.tensor(
                [1, 2, 3, 4], dtype=torch_dtype, device="cuda"
            )
            b_torch = torch.tensor(
                [1, 5, 3, 6], dtype=torch_dtype, device="cuda"
            )

        a_gpu = Buffer.from_dlpack(a_torch)
        b_gpu = Buffer.from_dlpack(b_torch)

        interp = MOInterpreter()
        result = interp.execute(graph, [a_gpu, b_gpu])[0]

        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == DType.bool

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(result)
        torch.testing.assert_close(result_torch, expected)


class TestBooleanLogicOpsGPU:
    """Tests for GPU boolean logic operations in the interpreter."""

    @pytest.mark.parametrize(
        "op_func,torch_func",
        [
            (ops.logical_and, torch.logical_and),
            (ops.logical_or, torch.logical_or),
            (ops.logical_xor, torch.logical_xor),
        ],
    )
    def test_binary_logical_ops_gpu(
        self, op_func: Any, torch_func: Any
    ) -> None:
        """Test binary logical ops on GPU."""
        gpu = Accelerator()
        shape = [4]
        input_type = TensorType(DType.bool, shape, gpu)

        with Graph(
            f"gpu_logical_op_{op_func.__name__}",
            input_types=[input_type, input_type],
        ) as graph:
            a, b = graph.inputs
            c = op_func(a, b)
            graph.output(c)

        # Test data covers all truth table combinations
        a_torch = torch.tensor(
            [True, True, False, False], dtype=torch.bool, device="cuda"
        )
        b_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )

        a_gpu = Buffer.from_dlpack(a_torch)
        b_gpu = Buffer.from_dlpack(b_torch)

        interp = MOInterpreter()
        result = interp.execute(graph, [a_gpu, b_gpu])[0]

        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == DType.bool

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(result)
        torch.testing.assert_close(result_torch, expected)

    def test_logical_not_gpu(self) -> None:
        """Test logical not op on GPU."""
        gpu = Accelerator()
        shape = [4]
        input_type = TensorType(DType.bool, shape, gpu)

        with Graph("gpu_logical_not", input_types=[input_type]) as graph:
            (a,) = graph.inputs
            c = ops.logical_not(a)
            graph.output(c)

        a_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )
        a_gpu = Buffer.from_dlpack(a_torch)

        interp = MOInterpreter()
        result = interp.execute(graph, [a_gpu])[0]

        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == DType.bool

        expected = torch.logical_not(a_torch)
        result_torch = torch.from_dlpack(result)
        torch.testing.assert_close(result_torch, expected)


class TestElementwiseGPU:
    """Tests for GPU elementwise operations in the interpreter."""

    def test_gpu_inputs_with_cpu_target_raises_error(self) -> None:
        """Test that GPU inputs with CPU-target ops raise an error."""
        gpu = Accelerator()
        cpu = CPU()
        shape = [4]
        # Graph expects CPU output
        input_type = TensorType(DType.float32, shape, cpu)

        with Graph(
            "gpu_to_cpu_test", input_types=[input_type, input_type]
        ) as graph:
            a, b = graph.inputs
            c = ops.add(a, b)
            graph.output(c)

        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda"
        )

        # Both inputs on GPU, but target is CPU - should raise error
        a_gpu = Buffer.from_dlpack(a_torch)
        b_gpu = Buffer.from_dlpack(b_torch)

        interp = MOInterpreter()

        with pytest.raises(ValueError):
            interp.execute(graph, [a_gpu, b_gpu])

    def test_mixed_device_inputs_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error."""
        gpu = Accelerator()
        cpu = CPU()
        shape = [4]
        # Graph expects GPU output
        input_type = TensorType(DType.float32, shape, gpu)

        with Graph(
            "mixed_input_test", input_types=[input_type, input_type]
        ) as graph:
            a, b = graph.inputs
            c = ops.add(a, b)
            graph.output(c)

        a_torch_cpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cpu"
        )
        b_torch_gpu = torch.tensor(
            [5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda"
        )

        # One CPU, one GPU input - should raise an error
        a_cpu = Buffer.from_dlpack(a_torch_cpu)
        b_gpu = Buffer.from_dlpack(b_torch_gpu)

        interp = MOInterpreter()

        with pytest.raises(Exception):
            interp.execute(graph, [a_cpu, b_gpu])

    def test_unsupported_op_raises_on_gpu(self) -> None:
        """Test that unsupported GPU ops raise an error."""
        gpu = Accelerator()
        cpu = CPU()
        shape = [4]
        input_type = TensorType(DType.float32, shape, gpu)

        # atanh uses libm and is not supported on GPU
        with Graph("gpu_atanh_test", input_types=[input_type]) as graph:
            (a,) = graph.inputs
            c = ops.atanh(a)
            graph.output(c)

        # Use values in valid range for atanh
        a_torch = torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cuda"
        )
        a_gpu = Buffer.from_dlpack(a_torch)

        interp = MOInterpreter()

        with pytest.raises(Exception, match="GPU execution not supported"):
            interp.execute(graph, [a_gpu])

    @pytest.mark.parametrize(
        "op_func,torch_func",
        [
            (ops.add, torch.add),
            (ops.sub, torch.sub),
            (ops.mul, torch.mul),
            (ops.div, torch.div),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_binary_ops_gpu(
        self, op_func: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test binary ops on GPU with various dtypes."""
        # Skip div for integer types (different semantics)
        if op_func == ops.div and dtype in (DType.int32, DType.int64):
            pytest.skip("Division not tested for integer types")

        gpu = Accelerator()
        cpu = CPU()
        shape = [4]
        torch_dtype = DTYPE_TO_TORCH[dtype]
        input_type = TensorType(dtype, shape, gpu)

        with Graph(
            f"gpu_binary_op_{dtype}", input_types=[input_type, input_type]
        ) as graph:
            a, b = graph.inputs
            c = op_func(a, b)
            graph.output(c)

        # Create test data using torch (supports bfloat16)
        a_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([5, 6, 7, 8], dtype=torch_dtype, device="cuda")

        a_gpu = Buffer.from_dlpack(a_torch)
        b_gpu = Buffer.from_dlpack(b_torch)

        interp = MOInterpreter()
        result = interp.execute(graph, [a_gpu, b_gpu])[0]

        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == dtype

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(result)
        torch.testing.assert_close(result_torch, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "op_func,torch_func",
        [
            (ops.negate, torch.neg),
            (ops.abs, torch.abs),
            (ops.exp, torch.exp),
            (ops.log, torch.log),
            (ops.sqrt, torch.sqrt),
            (ops.sin, torch.sin),
            (ops.cos, torch.cos),
            (ops.tanh, torch.tanh),
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_unary_ops_gpu(
        self, op_func: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test unary ops on GPU with various dtypes."""
        # Float-only ops: skip for integer dtypes
        float_only_ops = (
            ops.exp,
            ops.log,
            ops.sqrt,
            ops.sin,
            ops.cos,
            ops.tanh,
        )
        if op_func in float_only_ops and dtype in (DType.int32, DType.int64):
            pytest.skip(f"{op_func.__name__} not tested for integer types")

        gpu = Accelerator()
        cpu = CPU()
        shape = [4]
        torch_dtype = DTYPE_TO_TORCH[dtype]
        input_type = TensorType(dtype, shape, gpu)

        with Graph(f"gpu_unary_op_{dtype}", input_types=[input_type]) as graph:
            (a,) = graph.inputs
            c = op_func(a)
            graph.output(c)

        # Use positive values to avoid domain issues with log/sqrt
        # Use values with negatives for negate/abs with integers
        if dtype in (DType.int32, DType.int64):
            a_torch = torch.tensor(
                [-1, 2, -3, 4], dtype=torch_dtype, device="cuda"
            )
        else:
            a_torch = torch.tensor(
                [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"
            )
        a_gpu = Buffer.from_dlpack(a_torch)

        interp = MOInterpreter()
        result = interp.execute(graph, [a_gpu])[0]

        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == dtype

        expected = torch_func(a_torch)
        result_torch = torch.from_dlpack(result)
        # Use relaxed tolerance for lower precision dtypes
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestMatmulGPU:
    """Tests for GPU matmul operations in the interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_matmul_gpu(self, dtype: DType) -> None:
        """Test matmul on GPU with various dtypes."""
        gpu = Accelerator()
        torch_dtype = DTYPE_TO_TORCH[dtype]

        # Test (3, 4) @ (4, 5) -> (3, 5)
        m, k, n = 3, 4, 5
        lhs_shape = [m, k]
        rhs_shape = [k, n]
        lhs_type = TensorType(dtype, lhs_shape, gpu)
        rhs_type = TensorType(dtype, rhs_shape, gpu)

        with Graph(
            f"gpu_matmul_{dtype}", input_types=[lhs_type, rhs_type]
        ) as graph:
            lhs, rhs = graph.inputs
            c = ops.matmul(lhs, rhs)  # type: ignore[arg-type]
            graph.output(c)

        # Create test data - use randint for int types, randn for float types
        if dtype in (DType.int32, DType.int64, DType.uint32, DType.uint64):
            lhs_torch = torch.randint(
                0, 10, (m, k), dtype=torch_dtype, device="cuda"
            )
            rhs_torch = torch.randint(
                0, 10, (k, n), dtype=torch_dtype, device="cuda"
            )
        else:
            lhs_torch = torch.randn(m, k, dtype=torch_dtype, device="cuda")
            rhs_torch = torch.randn(k, n, dtype=torch_dtype, device="cuda")

        lhs_gpu = Buffer.from_dlpack(lhs_torch)
        rhs_gpu = Buffer.from_dlpack(rhs_torch)

        interp = MOInterpreter()
        result = interp.execute(graph, [lhs_gpu, rhs_gpu])[0]

        assert isinstance(result, Buffer)
        assert not result.device.is_host
        assert result.dtype == dtype
        assert result.shape == (m, n)

        # perform the reference matmul using torch
        expected = torch.matmul(lhs_torch, rhs_torch)
        result_torch = torch.from_dlpack(result)
        # Use relaxed tolerance for lower precision dtypes
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_matmul_gpu_mixed_device_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error for matmul."""
        gpu = Accelerator()
        cpu = CPU()

        m, k, n = 3, 4, 5
        lhs_type = TensorType(DType.float32, [m, k], gpu)
        rhs_type = TensorType(DType.float32, [k, n], gpu)

        with Graph(
            "gpu_matmul_mixed_device", input_types=[lhs_type, rhs_type]
        ) as graph:
            lhs, rhs = graph.inputs
            c = ops.matmul(lhs, rhs)  # type: ignore[arg-type]
            graph.output(c)

        # Create one CPU and one GPU tensor
        lhs_torch_cpu = torch.randn(m, k, dtype=torch.float32, device="cpu")
        rhs_torch_gpu = torch.randn(k, n, dtype=torch.float32, device="cuda")

        lhs_cpu = Buffer.from_dlpack(lhs_torch_cpu)
        rhs_gpu = Buffer.from_dlpack(rhs_torch_gpu)

        interp = MOInterpreter()

        with pytest.raises(Exception):
            interp.execute(graph, [lhs_cpu, rhs_gpu])


class TestStaticBroadcastToGPU:
    """Tests for GPU static broadcast_to operations in the interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
            DType.int32,
            DType.int64,
        ],
    )
    def test_broadcast_1d_to_2d(self, dtype: DType) -> None:
        """Test broadcasting 1D tensor to 2D on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        target_shape = [2, 3]

        x_torch = torch.tensor([1, 2, 3], dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=target_shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, target_shape)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_broadcast_2d_to_3d(self, dtype: DType) -> None:
        """Test broadcasting 2D tensor with size-1 dim to 3D on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        target_shape = [2, 4, 3]

        x_torch = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch_dtype, device="cuda"
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=target_shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, target_shape)
        torch.testing.assert_close(result_torch, expected)

    def test_broadcast_scalar_like(self) -> None:
        """Test broadcasting scalar-like tensor [1] to higher rank on GPU."""
        target_shape = [2, 3, 4]

        x_torch = torch.tensor([5.0], dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=target_shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, target_shape)
        torch.testing.assert_close(result_torch, expected)

    def test_broadcast_same_shape(self) -> None:
        """Test broadcasting to same shape (no-op) on GPU."""
        shape = [2, 3]

        x_torch = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device="cuda",
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.broadcast_to(shape=shape)

        result_torch = torch.from_dlpack(y)
        expected = torch.broadcast_to(x_torch, shape)
        torch.testing.assert_close(result_torch, expected)
