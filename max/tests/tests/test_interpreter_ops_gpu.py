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
from max._interpreter import MOInterpreter
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import Graph, TensorType, ops

# Mapping from MAX DType to torch dtype
DTYPE_TO_TORCH = {
    DType.float32: torch.float32,
    DType.float16: torch.float16,
    DType.bfloat16: torch.bfloat16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
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
