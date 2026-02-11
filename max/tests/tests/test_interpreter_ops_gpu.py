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
"""GPU tests for MO interpreter operations.

These tests verify that the Mojo op implementations produce correct results
on GPU by comparing against PyTorch reference implementations.
"""

import operator
from typing import Any

import pytest
import torch
from max import _realization_context as rc
from max import functional as F
from max import random as max_random
from max._interpreter import MOInterpreter
from max._realization_context import set_seed
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
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


class TestBasicGPUExecution:
    """Tests for basic GPU execution through the interpreter."""

    def test_add_on_gpu(self) -> None:
        """Test that basic add works on GPU tensors."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [4.0, 5.0, 6.0], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    def test_add_on_gpu_2d(self) -> None:
        """Test 2D add on GPU."""
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    @pytest.mark.parametrize("dtype", [DType.float32, DType.int32, DType.int64])
    def test_add_on_gpu_dtypes(self, dtype: DType) -> None:
        """Test GPU add with various dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([5, 6, 7, 8], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)


class TestPowGPU:
    """Tests for GPU pow operation."""

    def test_pow_on_gpu(self) -> None:
        """Test that pow works on GPU tensors."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [2.0, 3.0, 2.0, 0.5], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = torch.pow(a_torch, b_torch)
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_pow_on_gpu_dtypes(self, dtype: DType) -> None:
        """Test GPU pow with various float dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"
        )
        b_torch = torch.tensor(
            [2.0, 2.0, 2.0, 2.0], dtype=torch_dtype, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a**b

        expected = torch.pow(a_torch, b_torch)
        torch.testing.assert_close(torch.from_dlpack(c), expected)


class TestBinaryComparisonOpsGPU:
    """Tests for GPU binary comparison operations."""

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.eq, torch.eq),
            (operator.ne, torch.ne),
            (operator.gt, torch.gt),
            (operator.ge, torch.ge),
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
        self, op: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test comparison ops on GPU with various dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]

        # Use test data that exercises both equal and unequal cases
        if op in (operator.gt, operator.ge):
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

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = op(a, b)

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected)


class TestBooleanLogicOpsGPU:
    """Tests for GPU boolean logic operations."""

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.and_, torch.logical_and),
            (operator.or_, torch.logical_or),
            (operator.xor, torch.logical_xor),
        ],
    )
    def test_binary_logical_ops_gpu(self, op: Any, torch_func: Any) -> None:
        """Test binary logical ops on GPU."""
        a_torch = torch.tensor(
            [True, True, False, False], dtype=torch.bool, device="cuda"
        )
        b_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = op(a, b)

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected)

    def test_logical_not_gpu(self) -> None:
        """Test logical not op on GPU."""
        a_torch = torch.tensor(
            [True, False, True, False], dtype=torch.bool, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = ~a

        expected = torch.logical_not(a_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected)


class TestElementwiseGPU:
    """Tests for GPU elementwise operations."""

    def test_mixed_device_inputs_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error."""
        a_torch_cpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cpu"
        )
        b_torch_gpu = torch.tensor(
            [5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch_cpu)
        b = Tensor.from_dlpack(b_torch_gpu)

        with pytest.raises(Exception):
            with (
                rc.EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                a + b

    def test_unsupported_op_raises_on_gpu(self) -> None:
        """Test that unsupported GPU ops raise an error."""
        # atanh uses libm and is not supported on GPU
        a_torch = torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)

        with pytest.raises(Exception, match="GPU execution not supported"):
            with (
                rc.EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                b = F.atanh(a)

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.add, torch.add),
            (operator.sub, torch.sub),
            (operator.mul, torch.mul),
            (operator.truediv, torch.div),
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
        self, op: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test binary ops on GPU with various dtypes."""
        # Skip div for integer types (different semantics)
        if op == operator.truediv and dtype in (DType.int32, DType.int64):
            pytest.skip("Division not tested for integer types")

        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([5, 6, 7, 8], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = op(a, b)

        expected = torch_func(a_torch, b_torch)
        result_torch = torch.from_dlpack(c)
        torch.testing.assert_close(result_torch, expected, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        "op,torch_func",
        [
            (operator.neg, torch.neg),
            (abs, torch.abs),
            (F.exp, torch.exp),
            (F.log, torch.log),
            (F.sqrt, torch.sqrt),
            (F.sin, torch.sin),
            (F.cos, torch.cos),
            (F.tanh, torch.tanh),
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
        self, op: Any, torch_func: Any, dtype: DType
    ) -> None:
        """Test unary ops on GPU with various dtypes."""
        # Float-only ops: skip for integer dtypes
        float_only_ops = (F.exp, F.log, F.sqrt, F.sin, F.cos, F.tanh)
        if op in float_only_ops and dtype in (DType.int32, DType.int64):
            pytest.skip("Op not tested for integer types")

        torch_dtype = DTYPE_TO_TORCH[dtype]

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

        a = Tensor.from_dlpack(a_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = op(a)

        expected = torch_func(a_torch)
        result_torch = torch.from_dlpack(c)
        # Use relaxed tolerance for lower precision dtypes
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestMatmulGPU:
    """Tests for GPU matmul operations."""

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
        torch_dtype = DTYPE_TO_TORCH[dtype]

        m, k, n = 3, 4, 5
        lhs_torch = torch.randn(m, k, dtype=torch_dtype, device="cuda")
        rhs_torch = torch.randn(k, n, dtype=torch_dtype, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch)
        rhs = Tensor.from_dlpack(rhs_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = lhs @ rhs

        expected = torch.matmul(lhs_torch, rhs_torch)
        result_torch = torch.from_dlpack(c)
        # Use relaxed tolerance for lower precision dtypes
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_matmul_gpu_mixed_device_raises_error(self) -> None:
        """Test that mixed CPU/GPU inputs raise an error for matmul."""
        m, k, n = 3, 4, 5
        lhs_torch_cpu = torch.randn(m, k, dtype=torch.float32, device="cpu")
        rhs_torch_gpu = torch.randn(k, n, dtype=torch.float32, device="cuda")

        lhs = Tensor.from_dlpack(lhs_torch_cpu)
        rhs = Tensor.from_dlpack(rhs_torch_gpu)

        with pytest.raises(Exception):
            with (
                rc.EagerRealizationContext(use_interpreter=True) as ctx,
                realization_context(ctx),
            ):
                lhs @ rhs


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


class TestRangeGPU:
    """Tests for GPU range operations via Tensor.arange with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_range_basic_gpu(self, dtype: DType) -> None:
        """Test basic range op on GPU with float dtypes."""
        gpu = Accelerator()
        torch_dtype = DTYPE_TO_TORCH[dtype]

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(10, dtype=dtype, device=gpu)

        result_torch = torch.from_dlpack(t)
        expected = torch.arange(0, 10, 1, dtype=torch_dtype, device="cuda")
        torch.testing.assert_close(result_torch, expected)

    def test_range_with_step_gpu(self) -> None:
        """Test range op with custom step on GPU."""
        gpu = Accelerator()

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(0, 10, 2, dtype=DType.float32, device=gpu)

        result_torch = torch.from_dlpack(t)
        expected = torch.arange(0, 10, 2, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.int32,
            DType.int64,
        ],
    )
    def test_range_int_gpu(self, dtype: DType) -> None:
        """Test range op with integer dtypes on GPU."""
        gpu = Accelerator()
        torch_dtype = DTYPE_TO_TORCH[dtype]

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            t = Tensor.arange(10, dtype=dtype, device=gpu)

        result_torch = torch.from_dlpack(t)
        expected = torch.arange(0, 10, 1, dtype=torch_dtype, device="cuda")
        torch.testing.assert_close(result_torch, expected)


class TestReduceMaxGPU:
    """Tests for GPU reduce_max operations via Tensor.max with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_max_last_axis(self, dtype: DType) -> None:
        """Test reduce_max on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_max_first_axis(self, dtype: DType) -> None:
        """Test reduce_max on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_max_middle_axis(self, dtype: DType) -> None:
        """Test reduce_max on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    def test_reduce_max_2d(self) -> None:
        """Test reduce_max on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.max(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amax(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)


class TestBroadcastBinaryOpsGPU:
    """Tests for implicit broadcasting in binary ops on GPU.

    These tests exercise the ShapeOfOp -> BroadcastShapeOp -> BroadcastToOp
    chain that gets generated when binary elementwise ops have operands with
    different shapes.
    """

    def test_add_broadcast_1d_2d(self) -> None:
        """Test add with broadcasting: [3] + [2,3] -> [2,3] on GPU."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            dtype=torch.float32,
            device="cuda",
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    def test_mul_broadcast_scalar_like(self) -> None:
        """Test mul with broadcasting: [1] * [3,4] -> [3,4] on GPU."""
        a_torch = torch.tensor([2.0], dtype=torch.float32, device="cuda")
        b_torch = torch.randn(3, 4, dtype=torch.float32, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a * b

        expected = a_torch * b_torch
        torch.testing.assert_close(torch.from_dlpack(c), expected)

    def test_sub_broadcast_different_ranks(self) -> None:
        """Test sub with broadcasting: [4] - [2,3,4] -> [2,3,4] on GPU."""
        a_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        b_torch = torch.randn(2, 3, 4, dtype=torch.float32, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a - b

        expected = a_torch - b_torch
        torch.testing.assert_close(
            torch.from_dlpack(c), expected, rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_add_broadcast_size1_dim(self, dtype: DType) -> None:
        """Test add with broadcasting: [1,4] + [3,4] -> [3,4] on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.randn(1, 4, dtype=torch_dtype, device="cuda")
        b_torch = torch.randn(3, 4, dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            c = a + b

        expected = a_torch + b_torch
        torch.testing.assert_close(
            torch.from_dlpack(c), expected, rtol=1e-2, atol=1e-2
        )


class TestReduceMinGPU:
    """Tests for GPU reduce_min operations via Tensor.min with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_min_last_axis(self, dtype: DType) -> None:
        """Test reduce_min on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_min_first_axis(self, dtype: DType) -> None:
        """Test reduce_min on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_min_middle_axis(self, dtype: DType) -> None:
        """Test reduce_min on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)

    def test_reduce_min_2d(self) -> None:
        """Test reduce_min on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.min(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.amin(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected)


class TestReduceSumGPU:
    """Tests for GPU reduce_sum operations via Tensor.sum with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_sum_last_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_sum_first_axis(self, dtype: DType) -> None:
        """Test reduce_sum on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_sum_middle_axis(self, dtype: DType) -> None:
        """Test reduce_sum on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_reduce_sum_2d(self) -> None:
        """Test reduce_sum on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.sum(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.sum(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestMeanGPU:
    """Tests for GPU mean operations via Tensor.mean with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_mean_last_axis(self, dtype: DType) -> None:
        """Test mean on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_mean_first_axis(self, dtype: DType) -> None:
        """Test mean on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_mean_middle_axis(self, dtype: DType) -> None:
        """Test mean on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_mean_2d(self) -> None:
        """Test mean on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = torch.randn(shape, dtype=torch.float32, device="cuda")
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.mean(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.mean(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestReduceMulGPU:
    """Tests for GPU reduce_mul operations via Tensor.prod with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_mul_last_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the last axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        # Use values close to 1 to avoid overflow
        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda") * 0.3 + 1
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_mul_first_axis(self, dtype: DType) -> None:
        """Test reduce_mul on the first axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda") * 0.3 + 1
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=0)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=0, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_reduce_mul_middle_axis(self, dtype: DType) -> None:
        """Test reduce_mul on a middle axis on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        shape = [3, 4, 5]

        x_torch = torch.randn(shape, dtype=torch_dtype, device="cuda") * 0.3 + 1
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)

    def test_reduce_mul_2d(self) -> None:
        """Test reduce_mul on a 2D tensor on GPU."""
        shape = [4, 6]

        x_torch = (
            torch.randn(shape, dtype=torch.float32, device="cuda") * 0.3 + 1
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.prod(axis=-1)

        result_torch = torch.from_dlpack(y)
        expected = torch.prod(x_torch, dim=-1, keepdim=True)
        torch.testing.assert_close(result_torch, expected, rtol=1e-2, atol=1e-2)


class TestUnaryMixedOpsGPU:
    """Tests for GPU unary mixed-dtype ops (cast, is_nan, is_inf)."""

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (DType.float32, DType.int32),
            (DType.float32, DType.float16),
            (DType.float16, DType.float32),
            (DType.float32, DType.bfloat16),
            (DType.bfloat16, DType.float32),
            (DType.int32, DType.float32),
            (DType.int64, DType.float32),
            (DType.float32, DType.int64),
        ],
    )
    def test_cast(self, in_dtype: DType, out_dtype: DType) -> None:
        """Test cast op on GPU converts dtype correctly."""
        in_torch_dtype = DTYPE_TO_TORCH[in_dtype]
        out_torch_dtype = DTYPE_TO_TORCH[out_dtype]

        if in_dtype in (DType.int32, DType.int64):
            x_torch = torch.arange(12, dtype=in_torch_dtype, device="cuda")
        else:
            x_torch = torch.tensor(
                [0.0, 1.0, 2.5, -1.5, 3.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                dtype=in_torch_dtype,
                device="cuda",
            )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(out_dtype)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.to(out_torch_dtype)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_is_nan(self, dtype: DType) -> None:
        """Test is_nan op on GPU detects NaN values."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.tensor(
            [1.0, float("nan"), 3.0, float("nan"), float("inf"), 0.0],
            dtype=torch_dtype,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.is_nan(x)

        result_torch = torch.from_dlpack(y)
        expected = torch.isnan(x_torch)
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_is_inf(self, dtype: DType) -> None:
        """Test is_inf op on GPU detects Inf values."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.tensor(
            [1.0, float("inf"), float("-inf"), float("nan"), 0.0, 42.0],
            dtype=torch_dtype,
            device="cuda",
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.is_inf(x)

        result_torch = torch.from_dlpack(y)
        expected = torch.isinf(x_torch)
        torch.testing.assert_close(result_torch, expected)

    def test_cast_identity(self) -> None:
        """Test cast to same dtype on GPU is identity."""
        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.float32)

        result_torch = torch.from_dlpack(y)
        torch.testing.assert_close(result_torch, x_torch)

    def test_cast_float_to_int_truncation(self) -> None:
        """Test cast from float to int on GPU truncates toward zero."""
        x_torch = torch.tensor(
            [1.7, -2.3, 3.9, -4.1], dtype=torch.float32, device="cuda"
        )

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.cast(DType.int32)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.to(torch.int32)
        torch.testing.assert_close(result_torch, expected)


class TestRandomNormalGPU:
    """Tests for random normal op on GPU with interpreter."""

    def test_random_normal_gpu_shape_and_device(self) -> None:
        """Test random normal on GPU produces correct shape and device."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.gaussian(
                (3, 4), dtype=DType.float32, device=Accelerator()
            )

        result_torch = torch.from_dlpack(result)
        assert result_torch.shape == (3, 4)
        assert result_torch.dtype == torch.float32
        assert result_torch.is_cuda

    def test_random_normal_gpu_statistics(self) -> None:
        """Test random normal on GPU has approximately correct statistics."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.gaussian(
                (1000, 1000),
                mean=3.0,
                std=1.5,
                dtype=DType.float32,
                device=Accelerator(),
            )

        result_torch = torch.from_dlpack(result).float()
        torch.testing.assert_close(
            result_torch.mean(),
            torch.tensor(3.0, device="cuda"),
            atol=0.1,
            rtol=0.1,
        )
        torch.testing.assert_close(
            result_torch.std(),
            torch.tensor(1.5, device="cuda"),
            atol=0.1,
            rtol=0.1,
        )

    def test_random_normal_gpu_deterministic(self) -> None:
        """Test that same seed produces identical results on GPU."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.gaussian(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        torch.testing.assert_close(
            torch.from_dlpack(result1), torch.from_dlpack(result2)
        )


class TestRandomUniformGPU:
    """Tests for random uniform op on GPU with interpreter."""

    def test_random_uniform_gpu_shape_and_device(self) -> None:
        """Test random uniform on GPU produces correct shape and device."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result = max_random.uniform(
                (3, 4), dtype=DType.float32, device=Accelerator()
            )

        result_torch = torch.from_dlpack(result)
        assert result_torch.shape == (3, 4)
        assert result_torch.dtype == torch.float32
        assert result_torch.is_cuda

    def test_random_uniform_gpu_statistics(self) -> None:
        """Test random uniform on GPU has approximately correct statistics."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(123)
            result = max_random.uniform(
                (1000, 1000),
                range=(2.0, 5.0),
                dtype=DType.float32,
                device=Accelerator(),
            )

        result_torch = torch.from_dlpack(result).float()
        torch.testing.assert_close(
            result_torch.mean(),
            torch.tensor(3.5, device="cuda"),
            atol=0.1,
            rtol=0.1,
        )
        assert result_torch.min().item() >= 2.0
        assert result_torch.max().item() <= 5.0

    def test_random_uniform_gpu_deterministic(self) -> None:
        """Test that same seed produces identical results on GPU."""

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result1 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            set_seed(42)
            result2 = max_random.uniform(
                (5, 5), dtype=DType.float32, device=Accelerator()
            )

        torch.testing.assert_close(
            torch.from_dlpack(result1), torch.from_dlpack(result2)
        )


class TestTransferOpsGPU:
    """Tests for GPU transfer operations via Tensor.to() with interpreter."""

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
    def test_cpu_to_gpu(self, dtype: DType) -> None:
        """Test transferring a tensor from CPU to GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        gpu = Accelerator()

        x_torch_cpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cpu"
        )
        x = Tensor.from_dlpack(x_torch_cpu)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.to(gpu)

        result_torch = torch.from_dlpack(y)
        expected = x_torch_cpu.to("cuda")
        assert torch.equal(result_torch, expected)

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
    def test_gpu_to_cpu(self, dtype: DType) -> None:
        """Test transferring a tensor from GPU to CPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]

        x_torch_gpu = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch_dtype, device="cuda"
        )
        x = Tensor.from_dlpack(x_torch_gpu)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.to(CPU())

        result_torch = torch.from_dlpack(y)
        expected = x_torch_gpu.to("cpu")
        assert torch.equal(result_torch, expected)

    def test_cpu_to_gpu_2d(self) -> None:
        """Test transferring a 2D tensor from CPU to GPU."""
        gpu = Accelerator()

        x_torch_cpu = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device="cpu",
        )
        x = Tensor.from_dlpack(x_torch_cpu)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.to(gpu)

        result_torch = torch.from_dlpack(y)
        expected = x_torch_cpu.to("cuda")
        assert torch.equal(result_torch, expected)

    def test_gpu_to_cpu_2d(self) -> None:
        """Test transferring a 2D tensor from GPU to CPU."""
        x_torch_gpu = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device="cuda",
        )
        x = Tensor.from_dlpack(x_torch_gpu)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.to(CPU())

        result_torch = torch.from_dlpack(y)
        expected = x_torch_gpu.to("cpu")
        assert torch.equal(result_torch, expected)

    def test_cpu_to_gpu_large_tensor(self) -> None:
        """Test transferring a larger tensor from CPU to GPU."""
        gpu = Accelerator()

        x_torch_cpu = torch.randn(64, 128, dtype=torch.float32, device="cpu")
        x = Tensor.from_dlpack(x_torch_cpu)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.to(gpu)

        result_torch = torch.from_dlpack(y)
        expected = x_torch_cpu.to("cuda")
        assert torch.equal(result_torch, expected)

    def test_transfer_preserves_data_roundtrip(self) -> None:
        """Test CPU -> GPU -> CPU preserves data exactly."""
        gpu = Accelerator()

        x_torch = torch.tensor(
            [1.0, -2.5, 3.14, 0.0, -100.0],
            dtype=torch.float32,
            device="cpu",
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y_gpu = x.to(gpu)
            y_cpu = y_gpu.to(CPU())

        result_torch = torch.from_dlpack(y_cpu)
        assert torch.equal(result_torch, x_torch)

    def test_same_device_elide_true_aliases(self) -> None:
        """Test same-device transfer with alwaysElideSameDeviceCopy=True aliases."""
        from max._core.dialects import m, mo

        gpu_ref = DeviceRef.GPU(0)
        input_type = TensorType(DType.float32, [4], gpu_ref)

        with Graph("same_device_transfer", input_types=[input_type]) as g:
            x = g.inputs[0]
            in_chain = g.always_ready_chain
            (result, _) = g._add_op_generated(
                mo.TransferOp,
                x,
                m.DeviceRefAttr("gpu", 0),
                in_chain,
                always_elide_same_device_copy=True,
            )
            g.output(result)

        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        input_buf = Buffer.from_dlpack(x_torch)

        interp = MOInterpreter()
        outputs = interp.execute(g, [input_buf])

        out_buf = outputs[0]
        assert isinstance(out_buf, Buffer)
        result_torch = torch.from_dlpack(out_buf)
        assert torch.equal(result_torch, x_torch)
        # alwaysElideSameDeviceCopy=True should alias, not copy.
        assert out_buf._data_ptr() == input_buf._data_ptr()

    def test_same_device_elide_false_copies(self) -> None:
        """Test same-device transfer with alwaysElideSameDeviceCopy=False copies."""
        from max._core.dialects import m, mo

        gpu_ref = DeviceRef.GPU(0)
        input_type = TensorType(DType.float32, [4], gpu_ref)

        with Graph("same_device_elide_false", input_types=[input_type]) as g:
            x = g.inputs[0]
            in_chain = g.always_ready_chain
            (result, _) = g._add_op_generated(
                mo.TransferOp,
                x,
                m.DeviceRefAttr("gpu", 0),
                in_chain,
                always_elide_same_device_copy=False,
            )
            g.output(result)

        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device="cuda"
        )
        input_buf = Buffer.from_dlpack(x_torch)

        interp = MOInterpreter()
        outputs = interp.execute(g, [input_buf])

        out_buf = outputs[0]
        assert isinstance(out_buf, Buffer)
        result_torch = torch.from_dlpack(out_buf)
        assert torch.equal(result_torch, x_torch)
        # alwaysElideSameDeviceCopy=False should produce a copy, not alias.
        assert out_buf._data_ptr() != input_buf._data_ptr()


class TestReshapeOpsGPU:
    """Tests for reshape operations (squeeze, unsqueeze, reshape) on GPU."""

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16, DType.int32],
    )
    def test_squeeze_on_gpu(self, dtype: DType) -> None:
        """Test squeeze removes a size-1 dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 1, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.squeeze(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.squeeze(1)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16, DType.int32],
    )
    def test_unsqueeze_on_gpu(self, dtype: DType) -> None:
        """Test unsqueeze adds a dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.unsqueeze(axis=1)

        result_torch = torch.from_dlpack(y)
        expected = x_torch.unsqueeze(1)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_reshape_split_dim_on_gpu(self, dtype: DType) -> None:
        """Test reshape splitting a dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(36, dtype=torch_dtype, device="cuda").reshape(
            12, 3
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 4, 3])

        result_torch = torch.from_dlpack(y)
        expected = x_torch.reshape(3, 4, 3)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_reshape_merge_dims_on_gpu(self, dtype: DType) -> None:
        """Test reshape merging adjacent dimensions on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(24, dtype=torch_dtype, device="cuda").reshape(
            2, 3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([6, 4])

        result_torch = torch.from_dlpack(y)
        expected = x_torch.reshape(6, 4)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_reshape_add_singleton_on_gpu(self, dtype: DType) -> None:
        """Test reshape adding a singleton dimension on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.arange(12, dtype=torch_dtype, device="cuda").reshape(
            3, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = x.reshape([3, 1, 4])

        result_torch = torch.from_dlpack(y)
        expected = x_torch.reshape(3, 1, 4)
        assert result_torch.shape == expected.shape
        torch.testing.assert_close(result_torch, expected)

    def test_squeeze_then_unsqueeze_roundtrip_on_gpu(self) -> None:
        """Test squeeze then unsqueeze roundtrip on GPU."""
        x_torch = torch.arange(12, dtype=torch.float32, device="cuda").reshape(
            3, 1, 4
        )
        x = Tensor.from_dlpack(x_torch)

        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            squeezed = x.squeeze(axis=1)
            unsqueezed = squeezed.unsqueeze(axis=1)

        result_torch = torch.from_dlpack(unsqueezed)
        assert result_torch.shape == x_torch.shape
        torch.testing.assert_close(result_torch, x_torch)


class TestSoftmaxGPU:
    """Tests for softmax and logsoftmax on GPU."""

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_softmax_gpu(self, dtype: DType) -> None:
        """Test softmax on GPU matches torch reference."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.randn(3, 4, 5, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.softmax(x, axis=-1)

        expected = torch.softmax(x_torch, dim=-1)
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )

    @pytest.mark.parametrize(
        "dtype", [DType.float32, DType.float16, DType.bfloat16]
    )
    def test_logsoftmax_gpu(self, dtype: DType) -> None:
        """Test logsoftmax on GPU matches torch reference."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        x_torch = torch.randn(3, 4, 5, dtype=torch_dtype, device="cuda")

        x = Tensor.from_dlpack(x_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            y = F.logsoftmax(x, axis=-1)

        expected = torch.log_softmax(x_torch, dim=-1)
        tol = 1e-2 if dtype == DType.bfloat16 else 1e-3
        torch.testing.assert_close(
            torch.from_dlpack(y), expected, atol=tol, rtol=tol
        )


class TestSelectGPU:
    """Tests for GPU select (where) operations with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.float32,
            DType.float16,
            DType.bfloat16,
        ],
    )
    def test_select_basic_gpu(self, dtype: DType) -> None:
        """Test basic select op on GPU with float dtypes."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        cond_torch = torch.tensor(
            [True, False, True, False, True, False],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            dtype=torch_dtype,
            device="cuda",
        )
        y_torch = torch.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            dtype=torch_dtype,
            device="cuda",
        )

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            DType.int32,
            DType.int64,
        ],
    )
    def test_select_int_gpu(self, dtype: DType) -> None:
        """Test select op with integer dtypes on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        cond_torch = torch.tensor(
            [True, False, True, False],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.tensor([1, 2, 3, 4], dtype=torch_dtype, device="cuda")
        y_torch = torch.tensor(
            [10, 20, 30, 40], dtype=torch_dtype, device="cuda"
        )

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_select_2d_gpu(self) -> None:
        """Test select with 2D tensors on GPU."""
        cond_torch = torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
            device="cuda",
        )
        x_torch = torch.arange(
            1, 7, dtype=torch.float32, device="cuda"
        ).reshape(2, 3)
        y_torch = torch.arange(
            10, 70, 10, dtype=torch.float32, device="cuda"
        ).reshape(2, 3)

        cond = Tensor.from_dlpack(cond_torch)
        x = Tensor.from_dlpack(x_torch)
        y = Tensor.from_dlpack(y_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.where(cond, x, y)

        expected = torch.where(cond_torch, x_torch, y_torch)
        torch.testing.assert_close(torch.from_dlpack(result), expected)


class TestConcatGPU:
    """Tests for GPU concat operations with interpreter."""

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_concat_axis0_gpu(self, dtype: DType) -> None:
        """Test concat along axis 0 on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch_dtype, device="cuda"
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = torch.cat([a_torch, b_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_concat_axis1_gpu(self, dtype: DType) -> None:
        """Test concat along axis 1 on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch_dtype, device="cuda"
        )
        b_torch = torch.tensor(
            [[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
            dtype=torch_dtype,
            device="cuda",
        )

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=1)

        expected = torch.cat([a_torch, b_torch], dim=1)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_concat_negative_axis_gpu(self) -> None:
        """Test concat with negative axis on GPU."""
        a_torch = torch.arange(6, dtype=torch.float32, device="cuda").reshape(
            2, 3
        )
        b_torch = torch.arange(
            6, 10, dtype=torch.float32, device="cuda"
        ).reshape(2, 2)

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=-1)

        expected = torch.cat([a_torch, b_torch], dim=-1)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize(
        "dtype",
        [DType.float32, DType.float16, DType.bfloat16],
    )
    def test_concat_multiple_tensors_gpu(self, dtype: DType) -> None:
        """Test concat with more than two tensors on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([[1.0, 2.0]], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([[3.0, 4.0]], dtype=torch_dtype, device="cuda")
        c_torch = torch.tensor([[5.0, 6.0]], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        c = Tensor.from_dlpack(c_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b, c], axis=0)

        expected = torch.cat([a_torch, b_torch, c_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    def test_concat_3d_gpu(self) -> None:
        """Test concat with 3D tensors on GPU."""
        a_torch = torch.arange(24, dtype=torch.float32, device="cuda").reshape(
            2, 3, 4
        )
        b_torch = torch.arange(
            24, 48, dtype=torch.float32, device="cuda"
        ).reshape(2, 3, 4)

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = torch.cat([a_torch, b_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)

    @pytest.mark.parametrize("dtype", [DType.int32, DType.int64])
    def test_concat_int_dtypes_gpu(self, dtype: DType) -> None:
        """Test concat with integer dtypes on GPU."""
        torch_dtype = DTYPE_TO_TORCH[dtype]
        a_torch = torch.tensor([1, 2, 3], dtype=torch_dtype, device="cuda")
        b_torch = torch.tensor([4, 5, 6], dtype=torch_dtype, device="cuda")

        a = Tensor.from_dlpack(a_torch)
        b = Tensor.from_dlpack(b_torch)
        with (
            rc.EagerRealizationContext(use_interpreter=True) as ctx,
            realization_context(ctx),
        ):
            result = F.concat([a, b], axis=0)

        expected = torch.cat([a_torch, b_torch], dim=0)
        torch.testing.assert_close(torch.from_dlpack(result), expected)
