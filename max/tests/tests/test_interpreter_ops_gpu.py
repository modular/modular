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
from max.driver import Accelerator
from max.dtype import DType
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
