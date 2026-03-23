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

"""Backward (gradient) kernels for activation functions.

Provides compiler-registered backward operations for training:
- SiLUBackward: dX = dY * sigmoid(X) * (1 + X * (1 - sigmoid(X)))
- GeLUBackward: dX = dY * gelu_derivative(X) (approximate or exact)
- ReLUBackward: dX = dY * (X > 0)
"""

import std.math

import compiler_internal as compiler
from std.runtime.asyncrt import DeviceContextPtr
from std.utils.index import IndexList
from tensor import InputTensor, OutputTensor, foreach


# ===----------------------------------------------------------------------=== #
# Activation derivative helpers
# ===----------------------------------------------------------------------=== #


@always_inline
def _sigmoid[
    dtype: DType, width: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Compute sigmoid(x) = 1 / (1 + exp(-x))."""
    alias one = SIMD[dtype, width](1)
    return one / (one + std.math.exp(-x))


@always_inline
def _silu_backward[
    dtype: DType, width: Int
](
    grad_output: SIMD[dtype, width], x: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """Compute SiLU backward: dX = dY * sigmoid(X) * (1 + X * (1 - sigmoid(X)))."""
    alias one = SIMD[dtype, width](1)
    var sig = _sigmoid[dtype, width](x)
    return grad_output * sig * (one + x * (one - sig))


@always_inline
def _gelu_backward_approximate[
    dtype: DType, width: Int
](
    grad_output: SIMD[dtype, width], x: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """Compute approximate GeLU backward using tanh approximation.

    Uses the derivative of:
    GeLU_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    alias sqrt_2_over_pi = SIMD[dtype, width](0.7978845608028654)
    alias coeff = SIMD[dtype, width](0.044715)
    alias three_coeff = SIMD[dtype, width](3.0 * 0.044715)
    alias half = SIMD[dtype, width](0.5)
    alias one = SIMD[dtype, width](1)

    var inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    var tanh_inner = std.math.tanh(inner)
    var dtanh = one - tanh_inner * tanh_inner

    var grad = half * (one + tanh_inner) + half * x * dtanh * sqrt_2_over_pi * (
        one + three_coeff * x * x
    )
    return grad_output * grad


@always_inline
def _gelu_backward_exact[
    dtype: DType, width: Int
](
    grad_output: SIMD[dtype, width], x: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """Compute exact GeLU backward using erf.

    Uses the derivative of:
    GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    """
    alias inv_sqrt2 = SIMD[dtype, width](0.7071067811865476)
    alias inv_sqrt_2pi = SIMD[dtype, width](0.3989422804014327)
    alias half = SIMD[dtype, width](0.5)
    alias one = SIMD[dtype, width](1)

    var erf_val = std.math.erf(x * inv_sqrt2)
    var pdf = inv_sqrt_2pi * std.math.exp(
        SIMD[dtype, width](-0.5) * x * x
    )

    var grad = half * (one + erf_val) + x * pdf
    return grad_output * grad


@always_inline
def _relu_backward[
    dtype: DType, width: Int
](
    grad_output: SIMD[dtype, width], x: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    """Compute ReLU backward: dX = dY * (X > 0)."""
    return x.gt(0).select(grad_output, SIMD[dtype, width](0))


# ===----------------------------------------------------------------------=== #
# SiLU Backward Registration
# ===----------------------------------------------------------------------=== #


@compiler.register("training.silu_backward")
struct SiLUBackward:
    """Backward pass for SiLU (Swish) activation.

    Computes dX = dY * sigmoid(X) * (1 + X * (1 - sigmoid(X)))

    Tensor Shapes:
        - grad_output: [*, hidden] - Upstream gradient.
        - input: [*, hidden] - Original input to forward SiLU.
        - grad_input: [*, hidden] - Computed input gradient (same shape).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        grad_input: OutputTensor[dtype=dtype, rank=rank, ...],
        grad_output: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if grad_input.shape() != input.shape():
            raise Error("grad_input shape must match input shape")
        if grad_output.shape() != input.shape():
            raise Error("grad_output shape must match input shape")

        @parameter
        @always_inline
        def func[
            width: Int,
        ](idx: IndexList[rank]) capturing -> SIMD[dtype, width]:
            var dy = grad_output.load[width](idx)
            var x = input.load[width](idx)
            return _silu_backward[dtype, width](dy, x)

        foreach[func, target=target](grad_input, ctx)

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        grad_output: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# GeLU Backward Registration
# ===----------------------------------------------------------------------=== #


@compiler.register("training.gelu_backward")
struct GeLUBackward[approximate: Bool = True]:
    """Backward pass for GeLU activation.

    When approximate=True, uses the tanh-based GeLU approximation derivative.
    When approximate=False, uses the exact erf-based derivative.

    Parameters:
        approximate: Whether to use the tanh approximation (default True).

    Tensor Shapes:
        - grad_output: [*, hidden] - Upstream gradient.
        - input: [*, hidden] - Original input to forward GeLU.
        - grad_input: [*, hidden] - Computed input gradient (same shape).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        grad_input: OutputTensor[dtype=dtype, rank=rank, ...],
        grad_output: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if grad_input.shape() != input.shape():
            raise Error("grad_input shape must match input shape")
        if grad_output.shape() != input.shape():
            raise Error("grad_output shape must match input shape")

        @parameter
        @always_inline
        def func[
            width: Int,
        ](idx: IndexList[rank]) capturing -> SIMD[dtype, width]:
            var dy = grad_output.load[width](idx)
            var x = input.load[width](idx)

            comptime if Self.approximate:
                return _gelu_backward_approximate[dtype, width](dy, x)
            else:
                return _gelu_backward_exact[dtype, width](dy, x)

        foreach[func, target=target](grad_input, ctx)

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        grad_output: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ) -> IndexList[rank]:
        return input.shape()


# ===----------------------------------------------------------------------=== #
# ReLU Backward Registration
# ===----------------------------------------------------------------------=== #


@compiler.register("training.relu_backward")
struct ReLUBackward:
    """Backward pass for ReLU activation.

    Computes dX = dY * (X > 0), a simple mask multiply.

    Tensor Shapes:
        - grad_output: [*, hidden] - Upstream gradient.
        - input: [*, hidden] - Original input to forward ReLU.
        - grad_input: [*, hidden] - Computed input gradient (same shape).
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        target: StaticString,
    ](
        grad_input: OutputTensor[dtype=dtype, rank=rank, ...],
        grad_output: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContextPtr,
    ) capturing raises:
        if grad_input.shape() != input.shape():
            raise Error("grad_input shape must match input shape")
        if grad_output.shape() != input.shape():
            raise Error("grad_output shape must match input shape")

        @parameter
        @always_inline
        def func[
            width: Int,
        ](idx: IndexList[rank]) capturing -> SIMD[dtype, width]:
            var dy = grad_output.load[width](idx)
            var x = input.load[width](idx)
            return _relu_backward[dtype, width](dy, x)

        foreach[func, target=target](grad_input, ctx)

    @staticmethod
    def shape[
        dtype: DType,
        rank: Int,
    ](
        grad_output: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=dtype, rank=rank, ...],
    ) -> IndexList[rank]:
        return input.shape()
