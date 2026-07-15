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


# ===-----------------------------------------------------------------------===#
# General imports
# ===-----------------------------------------------------------------------===#

from std.math import (
    acos,
    atanh,
    ceil,
    cos,
    erf,
    exp,
    floor,
    rsqrt,
    log,
    log1p,
    sin,
    sqrt,
    tanh,
)
from std.sys import llvm_intrinsic
import extensibility

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from std.builtin.simd import _pow

from nn.activations import (
    gelu,
    gelu_quick,
    gelu_tanh,
    relu,
    sigmoid,
    silu,
)
from extensibility import (
    ElementwiseBinaryComparisonOp,
    ElementwiseBinaryOp,
    ElementwiseUnaryMixedOp,
    ElementwiseUnaryOp,
)
from layout import Idx, TileTensor
from layout.tile_layout import TensorLayout
from std.logger import Logger

comptime logger = Logger()

from std.utils.numerics import isinf, isnan

# ===-----------------------------------------------------------------------===#
from .kernels import *


@always_inline
def _elementwise_tile[
    Op: ElementwiseBinaryOp,
    dtype: DType,
    LayoutType: TensorLayout,
](
    lhs: TileTensor[dtype, LayoutType, MutAnyOrigin],
    rhs: TileTensor[dtype, LayoutType, MutAnyOrigin],
) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
    """Naive in-place element-wise binary op over two statically-shaped tiles.

    Applies `Op.elementwise` (the scalar overload) to each element of `lhs` and
    `rhs`, writing the result in place into `lhs`, which is returned. A
    statically known layout is required so the walk is unrolled at compile time.

    TODO(GEX-3906): This is currently a naive implementation: a scalar element
    walk that lowers to scalar loads/stores on both CPU and GPU. It will need to
    be optimized (e.g. vectorized copies).
    """
    comptime assert (
        LayoutType.shape_known
    ), "elementwise(TileTensor) requires a statically known layout"

    comptime element_size = type_of(lhs).element_size
    comptime num_elements = LayoutType.static_product

    comptime for i in range(num_elements):
        var lhs_off = lhs.layout(Idx[i])
        var rhs_off = rhs.layout(Idx[i])
        var lhs_val = lhs.raw_load[width=element_size](lhs_off)
        var rhs_val = rhs.raw_load[width=element_size](rhs_off)
        lhs.raw_store[width=element_size](
            lhs_off, Op.elementwise[dtype, element_size](lhs_val, rhs_val)
        )
    return lhs


@extensibility.register("mo.add")
struct Add(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs

    @staticmethod
    def elementwise[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        lhs: TileTensor[dtype, LayoutType, MutAnyOrigin],
        rhs: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        """Element-wise add two tiles in place into `lhs`; see `_elementwise_tile`.
        """
        return _elementwise_tile[Self, dtype, LayoutType](lhs, rhs)


@extensibility.register("mo.sub")
struct Sub(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs - rhs


@extensibility.register("mo.mul")
struct Mul(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs * rhs

    @staticmethod
    def elementwise[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        lhs: TileTensor[dtype, LayoutType, MutAnyOrigin],
        rhs: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[dtype, LayoutType, MutAnyOrigin]:
        """Element-wise multiply two tiles in place into `lhs`; see `_elementwise_tile`.
        """
        return _elementwise_tile[Self, dtype, LayoutType](lhs, rhs)


@extensibility.register("mo.div")
struct Div(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs / rhs


@extensibility.register("mo.mod")
struct Mod(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs % rhs


@extensibility.register("mo.equal")
struct Equal(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.eq(rhs)


@extensibility.register("mo.greater")
struct Greater(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.gt(rhs)


@extensibility.register("mo.greater_equal")
struct GreaterEqual(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.ge(rhs)


@extensibility.register("mo.not_equal")
struct NotEqual(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.ne(rhs)


@extensibility.register("mo.and")
struct And(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.and"
        return lhs & rhs


@extensibility.register("mo.or")
struct Or(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.oor"
        return lhs | rhs


@extensibility.register("mo.xor")
struct Xor(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.xor"
        return lhs ^ rhs


@extensibility.register("mo.pow")
struct Pow:
    @staticmethod
    def elementwise[
        dtype: DType,
        pow_dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[pow_dtype, width]) -> SIMD[
        dtype, width
    ]:
        return _pow(lhs, rhs)


@extensibility.register("mo.max")
struct Max(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(lhs, rhs)


@extensibility.register("mo.min")
struct Min(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return min(lhs, rhs)


@extensibility.register("mo.cast")
struct Cast(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@extensibility.register("mo.negative")
struct Negative(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return -x


@extensibility.register("mo.relu")
struct ReLU(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return relu(x)

    @staticmethod
    def elementwise[
        dtype: DType,
        LayoutType: TensorLayout,
    ](
        x: TileTensor[dtype, LayoutType, MutAnyOrigin],
    ) -> TileTensor[
        dtype, LayoutType, MutAnyOrigin
    ]:
        # TODO(GEX-3799): implement TileTensor element-wise relu.
        return x


@extensibility.register("mo.gelu")
struct Gelu(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return gelu(x)


@extensibility.register("mo.gelu_tanh")
struct GeluTanh(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return gelu_tanh(x)


@extensibility.register("mo.gelu_quick")
struct GeluQuick(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return gelu_quick(x)


@extensibility.register("mo.sigmoid")
struct Sigmoid(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return sigmoid(x)


@extensibility.register("mo.silu")
struct Silu(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return silu(x)


@extensibility.register("mo.ceil")
struct Ceil(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return ceil(x)


@extensibility.register("mo.floor")
struct Floor(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return floor(x)


@extensibility.register("mo.tanh")
struct Tanh(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return tanh(x)


@extensibility.register("mo.acos")
struct ACos(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return acos(x)


@extensibility.register("mo.atanh")
struct ATanh(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return atanh(x)


@extensibility.register("mo.cos")
struct Cos(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return cos(x)


@extensibility.register("mo.sin")
struct Sin(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return sin(x)


@extensibility.register("mo.erf")
struct Erf(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return erf(x)


@extensibility.register("mo.exp")
struct Exp(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return exp(x)


@extensibility.register("mo.round")
struct Round(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return round(x)


@extensibility.register("mo.sqrt")
struct Sqrt(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return sqrt(x)


@extensibility.register("mo.rsqrt")
struct Rsqrt(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return rsqrt(x)


@extensibility.register("mo.select")
struct Select:
    @staticmethod
    def elementwise[
        cond_dtype: DType,
        dtype: DType,
        width: SIMDSize,
    ](
        cond: SIMD[cond_dtype, width],
        tc: SIMD[dtype, width],
        fc: SIMD[dtype, width],
    ) -> SIMD[dtype, width]:
        return cond.select(tc, fc)


@extensibility.register("mo.trunc")
struct Trunc(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return llvm_intrinsic["llvm.trunc", type_of(x), has_side_effect=False](
            x
        )


@extensibility.register("mo.log")
struct Log(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return log(x)


@extensibility.register("mo.log1p")
struct Log1p(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert (
            dtype.is_floating_point()
        ), "dtype must be floating point"
        return log1p(x)


@extensibility.register("mo.is_nan")
struct IsNan(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        comptime assert (
            out_dtype == DType.bool
        ), "expected bool output type for mo.is_nan"
        return rebind[SIMD[out_dtype, width]](isnan(x))


@extensibility.register("mo.is_inf")
struct IsInf(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        comptime assert (
            out_dtype == DType.bool
        ), "expected bool output type for mo.is_inf"
        return rebind[SIMD[out_dtype, width]](isinf(x))


@extensibility.register("mo.not")
struct Not(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.not"
        return ~x


@extensibility.register("mo.abs")
struct Abs(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return abs(x)
