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
import extensibility as compiler

# ===-----------------------------------------------------------------------===#
# Kernel imports
# ===-----------------------------------------------------------------------===#
from std.builtin.simd import _pow

from nn.activations import relu
from extensibility import (
    ElementwiseBinaryComparisonOp,
    ElementwiseBinaryOp,
    ElementwiseUnaryMixedOp,
    ElementwiseUnaryOp,
)
from std.logger import Logger

comptime logger = Logger()

from std.utils.numerics import isinf, isnan

# ===-----------------------------------------------------------------------===#
from .kernels import *


@compiler.register("mo.add")
struct Add(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs + rhs


@compiler.register("mo.sub")
struct Sub(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs - rhs


@compiler.register("mo.mul")
struct Mul(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs * rhs


@compiler.register("mo.div")
struct Div(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs / rhs


@compiler.register("mo.mod")
struct Mod(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return lhs % rhs


@compiler.register("mo.equal")
struct Equal(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.eq(rhs)


@compiler.register("mo.greater")
struct Greater(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.gt(rhs)


@compiler.register("mo.greater_equal")
struct GreaterEqual(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.ge(rhs)


@compiler.register("mo.not_equal")
struct NotEqual(ElementwiseBinaryComparisonOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[
        DType.bool, width
    ]:
        return lhs.ne(rhs)


@compiler.register("mo.and")
struct And(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.and"
        return lhs & rhs


@compiler.register("mo.or")
struct Or(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.oor"
        return lhs | rhs


@compiler.register("mo.xor")
struct Xor(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.xor"
        return lhs ^ rhs


@compiler.register("mo.pow")
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


@compiler.register("mo.max")
struct Max(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return max(lhs, rhs)


@compiler.register("mo.min")
struct Min(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](lhs: SIMD[dtype, width], rhs: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return min(lhs, rhs)


@compiler.register("mo.cast")
struct Cast(ElementwiseUnaryMixedOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        return x.cast[out_dtype]()


@compiler.register("mo.negative")
struct Negative(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return -x


@compiler.register("mo.relu")
struct ReLU(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return relu(x)


@compiler.register("mo.ceil")
struct Ceil(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return ceil(x)


@compiler.register("mo.floor")
struct Floor(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return floor(x)


@compiler.register("mo.tanh")
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


@compiler.register("mo.acos")
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


@compiler.register("mo.atanh")
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


@compiler.register("mo.cos")
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


@compiler.register("mo.sin")
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


@compiler.register("mo.erf")
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


@compiler.register("mo.exp")
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


@compiler.register("mo.round")
struct Round(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return round(x)


@compiler.register("mo.sqrt")
struct Sqrt(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return sqrt(x)


@compiler.register("mo.rsqrt")
struct Rsqrt(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return rsqrt(x)


@compiler.register("mo.select")
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


@compiler.register("mo.trunc")
struct Trunc(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return llvm_intrinsic["llvm.trunc", type_of(x), has_side_effect=False](
            x
        )


@compiler.register("mo.log")
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


@compiler.register("mo.log1p")
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


@compiler.register("mo.is_nan")
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


@compiler.register("mo.is_inf")
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


@compiler.register("mo.not")
struct Not(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype == DType.bool, "expected bool operands for mo.not"
        return ~x


@compiler.register("mo.abs")
struct Abs(ElementwiseUnaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDSize,
    ](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return abs(x)
