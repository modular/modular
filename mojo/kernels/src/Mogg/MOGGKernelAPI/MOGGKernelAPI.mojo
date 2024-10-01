# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------===#
# General imports
# ===----------------------------------------------------------------------===#
from buffer import NDBuffer
from buffer.dimlist import DimList
from builtin.simd import _pow
import compiler_internal as compiler
from runtime.asyncrt import MojoCallContextPtr
from sys import llvm_intrinsic
from sys.info import simdwidthof
from tensor_utils import ManagedTensorSlice, foreach
from utils import StaticIntTuple

# ===----------------------------------------------------------------------===#
# Kernel imports
# ===----------------------------------------------------------------------===#
from algorithm import argmax, argmin
from linalg.matrix_solve import matrix_solve, matrix_solve_shape
from math import (
    ceil,
    cos,
    erf,
    exp,
    floor,
    fma,
    iota,
    isqrt,
    log,
    log1p,
    sin,
    sqrt,
    tanh,
)
from nn import arg_nonzero
from nn.activations import relu, gelu
from nn.arange import arange, arange_shape
from nn.nms import non_max_suppression, non_max_suppression_shape_func
from nn.pool import avg_pool, max_pool, pool_shape, pool_shape_ceil
from nn.resize import resize_nearest_neighbor, resize_linear
from nn.roi_align import roi_align_nhwc
from nn.tile import tile, tile_shape
from nn.topk import top_k, top_k_shape_impl
from nn.gather_scatter import (
    scatter_nd,
    scatter_nd_generator,
    gather_nd,
    gather_shape,
    gather,
    gather_reduce,
    Axis,
    scatter_elements,
    normalize_neg_index,
    scatter_elements_shape,
)
from random import randn, seed
from utils.numerics import isinf, isnan
from nn.softmax import softmax, logsoftmax


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#

# TODO(GRA-914): Properly support scalars.
alias Scalar = ManagedTensorSlice[rank=1]


# Until the kernel translation is complete, this is copied from MOGG.mojo.
@always_inline
fn managed_tensor_slice_to_ndbuffer[
    type: DType,
    rank: Int,
    static_shape: DimList = DimList.create_unknown[rank](),
](tensor: ManagedTensorSlice[type, rank]) -> NDBuffer[type, rank, static_shape]:
    return NDBuffer[type, rank, static_shape](
        tensor._ptr, tensor.get_static_spec().shape, tensor._strides
    )


# ===----------------------------------------------------------------------===#
# Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.range")
@compiler.elementwise
struct Range:
    @staticmethod
    fn execute[
        type: DType,
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=1],
        start: Scalar[type=type],
        stop: Scalar[type=type],
        step: Scalar[type=type],
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[1]) -> SIMD[type, width]:
            return start[0] + step[0] * (iota[type, width](idx[0]))

        foreach[func, synchronous, target](output, ctx)

    @staticmethod
    fn shape[
        type: DType
    ](
        start: Scalar[type=type],
        stop: Scalar[type=type],
        step: Scalar[type=type],
    ) raises -> StaticIntTuple[1]:
        return arange_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(start),
            managed_tensor_slice_to_ndbuffer(stop),
            managed_tensor_slice_to_ndbuffer(step),
        )


# ===----------------------------------------------------------------------===#
# Binary Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.add")
@compiler.elementwise
struct Add:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs + rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.sub")
@compiler.elementwise
struct Sub:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs - rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.mul")
@compiler.elementwise
struct Mul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs * rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.div")
@compiler.elementwise
struct Div:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs / rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.mod")
@compiler.elementwise
struct Mod:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return lhs % rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.equal")
@compiler.elementwise
struct Equal:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs == rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.greater")
@compiler.elementwise
struct Greater:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs > rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.greater_equal")
@compiler.elementwise
struct GreaterEqual:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs >= rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.not_equal")
@compiler.elementwise
struct NotEqual:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[x.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs != rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.and")
@compiler.elementwise
struct And:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs & rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.or")
@compiler.elementwise
struct Or:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs | rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.xor")
@compiler.elementwise
struct Xor:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y._fused_load[width](idx))
            return rebind[SIMD[z.type, width]](lhs ^ rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.pow")
@compiler.elementwise
struct Pow:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = y.load[width](idx)
            return _pow(lhs, rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.max")
@compiler.elementwise
struct Max:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return max(lhs, rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.min")
@compiler.elementwise
struct Min:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        z: ManagedTensorSlice,
        x: ManagedTensorSlice,
        y: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
            var lhs = rebind[SIMD[z.type, width]](x._fused_load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y._fused_load[width](idx))
            return min(lhs, rhs)

        foreach[func, synchronous, target](z, ctx)


# ===----------------------------------------------------------------------===#
# Unary Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.cast")
@compiler.elementwise
struct Cast:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).cast[y.type]()
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.negative")
@compiler.elementwise
struct Negative:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](-x._fused_load[width](idx))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.relu")
@compiler.elementwise
struct ReLU:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](relu(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.gelu")
@compiler.elementwise
struct GeLU:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](gelu(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.ceil")
@compiler.elementwise
struct Ceil:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](ceil(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.floor")
@compiler.elementwise
struct Floor:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](floor(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.tanh")
@compiler.elementwise
struct Tanh:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](tanh(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.cos")
@compiler.elementwise
struct Cos:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](cos(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.sin")
@compiler.elementwise
struct Sin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sin(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.erf")
@compiler.elementwise
struct Erf:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](erf(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.exp")
@compiler.elementwise
struct Exp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](exp(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.round")
@compiler.elementwise
struct Round:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](round(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.roundeven")
@compiler.elementwise
struct RoundEven:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](
                x._fused_load[width](idx).roundeven()
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.sqrt")
@compiler.elementwise
struct Sqrt:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sqrt(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.isqrt")
@compiler.elementwise
struct Isqrt:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isqrt(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.select")
@compiler.elementwise
struct Select:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        out: ManagedTensorSlice,
        condition: ManagedTensorSlice,
        true_case: ManagedTensorSlice,
        false_case: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[
            width: Int
        ](idx: StaticIntTuple[out.rank]) -> SIMD[out.type, width]:
            var cond = condition._fused_load[width](idx)
            var tc = rebind[SIMD[out.type, width]](
                true_case._fused_load[width](idx)
            )
            var fc = rebind[SIMD[out.type, width]](
                false_case._fused_load[width](idx)
            )
            return cond.select(tc, fc)

        foreach[func, synchronous, target](out, ctx)


@compiler.register("mo.trunc")
@compiler.elementwise
struct Trunc:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            var val = x._fused_load[width](idx)
            return rebind[SIMD[y.type, width]](
                llvm_intrinsic[
                    "llvm.trunc", __type_of(val), has_side_effect=False
                ](val)
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.log")
@compiler.elementwise
struct Log:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.log1p")
@compiler.elementwise
struct Log1p:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log1p(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.is_nan")
@compiler.elementwise
struct IsNan:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isnan(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.is_inf")
@compiler.elementwise
struct IsInf:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isinf(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.not")
@compiler.elementwise
struct Not:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            var val = rebind[SIMD[DType.bool, width]](x._fused_load[width](idx))
            return rebind[SIMD[y.type, width]](~val)

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.abs")
@compiler.elementwise
struct Abs:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](abs(x._fused_load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


# ===----------------------------------------------------------------------===#
# ScatterND kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.scatter_nd")
struct ScatterND:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        scatter_nd[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.add")
struct ScatterNDAdd:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return lhs + rhs

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.mul")
struct ScatterNDMul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return lhs * rhs

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.min")
struct ScatterNDMin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return min(lhs, rhs)

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


@compiler.register("mo.scatter_nd.max")
struct ScatterNDMax:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)

        @always_inline
        @parameter
        fn reduce_fn[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return max(lhs, rhs)

        scatter_nd_generator[
            output_ndbuffer.type,
            indices_ndbuffer.type,
            output_ndbuffer.rank,
            indices_ndbuffer.rank,
            updates_ndbuffer.rank,
            synchronous,
            target,
            reduce_fn=reduce_fn,
        ](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            output_ndbuffer,
            context=ctx,
        )


# ===----------------------------------------------------------------------===#
# Scatter kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.scatter")
struct Scatter:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return rhs  # always return the latest update element

        scatter_elements[reduce_func](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output_ndbuffer,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.add")
struct ScatterAdd:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return lhs + rhs

        scatter_elements[reduce_func](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output_ndbuffer,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.max")
struct ScatterMax:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return max(lhs, rhs)

        scatter_elements[reduce_func](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output_ndbuffer,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.min")
struct ScatterMin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return min(lhs, rhs)

        scatter_elements[reduce_func](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output_ndbuffer,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


@compiler.register("mo.scatter.mul")
struct ScatterMul:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        updates: ManagedTensorSlice[output.type, output.rank],
        indices: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var scalar_axis = managed_tensor_slice_to_ndbuffer(axis)[0]

        @always_inline
        @parameter
        fn reduce_func[
            type: DType, width: Int
        ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
            return lhs * rhs

        scatter_elements[reduce_func](
            input_ndbuffer,
            indices_ndbuffer,
            updates_ndbuffer,
            int(normalize_neg_index(scalar_axis, output.rank)),
            output_ndbuffer,
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        updates: ManagedTensorSlice[input.type, input.rank],
        indices: ManagedTensorSlice[rank = input.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        var updates_ndbuffer = managed_tensor_slice_to_ndbuffer(updates)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer(axis)
        return scatter_elements_shape[
            input.rank,
            input.type,
            indices.type,
            axis.type,
            single_thread_blocking_override=True,
        ](input_ndbuffer, updates_ndbuffer, indices_ndbuffer, axis_ndbuffer)


# ===----------------------------------------------------------------------===#
# View kernels
# ===----------------------------------------------------------------------===#


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@no_inline
fn view_copy_impl[
    synchronous: Bool, target: StringLiteral, type: DType, rank: Int
](z: ManagedTensorSlice[type, rank], x: ManagedTensorSlice[type, rank]):
    @parameter
    @always_inline
    fn func[width: Int](idx: StaticIntTuple[z.rank]) -> SIMD[z.type, width]:
        return x._simd_load_internal[width](idx)

    foreach[func](z)


@compiler.register("mo.static.broadcast_to")
@compiler.view_kernel
struct BroadcastTo:
    @staticmethod
    fn build_view[
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        x: ManagedTensorSlice[type, in_rank],
        output_shape: StaticIntTuple[out_rank],
    ) -> ManagedTensorSlice[type, out_rank]:
        var new_strides = StaticIntTuple[out_rank]()
        alias delta = out_rank - in_rank

        @parameter
        for i in range(out_rank):
            if i < delta:
                new_strides[i] = 0
            elif x.dim_size(i - delta) <= 1:
                new_strides[i] = 0
            else:
                new_strides[i] = x._strides[i - delta]

        return ManagedTensorSlice[type, out_rank](
            x._ptr, output_shape, new_strides
        )

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        in_rank: Int,
        out_rank: Int,
    ](
        z: ManagedTensorSlice[type, out_rank],
        x: ManagedTensorSlice[type, in_rank],
        output_shape: StaticIntTuple[out_rank],
    ):
        # We need the extra output_shape argument.
        # Using `z.shape` instead will prevent the compiler from fusing the kernels.
        var x_view = Self.build_view(x, output_shape)
        view_copy_impl[synchronous, target](z, x_view)


# ===----------------------------------------------------------------------===#
# Data dependent kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.arg_max")
struct ArgMax:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        alias axis_shape = compiler.specsof[axis.type, axis.rank]("axis").shape
        alias input_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape

        var output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=output_shape
        ](output)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=axis_shape
        ](axis)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=input_shape
        ](input)

        argmax(input_ndbuffer, axis_ndbuffer, output_ndbuffer)


@compiler.register("mo.arg_min")
struct ArgMin:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
    ) raises:
        alias output_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        alias axis_shape = compiler.specsof[axis.type, axis.rank]("axis").shape
        alias input_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape

        var output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=output_shape
        ](output)
        var axis_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=axis_shape
        ](axis)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=input_shape
        ](input)

        argmin(input_ndbuffer, axis_ndbuffer, output_ndbuffer)


@compiler.register("mo.arg_nonzero")
struct ArgNonZero:
    @staticmethod
    fn execute(
        output_buffer: ManagedTensorSlice[rank=2],
        input_buffer: ManagedTensorSlice,
    ):
        var out_ndbuffer = managed_tensor_slice_to_ndbuffer(output_buffer)
        var in_ndbuffer = managed_tensor_slice_to_ndbuffer(input_buffer)

        arg_nonzero.arg_nonzero(in_ndbuffer, out_ndbuffer)

    @staticmethod
    fn shape(input_buffer: ManagedTensorSlice) -> StaticIntTuple[2]:
        return arg_nonzero.arg_nonzero_shape[
            single_thread_blocking_override=True
        ](managed_tensor_slice_to_ndbuffer(input_buffer))


# ===----------------------------------------------------------------------===#
# Pooling kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.avg_pool")
struct AvgPool:
    @staticmethod
    fn execute[
        count_boundary: Bool,
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        avg_pool[count_boundary=count_boundary](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            False,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ) raises -> StaticIntTuple[input.rank]:
        return pool_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


@compiler.register("mo.avg_pool_ceil_mode_true")
struct AvgPoolCeilModeTrue:
    @staticmethod
    fn execute[
        count_boundary: Bool,
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        avg_pool[count_boundary=count_boundary](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            True,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
        ctx: MojoCallContextPtr,
    ) raises -> StaticIntTuple[input.rank]:
        return pool_shape_ceil[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


@compiler.register("mo.max_pool")
struct MaxPool:
    @staticmethod
    fn execute[
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        max_pool(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            False,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ) raises -> StaticIntTuple[input.rank]:
        return pool_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


@compiler.register("mo.max_pool_ceil_mode_true")
struct MaxPoolCeilModeTrue:
    @staticmethod
    fn execute[
        type: DType,
        int_type: DType,
    ](
        output: ManagedTensorSlice[type, 4],
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ):
        max_pool(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(output),
            True,
        )

    @staticmethod
    fn shape[
        type: DType,
        int_type: DType,
    ](
        input: ManagedTensorSlice[type, 4],
        filter: ManagedTensorSlice[int_type, 1],
        strides: ManagedTensorSlice[int_type, 1],
        dilations: ManagedTensorSlice[int_type, 1],
        paddings: ManagedTensorSlice[int_type, 1],
    ) raises -> StaticIntTuple[input.rank]:
        return pool_shape_ceil[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
        )


# ===----------------------------------------------------------------------===#
# Gather kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.gather_nd")
struct GatherND:
    @staticmethod
    fn execute[
        batchDims: Int,
    ](
        output: ManagedTensorSlice,
        data: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var data_ndbuffer = managed_tensor_slice_to_ndbuffer(data)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)
        gather_nd[batch_dims=batchDims,](
            data_ndbuffer,
            indices_ndbuffer,
            output_ndbuffer,
        )


@compiler.register("mo.gather")
struct Gather:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice,
        axis: Scalar,
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[output.type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn indices_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[indices.type, width]:
            return indices.load[width=width](
                rebind[StaticIntTuple[indices.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = axis._ptr.load(0)

        gather[
            type = output.type,
            indices_type = indices.type,
            input_fn=input_fn,
            indices_fn=indices_fn,
            output_fn=output_fn,
            input_rank = input.rank,
            indices_rank = indices.rank,
            output_rank = output.rank,
            target=target,
            single_thread_blocking_override=synchronous,
        ](
            Axis(axis_val, input.rank),
            input._spec.shape,
            indices._spec.shape,
            output._spec.shape,
            ctx,
        )

    @staticmethod
    fn shape[
        output_rank: Int,
    ](
        input: ManagedTensorSlice,
        indices: ManagedTensorSlice,
        axis: Scalar,
    ) raises -> StaticIntTuple[output_rank]:
        return gather_shape[
            output_rank=output_rank,
            single_thread_blocking_override=True,
        ](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(indices),
            managed_tensor_slice_to_ndbuffer(axis),
        )


@compiler.register("mo.gather_sum")
struct GatherSum:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, *_],
        indices: ManagedTensorSlice[DType.int32, *_],
    ) raises:
        # Existing implementations do not require static shape information
        var output_ndbuffer = managed_tensor_slice_to_ndbuffer(output)
        var input_ndbuffer = managed_tensor_slice_to_ndbuffer(input)
        var indices_ndbuffer = managed_tensor_slice_to_ndbuffer(indices)

        fn add[
            type: DType, simd_width: Int
        ](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
            type, simd_width
        ]:
            return x + y

        gather_reduce[output.type, 0, 1, simdwidthof[output.type](), add](
            output_ndbuffer, input_ndbuffer, indices_ndbuffer, 0
        )


# ===----------------------------------------------------------------------===#
# TopK kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.bottom_k")
struct BottomK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[type = DType.bool],
    ):
        top_k(
            managed_tensor_slice_to_ndbuffer(input),
            int(k[0]),
            int(axis[0]),
            False,
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted[0],
        )

    @staticmethod
    fn shape[
        axis_type: DType
    ](
        input: ManagedTensorSlice,
        k: Scalar[axis_type],
        axis: Scalar[axis_type],
        sorted: Scalar[type = DType.bool],
    ) raises -> StaticIntTuple[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(axis),
        )


@compiler.register("mo.top_k")
struct TopK:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        values: ManagedTensorSlice[type=type, rank=rank],
        indices: ManagedTensorSlice[type = DType.int64, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        k: Scalar,
        axis: Scalar,
        sorted: Scalar[type = DType.bool],
    ):
        top_k(
            managed_tensor_slice_to_ndbuffer(input),
            int(k[0]),
            int(axis[0]),
            True,
            managed_tensor_slice_to_ndbuffer(values),
            managed_tensor_slice_to_ndbuffer(indices),
            sorted[0],
        )

    @staticmethod
    fn shape[
        axis_type: DType
    ](
        input: ManagedTensorSlice,
        k: Scalar[axis_type],
        axis: Scalar[axis_type],
        sorted: Scalar[type = DType.bool],
    ) raises -> StaticIntTuple[input.rank]:
        return top_k_shape_impl[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(k),
            managed_tensor_slice_to_ndbuffer(axis),
        )


# ===----------------------------------------------------------------------===#
# Non maximum suppression kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.non_maximum_suppression")
struct NonMaximumSupression:
    @staticmethod
    fn execute[
        type: DType
    ](
        output: ManagedTensorSlice[type = DType.int64, rank=2],
        boxes: ManagedTensorSlice[type=type, rank=3],
        scores: ManagedTensorSlice[type, rank=3],
        max_output_boxes_per_class: Scalar[DType.int64],
        iou_threshold: Scalar[DType.float32],
        score_threshold: Scalar[DType.float32],
    ):
        var max_output_boxes_int = int(max_output_boxes_per_class[0])
        var iou_threshold_float = iou_threshold[0]
        var score_threshold_float = score_threshold[0]

        non_max_suppression(
            managed_tensor_slice_to_ndbuffer(boxes),
            managed_tensor_slice_to_ndbuffer(scores),
            managed_tensor_slice_to_ndbuffer(output),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )

    @staticmethod
    fn shape[
        type: DType
    ](
        boxes: ManagedTensorSlice[type=type, rank=3],
        scores: ManagedTensorSlice[type=type, rank=3],
        max_output_boxes_per_class: Scalar[DType.int64],
        iou_threshold: Scalar[DType.float32],
        score_threshold: Scalar[DType.float32],
    ) -> StaticIntTuple[2]:
        var max_output_boxes_int = int(max_output_boxes_per_class[0])
        var iou_threshold_float = iou_threshold[0]
        var score_threshold_float = score_threshold[0]

        return non_max_suppression_shape_func(
            managed_tensor_slice_to_ndbuffer(boxes),
            managed_tensor_slice_to_ndbuffer(scores),
            max_output_boxes_int,
            iou_threshold_float,
            score_threshold_float,
        )


# ===----------------------------------------------------------------------===#
# Linalg kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.linalg.solve")
struct LinalgSolve:
    @staticmethod
    fn execute[
        synchronous: Bool,
        type: DType,
    ](
        x: ManagedTensorSlice[type=type],
        a: ManagedTensorSlice[type=type],
        b: ManagedTensorSlice[type=type],
    ) raises:
        matrix_solve[single_thread_blocking_override=synchronous](
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b),
            managed_tensor_slice_to_ndbuffer(x),
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        a: ManagedTensorSlice[type=type, rank=rank],
        b: ManagedTensorSlice[type=type, rank=rank],
    ) raises -> StaticIntTuple[a.rank]:
        return matrix_solve_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(a),
            managed_tensor_slice_to_ndbuffer(b),
        )


# ===----------------------------------------------------------------------===#
# Resize kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.resize.nearest")
struct ResizeNearest:
    @staticmethod
    fn execute[
        coordinate_transform_mode: Int,
        round_mode: Int,
        rank: Int,
        type: DType,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        size: ManagedTensorSlice[rank=1],
    ):
        resize_nearest_neighbor[coordinate_transform_mode, round_mode](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape[
        rank: Int
    ](
        input: ManagedTensorSlice[rank=rank],
        size: ManagedTensorSlice[rank=1],
    ) -> StaticIntTuple[rank]:
        var shape = StaticIntTuple[rank]()
        for i in range(rank):
            shape[i] = int(size[i])

        return shape


@compiler.register("mo.resize.linear")
struct ResizeLinear:
    @staticmethod
    fn execute[
        coordinate_transform_mode: Int,
        antialias: Bool,
        rank: Int,
        type: DType,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        size: ManagedTensorSlice[rank=1],
    ):
        resize_linear[coordinate_transform_mode, antialias](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape[
        rank: Int
    ](
        input: ManagedTensorSlice[rank=rank],
        size: ManagedTensorSlice[rank=1],
    ) -> StaticIntTuple[rank]:
        var shape = StaticIntTuple[rank]()
        for i in range(rank):
            shape[i] = int(size[i])

        return shape


# ===----------------------------------------------------------------------===#
# ROI align kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.roi_align")
struct ROIAlign:
    @staticmethod
    fn execute[
        aligned: Bool,
        mode: StringLiteral,
        type: DType,
    ](
        output: ManagedTensorSlice[type=type, rank=4],
        input: ManagedTensorSlice[type=type, rank=4],
        rois: ManagedTensorSlice[type=type, rank=2],
        output_height: Scalar[DType.int64],
        output_width: Scalar[DType.int64],
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ):
        roi_align_nhwc[aligned, mode](
            managed_tensor_slice_to_ndbuffer(output),
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(rois),
            int(output_height[0]),
            int(output_width[0]),
            spatial_scale[0],
            sampling_ratio[0],
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice[rank=4],
        rois: ManagedTensorSlice[rank=2],
        output_height: Scalar[DType.int64],
        output_width: Scalar[DType.int64],
        spatial_scale: Scalar,
        sampling_ratio: Scalar,
    ) -> StaticIntTuple[4]:
        var shape = StaticIntTuple[4]()
        # input shape is [N, H, W, C]
        # rois shape is [M, 5]
        # output shape is [M, output_height, output_width, C]
        shape[0] = rois.spec().shape[0]
        shape[1] = int(output_height[0])
        shape[2] = int(output_width[0])
        shape[3] = input.spec().shape[3]

        return shape


# ===----------------------------------------------------------------------===#
# Tile kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.tile")
struct Tile:
    @staticmethod
    fn execute[
        type: DType, rank: Int
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        repeats: ManagedTensorSlice,
    ) raises:
        tile(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(repeats),
            managed_tensor_slice_to_ndbuffer(output),
        )

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        repeats: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        return tile_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(repeats),
        )


# ===----------------------------------------------------------------------===#
# Random kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.random.normal")
struct RandomNormal:
    @staticmethod
    fn execute[
        mean_var_type: DType
    ](
        output: ManagedTensorSlice,
        shape: ManagedTensorSlice[rank=1],
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ):
        seed(int(seed_value[0]))
        var num_elements = 1
        # TODO: Add __len__ support in ManagedTensorSlice.
        for i in range(shape.spec().shape[0]):
            num_elements *= int(shape[i])
        randn(
            output._ptr,
            num_elements,
            mean[0].cast[DType.float64](),
            variance[0].cast[DType.float64](),
        )

    @staticmethod
    fn shape[
        output_rank: Int
    ](shape: ManagedTensorSlice[rank=1]) -> StaticIntTuple[output_rank]:
        var unrolled_shape = StaticIntTuple[output_rank]()
        for i in range(output_rank):
            unrolled_shape[i] = int(shape[i])

        return unrolled_shape


@compiler.register("mo.static.random.normal")
struct StaticRandomNormal:
    @staticmethod
    fn execute[
        mean_var_type: DType
    ](
        output: ManagedTensorSlice,
        mean: Scalar,
        variance: Scalar,
        seed_value: Scalar,
    ):
        seed(int(seed_value[0]))
        var num_elements = output.spec().shape.num_elements()
        randn(
            output._ptr,
            num_elements,
            mean[0].cast[DType.float64](),
            variance[0].cast[DType.float64](),
        )


@compiler.register("mo.softmax")
struct Softmax:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
        ctx: MojoCallContextPtr,
    ) raises:
        # shape should be the same between the two inputs
        alias static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=static_shape
        ](output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[output.type, width]:
            @parameter
            if compiler.specsof[output.type, output.rank]("input").in_lambda:
                return input._fused_load[width=width](
                    rebind[StaticIntTuple[input.rank]](coords)
                )
            else:
                return input.load[width=width](
                    rebind[StaticIntTuple[input.rank]](coords)
                )

        softmax[
            output.type,
            simdwidthof[output.type](),
            output.rank,
            static_shape,
            input_fn,
            target,
        ](
            output.get_static_spec().shape,
            output_ndbuffer,
            output.rank - 1,
            context=ctx,
        )


@compiler.register("mo.logsoftmax")
struct LogSoftmax:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[output.type, output.rank],
    ) raises:
        # shape should be the same between the two inputs
        alias static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape
        output_ndbuffer = managed_tensor_slice_to_ndbuffer[
            static_shape=static_shape
        ](output)

        # For adapting input fusion lambda required by call
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[output.type, width]:
            @parameter
            if compiler.specsof[output.type, output.rank]("input").in_lambda:
                return input._fused_load[width=width](
                    rebind[StaticIntTuple[input.rank]](coords)
                )
            else:
                return input.load[width=width](
                    rebind[StaticIntTuple[input.rank]](coords)
                )

        logsoftmax[
            output.type,
            simdwidthof[output.type](),
            output.rank,
            static_shape,
            input_fn,
        ](output.get_static_spec().shape, output_ndbuffer, output.rank - 1)
