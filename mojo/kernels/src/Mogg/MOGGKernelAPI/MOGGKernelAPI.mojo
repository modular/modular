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
from collections import OptionalReg
import compiler_internal as compiler
from runtime.asyncrt import MojoCallContextPtr
from sys import llvm_intrinsic
from sys.info import simdwidthof
from tensor_utils import ManagedTensorSlice, foreach
from utils import StaticIntTuple
from utils.index import Index

# ===----------------------------------------------------------------------===#
# Kernel imports
# ===----------------------------------------------------------------------===#
from algorithm import argmax, argmin, mean, product, sum
from algorithm import max as reduce_max
from algorithm import min as reduce_min
from linalg.bmm import batched_matmul, batched_matmul_shape
from linalg.matmul import matmul
from linalg.bmm import (
    elementwise_epilogue_type as batched_matmul_elementwise_epilogue_type,
)
from linalg.matrix_solve import matrix_solve, matrix_solve_shape
from linalg.matrix_band_part import matrix_band_part
from linalg.utils import (
    elementwise_epilogue_type as matmul_elementwise_epilogue_type,
)
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
from nn.reshape import reshape, reshape_shape
from nn.resize import resize_nearest_neighbor, resize_linear
from nn.roi_align import roi_align_nhwc
from nn.slice import slice_as_view, slice_shape, slice_dim_as_view
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
from nn.pad import pad_constant, pad_repeat, pad_reflect, pad_shape
from nn.cumsum import cumsum
from nn.conv import ConvInfoStatic, conv_nhwc_direct, conv_shape
from nn.normalization import layer_norm, rms_norm

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


@always_inline("nodebug")
fn reduce_shape[
    input_rank: Int,
    input_type: DType,
](
    input_buf: ManagedTensorSlice[input_type, input_rank],
    axis0: Scalar,
) raises -> StaticIntTuple[input_rank]:
    """
    Compute the output shape of a `reduce` operation, and assert the inputs are
    compatible.

    Parameters:
        input_rank: Input_rank of the input tensor.
        input_type: Type of the input tensor.

    Args:
        input_buf: The input tensor.
        axis0: The axis tensor.

    Returns:
        The output shape.
    """

    var axis_scalar = axis0._ptr.load(0)
    var axis = int(normalize_neg_index(axis_scalar, input_rank))

    if axis < 0 or input_rank <= axis:
        raise Error(
            "[reduction] normalized axis must be within range [0, input_rank)"
        )

    # compute and return the output shape
    var output_shape = input_buf.get_static_spec().shape
    output_shape[axis] = 1
    return output_shape


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


@compiler.register("mo.static.reshape")
@compiler.view_kernel
struct StaticReshape:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        output_rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=output_rank],
        input: ManagedTensorSlice[type=type],
        shape: StaticIntTuple[output_rank],
    ):
        var view_buffer = reshape(
            managed_tensor_slice_to_ndbuffer(input), shape
        )
        var view_tensor = ManagedTensorSlice[type, output_rank](
            view_buffer.data, shape, view_buffer.dynamic_stride
        )
        view_copy_impl[synchronous, target](output, view_tensor)


@compiler.register("mo.reshape")
struct Reshape:
    # The `execute` method should never be used in the graph compiler.
    @staticmethod
    fn execute():
        pass

    @staticmethod
    fn shape[
        output_rank: Int
    ](
        input: ManagedTensorSlice, shape: ManagedTensorSlice[rank=1]
    ) raises -> StaticIntTuple[output_rank]:
        return reshape_shape[
            output_rank=output_rank, single_thread_blocking_override=True
        ](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(shape),
        )


@compiler.register("mo.transpose")
@compiler.view_kernel
struct Transpose:
    @staticmethod
    fn transpose_in_place(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) -> ManagedTensorSlice[type = input.type, rank = input.rank]:
        var new_shape = StaticIntTuple[input.rank]()
        var new_stride = StaticIntTuple[input.rank]()

        @parameter
        for i in range(input.rank):
            var dim = int(permutations[i])
            new_shape[i] = input.spec().shape[dim]
            new_stride[i] = input._strides[dim]

        return ManagedTensorSlice[type = input.type, rank = input.rank](
            input._ptr, new_shape, new_stride
        )

    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        permutations: ManagedTensorSlice[rank=1],
    ):
        view_copy_impl[synchronous, target](
            output, Self.transpose_in_place(input, permutations)
        )

    # TODO(GRA-1033) Make it possible to have multiple raises.
    @no_inline
    @staticmethod
    fn shape_impl(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        if permutations.spec().shape[0] != input.rank:
            raise Error("[transpose] permutation size must match input rank")

        @parameter
        for i in range(input.rank):
            var perm = int(permutations[i])
            if perm < 0 or input.rank <= perm:
                raise Error(
                    "[transpose] each permutation must be within range [0,"
                    " rank)"
                )

        var view_tensor = Self.transpose_in_place(input, permutations)
        var out = StaticIntTuple[input.rank]()

        @parameter
        for i in range(input.rank):
            out[i] = view_tensor.spec().shape[i]

        return out

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        permutations: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        return Self.shape_impl(input, permutations)


@compiler.register("mo.slice")
@compiler.view_kernel
struct Slice:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
    ):
        var view_buffer = slice_as_view(
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )
        var view_tensor = ManagedTensorSlice[type, rank](
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.dynamic_stride,
        )
        view_copy_impl[synchronous, target](output, view_tensor)

    @staticmethod
    fn shape(
        input: ManagedTensorSlice,
        starts: ManagedTensorSlice[rank=1],
        stops: ManagedTensorSlice[rank=1],
        steps: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        return slice_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(starts),
            managed_tensor_slice_to_ndbuffer(stops),
            managed_tensor_slice_to_ndbuffer(steps),
        )


@compiler.register("mo.slice_dim")
@compiler.view_kernel
struct SliceDim:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        rank: Int,
        axis: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        starts: Scalar,
        stops: Scalar,
        steps: Scalar,
    ):
        var view_buffer = slice_dim_as_view[dim=axis](
            managed_tensor_slice_to_ndbuffer(input),
            int(starts[0]),
            int(stops[0]),
            int(steps[0]),
        )
        var view_tensor = ManagedTensorSlice[type, rank](
            view_buffer.data,
            view_buffer.get_shape(),
            view_buffer.dynamic_stride,
        )
        view_copy_impl[synchronous, target](output, view_tensor)


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


@compiler.register("mo.mean")
struct Mean:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank]) -> SIMD[input.type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        mean[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val, output._spec.shape)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> StaticIntTuple[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.add")
struct ReduceAdd:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank]) -> SIMD[input.type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        sum[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> StaticIntTuple[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.mul")
struct ReduceMul:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank]) -> SIMD[input.type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        product[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> StaticIntTuple[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.max")
struct ReduceMax:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank]) -> SIMD[input.type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        reduce_max[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> StaticIntTuple[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


@compiler.register("mo.reduce.min")
struct ReduceMin:
    @staticmethod
    fn execute[
        synchronous: Bool, target: StringLiteral
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[type = output.type, rank = output.rank],
        axis: Scalar,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank]) -> SIMD[input.type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn output_fn[
            width: Int, rank: Int
        ](coords: StaticIntTuple[rank], val: SIMD[output.type, width]):
            output.store[width=width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, width]](val),
            )

        var axis_val = int(axis[0])

        reduce_min[
            output.type,
            input_fn,
            output_fn,
            single_thread_blocking_override=synchronous,
            target=target,
        ](input._spec.shape, axis_val)

    @staticmethod
    fn shape[
        input_rank: Int,
        input_type: DType,
    ](
        input: ManagedTensorSlice[input_type, input_rank],
        axis: Scalar,
    ) raises -> StaticIntTuple[input_rank]:
        return reduce_shape[input_rank, input_type](input, axis)


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
# Padding kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.pad.constant")
struct PadConstant:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
        constant: ManagedTensorSlice[rank=1],
    ):
        var paddings_ptr = padding._ptr
        var constant_simd = constant._ptr.load(0)
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        pad_constant(output_buf, input_buf, paddings_ptr, constant_simd)

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[rank]:
        return pad_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(padding),
        )


@compiler.register("mo.pad.repeat")
struct PadRepeat:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ):
        var paddings_ptr = padding._ptr
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        pad_repeat(output_buf, input_buf, paddings_ptr)

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[rank]:
        return pad_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(padding),
        )


@compiler.register("mo.pad.reflect")
struct PadReflect:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ):
        var paddings_ptr = padding._ptr
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        pad_reflect(output_buf, input_buf, paddings_ptr)

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        padding: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[rank]:
        return pad_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(padding),
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
# Normalization kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.layer_norm")
struct LayerNorm:
    @compiler.enable_fusion_for("input", "gamma")
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        beta: ManagedTensorSlice[type=type, rank=1],
        epsilon: Scalar[type=type],
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
            return input._fused_load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        @parameter
        @always_inline
        fn gamma_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
            return gamma._fused_load[width=width](
                rebind[StaticIntTuple[1]](coords)
            )

        var beta_buf = managed_tensor_slice_to_ndbuffer(beta)
        var epsilon_val = epsilon._ptr.load(0)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        layer_norm[type, rank, input_fn, gamma_fn, target=target,](
            input._spec.shape,
            gamma._spec.shape,
            beta_buf,
            epsilon_val,
            output_buf,
            ctx,
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        beta: ManagedTensorSlice[type=type, rank=1],
        epsilon: Scalar[type=type],
    ) -> StaticIntTuple[rank]:
        return input._spec.shape


@compiler.register("rms_norm")
struct RMSNorm:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        target: StringLiteral,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        epsilon: Scalar[type=type],
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[type, width]:
            return input.load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        var gamma_buf = managed_tensor_slice_to_ndbuffer(gamma)
        var epsilon_val = epsilon._ptr.load(0)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        rms_norm[type, rank, input_fn, target=target](
            input._spec.shape, gamma_buf, epsilon_val, output_buf, ctx
        )

    @staticmethod
    fn shape[
        type: DType,
        rank: Int,
    ](
        input: ManagedTensorSlice[type=type, rank=rank],
        gamma: ManagedTensorSlice[type=type, rank=1],
        epsilon: Scalar[type=type],
    ) -> StaticIntTuple[rank]:
        return input._spec.shape


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


@compiler.register("mo.matmul")
struct Matmul:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        transpose_b: Bool,
        packed_b: Bool,
        lambdas_have_fusion: Bool,
        synchronous: Bool,
        target: StringLiteral = "cpu",
    ](
        c: ManagedTensorSlice[rank=2],
        a: ManagedTensorSlice[rank=2],
        b: ManagedTensorSlice[rank=2],
        ctx: MojoCallContextPtr,
    ):
        constrained[
            not (packed_b and transpose_b),
            (
                "transpose_b and b_packed cannot both be true because"
                " pre-packing transposes B"
            ),
        ]()

        alias transposed_a = False

        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        var c_buffer = managed_tensor_slice_to_ndbuffer(c)

        alias out_lambda = compiler.specsof[c.type, c.rank]("c").out_lambda

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, *, alignment: Int = 1
        ](coords: StaticIntTuple[2], val: SIMD[_type, _width]):
            c._fused_store[width=_width](
                rebind[StaticIntTuple[c.rank]](coords),
                rebind[SIMD[c.type, _width]](val),
            )

        matmul[
            transposed_a,
            transpose_b,
            packed_b,
            OptionalReg[matmul_elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](c_buffer, a_buffer, b_buffer, ctx)


@compiler.register("mo.batch_matmul")
struct BatchMatmul:
    @compiler.enable_fusion_for("c")
    @staticmethod
    fn execute[
        lambdas_have_fusion: Bool,
        rank: Int,
        transpose_b: Bool,
        synchronous: Bool,
        target: StringLiteral = "cpu",
    ](
        c: ManagedTensorSlice[rank=rank],
        a: ManagedTensorSlice[rank=rank],
        b: ManagedTensorSlice[rank=rank],
        ctx: MojoCallContextPtr,
    ):
        alias transposed_a = False

        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        var c_buffer = managed_tensor_slice_to_ndbuffer(c)

        alias out_lambda = compiler.specsof[c.type, c.rank]("c").out_lambda

        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _width: Int, _rank: Int, *, alignment: Int = 1
        ](coords: StaticIntTuple[_rank], val: SIMD[_type, _width]):
            c._fused_store[width=_width](
                rebind[StaticIntTuple[c.rank]](coords),
                rebind[SIMD[c.type, _width]](val),
            )

        batched_matmul[
            c.rank,
            a.type,
            b.type,
            c.type,
            transposed_a,
            transpose_b,
            OptionalReg[batched_matmul_elementwise_epilogue_type](
                output_fn
            ) if lambdas_have_fusion else None,
            saturated_vnni=False,
            single_thread_blocking_override=synchronous,
            target=target,
        ](c_buffer, a_buffer, b_buffer, context=ctx)

    @staticmethod
    fn shape[
        rank: Int,
        a_type: DType,
        b_type: DType,
    ](
        a: ManagedTensorSlice[a_type, rank],
        b: ManagedTensorSlice[b_type, rank],
    ) raises -> StaticIntTuple[rank]:
        var a_buffer = managed_tensor_slice_to_ndbuffer(a)
        var b_buffer = managed_tensor_slice_to_ndbuffer(b)
        return batched_matmul_shape[single_thread_blocking_override=True](
            a_buffer, b_buffer
        )


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


@compiler.register("mo.linalg.band_part")
struct LinalgBandPart:
    @compiler.enable_fusion_for("input")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
        type: DType,
        int_type: DType,
        rank: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        num_lower: ManagedTensorSlice[type=int_type, rank=1],
        num_upper: ManagedTensorSlice[type=int_type, rank=1],
        exclude: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn input_fn[
            width: Int, _rank: Int
        ](coords: StaticIntTuple[_rank]) -> SIMD[output.type, width]:
            return input._fused_load[width=width](
                rebind[StaticIntTuple[input.rank]](coords)
            )

        var num_lower_buf = managed_tensor_slice_to_ndbuffer(num_lower)
        var num_upper_buf = managed_tensor_slice_to_ndbuffer(num_upper)
        var exclude_buf = managed_tensor_slice_to_ndbuffer(exclude)
        var output_buf = managed_tensor_slice_to_ndbuffer(output)

        matrix_band_part[
            input_0_fn=input_fn,
            simd_width = simdwidthof[type](),
            single_thread_blocking_override=synchronous,
            target=target,
        ](
            input.get_static_spec().shape,
            num_lower_buf,
            num_upper_buf,
            exclude_buf,
            output_buf,
            ctx,
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


# ===----------------------------------------------------------------------===#
# Cumsum kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.cumsum")
struct CumSum:
    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        exclusive: Int,
        reverse: Int,
    ](
        output: ManagedTensorSlice[type=type, rank=rank],
        input: ManagedTensorSlice[type=type, rank=rank],
        axis: Scalar,
        ctx: MojoCallContextPtr,
    ):
        var output_buf = managed_tensor_slice_to_ndbuffer(output)
        var input_buf = managed_tensor_slice_to_ndbuffer(input)
        var axis_val = axis._ptr.load(0)

        cumsum[rank, type, exclusive, reverse](
            output_buf, input_buf, int(normalize_neg_index(axis_val, rank))
        )


# ===----------------------------------------------------------------------===#
# Convolution kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.conv")
struct Conv:
    @staticmethod
    fn execute[
        filter_packed: Bool,
        lambdas_have_fusion: Bool,
        static_strides: DimList,
        static_dilations: DimList,
        static_padding: DimList,
    ](
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        filter: ManagedTensorSlice,
        strides: ManagedTensorSlice,
        dilation: ManagedTensorSlice,
        paddings: ManagedTensorSlice,
        num_groups: Scalar,
    ) raises:
        @parameter
        @always_inline
        fn output_fn[
            _type: DType, _rank: Int, _width: Int
        ](coords: StaticIntTuple[_rank], val: SIMD[_type, _width]):
            output.store[width=_width](
                rebind[StaticIntTuple[output.rank]](coords),
                rebind[SIMD[output.type, _width]](val),
            )

        alias input_static_shape = compiler.specsof[input.type, input.rank](
            "input"
        ).shape
        alias filter_static_shape = compiler.specsof[filter.type, filter.rank](
            "filter"
        ).shape
        alias output_static_shape = compiler.specsof[output.type, output.rank](
            "output"
        ).shape

        constrained[
            strides.type.is_integral() and dilation.type.is_integral(),
            "stride and dilation must have integral type",
        ]()

        if strides.size() != input.rank - 2:
            raise Error("$(input_rank-2) values expected in conv strides")

        if dilation.size() != input.rank - 2:
            raise Error("$(input_rank-2) values expected in conv dilation")

        if paddings.size() != 2 * (input.rank - 2):
            raise Error("$(2*(input_rank-2)) value expected in conv paddings")

        var stride_tuple = StaticIntTuple[input.rank - 2](0)
        var dilation_tuple = StaticIntTuple[input.rank - 2](0)

        @parameter
        for i in range(input.rank - 2):
            stride_tuple[i] = int(strides._ptr[i])
            dilation_tuple[i] = int(dilation._ptr[i])

        if dilation_tuple != StaticIntTuple[input.rank - 2](1):
            raise Error("Non-unit dilation is not supported yet.")

        var pad_d_tuple = StaticIntTuple[2](0)
        var pad_h_tuple = StaticIntTuple[2](0)
        var pad_w_tuple = StaticIntTuple[2](0)

        @parameter
        if input.rank == 3:
            pad_w_tuple = Index(paddings._ptr[0], paddings._ptr[1])
        elif input.rank == 4:
            pad_h_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_w_tuple = Index(paddings._ptr[2], paddings._ptr[3])
        elif input.rank == 5:
            pad_d_tuple = Index(paddings._ptr[0], paddings._ptr[1])
            pad_h_tuple = Index(paddings._ptr[2], paddings._ptr[3])
            pad_w_tuple = Index(paddings._ptr[4], paddings._ptr[5])

        alias conv_attr = ConvInfoStatic[input.rank - 2](
            static_padding,
            static_strides,
            static_dilations,
            input_static_shape.at[input.rank - 1](),  # input C, NHWC
            filter_static_shape.at[
                filter.rank - 2
            ](),  # filter C, RSCF or FRSCf
        )

        var input_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=input_static_shape
        ](input)
        var filter_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=filter_static_shape
        ](filter)
        var output_buf = managed_tensor_slice_to_ndbuffer[
            static_shape=output_static_shape
        ](output)

        conv_nhwc_direct[
            input.rank,
            filter.rank,
            input_static_shape,  # input shape
            filter_static_shape,  # filter shape
            output_static_shape,  # output shape
            input.type,
            filter.type,
            output.type,
            filter_packed,
            conv_attr,
            lambdas_have_fusion,
            output_fn,
        ](
            input_buf,
            filter_buf,
            output_buf,
            stride_tuple,
            dilation_tuple,
            pad_d_tuple,
            pad_h_tuple,
            pad_w_tuple,
            int(num_groups._ptr[0]),
        )

    @staticmethod
    fn shape[
        type: DType
    ](
        input: ManagedTensorSlice,
        filter: ManagedTensorSlice,
        strides: ManagedTensorSlice[rank=1],
        dilations: ManagedTensorSlice[rank=1],
        paddings: ManagedTensorSlice[rank=1],
        num_groups: ManagedTensorSlice[rank=1],
    ) raises -> StaticIntTuple[input.rank]:
        return conv_shape[single_thread_blocking_override=True](
            managed_tensor_slice_to_ndbuffer(input),
            managed_tensor_slice_to_ndbuffer(filter),
            managed_tensor_slice_to_ndbuffer(strides),
            managed_tensor_slice_to_ndbuffer(dilations),
            managed_tensor_slice_to_ndbuffer(paddings),
            managed_tensor_slice_to_ndbuffer(num_groups),
        )
