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
import compiler_internal as compiler
from runtime.asyncrt import MojoCallContextPtr
from sys import llvm_intrinsic
from tensor_utils import ManagedTensorSlice, foreach
from utils import StaticIntTuple

# ===----------------------------------------------------------------------===#
# Kernel imports
# ===----------------------------------------------------------------------===#
from algorithm import argmax, argmin
from math import (
    ceil,
    cos,
    erf,
    exp,
    floor,
    fma,
    isqrt,
    log,
    log1p,
    sin,
    sqrt,
    tanh,
)
from nn import arg_nonzero
from utils.numerics import isinf, isnan


# ===----------------------------------------------------------------------===#
# Helpers
# ===----------------------------------------------------------------------===#


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
# Binary Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.add")
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
            var lhs = rebind[SIMD[z.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y.load[width](idx))
            return lhs + rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.sub")
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
            var lhs = rebind[SIMD[z.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y.load[width](idx))
            return lhs - rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.mul")
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
            var lhs = rebind[SIMD[z.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y.load[width](idx))
            return lhs * rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.div")
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
            var lhs = rebind[SIMD[z.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y.load[width](idx))
            return lhs / rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.mod")
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
            var lhs = rebind[SIMD[z.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[z.type, width]](y.load[width](idx))
            return lhs % rhs

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.equal")
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
            var lhs = rebind[SIMD[x.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs == rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.greater")
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
            var lhs = rebind[SIMD[x.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs > rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.greater_equal")
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
            var lhs = rebind[SIMD[x.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs >= rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.not_equal")
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
            var lhs = rebind[SIMD[x.type, width]](x.load[width](idx))
            var rhs = rebind[SIMD[x.type, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs != rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.and")
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
            var lhs = rebind[SIMD[DType.bool, width]](x.load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs & rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.or")
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
            var lhs = rebind[SIMD[DType.bool, width]](x.load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs | rhs)

        foreach[func, synchronous, target](z, ctx)


@compiler.register("mo.xor")
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
            var lhs = rebind[SIMD[DType.bool, width]](x.load[width](idx))
            var rhs = rebind[SIMD[DType.bool, width]](y.load[width](idx))
            return rebind[SIMD[z.type, width]](lhs ^ rhs)

        foreach[func, synchronous, target](z, ctx)


# ===----------------------------------------------------------------------===#
# Unary Elementwise Kernels
# ===----------------------------------------------------------------------===#


@compiler.register("mo.ceil")
struct Ceil:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](ceil(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.floor")
struct Floor:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](floor(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.tanh")
struct Tanh:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](tanh(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.cos")
struct Cos:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](cos(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.sin")
struct Sin:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](sin(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.erf")
struct Erf:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](erf(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.exp")
struct Exp:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](exp(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.round")
struct Round:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](round(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.roundeven")
struct RoundEven:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](x.load[width](idx).roundeven())

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.isqrt")
struct Isqrt:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isqrt(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.select")
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
            var cond = condition.load[width](idx)
            var tc = rebind[SIMD[out.type, width]](true_case.load[width](idx))
            var fc = rebind[SIMD[out.type, width]](false_case.load[width](idx))
            return cond.select(tc, fc)

        foreach[func, synchronous, target](out, ctx)


@compiler.register("mo.trunc")
struct Trunc:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            var val = x.load[width](idx)
            return rebind[SIMD[y.type, width]](
                llvm_intrinsic[
                    "llvm.trunc", __type_of(val), has_side_effect=False
                ](val)
            )

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.log")
struct Log:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.log1p")
struct Log1p:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](log1p(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.is_nan")
struct IsNan:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isnan(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.is_inf")
struct IsInf:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](isinf(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.not")
struct Not:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            var val = rebind[SIMD[DType.bool, width]](x.load[width](idx))
            return rebind[SIMD[y.type, width]](~val)

        foreach[func, synchronous, target](y, ctx)


@compiler.register("mo.abs")
struct Abs:
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](y: ManagedTensorSlice, x: ManagedTensorSlice, ctx: MojoCallContextPtr):
        @parameter
        @always_inline
        fn func[width: Int](idx: StaticIntTuple[y.rank]) -> SIMD[y.type, width]:
            return rebind[SIMD[y.type, width]](abs(x.load[width](idx)))

        foreach[func, synchronous, target](y, ctx)


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
        ctx: MojoCallContextPtr,
    ):
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

        try:
            argmax(input_ndbuffer, axis_ndbuffer, output_ndbuffer)
        except err:
            ctx.set_to_error(err)


@compiler.register("mo.arg_min")
struct ArgMin:
    @staticmethod
    fn execute(
        output: ManagedTensorSlice,
        input: ManagedTensorSlice[rank = output.rank],
        axis: ManagedTensorSlice[rank=1],
        ctx: MojoCallContextPtr,
    ):
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

        try:
            argmin(input_ndbuffer, axis_ndbuffer, output_ndbuffer)
        except err:
            ctx.set_to_error(err)


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
