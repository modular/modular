# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from buffer import NDBuffer
from buffer.dimlist import DimList
from register import mogg_register_override, mogg_register_shape_func
from runtime.asyncrt import MojoCallContextPtr
from runtime.tracing import Trace, TraceLevel, trace_arg
from utils.index import StaticIntTuple

from .matmul import matmul
from .utils import GemmShape


@mogg_register_override("linear_op", 1)
@always_inline
@export
fn linear_op[
    a_type: DType,
    input_0_static_shape: DimList,
    b_type: DType,
    input_1_static_shape: DimList,
    c_type: DType,
    input_2_static_shape: DimList,
    /,
    target: StringLiteral = "cpu",
](
    a: NDBuffer[a_type, 2, input_0_static_shape],
    b: NDBuffer[b_type, 2, input_1_static_shape],
    c: NDBuffer[c_type, 2, input_2_static_shape],
    ctx: MojoCallContextPtr,
) raises:
    @always_inline
    @parameter
    fn description_fn() -> String:
        var shape = GemmShape.get[transpose_b=True](c, a, b)
        return String(";").join(
            trace_arg("A", StaticIntTuple[2](shape.M, shape.K), a_type),
            trace_arg("B", StaticIntTuple[2](shape.K, shape.N), b_type),
            trace_arg("C", StaticIntTuple[2](shape.M, shape.N), c_type),
        )

    with Trace[TraceLevel.OP](
        "linear.op", Trace[TraceLevel.OP]._get_detail_str[description_fn]()
    ):
        matmul[
            a_type,
            input_0_static_shape,
            b_type,
            input_1_static_shape,
            c_type,
            input_2_static_shape,
            transpose_b=True,
            target=target,
        ](c, a, b, ctx)


@mogg_register_shape_func("linear_op")
@always_inline
@export
fn linear_op_shape_func[
    a_type: DType, b_type: DType, single_thread_blocking_override: Bool
](a: NDBuffer[a_type, 2], b: NDBuffer[b_type, 2]) -> StaticIntTuple[2]:
    return StaticIntTuple[2](a.dim[0](), b.dim[0]())
