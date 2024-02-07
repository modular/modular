# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max

from max.graph.type import Dim, ElementType, MOTensor
from max.graph.type_promotion import promote


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def elementwise_broadcast(lhs: Symbol, rhs: Symbol) -> SymbolTuple:
    var g = lhs.graph()
    let lhs_type = lhs.tensor_type()
    let rhs_type = rhs.tensor_type()

    if lhs_type == rhs_type and lhs_type.is_static():
        return (lhs, rhs)

    let lhs_rank = lhs_type.rank()
    let rhs_rank = rhs_type.rank()
    let bcast_rank = max(lhs_rank, rhs_rank)

    let lhs_shape = shape_of(lhs)
    let rhs_shape = shape_of(rhs)
    let broadcast_shape = g.op(
        "mo.broadcast_shape",
        (lhs_shape, rhs_shape),
        MOTensor(DType.int64, bcast_rank),
    )

    # This follows NumPy broadcasting semantics:
    #   1. The smaller shape is filled with 1 from the left
    #   2. Dimensions are promoted by the rule 1 -> N -> dynamic
    # TODO: Raise error if static dumensions don't match and can't be promoted.
    var broadcast_dims = DynamicVector[Dim]()
    let larger = lhs_type if lhs_rank > rhs_rank else rhs_type
    let smaller = rhs_type if lhs_rank > rhs_rank else lhs_type
    let offset = larger.rank() - smaller.rank()
    for i in range(offset):
        broadcast_dims.push_back(larger.dims[i])
    for i in range(offset, bcast_rank):
        let d1 = larger.dims[i]
        let d2 = smaller.dims[i - offset]
        broadcast_dims.push_back(
            d1 if d1 == d2 or d2 == 1 else (d2 if d1 == 1 else Dim.dynamic())
        )

    let broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, broadcast_shape),
        MOTensor(lhs_type.dtype, broadcast_dims),
    )
    let broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, broadcast_shape),
        MOTensor(rhs_type.dtype, broadcast_dims),
    )
    return (broadcast_lhs, broadcast_rhs)


# ===----------------------------------------------------------------------=== #
# Binary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _binary_op[op_name: StringLiteral](lhs: Symbol, rhs: Symbol) -> Symbol:
    let broadcast_operands = elementwise_broadcast(lhs, rhs)
    let operands = promote(broadcast_operands[0], broadcast_operands[1])
    return lhs.graph().op(op_name, operands, operands[0].tensor_type())


def _binary_comparison_op[
    op_name: StringLiteral
](lhs: Symbol, rhs: Symbol) -> Symbol:
    let operands = elementwise_broadcast(lhs, rhs)
    let result_type = operands[0].tensor_type().cast(DType.bool)
    return lhs.graph().op(op_name, operands, result_type)


alias add = _binary_op["mo.add"]


def div(lhs: Symbol, rhs: Symbol) -> Symbol:
    # div requires its operands to be the same dtype
    return _binary_op["mo.div"](lhs, cast(rhs, lhs.tensor_type().dtype))


# alias max = _binary_op["mo.max"]  # TODO: namespace problem
alias min = _binary_op["mo.min"]
alias mod = _binary_op["mo.mod"]
alias mul = _binary_op["mo.mul"]
alias pow = _binary_op["mo.pow"]
alias sub = _binary_op["mo.sub"]

alias equal = _binary_comparison_op["mo.equal"]
alias greater = _binary_comparison_op["mo.greater"]
alias greater_equal = _binary_comparison_op["mo.greater_equal"]
alias not_equal = _binary_comparison_op["mo.not_equal"]


# ===----------------------------------------------------------------------=== #
# Unary Ops
# ===----------------------------------------------------------------------=== #
# Note: Keep alphabetized.


def _unary_op[op_name: StringLiteral](value: Symbol) -> Symbol:
    return value.graph().op(op_name, value, value.tensor_type())


def _unary_float_op[op_name: StringLiteral](value: Symbol) -> Symbol:
    let float_v = cast(value, DType.float32)
    return value.graph().op(op_name, float_v, float_v.tensor_type())


def _unary_comparison_op[op_name: StringLiteral](value: Symbol) -> Symbol:
    let result_type = value.tensor_type().cast(DType.bool)
    return value.graph().op(op_name, value, result_type)


alias abs = _unary_op["mo.abs"]
alias exp = _unary_op["mo.exp"]
alias erf = _unary_op["mo.erf"]
alias gelu = _unary_op["mo.gelu"]
alias log = _unary_op["mo.log"]
alias log1p = _unary_op["mo.log1p"]
alias logsoftmax = _unary_op["mo.logsoftmax"]
alias relu = _unary_op["mo.relu"]
alias softmax = _unary_op["mo.softmax"]
alias sigmoid = _unary_op["mo.sigmoid"]


def silu(v: Symbol) -> Symbol:
    return mul(v, sigmoid(v))


alias cos = _unary_float_op["mo.cos"]
alias floor = _unary_float_op["mo.floor"]
alias round = _unary_float_op["mo.round"]
alias roundeven = _unary_float_op["mo.roundeven"]
alias rsqrt = _unary_float_op["mo.rsqrt"]  # TODO: add missing rsqrt coverage.
alias sqrt = _unary_float_op["mo.sqrt"]
alias sin = _unary_float_op["mo.sin"]
alias tanh = _unary_float_op["mo.tanh"]
alias trunc = _unary_float_op["mo.trunc"]


alias is_nan = _unary_comparison_op["mo.is_nan"]
alias is_inf = _unary_comparison_op["mo.is_inf"]
