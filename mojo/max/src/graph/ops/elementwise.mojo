# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max

from max.graph.type import ElementType, MOTensor, dyn


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


def elementwise_broadcast(lhs: Symbol, rhs: Symbol) -> (Symbol, Symbol):
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
    var broadcast_dims = DynamicVector[Int64]()
    let larger = lhs_type if lhs_rank > rhs_rank else rhs_type
    let smaller = rhs_type if lhs_rank > rhs_rank else lhs_type
    let offset = larger.rank() - smaller.rank()
    for i in range(offset):
        broadcast_dims.push_back(larger.dims[i])
    for i in range(offset, bcast_rank):
        let d1 = larger.dims[i]
        let d2 = smaller.dims[i - offset]
        broadcast_dims.push_back(
            d1 if d1 == d2 or d2 == 1 else (d2 if d1 == 1 else dyn())
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
# Ops
# ===----------------------------------------------------------------------=== #


# Note: Keep alphabetized.


def add(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let opnds = elementwise_broadcast(lhs, rhs)
    return g.op("mo.add", opnds, opnds.get[0, Symbol]().tensor_type())


def cos(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.cos", v, v.tensor_type())


def div(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    # TODO: This needs proper type promotion, as do all binary ops.
    let cast_rhs = cast(rhs, lhs.tensor_type().dtype)
    let opnds = elementwise_broadcast(lhs, cast_rhs)
    return g.op("mo.div", opnds, opnds.get[0, Symbol]().tensor_type())


def mul(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let opnds = elementwise_broadcast(lhs, rhs)
    return g.op("mo.mul", opnds, opnds.get[0, Symbol]().tensor_type())


def pow(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let opnds = elementwise_broadcast(lhs, rhs)
    return g.op("mo.pow", opnds, opnds.get[0, Symbol]().tensor_type())


def rsqrt(v: Symbol) -> Symbol:
    var g = v.graph()
    let f32_v = cast(v, DType.float32)  # TODO: add missing rsqrt coverage.
    return g.op("mo.rsqrt", f32_v, f32_v.tensor_type())


def softmax(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.softmax", v, v.tensor_type())


def sigmoid(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.sigmoid", v, v.tensor_type())


def silu(v: Symbol) -> Symbol:
    return mul(v, sigmoid(v))


def sin(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.sin", v, v.tensor_type())


def sub(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let opnds = elementwise_broadcast(lhs, rhs)
    return g.op("mo.sub", opnds, opnds.get[0, Symbol]().tensor_type())
