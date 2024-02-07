# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import max

from max.graph.type import Dim, ElementType, MOTensor
from max.graph.ops.casting import reshape


def outer(lhs: Symbol, rhs: Symbol) -> Symbol:
    """Outer product of two vectors. This function does not broadcast."""
    return reshape(lhs, -1, 1) * reshape(rhs, 1, -1)


def batch_matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let broadcast_pair = matmul_broadcast(lhs, rhs)
    let broadcast_lhs = broadcast_pair[0]
    let broadcast_rhs = broadcast_pair[1]

    let lhs_type = broadcast_lhs.tensor_type()
    let rhs_type = broadcast_rhs.tensor_type()
    var dims = DynamicVector[Dim]()
    for i in range(lhs_type.rank() - 1):
        dims.push_back(lhs_type.dims[i])
    dims.push_back(rhs_type.dim(-1))
    let out_type = MOTensor(lhs_type.dtype, dims)

    return g.op("mo.batch_matmul", (broadcast_lhs, broadcast_rhs), out_type)


def matmul(lhs: Symbol, rhs: Symbol) -> Symbol:
    let rhs_type = rhs.tensor_type()
    if rhs_type.rank() > 2:
        return batch_matmul(lhs, rhs)
    else:
        return matmul_by_matrix(lhs, rhs)


def matmul_broadcast(lhs: Symbol, rhs: Symbol) -> SymbolTuple:
    var g = lhs.graph()
    let lhs_type = lhs.tensor_type()
    let rhs_type = rhs.tensor_type()

    let lhs_rank = lhs_type.rank()
    let rhs_rank = rhs_type.rank()

    let broadcast_rank = max(lhs_rank, rhs_rank)
    let lhs_shape = shape_of(lhs)
    let rhs_shape = shape_of(rhs)

    let lhs_broadcast_dims = lhs_shape[: lhs_rank - 2]
    let lhs_matrix_dims = lhs_shape[lhs_rank - 2 : lhs_rank]

    let rhs_broadcast_dims = rhs_shape[: rhs_rank - 2]
    let rhs_matrix_dims = rhs_shape[rhs_rank - 2 : rhs_rank]

    let broadcast_dims_shape = g.op(
        "mo.broadcast_shape",
        (lhs_broadcast_dims, rhs_broadcast_dims),
        MOTensor(DType.int64, broadcast_rank - 2),
    )

    var lhs_final_dims = DynamicVector[Dim]()
    var rhs_final_dims = DynamicVector[Dim]()
    for _ in range(broadcast_rank - 2):
        lhs_final_dims.push_back(Dim.dynamic())
        rhs_final_dims.push_back(Dim.dynamic())
    lhs_final_dims.push_back(lhs_type.dim(-2))
    lhs_final_dims.push_back(lhs_type.dim(-1))
    rhs_final_dims.push_back(rhs_type.dim(-2))
    rhs_final_dims.push_back(rhs_type.dim(-1))

    let lhs_broadcast_shape = concat((broadcast_dims_shape, lhs_matrix_dims))

    let broadcast_lhs = g.op(
        "mo.broadcast_to",
        (lhs, lhs_broadcast_shape),
        MOTensor(lhs_type.dtype, lhs_final_dims),
    )

    let rhs_broadcast_shape = concat((broadcast_dims_shape, rhs_matrix_dims))

    let broadcast_rhs = g.op(
        "mo.broadcast_to",
        (rhs, rhs_broadcast_shape),
        MOTensor(rhs_type.dtype, rhs_final_dims),
    )

    return (broadcast_lhs, broadcast_rhs)


def matmul_by_matrix(lhs: Symbol, rhs: Symbol) -> Symbol:
    var g = lhs.graph()
    let lhs_type = lhs.tensor_type()
    let rhs_type = rhs.tensor_type()
    if rhs_type.rank() != 2:
        raise "rhs must be a matrix"

    let reshape_shape = stack((g.scalar(Int64(-1)), dim(lhs, -1)))

    let final_shape = concat(
        (shape_of(lhs)[: lhs_type.rank() - 1], dims(rhs, 1, 2))
    )

    var final_dims = DynamicVector[Dim]()
    for i in range(lhs_type.rank() - 1):
        final_dims.push_back(lhs_type.dim(i))
    final_dims.push_back(rhs_type.dim(-1))

    var matmul_dims = DynamicVector[Dim]()
    matmul_dims.append(Dim.dynamic())
    matmul_dims.append(lhs_type.dim(-1))
    let matmul_out = g.op(
        "mo.matmul",
        (reshape(lhs, reshape_shape, matmul_dims), rhs),
        MOTensor(lhs_type.dtype, Dim.dynamic(), rhs_type.dim(-1)),
    )

    return reshape(matmul_out, final_shape, final_dims)
