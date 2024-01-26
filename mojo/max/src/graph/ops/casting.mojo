# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorShape

from max.graph import ops
from max.graph.type import ElementType


# TODO: Add checks or extend to unranked support, where static shapes assumed.


# ===----------------------------------------------------------------------=== #
# Shape accessors
# ===----------------------------------------------------------------------=== #


def dim(v: Symbol, dim: Int) -> Symbol:
    if dim < 0:
        dim += v.tensor_type().rank()
    return ops.slice(shape_of(v), dim)


def dims(v: Symbol, start: Int, stop: Int) -> Symbol:
    if start < 0:
        start += v.tensor_type().rank()
    if stop < 0:
        stop += v.tensor_type().rank()
    return shape_of(v)[start:stop]


def shape_of(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.shape_of", v, MOTensor(DType.int64, v.tensor_type().rank()))


# ===----------------------------------------------------------------------=== #
# Casters
# ===----------------------------------------------------------------------=== #


def cast(v: Symbol, dtype: ElementType) -> Symbol:
    var g = v.graph()
    return g.op("mo.cast", v, MOTensor(dtype, v.tensor_type().dims))


# ===----------------------------------------------------------------------=== #
# Reshapes
# ===----------------------------------------------------------------------=== #


def squeeze(v: Symbol, axis: Int) -> Symbol:
    var g = v.graph()
    let v_type = v.tensor_type()
    let rank = v_type.rank()
    # TODO: This should be a mo.select
    if axis < 0:
        axis += rank

    let new_shape = g.op(
        "mo.squeeze_shape",
        (shape_of(v), g.scalar(Int64(axis), rank=1)),
        MOTensor(DType.int64, rank - 1),
    )

    var squeezed_dims = DynamicVector[Int64]()
    for i in range(rank):
        if i != axis:
            squeezed_dims.push_back(v_type.dims[i])

    return reshape(v, new_shape, squeezed_dims)


def unsqueeze(v: Symbol, axis: Int) -> Symbol:
    var g = v.graph()
    let v_type = v.tensor_type()
    # Negative values add an extra 1, as -1 adds a new dim at the _end_.
    if axis < 0:
        axis += v_type.rank() + 1
    if axis < 0 or axis > v_type.rank():
        raise (
            "unsqueeze axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(v_type.rank())
        )

    # TODO: Bug - passing v_type.rank() + 1 into a variadic Int64 corrupts it.
    let new_shape = g.op(
        "mo.unsqueeze_shape",
        (shape_of(v), g.scalar(Int64(axis), rank=1)),
        MOTensor(DType.int64, v_type.rank() + 1),
    )

    var dims = DynamicVector[Int64]()
    for i in range(v_type.rank()):
        if i == axis:
            dims.push_back(1)
        dims.push_back(v_type.dims[i])
    if axis == v_type.rank():
        dims.push_back(1)

    return reshape(v, new_shape, dims)


fn reshape(
    v: Symbol, shape: Symbol, out_dims: DynamicVector[Int64]
) raises -> Symbol:
    var g = v.graph()
    return g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_dims)
    )


fn reshape(v: Symbol, shape: Symbol) raises -> Symbol:
    var g = v.graph()

    var shape_t = shape.tensor_type()
    if (shape_t.rank() != 1) or (shape_t.dims[0] == dyn()):
        raise "reshape shape requires static shape shape"
    var out_dims = DynamicVector[Int64]()
    for _ in range(shape_t.dims[0]):
        out_dims.append(dyn())

    return reshape(v, shape, out_dims)


def reshape(v: Symbol, dims: DynamicVector[Int64]) -> Symbol:
    var g = v.graph()

    var out_dims = DynamicVector[Int64]()
    for i in range(len(dims)):
        out_dims.append(dims[i] if dims[i] >= 0 else dyn())

    return reshape(v, g.vector[DType.int64](dims), out_dims)


def reshape(v: Symbol, *dims: Int) -> Symbol:
    var shape = DynamicVector[Int64]()
    for d in dims:
        shape.append(d)
    return reshape(v, shape)


fn reshape_like(v: Symbol, like: Symbol) raises -> Symbol:
    return reshape(v, shape_of(like), like.tensor_type().dims)


# ===----------------------------------------------------------------------=== #
# Transpositions
# ===----------------------------------------------------------------------=== #


def transpose(input: Symbol, x: Int, y: Int) -> Symbol:
    var g = input.graph()
    let input_type = input.tensor_type()
    if input_type.rank() < 2:
        raise "transpose input must have rank >= 2"
    if x < 0:
        x += input_type.rank()
    if y < 0:
        y += input_type.rank()
    if x < 0 or x >= input_type.rank() or y < 0 or y >= input_type.rank():
        raise "transpose dim outside range"

    var dims = DynamicVector[Int64]()
    let ptr = DTypePointer[DType.int64].alloc(input_type.rank())
    for i in range(input_type.rank()):
        dims.push_back(input_type.dims[i])
        ptr.store(i, i)

    dims[x] = input_type.dims[y]
    dims[y] = input_type.dims[x]
    ptr.store(x, y)
    ptr.store(y, x)

    let transpose_indices = g.constant(
        Tensor[DType.int64](ptr, TensorShape(input_type.rank()))
    )

    return g.op(
        "mo.transpose",
        (input, transpose_indices),
        MOTensor(input_type.dtype, dims),
    )


def transpose_matrix(matrix: Symbol) -> Symbol:
    return transpose(matrix, -1, -2)
