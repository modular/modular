# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorShape
from utils.variant import Variant

from max.graph.type import Dim, ElementType


# TODO: Add checks or extend to unranked support, where static shapes assumed.


# ===----------------------------------------------------------------------=== #
# Shape accessors
# ===----------------------------------------------------------------------=== #


def shape_of(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.shape_of", v, MOTensor(DType.int64, v.tensor_type().rank()))


# ===----------------------------------------------------------------------=== #
# Casters
# ===----------------------------------------------------------------------=== #


def cast(v: Symbol, dtype: ElementType) -> Symbol:
    return v.graph().op("mo.cast", v, v.tensor_type().cast(dtype))


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

    var squeezed_dims = DynamicVector[Dim]()
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

    var dims = DynamicVector[Dim]()
    for i in range(v_type.rank()):
        if i == axis:
            dims.push_back(1)
        dims.push_back(v_type.dims[i])
    if axis == v_type.rank():
        dims.push_back(1)

    return reshape(v, new_shape, dims)


fn reshape(
    v: Symbol, shape: Symbol, out_dims: DynamicVector[Dim]
) raises -> Symbol:
    var g = v.graph()
    let dtype = shape.tensor_type().dtype.dtype
    if not (dtype == DType.int64 or dtype == DType.int32):
        raise "reshape shape must be int32 or int64"
    if shape.tensor_type().rank() != 1:
        raise "reshape shape must be rank 1"
    return g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_dims)
    )


fn reshape(v: Symbol, shape: SymbolTuple) raises -> Symbol:
    var g = v.graph()

    if len(shape) == 0:  # Can't `stack` an empty tuple
        let dims = DynamicVector[Dim]()
        return reshape(v, g.constant(Tensor[DType.int64](TensorShape(0))), dims)

    for i in range(len(shape)):
        if shape[i].tensor_type().rank() != 0:
            print(shape[i])
            raise "reshape requires 0-rank dims"

    return reshape(v, stack(shape))


fn reshape(v: Symbol, shape: Symbol) raises -> Symbol:
    var g = v.graph()

    var shape_t = shape.tensor_type()
    if (shape_t.rank() != 1) or (not shape_t.dims[0].is_static()):
        raise "reshape shape requires static shape shape"
    var out_dims = DynamicVector[Dim]()
    for _ in range(shape_t.dims[0].num_elements()):
        out_dims.append(Dim.dynamic())

    return reshape(v, shape, out_dims)


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

    var dims = DynamicVector[Dim]()
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
