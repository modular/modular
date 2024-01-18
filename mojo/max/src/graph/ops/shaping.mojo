# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec, TensorShape

from max.graph.symbol import Tup
from max.graph.type import ElementType


# TODO: Add checks or extend to unranked support, where static shapes assumed.


def dim(v: Symbol, dim: Int) -> Symbol:
    if dim < 0:
        dim += v.tensor_type().rank()
    return reshape(shape_of(v)[dim : dim + 1], TensorShape())


def cast(v: Symbol, dtype: ElementType) -> Symbol:
    var g = v.graph()
    var t = v.tensor_type()
    t.dtype = dtype
    return g.op("mo.cast", v, t)


def shape_of(v: Symbol) -> Symbol:
    var g = v.graph()
    return g.op("mo.shape_of", v, MOTensor(DType.int64, v.tensor_type().rank()))


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

    return g.op(
        "mo.reshape", (v, new_shape), MOTensor(v_type.dtype, squeezed_dims)
    )


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

    return g.op("mo.reshape", (v, new_shape), MOTensor(v_type.dtype, dims))


def reshape(v: Symbol, shape: TensorShape) -> Symbol:
    var g = v.graph()

    let shape_data = DTypePointer[DType.int64].alloc(shape.rank())
    var out_shape = DynamicVector[Int64]()

    for i in range(shape.rank()):
        let dim = shape[i]
        out_shape.push_back(dyn() if dim == -1 else dim)
        shape_data.store(i, dim)

    let shape_ = g.constant(Tensor(shape_data, TensorShape(shape.rank())))

    return g.op(
        "mo.reshape", (v, shape_), MOTensor(v.tensor_type().dtype, out_shape)
    )


def reshape(v: Symbol, *dims: Int) -> Symbol:
    return reshape(v, TensorShape(dims))


def reshape(v: Symbol, *dims: Symbol) -> Symbol:
    var g = v.graph()
    for dim in dims:
        if dim.tensor_type().rank() > 1:
            raise "reshape: got multi-dimensional shape"

    if len(dims) == 1:
        # Special case: we're maybe passing in a dynamic reshape shape
        let dim = dims[0]
        let dim_t = dim.tensor_type()
        if dim_t.dims[0] > 1:  # only specialize for statically known rank > 1
            return reshape(dim_t.dims[0], v, dim)

    let rank = len(dims)
    var vec = DynamicVector[Symbol](rank)
    for dim in dims:
        vec.append(dim)
    let shape = stack[axis=0](vec)

    var out_shape = DynamicVector[Int64]()
    for _ in range(rank):
        out_shape.push_back(dyn())

    let dtype = v.tensor_type().dtype
    return g.op("mo.reshape", (v, shape), MOTensor(dtype, out_shape))


fn reshape(v: Symbol, shape: Symbol, out_shape: TensorShape) raises -> Symbol:
    var g = v.graph()
    var out_dims = DynamicVector[Int64]()
    for i in range(out_shape.rank()):
        out_dims.push_back(out_shape[i])
    let result = g.op(
        "mo.reshape",
        (v, shape),
        MOTensor(v.tensor_type().dtype.dtype, out_dims),
    )
    return result


fn reshape(
    v: Symbol, shape: Symbol, out_shape: DynamicVector[Int64]
) raises -> Symbol:
    var g = v.graph()
    let result = g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_shape)
    )
    return result


fn reshape(v: Symbol, shape: Symbol, *out_shape: Int64) raises -> Symbol:
    var g = v.graph()
    let result = g.op(
        "mo.reshape", (v, shape), MOTensor(v.tensor_type().dtype, out_shape)
    )
    return result


fn reshape(rank: Int64, v: Symbol, shape: Symbol) raises -> Symbol:
    var g = v.graph()
    var out_shape = DynamicVector[Int64]()
    for _ in range(rank):
        out_shape.push_back(dyn())
    let result = g.op(
        "mo.reshape",
        (v, shape),
        MOTensor(v.tensor_type().dtype.dtype, out_shape),
    )
    return result


fn reshape_like(v: Symbol, like: Symbol) raises -> Symbol:
    return reshape(v, shape_of(like), like.tensor_type().dims)


def transpose_matrix(matrix: Symbol) -> Symbol:
    return transpose(matrix, -1, -2)


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
