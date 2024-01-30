# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorShape

from max.graph.symbol import SymbolTuple
from max.graph.type import *


# TODO: Add checks or extend to unranked support, where static shapes assumed.


# ===----------------------------------------------------------------------=== #
# Slicing and indexing
# ===----------------------------------------------------------------------=== #


def gather(input: Symbol, indices: Symbol, axis: Int = 0) -> Symbol:
    var g = input.graph()
    let input_type = input.tensor_type()
    let indices_type = indices.tensor_type()

    if axis < 0:
        axis += input_type.rank()
    if axis < 0 or axis >= input_type.rank():
        raise (
            "gather axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(input_type.rank())
        )

    var dims = DynamicVector[Int64]()
    for i in range(input_type.rank()):
        if i == axis:
            for j in range(indices_type.rank()):
                dims.push_back(indices_type.dims[j])
        else:
            dims.push_back(input_type.dims[i])

    return g.op(
        "mo.gather",
        (input, indices, g.scalar(Int64(axis))),
        MOTensor(input_type.dtype, dims),
    )


# TODO: Come up with a satisfactory name for this single-slice thing
def slice(input: Symbol, idx: Symbol, axis: Int = 0) -> Symbol:
    var g = input.graph()
    let input_type = input.tensor_type()
    let rank = input_type.rank()

    let input_shape = shape_of(input)
    if axis < 0:
        axis = rank + axis

    var slice_dims = DynamicVector[Int64]()
    var start = DynamicVector[Symbol]()
    var stop = DynamicVector[Symbol]()
    var step = DynamicVector[Int64]()
    for i in range(rank):
        if i == axis:
            slice_dims.append(1)
            start.append(idx)
            stop.append(idx + 1)
        else:
            slice_dims.append(input_type.dims[i])
            start.append(g.scalar(Int64(0)))
            stop.append(slice(input_shape, g.scalar(Int64(i))))
        step.append(1)

    let slice = g.op(
        "mo.slice",
        (input, stack(start), stack(stop), g.vector[DType.int64](step)),
        MOTensor(input_type.dtype, slice_dims),
    )

    return squeeze(slice, axis)


def slice(input: Symbol, *slices: SymbolTuple) -> Symbol:
    let g = input.graph()
    let input_type = input.tensor_type()
    let input_shape = shape_of(input)

    var dims = DynamicVector[Int64]()
    for slice in slices:
        # TODO: This can actually be calculated.
        dims.push_back(dyn())
    for i in range(len(slices), input_type.rank()):
        dims.push_back(input_type.dims[i])

    var starts = DynamicVector[Symbol]()
    var stops = DynamicVector[Symbol]()
    var steps = DynamicVector[Symbol]()

    let input_t = input.tensor_type()

    for s in slices:
        starts.push_back(s[][0])
        stops.push_back(s[][1])
        if len(s[]) == 3:
            steps.push_back(s[][2])
        else:
            steps.push_back(g.scalar(Int64((1))))
    for dim in range(len(slices), input_t.rank()):
        starts.push_back(g.scalar(Int64(0)))
        stops.push_back(input_shape[dim])
        steps.push_back(g.scalar(Int64((1))))

    let start = stack(starts, axis=0)
    let stop = stack(stops, axis=0)
    let step = stack(steps, axis=0)

    return g.op(
        "mo.slice", (input, start, stop, step), MOTensor(input_t.dtype, dims)
    )


# ===----------------------------------------------------------------------=== #
# Splitting
# ===----------------------------------------------------------------------=== #


def split[
    n: Int
](x: Symbol, sizes: StaticIntTuple[n], axis: Int = 0) -> SymbolTuple:
    var g = x.graph()
    let x_type = x.tensor_type()
    let norm_axis = axis + x_type.rank() if axis < 0 else axis

    var split_sizes = DynamicVector[Int64]()
    var out_types = TypeTuple()
    for i in range(n):
        split_sizes.append(sizes[i])
        var out_dims = x_type.dims
        out_dims[norm_axis] = sizes[i]
        out_types.append(MOTensor(x_type.dtype, out_dims))

    return g.nvop(
        "mo.split",
        (x, g.vector[DType.int64](split_sizes), g.scalar(Int64(axis))),
        out_types,
    )


# ===----------------------------------------------------------------------=== #
# Concatenation
# ===----------------------------------------------------------------------=== #


def concat(values: SymbolTuple, axis: Int = 0) -> Symbol:
    if not len(values):
        raise "must concat at least 1 value"
    let v0 = values[0]
    var g = values[0].graph()

    let v0_type = v0.tensor_type()
    let rank = v0_type.rank()
    let norm_axis = axis + v0_type.rank() if axis < 0 else axis
    if norm_axis < 0 or norm_axis >= v0_type.rank():
        raise (
            "concat axis out of bounds: axis="
            + String(norm_axis)
            + ", rank="
            + String(v0_type.rank())
        )

    var concat_dim: Int64 = 0
    for i in range(len(values)):
        let v_type = values[i].tensor_type()
        if v_type.rank() != rank:
            raise (
                "all concat values must have same rank: rank[0]="
                + String(rank)
                + ", rank["
                + String(i)
                + "]="
                + String(v_type.rank())
            )
        if concat_dim == dyn() or v_type.dims[norm_axis] == dyn():
            concat_dim = dyn()
        else:
            concat_dim += v_type.dims[norm_axis]

    var concat_args = DynamicVector[Symbol]()
    concat_args.push_back(g.scalar(Int64(norm_axis)))
    for i in range(len(values)):
        concat_args.push_back(values[i])

    var dims = DynamicVector[Int64](rank)
    for i in range(rank):
        dims.push_back(v0_type.dims[i])
    dims[norm_axis] = concat_dim

    return g.op("mo.concat", concat_args, MOTensor(v0_type.dtype, dims))


def stack(values: SymbolTuple, axis: Int = 0) -> Symbol:
    var unsqueezed = DynamicVector[Symbol]()
    for i in range(len(values)):
        unsqueezed.push_back(unsqueeze(values[i], axis))
    return concat(unsqueezed, axis=axis)
