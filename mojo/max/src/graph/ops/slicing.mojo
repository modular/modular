# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections.optional import Optional
from tensor import TensorShape

from max.graph.symbol import SymbolTuple, SymbolicSlice


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

    var dims = DynamicVector[Dim]()
    for i in range(input_type.rank()):
        if i == axis:
            for j in range(indices_type.rank()):
                dims.append(indices_type.dims[j])
        else:
            dims.append(input_type.dims[i])

    return g.op(
        "mo.gather",
        (input, indices, g.scalar(Int64(axis))),
        MOTensor(input_type.dtype, dims),
    )


def slice(
    input: Symbol,
    slices: DynamicVector[SymbolicSlice],
    static_shape: Optional[DynamicVector[Dim]] = None,
) -> Symbol:
    let g = input.graph()
    let input_type = input.tensor_type()

    let out_shape: DynamicVector[Dim]
    if static_shape:
        out_shape = static_shape.value()
    else:
        var dims = DynamicVector[Dim]()
        for axis in range(input_type.rank()):
            if axis < len(slices):
                dims.append(Dim.dynamic())
            else:
                dims.append(input_type.dims[axis])
        out_shape = dims

    let input_shape = shape_of(input)
    var starts = DynamicVector[Symbol]()
    var stops = DynamicVector[Symbol]()
    var steps = DynamicVector[Symbol]()

    for axis in range(input_type.rank()):
        var start: Optional[Symbol] = None
        var stop: Optional[Symbol] = None
        var step: Optional[Symbol] = None
        if axis < len(slices):
            start = slices[axis].start
            stop = slices[axis].stop
            step = slices[axis].step

        # TODO: Fix start/stop for negative step. Needs a select op.
        if start:
            starts.append(start.value())
        else:
            starts.append(g.scalar(Int64(0)))
        if stop:
            stops.append(stop.value())
        else:
            stops.append(input_shape[axis])
        if step:
            steps.append(step.value())
        else:
            steps.append(g.scalar(Int64(1)))

    let start = stack(starts, axis=0)
    let stop = stack(stops, axis=0)
    let step = stack(steps, axis=0)

    return g.op(
        "mo.slice",
        (input, start, stop, step),
        MOTensor(input_type.dtype, out_shape),
    )


# TODO: Change to DynamicVector once Slice is a CollectionElement.
def slice(input: Symbol, s: Slice) -> Symbol:
    let t = input.tensor_type()
    var dims = DynamicVector[Dim]()
    var sym_slices = DynamicVector[SymbolicSlice]()
    for i in range(t.rank()):
        if i < 1:
            dims.append(len(s))
            sym_slices.append(SymbolicSlice(input.graph(), s))
        else:
            dims.append(t.dims[i])
    return slice(input, sym_slices, static_shape=dims)


def slice(input: Symbol, idx: Symbol, axis: Int = 0) -> Symbol:
    let input_type = input.tensor_type()
    let rank = input_type.rank()

    if axis < 0:
        axis = rank + axis

    var slices = DynamicVector[SymbolicSlice]()
    var dims = DynamicVector[Dim]()
    for i in range(rank):
        if i == axis:
            slices.append(SymbolicSlice(idx, idx + 1, None))
            dims.append(1)
        else:
            slices.append(SymbolicSlice(None, None, None))
            dims.append(input_type.dims[i])
    return squeeze(slice(input, slices, static_shape=dims), axis)


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
        out_dims[norm_axis] = Dim.static(sizes[i])
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

    var concat_dim: Dim = 0
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
        let dim = v_type.dims[norm_axis]
        if concat_dim.is_dynamic() or dim.is_dynamic():
            concat_dim = Dim.dynamic()
        elif dim.is_symbolic():
            raise "Concat doesn't yet support symbolic dimensions"
        else:
            concat_dim = Dim.static(
                concat_dim.num_elements() + dim.num_elements()
            )

    var concat_args = DynamicVector[Symbol]()
    concat_args.append(g.scalar(Int64(norm_axis)))
    for i in range(len(values)):
        concat_args.append(values[i])

    var dims = DynamicVector[Dim](capacity=rank)
    for i in range(rank):
        dims.append(v0_type.dims[i])
    dims[norm_axis] = concat_dim

    return g.op("mo.concat", concat_args, MOTensor(v0_type.dtype, dims))


def stack(values: SymbolTuple, axis: Int = 0) -> Symbol:
    var unsqueezed = DynamicVector[Symbol]()
    for i in range(len(values)):
        unsqueezed.append(unsqueeze(values[i], axis))
    return concat(unsqueezed, axis=axis)
