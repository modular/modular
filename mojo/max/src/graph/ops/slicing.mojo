# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops that slice, index, stack, concat etc."""

from collections.optional import Optional
from tensor import TensorShape

from max.graph.symbol import SymbolicSlice


# TODO: Add checks or extend to unranked support, where static shapes assumed.


# ===----------------------------------------------------------------------=== #
# Slicing and indexing
# ===----------------------------------------------------------------------=== #


def gather(input: Symbol, indices: Symbol, axis: Int = 0) -> Symbol:
    """Selects elements out of an input tensor by index.

    Args:
        input: The input symbolic tensor to select elements from.
        indices: A symbolic tensor of index values to use for selection.
        axis: The dimension which `indices` indexes from `input`.
            If negative, indexes relative to the end of the input tensor.
            For instance `gather(input, indices, axis=-1)` will index
            against the last dimension of `input`.

    Returns:
        A new symbolic tensor representing the result of the gather operation.
    """
    var g = input.graph()
    var input_type = input.tensor_type()
    var indices_type = indices.tensor_type()

    if axis < 0:
        axis += input_type.rank()
    if axis < 0 or axis >= input_type.rank():
        raise (
            "gather axis out of bounds: axis="
            + String(axis)
            + ", rank="
            + String(input_type.rank())
        )

    var dims = List[Dim]()
    for i in range(input_type.rank()):
        if i == axis:
            for j in range(indices_type.rank()):
                dims.append(indices_type.dims[j])
        else:
            dims.append(input_type.dims[i])

    return g.op(
        "mo.gather",
        List[Symbol](input, indices, g.scalar(Int64(axis))),
        MOTensor(input_type.dtype, dims),
    )


def slice(
    input: Symbol,
    slices: List[SymbolicSlice],
    static_shape: Optional[List[Dim]] = None,
) -> Symbol:
    """Slices a symbolic tensor along each dimension.

    Args:
        input: The symbolic tensor to slice.
        slices: Per-dimension slice specifiers. If smaller than the
            input rank, trivial slices (ie. ones which select the whole range)
            will be added to the end for the remaining dimensions.
        static_shape: An optional shape to use to hint the output type.

    Returns:
        A new symbolic tensor representing the result of slicing the
        input along each dimension using each slice in `slices`
        respectively. The output will have the same rank as the input,
        but fewer values depending on the slices. If `static_shape` is
        present it will be used as the output tensor's shape, otherwise
        each dimension will be dynamic size.
    """
    var g = input.graph()
    var input_type = input.tensor_type()

    var out_shape: List[Dim]
    if static_shape:
        out_shape = static_shape._value_copy()
    else:
        var dims = List[Dim]()
        for axis in range(input_type.rank()):
            if axis < len(slices):
                dims.append(Dim.dynamic())
            else:
                dims.append(input_type.dims[axis])
        out_shape = dims

    var input_shape = shape_of(input)
    var starts = List[Symbol]()
    var stops = List[Symbol]()
    var steps = List[Symbol]()

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
            starts.append(start._value_copy())
        else:
            starts.append(g.scalar(Int64(0)))
        if stop:
            stops.append(stop._value_copy())
        else:
            stops.append(input_shape[axis])
        if step:
            steps.append(step._value_copy())
        else:
            steps.append(g.scalar(Int64(1)))

    var start = stack(starts, axis=0)
    var stop = stack(stops, axis=0)
    var step = stack(steps, axis=0)

    return g.op(
        "mo.slice",
        List[Symbol](input, start, stop, step),
        MOTensor(input_type.dtype, out_shape),
    )


# TODO: Change to List once Slice is a CollectionElement.
def slice(input: Symbol, s: Slice) -> Symbol:
    """Slices a symbolic tensor along its first dimension.

    Args:
        input: The symbolic tensor to slice.
        s: A slice applied to the first dimension of the input.

    Returns:
        A new symbolic tensor representing the result of slicing the
        input along its major dimension according to `s`. The output will
        have the same rank as the input, but fewer values depending on the
        slice.
    """
    var t = input.tensor_type()
    var dims = List[Dim]()
    var sym_slices = List[SymbolicSlice]()
    # There's a lot of corner cases we're getting wrong here:
    # - if `s` has no `end` then it uses math.limit.max_finite[DType.index]
    #    which causes `len` to be very wrong
    # - slices are allowed to negative-index, and for instance `len(Slice(0, -1)) is -1`
    dims.append(len(s))
    sym_slices.append(SymbolicSlice(input.graph(), s))
    for i in range(1, t.rank()):
        dims.append(t.dims[i])
    return slice(input, sym_slices, static_shape=dims)


def slice[
    keep_dims: Bool = False
](input: Symbol, idx: Symbol, axis: Int = 0) -> Symbol:
    """Slices out a `n-1`-d plane from the input symbolic tensor.

    Args:
        input: The symbolic tensor to slice.
        idx: The index to select along the given axis.
        axis: The axis to select using the index.

    Returns:
        A new symbolic tensor representing the result of selecting every
        value having the specified `index` in the specified `axis`. The result
        will have rank `n-1` where `n` is the rank of the input tensor,
        with the `axis` dimension removed.
    """
    var input_type = input.tensor_type()
    var rank = input_type.rank()

    if axis < 0:
        axis = rank + axis

    var slices = List[SymbolicSlice]()
    var dims = List[Dim]()
    for i in range(rank):
        if i == axis:
            slices.append(SymbolicSlice(idx, idx + 1, None))
            dims.append(1)
        else:
            slices.append(SymbolicSlice(None, None, None))
            dims.append(input_type.dims[i])

    var out_sliced = slice(input, slices, static_shape=dims)

    @parameter
    if keep_dims:
        return out_sliced
    else:
        return squeeze(out_sliced, axis)


# ===----------------------------------------------------------------------=== #
# Splitting
# ===----------------------------------------------------------------------=== #


def split[
    n: Int
](input: Symbol, sizes: StaticIntTuple[n], axis: Int = 0) -> List[Symbol]:
    """Splits a symbolic tensor into specified bucket sizes along the axis.

    Parameters:
        n: The number of symbolic tensors to split into.

    Args:
        input: The symbolic tensor to split.
        sizes: The list of sizes for each split.
        axis: The axis to split along.

    Returns:
        `n` symbolic tensor values. The `i`th result corresponds to the
        `sizes[i]`. Each tensor will have the same rank as the input,
        and will be the result of slicing the input of `sizes[i]` elements
        along `axis`, starting at the offset of the cumulative sum of the
        previous sizes.
    """
    var g = input.graph()
    var type = input.tensor_type()
    var norm_axis = axis + type.rank() if axis < 0 else axis

    var split_sizes = List[Int64]()
    var out_types = List[AnyMOType]()
    for i in range(n):
        split_sizes.append(sizes[i])
        var out_dims = type.dims
        out_dims[norm_axis] = Dim.static(sizes[i])
        out_types.append(MOTensor(type.dtype, out_dims))

    return g.nvop(
        "mo.split",
        List[Symbol](
            input, g.vector[DType.int64](split_sizes), g.scalar(Int64(axis))
        ),
        out_types,
    )


# ===----------------------------------------------------------------------=== #
# Concatenation
# ===----------------------------------------------------------------------=== #


def concat(values: List[Symbol], axis: Int = 0) -> Symbol:
    """Concatenates a list of symbolic tensors along an axis.

    Args:
        values: A list of symbolic tensor values. Each tensor must have the same
            dtype and rank, and must have the same dimension size for each
            dimension other than `axis`.
        axis: The axis to concatenate along. If negative, indexes relative
            to the end of the tensor shape. For instance, `concat(vs, -1)`
            will concat along the last dimension.

    Returns:
        A new symbolic tensor representing the concatenation result. It will
        have the same rank as each input tensor, and its dimenions will be the same
        as each input tensor's for each dimension other than `axis`, which will
        have size equal to the sum of all tensor's size for that dimension.
    """
    if not len(values):
        raise "must concat at least 1 value"
    var v0 = values[0]
    var g = values[0].graph()

    var v0_type = v0.tensor_type()
    var rank = v0_type.rank()
    var norm_axis = axis + v0_type.rank() if axis < 0 else axis
    if norm_axis < 0 or norm_axis >= v0_type.rank():
        raise (
            "concat axis out of bounds: axis="
            + String(norm_axis)
            + ", rank="
            + String(v0_type.rank())
        )

    var concat_dim: Dim = 0
    for i in range(len(values)):
        var v_type = values[i].tensor_type()
        if v_type.rank() != rank:
            raise (
                "all concat values must have same rank: rank[0]="
                + String(rank)
                + ", rank["
                + String(i)
                + "]="
                + String(v_type.rank())
            )
        var dim = v_type.dims[norm_axis]
        if concat_dim.is_dynamic() or dim.is_dynamic():
            concat_dim = Dim.dynamic()
        elif dim.is_symbolic():
            raise "Concat doesn't yet support symbolic dimensions"
        else:
            concat_dim = Dim.static(
                concat_dim.num_elements() + dim.num_elements()
            )

    var concat_args = List[Symbol]()
    concat_args.append(g.scalar(Int64(norm_axis)))
    for i in range(len(values)):
        concat_args.append(values[i])

    var dims = List[Dim](capacity=rank)
    for i in range(rank):
        dims.append(v0_type.dims[i])
    dims[norm_axis] = concat_dim

    return g.op("mo.concat", concat_args, MOTensor(v0_type.dtype, dims))


def stack(values: List[Symbol], axis: Int = 0) -> Symbol:
    """Stacks a list of tensors along a new axis.

    Args:
        values: A list of symbolic tensor values. Each tensor must have the same
            dtype and rank, and must have the same dimension size for each
            dimension.
        axis: The axis to concatenate along. If negative, indexes relative
            to the end of the tensor shape _plus 1_. For instance,
            `stack(vs, -1)` will create and stack along a new axis as the
            last dimension, aad `stack(vs, -2)` will create and stack along a new
            dimension which is inserted immediately before the last dimension.

    Returns:
        A new symbolic tensor representing the result of the stack. It will
        have rank `n+1` where `n` is the rank of each input tensor. Its size
        on each dimension other than `axis` will be the same as each input tensors',
        with the new axis inserted. Along the new dimension it will have size
        `len(values)`.
    """
    if axis < 0:
        axis += values[0].tensor_type().rank() + 1
    var unsqueezed = List[Symbol]()
    for i in range(len(values)):
        unsqueezed.append(unsqueeze(values[i], axis))
    return concat(unsqueezed, axis=axis)
