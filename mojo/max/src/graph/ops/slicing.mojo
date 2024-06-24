# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops that slice, index, stack, concat etc."""

from builtin._location import __call_location, _SourceLocation
from collections.optional import Optional
from utils.numerics import max_finite

from _mlir.ir import Attribute, Identifier, NamedAttribute
from ..error import error
from ..symbol import SymbolicSlice

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
        raise error(
            g,
            "gather axis out of bounds: axis="
            + str(axis)
            + ", rank="
            + str(input_type.rank()),
        )

    var dims = List[Dim]()
    for i in range(input_type.rank()):
        if i == axis:
            for j in range(indices_type.rank()):
                dims.append(indices_type.dims[j])
        else:
            dims.append(input_type.dims[i])

    return g.op(
        "rmo.mo.gather",
        List[Symbol](input, indices, g.scalar(Int64(axis))),
        TensorType(input_type.dtype, dims),
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
        out_shape = static_shape.value()
    else:
        var dims = List[Dim]()
        for axis in range(input_type.rank()):
            if axis < len(slices):
                dims.append(Dim.dynamic())
            else:
                dims.append(input_type.dims[axis])
        out_shape = dims

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
            starts.append(start.value())
        else:
            starts.append(g.scalar(Int64(0)))
        if stop:
            stops.append(stop.value())
        else:
            # If we pass in the max int64 here, MO will automatically scale it to the input axis size.
            # This greatly reduces generated MO ir.
            stops.append(g.scalar(max_finite[DType.int64]()))
        if step:
            steps.append(step.value())
        else:
            steps.append(g.scalar(Int64(1)))

    var start = stack(starts, axis=0)
    var stop = stack(stops, axis=0)
    var step = stack(steps, axis=0)

    return g.op(
        "rmo.mo.slice",
        List[Symbol](input, start, stop, step),
        TensorType(input_type.dtype, out_shape),
    )


def select(condition: Symbol, x: Symbol, y: Symbol) -> Symbol:
    """Returns `condition ? x : y` (element-wise), where `cond`, `x` and `y`
    are input tensors.

    Args:
        condition: The condition tensor to use for selecting elementwise
                   values.
        x: If the condition is true at a position, the value from the same
           position in this tensor will be selected.
        y: If the condition is false at a position, the value from the same
           position in this tensor will be selected.

    Returns:
        A new symbolic tensor holding either values from either `x` or `y`,
        based on the elements in `condition`.
    """
    var g = condition.graph()
    return g.op(
        "rmo.mo.select",
        List[Symbol](condition, x, y),
        x.tensor_type(),
    )


def _slice_size(s: Slice, length: Optional[Int64]) -> Optional[Int]:
    """Calculates the size of a slice into a tensor or returns None."""

    def sign(x: Int64) -> Bool:
        return x > 0

    if length:
        start, stop, step = s.indices(int(length.value()))
        return len(range(start, stop, step))
    elif s.end and sign((s.start or 0).value()) == sign(s.end.value()):
        return s.unsafe_indices()
    return None


@always_inline
def slice(
    input: Symbol,
    *slices: Slice,
    out_dims: List[Dim] = List[Dim](),
    location: Optional[_SourceLocation] = None,
) -> Symbol:
    """Slices a symbolic tensor with `Int` ranges.

    Args:
        input: The symbolic tensor to slice.
        slices: Slices across the tensor's dimensions. If fewer than
            `input.rank()` slices are provided, the remaining dimensions
            will be trivially sliced.
        out_dims: The expected output dimensions returned by slicing.
          These will be assert at graph execution time to be correct.
        location: An optional location for a more specific error message.

    Returns:
        A new symbolic tensor representing the result of slicing the
        input along its dimension according to `slices`. The output will
        have the same rank as the input, but fewer values depending on the
        slice values.

    Raises:
        An exception if out_dims is empty and can't be calculated at graph build time.
    """
    return slice(input, slices, out_dims, location)


@always_inline
def slice(
    input: Symbol,
    slices: VariadicList[Slice],
    out_dims: List[Dim] = List[Dim](),
    location: Optional[_SourceLocation] = None,
) -> Symbol:
    """Slices a symbolic tensor with `Int` ranges.

    Will raise an exception if out_dim is not set and can't be calculated at graph build time.

    Args:
        input: The symbolic tensor to slice.
        slices: Slices across the tensor's dimensions. If fewer than
            `input.rank()` slices are provided, the remaining dimensions
            will be trivially sliced.
        out_dims: The expected output dimensions returned by slicing.
          These will be assert at graph execution time to be correct.
        location: An optional location for a more specific error message.

    Returns:
        A new symbolic tensor representing the result of slicing the
        input along its dimension according to `slices`. The output will
        have the same rank as the input, but fewer values depending on the
        slice values.

    Raises:
        An exception if out_dims is empty and can't be calculated at graph build time.
    """
    g = input.graph()
    t = input.tensor_type()
    if len(slices) > t.rank():
        message = str("got {} slices, tensor only has rank {}")
        raise error(g, message.format(len(slices), t.rank()))

    slice_max = int(Int64.MAX)
    empty_slice = Slice(start=None, end=None, step=1)

    dims = List[Dim]()
    starts = List[Int64]()
    stops = List[Int64]()
    steps = List[Int64]()

    loc = location or __call_location()
    if out_dims:
        dims = out_dims
        if len(dims) != len(slices):
            raise error(
                input.graph(),
                "Must specify an output dim for every slice dimension",
                loc,
            )

    for i in range(t.rank()):
        slice = slices[i] if i < len(slices) else empty_slice
        dim = t.dims[i]

        start, stop, step = slice.indices(
            int(dim.num_elements() if dim.is_static() else slice_max)
        )
        if step < 1:
            raise error(g, "negative slices unsupported")

        starts.append(start)
        stops.append(stop)
        steps.append(step)

        if i < len(dims):
            continue

        if slice == empty_slice:
            dims.append(dim)
            continue

        length = dim.maybe_num_elements()
        size = _slice_size(slice, length)

        if not size:
            raise error(
                input.graph(),
                "Could not calculate slice size at graph build time for dim="
                + str(i)
                + ". Please set out_dims.",
                loc,
            )
        # TODO(GRA-578): This should be handled by the slice op builder.
        # It should raise if the slice may load the wrong number of elements.
        # Technically this is based on the last loaded index rather than the end.
        if length and slice.end and slice.end.value() > int(length.value()):
            raise error(
                input.graph(),
                "Calculate slice end for dim="
                + str(i)
                + " was "
                + str(slice.end.value())
                + ", but the dimensions only has "
                + str(length.value())
                + " elements.",
                loc,
            )
        dims.append(size.value())

    return g.op(
        "rmo.mo.slice",
        List(input, g.vector(starts), g.vector(stops), g.vector(steps)),
        TensorType(t.dtype, dims),
    )


def slice(
    input: Symbol, idx: Symbol, axis: Int = 0, keep_dims: Bool = False
) -> Symbol:
    """Slices out a `n-1`-d plane from the input symbolic tensor.

    Args:
        input: The symbolic tensor to slice.
        idx: The index to select along the given axis.
        axis: The axis to select using the index.
        keep_dims: If True, returns a tensor of the same rank as the input.

    Returns:
        A new symbolic tensor representing the result of selecting every
        value having the specified `index` in the specified `axis`. If `keep_dims` is `False`,
        The result will have rank `n-1` where `n` is the rank of the input tensor,
        with the `axis` dimension removed.
    """
    var g = input.graph()
    var input_type = input.tensor_type()
    var rank = input_type.rank()

    if axis < 0:
        axis = rank + axis

    var slices = List[SymbolicSlice]()
    var dims = List[Dim]()
    for i in range(rank):
        if i == axis:
            # Handle edge case where the index is `-1`.
            # Slicing from `-1` to `0` returns no inputs.
            # Instead slice from `-1` to `Int64.MAX`.
            is_neg_one = ops.equal(idx, g.scalar(Int64(-1)))
            end = select(is_neg_one, g.scalar(Int64.MAX), idx + 1)
            slices.append(SymbolicSlice(idx, end, None))
            dims.append(1)
        else:
            slices.append(SymbolicSlice(None, None, None))
            dims.append(input_type.dims[i])

    var out_sliced = slice(input, slices, static_shape=dims)

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
    var out_types = List[Type]()
    for i in range(n):
        split_sizes.append(sizes[i])
        var out_dims = type.dims
        out_dims[norm_axis] = Dim.static(sizes[i])
        out_types.append(TensorType(type.dtype, out_dims))

    return g.nvop(
        "rmo.mo.split",
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
    var g = values[0].graph()
    if not len(values):
        raise error(g, "must concat at least 1 value")

    var ctx = g._context()
    var axisAttr = Attribute.parse(ctx, str(axis))
    var namedAxisAttr = NamedAttribute(Identifier(ctx, "axis"), axisAttr)
    return g.nvop(
        "rmo.concat",
        values,
        attrs=List[NamedAttribute](namedAxisAttr),
        enable_result_type_inference=True,
    )[0]


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
