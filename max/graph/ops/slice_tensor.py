# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for slice_tensor."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Optional, Union

from max import mlir
from max.mlir.dialects import rmo

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

if TYPE_CHECKING:
    # EllipsisType was added in 3.10, but we support down to 3.9.
    # Make this import unconditional when we drop 3.9 (MSDK-756).
    from types import EllipsisType  # type: ignore

import numpy as np
from max.dtype import DType

from ..graph import Graph
from ..type import DeviceRef, Dim, DimLike, Shape, StaticDim, TensorType
from ..value import BufferValue, TensorValue
from .constant import constant
from .where import where

# Currently slicing does not have any shape inference in RMO. Instead, it is
# done in python.
# Long term, it would be great to get more dimension wraggling and api work to
# move this, but this is the simplest path to get us nice results.


SliceIndex = Union[TensorValue, int, slice, tuple[slice, DimLike]]
SliceIndices = Sequence[Union[SliceIndex, "EllipsisType"]]


def _concrete_static_slice(n: int, index: slice) -> slice:
    """Replaces None/-ve indices in the slice with numerical non -ve indices.

    Normalizes -ve indices and replaces None indices with numerical values
    bounded by the length of the sequence.

    Args:
        n: The size of the static dimension. Should bound the slice's indices.
        index: the slice index.

    Returns:
        Returns the slice with numerical indices.
    """
    start, step, stop = index.start, index.step, index.stop

    if step is None:
        step = 1
    if start is None:
        start = 0 if step >= 0 else n - 1
    if stop is None:
        stop = n if step >= 0 else -1

    if step == 0:
        raise ValueError(f"Index.step in index {index} can't be 0")
    if not -n <= start < n:
        raise IndexError(f"Index.start {start} out of range of dim {n}")
    if not -n <= stop <= n:
        raise IndexError(f"Index.start {stop} out of range of dim {n}")

    if start < 0:
        start = n + start
    if stop < 0 and index.stop is not None:
        stop = n + stop
    return slice(start, stop, step)


def _static_slice(n: int, index: slice) -> tuple[slice, DimLike]:
    """Calculates the output shape of the slice.

    Args:
        n: The size of the static dimension. Should bound the slice's indices.
        index: the slice index.

    Returns:
        Returns the slice with numerical indices and calculated output shape.
    """
    concrete_slice = _concrete_static_slice(n, index)
    start, stop, step = (
        concrete_slice.start,
        concrete_slice.stop,
        concrete_slice.step,
    )

    # Calculate the size of slice assuming index is static.
    # Assume an arithmetic set where a0 == start, d == step, an == stop
    # stop = start + (n-1)step, ceil to return an int, return 0 if -ve length
    output_shape = max(
        0, (stop - start + (step - (1 if step > 0 else -1))) // step
    )

    return (concrete_slice, output_shape)


def _slice_index_and_output(
    dim: Dim, index: SliceIndex
) -> tuple[slice, Optional[DimLike]]:
    # These are values within an index which contains at least one
    # shape. The returned values will be used as `start, stop, step`
    # values in a mo.slice op. slices can therefore be forwarded
    # directly, while `int` and `TensorValue` need to be converted
    # to a slice(v, v+1).

    int64_max = 2**63 - 1
    # If index is int, return slice(index, index+1, 1)
    # For -1 specifically, slice(-1, 0, 1) is empty,
    # so we need to special case it.
    if isinstance(index, int):
        if isinstance(dim, StaticDim):
            if not -dim.dim <= index < dim.dim:
                raise IndexError(f"Index {index} out of range of dim {dim.dim}")
        stop = int64_max if index == -1 else index + 1
        return (slice(index, stop, 1), None)
    elif isinstance(index, TensorValue):
        if index.type.num_elements() != 1:
            raise ValueError(
                f"Slice index value must be a scalar, had shape {index.shape}"
            )
        return (  # Same as int index.
            slice(index, where(index == -1, int64_max, index + 1), 1),
            None,
        )
    elif isinstance(index, slice):
        if index.start is None and index.stop is None and index.step is None:
            return (slice(0, int64_max, 1), dim)

        if (  # index is a slice [start:stop:step]
            (index.start is None or isinstance(index.start, (int)))
            and (index.stop is None or isinstance(index.stop, (int)))
            and (index.step is None or isinstance(index.step, (int)))
        ):
            if isinstance(dim, StaticDim):
                return _static_slice(dim.dim, index)
            else:  # TODO() support dynamic dim if length calculation is possible
                raise NotImplementedError(
                    "Can't yet support slicing with calculated output size for"
                    " dynamic dims. Please use a tuple like (slice(0, 10),"
                    ' "out_dim") to specify the output dimensions.'
                )
    elif (  # index is a tuple (slice(start, stop, step), "out_dim")
        isinstance(index, tuple)
        and len(index) == 2
        and isinstance(index[0], slice)
        and isinstance(index[1], (int, str, Dim, np.integer))
    ):
        start = index[0].start
        stop = index[0].stop
        step = index[0].step
        zero = constant(0, DType.int64, DeviceRef.CPU())
        if step is None:
            step = 1
        if start is None:
            start = where(step >= zero, zero, int64_max)
        if stop is None:
            stop = where(step >= zero, int64_max, zero)
        return (slice(start, stop, step), index[1])

    raise ValueError(f"Unsupported slice inputs {dim=}, {index=}")


def _has_no_ellipsis(indices: SliceIndices) -> TypeGuard[list[SliceIndex]]:
    return not any(index is Ellipsis for index in indices)


def _stack_scalars(vals: Iterable[TensorValue]) -> TensorValue:
    axis = mlir.IntegerAttr.get(mlir.IndexType.get(), 0)

    vals = [v.reshape([1]) if v.shape != [1] else v for v in vals]
    return Graph.current._add_op(rmo.concat, vals, axis=axis)[0].tensor


def _slice_and_output_tensors(
    x: BufferValue | TensorValue, indices: SliceIndices
):
    if not x.shape:
        raise ValueError("Slicing does not support scalar inputs")

    # The indices where Ellipsis appears in the indices list
    ellipsis_indices = [
        i for i, index in enumerate(indices) if index is Ellipsis
    ]

    if len(ellipsis_indices) > 1:
        raise ValueError("Slicing index can contain at most one ellipsis")

    if len(x.shape) < len(indices) - len(ellipsis_indices):
        raise ValueError(
            f"Too many indices supplied to slice for shape {x.shape}"
        )

    ellipsis_index = ellipsis_indices[0] if ellipsis_indices else len(indices)
    before = indices[:ellipsis_index]
    after = indices[ellipsis_index + 1 :]
    assert _has_no_ellipsis(before)
    assert _has_no_ellipsis(after)

    remaining = len(x.shape) - len(before) - len(after)
    # Create a slice(None, None, None) index, for each dim indexed by ellipsis.
    full_index = [*before, *([slice(None, None, None)] * remaining), *after]
    # For each dim, convert idx (if int or TensorValue) to slice(idx, idx+1, 1).
    slices_and_outputs = [
        _slice_index_and_output(dim, index)
        for dim, index in zip(x.shape, full_index)
    ]
    slices = [s for s, _ in slices_and_outputs]
    unsqueezed_shape = Shape(
        d if d is not None else 1 for _, d in slices_and_outputs
    )
    squeezed_shape = Shape(d for _, d in slices_and_outputs if d is not None)

    # If type(dim,int), convert to an int constant TensorValue.
    def value(dim: Union[TensorValue, int]) -> TensorValue:
        assert isinstance(dim, (TensorValue, int))
        return (
            dim
            if isinstance(dim, TensorValue)
            else constant(dim, DType.int64, DeviceRef.CPU())
        )

    # Create starts, stops, and steps tensors.
    starts = _stack_scalars(value(s.start) for s in slices)
    stops = _stack_scalars(value(s.stop) for s in slices)
    steps = _stack_scalars(value(s.step) for s in slices)

    return starts, stops, steps, unsqueezed_shape, squeezed_shape


def expand_ellipsis(indices: SliceIndices, input_rank: int) -> SliceIndices:
    """Expands a potential Ellipsis in indices with slice(None).

    Returns:
        Indices with Ellipsis expanded.
        The length of the output indices is equal to the rank of the output of
        the slice, including new (None) indices.

    Raises:
        ValueError: if there is more than one Ellipsis.
    """
    num_ellipsis = indices.count(Ellipsis)
    if num_ellipsis > 1:
        msg = f"more than one Ellipsis in slice indices {indices}"
        raise ValueError(msg)

    # Handle Ellipsis by expanding remaining indices with slice(None).
    ellipsis_index = (
        indices.index(Ellipsis) if Ellipsis in indices else len(indices)
    )
    num_regular_indices = len(indices) - num_ellipsis - indices.count(None)
    remaining = input_rank - num_regular_indices
    return (
        list(indices[:ellipsis_index])
        + (remaining * [slice(None)])
        + list(indices[ellipsis_index + 1 :])
    )


def slice_arguments(
    indices: SliceIndices, input_shape: Shape
) -> tuple[list[int | Dim], list[int | Dim], list[int]]:
    """Computes starts, stops, and steps args from indices and input shape.

    Returns:
        A tuple of (starts, stops, steps) where starts and stops can be symbolic
        or algebraic expressions, but steps must be constant integers.

    Raises:
        TypeError: if any index is not a slice or integer.
        ValueError: for non-positive step.
    """
    # Expand the None indices after computing starts, stops, and steps.
    not_none_indices = [i for i in indices if i is not None]

    starts: list[int | Dim] = []
    stops: list[int | Dim] = []
    steps: list[int] = []
    for i, subslice in enumerate(not_none_indices):
        if not isinstance(subslice, (slice, int)):
            msg = (
                f"slice of tensor with symbolic shape {input_shape} "
                f"unsupported with indices {indices}. Currently, only slices "
                "and integers are supported"
            )
            raise TypeError(msg)

        if isinstance(subslice, int):
            # Create a single-element slice that will be squeezed later.
            # Set stop to input dim when index is -1 since x[-1:0] is wrong.
            stop = input_shape[i] if subslice == -1 else subslice + 1
            subslice = slice(subslice, stop)

        step = subslice.step if subslice.step is not None else 1
        if not isinstance(step, int) or step <= 0:
            # TODO(AIPIPE-109): allow negative step after improving rmo.slice.
            msg = f"expected positive integer step but got {step}"
            raise ValueError(msg)

        # Handle setting default start and stop depending on sign of step.
        if step < 0:
            start = subslice.start if subslice.start is not None else -1
            stop = (
                subslice.stop
                if subslice.stop is not None
                else -input_shape[i] - 1
            )
        elif step > 0:
            start = subslice.start if subslice.start is not None else 0
            stop = (
                subslice.stop if subslice.stop is not None else input_shape[i]
            )

        starts.append(start)
        assert isinstance(stop, (int, Dim))
        stops.append(stop)
        steps.append(step)

    return starts, stops, steps


def expand_none_indices(
    indices: SliceIndices, sliced_shape: Shape
) -> list[int | Dim]:
    """Expands a sliced shape with new None indices.

    Returns:
        The expanded shape, accounting for all the new dims (None indices).
    """
    expanded_shape: list[int | Dim] = []
    shape_iter = iter(sliced_shape)
    for slice_idx in indices:
        if slice_idx is None:
            # Replace None indices with 1.
            expanded_shape.append(1)
        elif isinstance(slice_idx, int):
            # Squeeze integer indices.
            next(shape_iter)
        else:
            # Otherwise, append the incoming dim.
            expanded_shape.append(next(shape_iter))

    return expanded_shape


def _slice_symbolic_tensor(
    x: TensorValue, indices: SliceIndices
) -> TensorValue:
    """Slices a tensor by a set of shape-like indices.

    Returns:
        A tensor sliced according to the passed indices.

    Raises:
        ValueError: if the indices vector's length exceed's the input's rank.
    """
    num_regular_indices = len(
        [i for i in indices if i is not None and i is not Ellipsis]
    )
    if num_regular_indices > x.rank:
        msg = (
            f"expected slice indices length {len(indices)} to be less than or "
            f"equal to input rank {x.rank}"
        )
        raise ValueError(msg)

    indices = expand_ellipsis(indices, x.rank)

    starts, stops, steps = slice_arguments(indices, x.shape)

    sliced_tensor = Graph.current._add_op(
        rmo.slice,
        x,
        Shape(starts).to_mlir(),
        Shape(stops).to_mlir(),
        Shape(steps).to_mlir(),
    )[0].tensor

    # Account for None entries in the slice indices by unsqueezing those dims.
    expanded_shape = expand_none_indices(indices, sliced_tensor.shape)

    if expanded_shape != sliced_tensor.shape:
        # Only reshape when necessary due to None dims or int slices.
        return sliced_tensor.reshape(expanded_shape)

    return sliced_tensor


def slice_tensor(x: TensorValue, indices: SliceIndices) -> TensorValue:
    """Slices out a subtensor view of the input tensor based on `indices`.

    The semantics of :obj:`slice_tensor()` follow NumPy slicing semantics with the
    following restrictions:

    - Slice indices must not index out of ``[-dim - 1, dim - 1]`` for negative step,
      or ``[-dim, dim]`` for positive step.

    .. code-block:: python

        # Reverse a tensor.
        slice_tensor(x, [slice(None, None, -1)])
        # Unsqueeze the second last dimension of a tensor.
        slice_tensor(x, [..., None, slice(None)])

    Returns:
        The sliced subtensor of `x`.
    """
    if not any(
        isinstance(subslice, (TensorValue, tuple)) for subslice in indices
    ):
        # For symbolic tensors, take a special path that emits rmo.slice.
        # This path doesn't support tuples or SSA values, so send those down
        # the other path.
        return _slice_symbolic_tensor(x, indices)

    starts, stops, steps, unsqueezed_shape, squeezed_shape = (
        _slice_and_output_tensors(x, indices)
    )

    unsqueezed_type = TensorType(x.dtype, unsqueezed_shape, x.device)
    return Graph.current._add_op(
        rmo.mo_slice, unsqueezed_type.to_mlir(), x, starts, stops, steps
    )[0].tensor.reshape(squeezed_shape)
