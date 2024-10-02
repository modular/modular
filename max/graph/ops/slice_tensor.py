# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for slice_tensor."""

import sys
from typing import TYPE_CHECKING, Union

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
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue
from ..type import Dim, DimLike, Shape, StaticDim, TensorType
from .constant import constant
from .select import select
from .stack import stack_scalars


# Currently slicing does not have any shape inference in RMO. Instead, it is
# done in python.
# Long term, it would be great to get more dimension wraggling and api work to
# move this, but this is the simplest path to get us nice results.


SliceIndex = Union[TensorValue, int, slice, tuple[slice, DimLike]]
SliceIndices = list[Union[SliceIndex, "EllipsisType"]]


def _slice_index_and_output(
    dim: Dim, index: SliceIndex
) -> tuple[slice, DimLike]:
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
        return (slice(index, stop, 1), 1)
    elif isinstance(index, TensorValue):
        if index.type.num_elements() != 1:
            raise ValueError(
                f"Slice index value must be a scalar, had shape {index.shape}"
            )
        return (
            slice(
                index,
                select(index == -1, int64_max, constant(0, DType.int64)),
                1,
            ),
            1,
        )
    elif isinstance(index, slice):
        if index.start is None and index.stop is None and index.step is None:
            return (slice(0, int64_max, 1), dim)

        raise NotImplementedError(
            "Can't yet support slicing with calculated output size. Please use"
            ' a tuple like (slice(0, 10), "out_dim") to specify the output'
            " dimensions."
        )
    elif (  # slice is a tuple of elements: e.g. (slice(start, stop), "out_dim")
        isinstance(index, tuple)
        and len(index) == 2
        and isinstance(index[0], slice)
        and isinstance(index[1], (int, str, Dim, np.integer))
    ):
        start = index[0].start
        stop = index[0].stop
        step = index[0].step
        zero = constant(0, DType.int64)

        if step is None:
            step = 1
        if start is None:
            start = select(step >= zero, zero, int64_max)
        if stop is None:
            stop = select(step >= zero, int64_max, zero)

        return (slice(start, stop, step), index[1])

    raise ValueError(f"Unsupported slice inputs {dim=}, {index=}")


def _has_no_ellipsis(indices: SliceIndices) -> TypeGuard[list[SliceIndex]]:
    return not any(index is Ellipsis for index in indices)


def _slice_and_output_tensors(x: TensorValue, indices: SliceIndices):
    if not x.shape:
        raise ValueError("Slicing does not support scalar inputs")

    # Replace ellipsis by trivial indices.
    ellipsis_count = indices.count(Ellipsis)
    if ellipsis_count > 1:
        raise ValueError("Slicing index can contain at most one ellipsis")

    if len(x.shape) < len(indices) - ellipsis_count:
        raise ValueError(
            f"Too many indices supplied to slice for shape {x.shape}"
        )

    ellipsis_index = indices.index(Ellipsis) if Ellipsis in indices else len(
        indices
    )
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
    output_shape = Shape(d for _, d in slices_and_outputs)

    # If type(dim,int), convert to an int constant TensorValue.
    def value(dim: Union[TensorValue, int]) -> TensorValue:
        assert isinstance(dim, (TensorValue, int))
        return dim if isinstance(dim, TensorValue) else constant(
            dim, DType.int64
        )

    # Create starts, stops, and steps tensors.
    starts = stack_scalars(value(s.start) for s in slices)
    stops = stack_scalars(value(s.stop) for s in slices)
    steps = stack_scalars(value(s.step) for s in slices)

    output_type = TensorType(x.dtype, output_shape)

    return starts, stops, steps, output_type


def slice_tensor(x: TensorValue, indices: SliceIndices) -> TensorValue:
    starts, stops, steps, output_type = _slice_and_output_tensors(x, indices)

    return Graph.current._add_op(
        rmo.mo_slice, output_type.to_mlir(), x, starts, stops, steps
    )[0].tensor
