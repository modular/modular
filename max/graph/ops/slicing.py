# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Slicing ops."""

import typing
from typing import TYPE_CHECKING, Iterable, Union, TypeGuard

if TYPE_CHECKING:
    # EllipsisType was added in 3.10, but we support down to 3.9.
    # Make this import unconditional when we drop 3.9 (MSDK-756).
    from types import EllipsisType

from max import mlir
from max.mlir.dialects import rmo

from .. import ops
from ..graph import DType, Graph
from ..graph_value import GraphValue, ValueLike
from ..type import Dim, StaticDim, TensorType
from .casting import unsqueeze
from .constant import scalar


def concat(vals: Iterable[ValueLike], axis: int = 0):
    vals = [GraphValue(v) for v in vals]

    # Check if length of vals is greater than 1
    if len(vals) < 1:
        raise ValueError("Must provide at least one value to concat.")

    # TODO: assert that all vals have the same rank

    axis_attr = mlir.IntegerAttr.get(mlir.IndexType.get(), axis)

    return Graph.current._add_op(rmo.concat, vals, axis=axis_attr)[0]


def gather(input: ValueLike, indices: ValueLike, axis: int = -1) -> GraphValue:
    input, indices = GraphValue(input), GraphValue(indices)
    shape = input.tensor_type.shape
    indices_shape = indices.tensor_type.shape
    output_shape = [*shape[:axis], *indices_shape, *shape[axis + 1 :]]
    return Graph.current._add_op(
        rmo.mo_gather,
        TensorType(input.tensor_type.dtype, output_shape).to_mlir(),
        input,
        indices,
        scalar(axis, DType.int64),
    )[0]


def select(cond: ValueLike, x: ValueLike, y: ValueLike):
    return Graph.current._add_op(
        rmo.select, GraphValue(cond), GraphValue(x), GraphValue(y)
    )[0]


# Currently slicing does not have any shape inference in RMO. Instead, it is done in python.
# Long term, it would be great to get more dimension wraggling and api work to move this,
# but this is the simplest path to get us nice results.


SliceIndex = Union[GraphValue, int, slice]
SliceIndices = list[Union[SliceIndex, "EllipsisType"]]


def _slice_index(dim: Dim, index: SliceIndex) -> slice:
    # These are values within an index which contains at least one
    # shape. The returned values will be used as `start, stop, stop`
    # values in a mo.slice op. slices can therefore be forwarded
    # directly, while `int` and `GraphValue` need to be converted
    # to a slice(v, v+1).

    int64_max = 2**63 - 1
    # For -1 specifically, slice(-1, 0, 1) is empty,
    # so we need to special case it.
    if isinstance(index, int):
        if isinstance(dim, StaticDim):
            if not -dim.dim <= index < dim.dim:
                raise IndexError(f"Index {index} out of range of dim {dim.dim}")
        return slice(index, (index + 1) or None)
    elif isinstance(index, GraphValue):
        if index.tensor_type.num_elements() != 1:
            raise ValueError(
                f"Slice index value must be a scalar, had shape {index.shape}"
            )
        # TODO (MSDK-751): remove scalar call
        return slice(
            index, ops.select(index == scalar(-1, DType.int64), int64_max, 0)
        )
    elif isinstance(index, slice):
        if index != slice(None, None, None):
            raise NotImplementedError(
                "Can't yet support slicing with non-static output size."
            )
        return slice(int64_max)
    typing.assert_never("unreachable")


def stack(vals: Iterable[ValueLike], axis: int = 0) -> GraphValue:
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
    vals_coerced = [GraphValue(v) for v in vals]
    if len(vals_coerced) == 0:
        raise ValueError("Expected at least one value to stack")

    rank = len(vals_coerced[0].shape)
    if any(len(v.shape) != rank for v in vals_coerced):
        raise ValueError("all inputs to stack must be the same rank")

    unsqueezed = [unsqueeze(v, axis) for v in vals_coerced]

    # Short circuit to avoid bloating graph with unneeded op.
    if len(unsqueezed) == 1:
        return unsqueezed[0]

    return concat(unsqueezed, axis=axis)


def stack_scalars(vals: Iterable[GraphValue]):
    axis = mlir.IntegerAttr.get(mlir.IndexType.get(), 0)

    vals = [v.reshape([1]) if v.shape != [1] else v for v in vals]
    return Graph.current._add_op(rmo.concat, vals, axis=axis)[0]


def _has_no_ellipsis(indices: SliceIndices) -> TypeGuard[list[SliceIndex]]:
    return not any(index is Ellipsis for index in indices)


def slice_tensor(x: GraphValue, indices: SliceIndices) -> GraphValue:
    ellipsis_count = indices.count(Ellipsis)
    if ellipsis_count > 1:
        raise ValueError("Slicing index can contain at most one ellipsis")

    if not x.shape:
        raise ValueError("Slicing does not support scalar inputs")

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
    full_index = [*before, *([slice(None, None, None)] * remaining), *after]
    output_shape = [
        dim if isinstance(index, slice) else StaticDim(1)
        for dim, index in zip(x.shape, full_index)
    ]
    slices = [
        _slice_index(dim, index) for dim, index in zip(x.shape, full_index)
    ]

    def value(dim: Union[GraphValue, int]) -> GraphValue:
        return dim if isinstance(dim, GraphValue) else scalar(dim, DType.int64)

    starts = stack_scalars(value(s.start) for s in slices)
    stops = stack_scalars(value(s.stop) for s in slices)
    steps = stack_scalars(value(s.step) for s in slices)

    output_type = TensorType(x.tensor_type.dtype, output_shape)
    return Graph.current._add_op(
        rmo.mo_slice, output_type.to_mlir(), x, starts, stops, steps
    )[0]
