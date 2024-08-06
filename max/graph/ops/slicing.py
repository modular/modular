# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Slicing ops."""

from typing import Iterable, Union, overload
from itertools import chain, repeat, starmap, tee

import numpy as np
from max import _graph, mlir
from ..type import Dim
from max.mlir.dialects import rmo, mo

from .. import ops
from .constant import scalar
from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from ..type import Dim, ShapeLike, StaticDim, TensorType, dim, DimLike
from ..dtype import DType


def concat(vals: Iterable[ValueLike], axis: int = 0):
    vals = [GraphValue(v) for v in vals]

    # Check if length of vals is greater than 1
    if len(vals) < 1:
        raise ValueError("Must provide at least one value to concat.")

    # TODO: assert that all vals have the same rank

    axis = mlir.IntegerAttr.get(mlir.IndexType.get(), axis)

    return Graph.current._add_op(rmo.concat, vals, axis=axis)[0]


def select(cond: ValueLike, x: ValueLike, y: ValueLike):
    return Graph.current._add_op(
        rmo.select, GraphValue(cond), GraphValue(x), GraphValue(y)
    )[0]


# Currently slicing does not have any shape inference in RMO. Instead, it is done in python.
# Long term, it would be great to get more dimension wraggling and api work to move this,
# but this is the simplest path to get us nice results.


SliceIndex = Union[ValueLike, int, slice]
SliceIndices = Iterable[Union[SliceIndex, Ellipsis]]


# For slicing, we calculate everything a single dimension at a time. Then merge that into a full slice op.
def _slice_dim_helper(input_dim: Dim, index: SliceIndex) -> (Dim, slice):
    """Calculates the output dim and slice[ValueLike] for a given index."""
    max_i64 = scalar(2 ^ 63 - 1, DType.int64)
    one = scalar(1, DType.int64)
    if isinstance(index, int):
        if input_dim.is_static():
            size = input_dim.dim
            if index >= size or index < -size:
                raise IndexError(
                    f"Index {index} out of range of dim with size {size}"
                )
        index = scalar(index, DType.int64)
    if isinstance(index, ValueLike):
        # static single index slicing.
        gv = GraphValue(index)
        type = gv.tensor_type
        if type.num_elements() != 1:
            raise ValueError(
                "Tensor can only be sliced by scalar indices. Instead got"
                f" shape: {gv.shape}"
            )

        # Handle edge case where the index is `-1`.
        # Slicing from `-1` to `0` returns no inputs.
        # Instead slice from `-1` to the max i64 (the compiler will limit the value to the tensor length).
        is_neg_one = ops.equal(gv, scalar(-1, DType.int64))
        end = select(is_neg_one, max_i64, gv + one)
        return (dim(1), slice(gv, end, one))
    elif isinstance(index, slice):
        if index == slice(None, None, None):
            return (input_dim, slice(scalar(0, DType.int64), max_i64, one))
        return NotImplementedError("Slicing with int and GraphValue slices")
    else:
        return TypeError("Slicing does not support index type of {type(index)}")


def stack_scalars(vals: Iterable[GraphValue]):
    axis = mlir.IntegerAttr.get(mlir.IndexType.get(), 0)

    vals = [v.reshape([1]) if v.shape != [1] else v for v in vals]
    if len(vals) == 1:
        return vals[0]
    else:
        return Graph.current._add_op(rmo.concat, vals, axis=axis)[0]


def slice_tensor(x: GraphValue, index: SliceIndices) -> GraphValue:
    ellipsis_count = index.count(Ellipsis)
    if ellipsis_count > 1:
        return ValueError("Slicing index can contain at most one ellipsis")

    if not x.shape:
        return ValueError("Slicing does not support scalar inputs")

    if len(index) - ellipsis_count > len(x.shape):
        return ValueError(
            "Slicing shape has less dimensions than required for indexing"
        )

    ellipsis_index = index.index(Ellipsis) if Ellipsis in index else len(index)
    before = index[:ellipsis_index]
    after = index[ellipsis_index + 1 :]

    remaining = len(x.shape) - len(before) - len(after)
    full_index = [*before, *([slice(None, None, None)] * remaining), *after]
    output_shape, slices = zip(
        *(starmap(_slice_dim_helper, zip(x.shape, full_index)))
    )

    starts = stack_scalars(s.start for s in slices)
    stops = stack_scalars(s.stop for s in slices)
    steps = stack_scalars(s.step for s in slices)

    output_type = TensorType(x.tensor_type.dtype, output_shape)
    return Graph.current._add_op(
        rmo.mo_slice, output_type.to_mlir(), x, starts, stops, steps
    )[0]
