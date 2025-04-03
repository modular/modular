# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements ops used when staging a graph.

Although the following modules provide a lot of the ops you want when building
a graph, you can also use functions in
[`Graph`](/max/api/python/graph/graph/Graph) to add constant values,
such as [`constant()`](/max/api/python/graph/graph/Graph#constant),
[`vector()`](/max/api/python/graph/graph/Graph#vector), and
[`scalar()`](/max/api/python/graph/graph/Graph#scalar).

The [`TensorValue`](/max/api/python/graph/value/TensorValue) type (returned
by most ops) also implements various dunder methods to support operations
between TensorValues, such as `+` add, `*` multiply, and `@` matmul, plus
convenience methods such as
[`reshape()`](/max/api/python/graph/value/TensorValue#reshape) and
[`swapaxes()`](/max/api/python/graph/value/TensorValue#swapaxes).
"""

from __future__ import annotations

from . import allreduce
from .allgather import allgather
from .argsort import argsort
from .band_part import band_part
from .broadcast_to import broadcast_to
from .buffer import buffer_load, buffer_store, buffer_store_slice
from .cast import cast
from .chunk import chunk
from .complex import as_interleaved_complex
from .concat import concat
from .conditional import cond
from .constant import constant
from .conv import conv2d, conv3d
from .cumsum import cumsum
from .custom import custom, inplace_custom
from .debug import print
from .elementwise import *
from .elementwise import max as _elementwise_max
from .elementwise import min as _elementwise_min
from .flatten import flatten
from .gather import gather, gather_nd
from .layer_norm import layer_norm
from .matmul import matmul
from .nonzero import nonzero
from .outer import outer
from .permute import permute
from .quantized import dequantize, qmatmul
from .range import range
from .rebind import rebind
from .reduction import argmax, argmin, mean, sum
from .reduction import max as _reduce_max
from .reduction import min as _reduce_min
from .repeat_interleave import repeat_interleave
from .reshape import reshape
from .scatter import masked_scatter
from .select import select
from .shape_to_tensor import shape_to_tensor
from .slice_tensor import slice_tensor
from .split import split
from .squeeze import squeeze
from .stack import stack
from .tile import tile
from .top_k import top_k
from .transfer_to import transfer_to
from .transpose import transpose
from .unsqueeze import unsqueeze
from .while_loop import while_loop


def min(  # type: ignore[no-redef]
    x: TensorValueLike,
    y: TensorValueLike | None = None,
    /,
    axis: int | None = None,
) -> TensorValue:
    """Overload for ops.elementwise.min and ops.reduction.min.

    - If two tensors are provided, `axis` is ignored and returns an elementwise minimum.
    - If one tensor is provided, compute `ops.reduction.min` on the tensor and axis.
    """
    if y is not None and axis is not None:
        raise ValueError("Axis not allowed for elementwise min.")
    axis = -1 if axis is None else axis
    return _reduce_min(x, axis=axis) if y is None else _elementwise_min(x, y)


def max(  # type: ignore[no-redef]
    x: TensorValueLike,
    y: TensorValueLike | None = None,
    /,
    axis: int | None = None,
) -> TensorValue:
    """Overload for ops.elementwise.max and ops.reduction.max.

    - If two tensors are provided, `axis` is ignored and returns an elementwise maximum.
    - If one tensor is provided, compute `ops.reduction.max` on the tensor and axis.
    """
    if y is not None and axis is not None:
        raise ValueError("Axis not allowed for elementwise max.")
    axis = -1 if axis is None else axis
    return _reduce_max(x, axis=axis) if y is None else _elementwise_max(x, y)
