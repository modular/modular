# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements operations used when staging a graph.

This module provides operations for building computational graphs in MAX. These
operations create, transform, and manipulate tensor values within the graph.

You can also use functions in [`Graph`](/max/api/python/graph/Graph) to add
constant values to your graph with operations like
[`constant()`](/max/api/python/graph/ops#max.graph.ops.constant).

The [`TensorValue`](/max/api/python/graph/TensorValue/) type (returned by most
operations) implements various dunder methods to support operations between
TensorValues, such as `+` for addition, `*` for multiplication, and `@` for
matrix multiplication. It also provides convenience methods like
[`reshape()`](/max/api/python/graph/TensorValue/#max.graph.TensorValue.reshape)
and
[`flatten()`](/max/api/python/graph/TensorValue/#max.graph.TensorValue.flatten).
"""

from __future__ import annotations

from . import allreduce, random
from .allgather import allgather
from .argsort import argsort
from .band_part import band_part
from .broadcast_to import broadcast_to
from .buffer import buffer_load, buffer_store, buffer_store_slice
from .call import call
from .cast import cast
from .chunk import chunk
from .complex import as_interleaved_complex
from .concat import concat
from .conditional import cond
from .constant import constant
from .conv import conv2d, conv3d
from .conv_transpose import conv2d_transpose
from .cumsum import cumsum
from .custom import custom, inplace_custom
from .debug import print
from .elementwise import *
from .elementwise import max as _elementwise_max
from .elementwise import min as _elementwise_min
from .flatten import flatten
from .fold import fold
from .gather import gather, gather_nd
from .hann_window import hann_window
from .irfft import irfft
from .layer_norm import layer_norm
from .matmul import matmul
from .nonzero import nonzero
from .outer import outer
from .pad import pad
from .permute import permute
from .quantized import dequantize, qmatmul
from .range import range
from .rebind import rebind
from .reduction import argmax, argmin, mean, sum
from .reduction import max as _reduce_max
from .reduction import min as _reduce_min
from .repeat_interleave import repeat_interleave
from .reshape import reshape
from .scatter import masked_scatter, scatter
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
