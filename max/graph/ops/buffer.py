# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for load_buffer."""

from typing import Union
from max.dtype import DType

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph, location
from ..type import TensorType
from ..value import BufferValue, TensorValue, ValueLike
from .constant import constant

from .stack import stack_scalars
from .slice_tensor import SliceIndices, _slice_and_output_tensors


def load_buffer(
    x: BufferValue,
) -> TensorValue:
    """Loads the input buffer into a tensor.

    It loads the in-place mutable tensor to an immutable tensor graph value.
    This is semantically equivalent to a copy from the mutable tensor `x` to the
    mutable value-semantic tensor `output`.

    Args:
        x: The buffer to be loaded to a tensor.

    Returns:
        A tensor graph value representing a copy of the buffer loaded.
    """
    in_chain = Graph.current._current_chain

    # TODO(MSDK-975): Change this to use self._add_op().
    with Graph.current._context, mlir.InsertionPoint(
        Graph.current._body
    ), location():
        output = rmo.mo_mutable_load(
            mlir.Type.parse("!mo.chain"),
            TensorType(x.dtype, x.shape).to_mlir(),
            in_chain,
            x._mlir_value,
        )

    Graph.current._update_chain(output[0])

    return TensorValue(output[1])


def store_in_buffer(y: BufferValue, x: TensorValue) -> None:
    """Stores the input tensor into the inout buffer.

    It stores the immutable input tensor `x` in the mutable tensor `y`.
    This is semantically equivalent to a copy from `x` tensor to the `y` buffer.

    Args:
        x: The tensor to be stored in the buffer.
        y: The buffer to store the tensor in.
    """
    in_chain = Graph.current._current_chain

    # TODO(MSDK-975): Change this to use self._add_op().
    with Graph.current._context, mlir.InsertionPoint(
        Graph.current._body
    ), location():
        output = rmo.mo_mutable_store(in_chain, y._mlir_value, x._mlir_value)

    Graph.current._update_chain(output)


def set_slice(
    destination: BufferValue, source: ValueLike, indices: SliceIndices
) -> None:
    """Stores the input tensor to into a slice in the input buffer.

    It stores the immutable input tensor `x` in the mutable tensor `y`.
    This is semantically equivalent to a copy from `x` tensor to a slice in the
    `y` buffer at index specified by `indices`.

    Args:
        x: The tensor to be stored in the buffer.
        y: The buffer to store the tensor in.
        indices: The index in the buffer where the tensor should be stored
    """
    in_chain = Graph.current._current_chain

    starts, stops, steps, _ = _slice_and_output_tensors(source, indices)

    # TODO(MSDK-975): Change this to use self._add_op().
    with Graph.current._context, mlir.InsertionPoint(
        Graph.current._body
    ), location():
        output = rmo.mo_mutable_store_slice(
            in_chain,
            destination._mlir_value,
            source._mlir_value,
            starts._mlir_value,
            stops._mlir_value,
            steps._mlir_value,
        )

    Graph.current._update_chain(output)
