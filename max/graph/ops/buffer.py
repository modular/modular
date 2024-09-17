# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for load_buffer."""

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph, location
from ..type import TensorType
from ..value import BufferValue, TensorValue


def load_buffer(
    x: BufferValue,
) -> TensorValue:
    """Loads the input buffer into a tensor.

    It loads the in-place mutable tensor to an immutable tensor graph value.
    This is semantically equivalent to a copy from `x` buffer to the output tensor.

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


def store_buffer(x: BufferValue, y: TensorValue) -> None:
    raise NotImplementedError("TODO")
