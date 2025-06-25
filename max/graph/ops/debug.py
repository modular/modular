# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops to help with debugging."""

from __future__ import annotations

from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue


def print(value: str | TensorValue, label: str = "debug_tensor") -> None:
    """
    Prints the value of a tensor or a string during graph execution.

    This function is used to output the current value of a tensor and is
    primarily used for debugging purposes within the context of the Max
    Engine and its graph execution framework. This is particularly useful to
    verify the intermediate results of your computations are as expected.

    By printing the tensor values, you can visualize the data flowing through the
    graph, which helps in understanding how the operations are transforming
    the data.

    When labeling the function you can assign the output, making it easier to
    identify which tensor's value is being printed, especially when there are
    multiple print statements in a complex graph.

    .. code-block:: python

        def add_tensors(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
            input_type = TensorType(dtype=DType.float32, shape=(1,), device=DeviceRef.CPU())
            with Graph(
                "simple_add_graph", input_types=(input_type, input_type)
            ) as graph:
                lhs, rhs = graph.inputs
                out = ops.add(lhs, rhs)
                ops.print(out, label="addition_output")  # Pass the output tensor here

                graph.output(out)
                print("final graph:", graph)

    Args:
        value: The value to print. Can be either a string or a TensorValue.
        label: A label to identify the printed value. Defaults to
          ``debug_tensor``.
    """
    in_chain = Graph.current._current_chain

    op = mo.debug_print if isinstance(value, str) else mo.debug_tensor_print

    output = Graph.current._add_op(op, in_chain, value, label=label)[0]
    Graph.current._update_chain(output)
