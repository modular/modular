# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

import os
from tempfile import NamedTemporaryFile

from max.engine import InferenceSession
from max.graph import Graph, TensorType, ops
from max.graph._testing import assert_tensors_almost_equal
from max.tensor import Tensor, TensorShape


fn test_load(path: String) raises -> Tensor[DType.float32]:
    var session = InferenceSession()
    var model = session.load(path)
    # Execute the model:
    var input = Tensor[DType.float32](TensorShape(2, 6), 0.5)
    var results = model.execute("input0", input^)
    var output = results.get[DType.float32]("output0")
    return output


fn create_graph() raises -> Graph:
    var graph = Graph(TensorType(DType.float32, 2, 6))
    # Create a constant for usage in the matmul op below:
    var matmul_constant_value = Tensor[DType.float32](TensorShape(6, 1), 0.15)
    var matmul_constant = graph.constant(matmul_constant_value)
    # Start adding a sequence of operator calls to build the graph.
    # We can use the subscript notation to get the graph's first input tensor:
    var matmul = graph[0] @ matmul_constant
    var relu = ops.relu(matmul)
    var softmax = ops.softmax(relu)
    graph.output(softmax)
    return graph


fn test_store(path: String) raises -> Tensor[DType.float32]:
    var graph = create_graph()

    # Load the graph:
    var session = InferenceSession()
    var model = session.load(graph)
    model.export_compiled_model(path)
    # Execute the model:
    var input = Tensor[DType.float32](TensorShape(2, 6), 0.5)
    var results = model.execute("input0", input^)
    var output = results.get[DType.float32]("output0")
    return output


def main():
    with NamedTemporaryFile() as file_path:
        var ref_output = test_store(file_path.name)
        var load_output = test_load(file_path.name)
        assert_tensors_almost_equal[DType.float32](ref_output, load_output)
