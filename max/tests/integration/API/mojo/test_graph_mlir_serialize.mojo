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
from max.tensor import Tensor, TensorShape
from max.graph._testing import assert_tensors_almost_equal
from _mlir.ir import Module, Operation
import _mlir
from max.graph.type import Type
from max.graph.graph import _OwnedGraph


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


fn run_on_graph(
    graph: Graph, owned input: Tensor[DType.float32]
) raises -> Tensor[DType.float32]:
    var session = InferenceSession()
    var model = session.load(graph)
    var results = model.execute("input0", input^)
    var output = results.get[DType.float32]("output0")
    return output


fn execute_graph_and_write_mlir(
    filename: String, owned input: Tensor[DType.float32]
) raises -> Tensor[DType.float32]:
    var graph = create_graph()
    with open(filename, "w") as f:
        f.write(String(graph))
    return run_on_graph(graph, input)


def main():
    with NamedTemporaryFile() as file_path:
        var input = Tensor[DType.float32](TensorShape(2, 6), 0.5)
        var outputs = execute_graph_and_write_mlir(file_path.name, input)
        var graph = Graph(file_path.name)
        outputs_ref = run_on_graph(graph, input)
        assert_tensors_almost_equal(outputs, outputs_ref)
