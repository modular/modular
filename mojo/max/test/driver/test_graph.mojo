# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from max.graph import Graph, TensorType
from tensor import TensorSpec
from driver import compile_graph, cpu_device, CPUDescriptor


def test_graph_execution():
    graph = Graph(TensorType(DType.float32, 1))
    graph.output(graph[0])

    var cpu = cpu_device()
    compiled_graph = compile_graph(graph, cpu)
    _ = compiled_graph.load()


def main():
    test_graph_execution()
