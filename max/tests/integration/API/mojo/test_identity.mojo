# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo %s

from max.graph import Graph, TensorType
from max._driver import Tensor
from testing import assert_equal
from max import engine


def main():
    graph = Graph(TensorType(DType.int32, 1))
    graph.output(graph[0])
    session = engine.InferenceSession()
    model = session.load(graph)
    input = Tensor[DType.int32, rank=1]((1,))
    input[0] = 1
    ret = model._execute(input^)
    output = ret[0].take_tensor().to_tensor[DType.int32, rank=1]()
    assert_equal(output[0], 1)
