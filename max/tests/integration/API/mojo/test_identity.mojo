# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from max import engine
from max.driver import Tensor
from max.graph import Graph, TensorType
from max.tensor import TensorShape
from testing import assert_equal


def main():
    graph = Graph(TensorType(DType.int32, 1))
    graph.output(graph[0])
    session = engine.InferenceSession()
    model = session.load(graph)
    input = Tensor[DType.int32, rank=1](
        TensorShape(
            1,
        )
    )
    input[0] = 1
    ret = model.execute(input^)
    output = ret[0].take_tensor().to_tensor[DType.int32, rank=1]()
    assert_equal(output[0], 1)
