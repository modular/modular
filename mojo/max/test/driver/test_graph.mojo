# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/driver-graph-test
# RUN: mojo build %s -o %t/driver-graph-test
# RUN: %t/driver-graph-test

import tensor as tensor
from max.graph import Graph, TensorType, Type, ops
from max._driver import cpu_device, Tensor

from testing import assert_equal, assert_almost_equal


def test_graph_execution():
    graph = Graph(TensorType(DType.float32, 1))
    graph.output(graph[0])

    cpu = cpu_device()
    compiled_graph = cpu.compile(graph)
    executable_graph = cpu.load(compiled_graph)

    input = Tensor[DType.float32, 1]((1,))
    input[0] = 1.0
    outputs = executable_graph(input^)
    assert_equal(len(outputs), 1)

    for output in outputs:
        tensor = (
            output[].take().to_device_tensor().to_tensor[DType.float32, 1]()
        )
        assert_equal(
            tensor[0],
            1.0,
        )


def build_graph() -> Graph:
    g = Graph(
        "test_mnist_helpers",
        List[Type](TensorType(DType.float32, 1, 28, 28, 1)),
    )
    cst_data = tensor.Tensor[DType.float32](128, 10)
    cst_data._to_buffer().fill(0.5)
    cst = g.constant(cst_data)

    cst_0 = g.constant(
        tensor.Tensor[DType.float32](
            tensor.TensorShape(1, 10),
            -0.0675942451,
            0.0063267909,
            7.43086217e-4,
            -0.0126994187,
            0.0148473661,
            0.108896509,
            -0.0398316309,
            0.0461452715,
            -0.0281771384,
            -0.0431172103,
        )
    )

    cst_1_data = tensor.Tensor[DType.float32](784, 128)
    cst_1_data._to_buffer().fill(0.5)
    cst_1 = g.constant(cst_1_data)

    cst_2_data = tensor.Tensor[DType.float32](1, 128)
    cst_2_data._to_buffer().fill(0.5)
    cst_2 = g.constant(cst_2_data)

    p1 = g[0].reshape(1, 784)
    p2 = p1 @ cst_1
    p3 = p2 + cst_2
    p4 = ops.relu(p3)
    p5 = p4 @ cst
    p6 = p5 + cst_0
    _ = g.output(p6)

    return g


def test_mnist():
    g = build_graph()
    cpu = cpu_device()
    compiled_graph = cpu.compile(g)
    executable_graph = cpu.load(compiled_graph)

    input = Tensor[DType.float32, 4]((1, 28, 28, 1))
    for i in range(1):
        for j in range(28):
            for k in range(28):
                for l in range(1):
                    input[i, j, k, l] = 1.0
    outputs = executable_graph(input^)
    assert_equal(len(outputs), 1)

    for output in outputs:
        tensor = (
            output[].take().to_device_tensor().to_tensor[DType.float32, 2]()
        )

        expected_output = Float32(2.511993e04)
        for i in range(10):
            assert_almost_equal(tensor[0, i], expected_output, atol=0.1)
        assert_equal(tensor.rank, 2)


def main():
    test_graph_execution()
    test_mnist()
