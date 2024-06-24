# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from max.graph import Graph, TensorType, Symbol, Type, ops
from max._driver import compile_graph
from tensor import TensorSpec
from driver import cpu_device, CPUDescriptor, AnyTensor, Tensor
from testing import assert_equal, assert_true, assert_raises
import tensor


def test_graph_execution():
    graph = Graph(TensorType(DType.float32, 1))
    graph.output(graph[0])

    cpu = cpu_device()
    compiled_graph = compile_graph(graph, cpu)
    executable_graph = compiled_graph.load()

    input_dt = cpu.allocate(TensorSpec(DType.float32, 1))
    input = input_dt^.to_tensor[DType.float32, 1]()
    input[0] = 1.0
    outputs = executable_graph.execute(input^)
    assert_equal(len(outputs), 1)

    def _assert_values(inout memory: AnyTensor):
        new = AnyTensor()
        tmp = memory^
        memory = new^
        tensor = tmp^.to_device_tensor().to_tensor[DType.float32, 1]()
        val = tensor[0]
        memory = tensor^.to_device_tensor()
        assert_equal(val, 1.0)

    for output in outputs:
        _assert_values(output[])


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
    compiled_graph = compile_graph(g, cpu)
    executable_graph = compiled_graph.load()

    input_dt = cpu.allocate(TensorSpec(DType.float32, 1, 28, 28, 1))
    input = input_dt^.to_tensor[DType.float32, 4]()
    for i in range(1):
        for j in range(28):
            for k in range(28):
                for l in range(1):
                    input[i, j, k, l] = 1.0
    outputs = executable_graph.execute(input^)
    assert_equal(len(outputs), 1)

    def _assert_values(inout memory: AnyTensor):
        new = AnyTensor()
        tmp = memory^
        memory = new^
        var tensor = tmp^.to_device_tensor().to_tensor[DType.float32, 2]()
        var rank = tensor.rank
        var output_list = List[Float32]()
        output_list.reserve(10)
        for i in range(10):
            output_list.append(tensor[0, i])
        memory = tensor^.to_device_tensor()
        assert_equal(rank, 2)

        expected_outputs = List[Float32](
            2.511993e04,
            2.512001e04,
            2.512000e04,
            2.511999e04,
            2.512002e04,
            2.512011e04,
            2.511996e04,
            2.512005e04,
            2.511997e04,
            2.511996e04,
        )

        for i in range(10):
            assert_true(abs(output_list[i] - expected_outputs[i]) < 0.1)

    for output in outputs:
        _assert_values(output[])


def main():
    test_graph_execution()
    test_mnist()
