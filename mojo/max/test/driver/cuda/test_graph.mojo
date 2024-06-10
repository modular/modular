# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: cuda
# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s %S/Inputs/cuda-graph.mlir

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/driver-graph-test
# RUN: mojo build %s -o %t/driver-graph-test
# RUN: %t/driver-graph-test %S/Inputs/cuda-graph.mlir

from max.graph import Graph, TensorType, Symbol, Type
from tensor import TensorSpec
from driver import (
    compile_graph,
    cuda_device,
    cpu_device,
    AnyMemory,
    Tensor,
    Device,
)
from testing import assert_equal
from sys import argv


def test_graph_execution():
    var args = argv()
    if len(args) < 2:
        print("Usage: program.exe <model_path>")
        raise "ArgumentError: Expected model path"

    var graph_path = args[1]
    var gpu = cuda_device()
    var cpu = cpu_device()
    compiled_graph = compile_graph(graph_path, gpu)
    var executable_graph = compiled_graph.load()

    var input_dt = cpu.allocate(TensorSpec(DType.float32, 5))
    var input_cpu = input_dt^.get_tensor[DType.float32, 1]()
    for i in range(5):
        input_cpu[i] = i
    var gpu_tensor = input_cpu^.get_device_memory().copy_to(gpu)
    var outputs = executable_graph.execute(gpu_tensor^)
    assert_equal(len(outputs), 1)

    def _assert_values(inout memory: AnyMemory, device: Device):
        var new = AnyMemory()
        var tmp = memory^
        memory = new^
        var gpu_tensor = tmp^.device_memory()
        var cpu_tensor = gpu_tensor.copy_to(device).get_tensor[
            DType.float32, 1
        ]()
        for i in range(5):
            assert_equal(cpu_tensor[i], 2 * i)

    for output in outputs:
        _assert_values(output[], cpu)


def main():
    test_graph_execution()
