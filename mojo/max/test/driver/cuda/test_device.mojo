# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# REQUIRES: cuda
# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from driver import Device, CPUDescriptor, get_cuda_device
from testing import assert_equal
from tensor import TensorSpec
from utils import Index


def test_cuda_device():
    var gpu = get_cuda_device(gpu_id=0)
    assert_equal(str(gpu), "Device(type=CUDA,gpu_id=0)")


def test_copy_d2h():
    var cpu = Device(CPUDescriptor())
    var gpu = get_cuda_device()

    var input_cpu = cpu.allocate(
        TensorSpec(DType.float32, 10, 2),
    )

    var input = input_cpu^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(2):
            input[Index(i, j)] = val
            val += 1

    input_cpu = input^.get_device_memory()
    var gpu_tensor = input_cpu.copy_to(gpu)
    var output_cpu = gpu_tensor.copy_to(cpu)

    input = input_cpu^.get_tensor[DType.float32, 2]()
    var output = output_cpu^.get_tensor[DType.float32, 2]()

    for i in range(10):
        for j in range(2):
            assert_equal(input[i, j], output[i, j])


def test_copy_d2d():
    var cpu = Device(CPUDescriptor())
    var gpu = get_cuda_device()

    var input_cpu = cpu.allocate(
        TensorSpec(DType.float32, 10, 2),
    )

    var input = input_cpu^.get_tensor[DType.float32, 2]()

    var val = 1
    for i in range(10):
        for j in range(2):
            input[Index(i, j)] = val
            val += 1

    input_cpu = input^.get_device_memory()
    var gpu_tensor1 = input_cpu.copy_to(gpu)
    var gpu_tensor2 = gpu.allocate(gpu_tensor1.get_tensor_spec())
    gpu_tensor1.copy_into(gpu_tensor2)
    var output_cpu = gpu_tensor2.copy_to(cpu)

    input = input_cpu^.get_tensor[DType.float32, 2]()
    var output = output_cpu^.get_tensor[DType.float32, 2]()

    for i in range(10):
        for j in range(2):
            assert_equal(input[i, j], output[i, j])


def main():
    test_cuda_device()
    test_copy_d2h()
    test_copy_d2d()
