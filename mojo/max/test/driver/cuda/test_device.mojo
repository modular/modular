# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/cuda-test-device
# RUN: %mojo-build %s -o %t/cuda-test-device
# RUN: %t/cuda-test-device

from max.driver import cpu_device, Tensor, AnyTensor
from max.driver._cuda import cuda_device
from testing import assert_equal, assert_not_equal, assert_true
from max.tensor import TensorSpec, TensorShape
from utils import Index
from gpu.host import DeviceContext


def _to_device_str(gpu_id: Int, sm_ver: Int) -> String:
    return (
        "Device(type=cuda,gpu_id=0,target_info(triple=nvptx64-nvidia-cuda,arch=sm_"
        + str(sm_ver)
        + ",features=[])"
    )


def test_cuda_device():
    gpu = cuda_device(gpu_id=0)
    assert_equal(
        str(gpu),
        _to_device_str(0, DeviceContext(0).compute_capability()),
    )


def test_copy_d2h():
    cpu = cpu_device()
    gpu = cuda_device()

    input = Tensor[DType.float32, 2](TensorShape(10, 2))

    val = 1
    for i in range(10):
        for j in range(2):
            input[i, j] = val
            val += 1

    input_cpu = input^.to_device_tensor()
    gpu_tensor = input_cpu.copy_to(gpu)
    output_cpu = gpu_tensor.copy_to(cpu)

    input = input_cpu^.to_tensor[DType.float32, 2]()
    output = output_cpu^.to_tensor[DType.float32, 2]()

    for i in range(10):
        for j in range(2):
            assert_equal(input[i, j], output[i, j])


def test_copy_empty():
    cpu = cpu_device()
    gpu = cuda_device()

    input_cpu = cpu.allocate(
        TensorSpec(DType.float32, 0, 2),
    )
    assert_equal(input_cpu.bytecount(), 0)

    gpu_tensor = input_cpu.copy_to(gpu)

    assert_equal(gpu_tensor.bytecount(), 0)


def test_copy_d2d():
    cpu = cpu_device()
    gpu = cuda_device()

    input = Tensor[DType.float32, 2](TensorShape(10, 2))

    val = 1
    for i in range(10):
        for j in range(2):
            input[i, j] = val
            val += 1

    input_cpu = input^.to_device_tensor()
    gpu_tensor1 = input_cpu.copy_to(gpu)
    gpu_tensor2 = gpu.allocate(gpu_tensor1.spec)
    gpu_tensor1.copy_into(gpu_tensor2)
    output_cpu = gpu_tensor2.copy_to(cpu)

    input = input_cpu^.to_tensor[DType.float32, 2]()
    output = output_cpu^.to_tensor[DType.float32, 2]()

    for i in range(10):
        for j in range(2):
            assert_equal(input[i, j], output[i, j])


def test_move():
    cpu = cpu_device()
    gpu = cuda_device()

    tensor = cpu.allocate(TensorSpec(DType.float32, (2, 2)))
    addr = tensor.unsafe_ptr()
    moved_tensor = tensor^.move_to(cpu)
    assert_equal(addr, moved_tensor.unsafe_ptr())

    tensor = cpu.allocate(TensorSpec(DType.float32, (2, 2)))
    addr = tensor.unsafe_ptr()
    moved_tensor = moved_tensor^.move_to(gpu)
    assert_not_equal(addr, moved_tensor.unsafe_ptr())


def test_print():
    gpu = cuda_device()

    input = Tensor[DType.float32, 2](TensorShape(10, 2))

    val = 1
    for i in range(10):
        for j in range(2):
            input[i, j] = val
            val += 1

    input_cpu = input^.to_device_tensor()
    gpu_tensor1 = input_cpu.copy_to(gpu)

    assert_true("DeviceTensor(Device(" in str(gpu_tensor1))
    assert_true("Spec(10x2xfloat32))" in str(gpu_tensor1))

    # AnyTensor
    any_tensor = AnyTensor(gpu_tensor1^)

    assert_true("Tensor(<Unable to print device tensor>," in str(any_tensor))
    assert_true("Device(" in str(any_tensor))
    assert_true("dtype=float32, shape=10x2" in str(any_tensor))

    # Tensor
    tensor = any_tensor^.to_device_tensor().to_tensor[DType.float32, 2]()
    assert_true("Tensor(<Unable to print device tensor>," in str(tensor))
    assert_true("Device(" in str(tensor))
    assert_true("dtype=float32, shape=10x2" in str(tensor))


def main():
    test_cuda_device()
    test_copy_d2h()
    test_copy_empty()
    test_copy_d2d()
    test_move()
    test_print()
