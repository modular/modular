# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s

from driver import Device, CPUDescriptor
from tensor import TensorSpec
from utils.index import Index
from testing import assert_equal


def test_device():
    var dev = Device(CPUDescriptor(numa_id=2))
    assert_equal(str(dev), "Device(type=CPU,numa_id=2)")

    var dev2 = dev
    assert_equal(str(dev), str(dev2))


def test_device_tensor():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    assert_equal(
        str(dt1), "DeviceTensor(Device(type=CPU,numa_id=2),Spec(2x2xfloat32))"
    )

    var dt2 = dev.allocate(
        TensorSpec(DType.float32, 3, 2),
    )
    assert_equal(
        str(dt2), "DeviceTensor(Device(type=CPU,numa_id=2),Spec(3x2xfloat32))"
    )


def test_tensor():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    assert_equal(tensor.get_rank(), 2)
    assert_equal(tensor.get_dtype(), DType.float32)

    tensor[Index(0, 0)] = 0
    tensor[Index(0, 1)] = 1
    tensor[Index(1, 0)] = 2
    tensor[Index(1, 1)] = 3

    assert_equal(tensor[0, 0], 0)
    assert_equal(tensor[1, 1], 3)


def test_tensor_slice():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt = dev.allocate(
        TensorSpec(DType.float32, 3, 3),
    )
    var tensor = dt^.get_tensor[DType.float32, 2]()

    assert_equal(tensor.get_rank(), 2)
    assert_equal(tensor.get_dtype(), DType.float32)

    for i in range(3):
        for j in range(3):
            tensor[Index(i, j)] = i + j

    # tensor
    # 0 1 2
    # 1 2 3
    # 2 3 4

    var slice = tensor[0:2, :]

    # slice
    # 0 1 2
    # 1 2 3

    assert_equal(slice[0, 0], 0)
    assert_equal(slice[1, 2], 3)

    var inner_slice = tensor[0:2, 1:2]

    # inner_slice
    # 1
    # 2

    assert_equal(inner_slice[0, 0], 1)
    assert_equal(inner_slice[1, 0], 2)


def main():
    test_device()
    test_device_tensor()
    test_tensor()
    test_tensor_slice()
