# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from driver import Device, CPUDescriptor, get_cuda_device
from testing import assert_equal
from tensor import TensorSpec


def test_device():
    var dev = Device(CPUDescriptor(numa_id=2))
    assert_equal(str(dev), "Device(type=CPU,numa_id=2)")

    var dev2 = dev
    assert_equal(str(dev), str(dev2))


def test_device_memory():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    assert_equal(
        str(dt1), "DeviceMemory(Device(type=CPU,numa_id=2),Spec(2x2xfloat32))"
    )

    var dt2 = dev.allocate(
        TensorSpec(DType.float32, 3, 2),
    )
    assert_equal(
        str(dt2), "DeviceMemory(Device(type=CPU,numa_id=2),Spec(3x2xfloat32))"
    )


def main():
    test_device()
    test_device_memory()
