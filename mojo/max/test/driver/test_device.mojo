# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/test-device
# RUN: %mojo-build %s -o %t/test-device
# RUN: %t/test-device

from max.driver import (
    cpu_device,
    DeviceMemory,
    DeviceTensor,
)
from memory import UnsafePointer
from testing import assert_equal, assert_true
from max.tensor import TensorSpec


def test_device():
    dev = cpu_device()
    assert_true("Device(type=cpu,target_info(" in str(dev))

    dev2 = dev
    assert_true("Device(type=cpu,target_info(" in str(dev2))


def test_device_memory():
    dev = cpu_device()
    alias type = DType.float32

    dt1 = dev.allocate(TensorSpec(DType.float32, 2, 2))
    assert_true("DeviceTensor(Device(type=cpu,target_info(" in str(dt1))
    assert_true("Spec(2x2xfloat32))" in str(dt1))

    dt2 = dev.allocate(TensorSpec(DType.float32, 3, 2))
    assert_true("DeviceTensor(Device(type=cpu,target_info(" in str(dt2))
    assert_true("Spec(3x2xfloat32))" in str(dt2))

    dt3 = dev.allocate(TensorSpec(type, 3, 2), str("foo"))
    assert_true("DeviceTensor(foo,Device(type=cpu,target_info(" in str(dt3))
    assert_true("Spec(3x2xfloat32))" in str(dt3))

    dt4 = dev.allocate(bytecount=128)
    assert_true("DeviceMemory(Device(type=cpu,target_info(" in str(dt4))
    assert_true("Bytecount(128)" in str(dt4))

    dt5 = dev.allocate(TensorSpec(type, 2))
    ptr = dt5.unsafe_ptr()
    t5 = dt5^.to_tensor[type, 1]()
    t5[0] = 22
    assert_equal(rebind[UnsafePointer[Scalar[type]]](ptr).load(), t5[0])


def test_take():
    var tensors = List[DeviceTensor]()
    var cpu = cpu_device()
    for _ in range(2):
        tensors.append(cpu.allocate(TensorSpec(DType.float32, 2, 2)))

    def consume_and_check(owned t: DeviceTensor):
        assert_true("DeviceTensor(Device(type=cpu,target_info" in str(t))
        assert_true("Spec(2x2xfloat32))" in str(t))

    for tensor in tensors:
        consume_and_check(tensor[].take())


def test_kv_cache():
    cpu = cpu_device()
    alias type = DType.float32
    alias shape = (2, 2)
    allocs = List[DeviceTensor]()
    for _ in range(2):
        allocs.append(cpu.allocate(TensorSpec(type, shape)))

    for t in allocs:
        assert_true("DeviceTensor(Device(type=cpu,target_info(" in str(t[]))
        assert_true("Spec(2x2xfloat32))" in str(t[]))


def main():
    test_device()
    test_device_memory()
    test_kv_cache()
    test_take()
