# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s

from max._driver import (
    cpu_device,
    DeviceMemory,
    DeviceTensor,
    cuda_device,
)
from testing import assert_equal
from max.tensor import TensorSpec


def test_device():
    dev = cpu_device()
    assert_equal(str(dev), "Device(type=CPU)")

    dev2 = dev
    assert_equal(str(dev), str(dev2))


def test_device_memory():
    dev = cpu_device()
    alias type = DType.float32

    dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    assert_equal(str(dt1), "DeviceTensor(Device(type=CPU),Spec(2x2xfloat32))")

    dt2 = dev.allocate(
        TensorSpec(DType.float32, 3, 2),
    )
    assert_equal(str(dt2), "DeviceTensor(Device(type=CPU),Spec(3x2xfloat32))")

    dt3 = dev.allocate(TensorSpec(type, 3, 2), str("foo"))
    assert_equal(
        str(dt3),
        "DeviceTensor(foo,Device(type=CPU),Spec(3x2xfloat32))",
    )

    dt4 = dev.allocate(bytecount=128)
    assert_equal(
        str(dt4),
        "DeviceMemory(Device(type=CPU),Bytecount(128))",
    )

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
        assert_equal(str(t), "DeviceTensor(Device(type=CPU),Spec(2x2xfloat32))")

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
        assert_equal(
            str(t[]),
            "DeviceTensor(Device(type=CPU),Spec(2x2xfloat32))",
        )


def main():
    test_device()
    test_device_memory()
    test_kv_cache()
    test_take()
