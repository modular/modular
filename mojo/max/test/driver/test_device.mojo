# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s

from driver import cpu_device, DeviceMemory, DeviceTensor, cuda_device, Tensor
from testing import assert_equal
from tensor import TensorSpec


def test_device():
    var dev = cpu_device()
    assert_equal(str(dev), "Device(type=CPU)")

    var dev2 = dev
    assert_equal(str(dev), str(dev2))


def test_device_memory():
    var dev = cpu_device()
    alias type = DType.float32

    var dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )
    assert_equal(str(dt1), "DeviceTensor(Device(type=CPU),Spec(2x2xfloat32))")

    var dt2 = dev.allocate(
        TensorSpec(DType.float32, 3, 2),
    )
    assert_equal(str(dt2), "DeviceTensor(Device(type=CPU),Spec(3x2xfloat32))")

    var dt3 = dev.allocate(TensorSpec(type, 3, 2), str("foo"))
    assert_equal(
        str(dt3),
        "DeviceTensor(foo,Device(type=CPU),Spec(3x2xfloat32))",
    )

    var dt4 = dev.allocate(bytecount=128)
    assert_equal(
        str(dt4),
        "DeviceMemory(Device(type=CPU),Bytecount(128))",
    )

    var dt5 = dev.allocate(TensorSpec(type, 2))
    var ptr = dt5.unsafe_ptr()
    var t5 = dt5^.get_tensor[type, 1]()
    t5[0] = 22
    assert_equal(Scalar.load(rebind[DTypePointer[type]](ptr)), t5[0])


def test_kv_cache():
    var cpu = cpu_device()
    alias type = DType.float32
    alias shape = (2, 2)
    var allocs = List[DeviceTensor]()
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
