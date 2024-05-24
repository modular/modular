# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s
from driver import AnyMemory, Device, CPUDescriptor
from testing import assert_equal
from tensor import TensorSpec


def test_from_device_memory():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    var anytensor = AnyMemory(dt^)

    assert_equal(anytensor.get_rank(), 2)


def test_from_tensor():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    var tensor = dt^.get_tensor[DType.float32, 2]()

    var anytensor = AnyMemory(tensor^)

    assert_equal(anytensor.get_rank(), 2)


def _function_that_takes_anymemory(owned t1: AnyMemory, owned t2: AnyMemory):
    return t1.get_rank() + t2.get_rank()


def test_implicit_conversion():
    var dev = Device(CPUDescriptor(numa_id=2))

    var dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    var tensor = dt1^.get_tensor[DType.float32, 2]()

    var dt2 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    assert_equal(_function_that_takes_anymemory(tensor^, dt2^), 4)


def main():
    test_from_device_memory()
    test_from_tensor()
    test_implicit_conversion()
