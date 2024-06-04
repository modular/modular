# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo -D MOJO_ENABLE_ASSERTIONS %s
from driver import AnyMemory, cpu_device
from testing import assert_equal
from tensor import TensorSpec


def test_from_device_memory():
    var dev = cpu_device()

    var dm = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    var anymemory = AnyMemory(dm^)

    assert_equal(anymemory.get_rank(), 2)


def test_from_tensor():
    var dev = cpu_device()

    var dm = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    var tensor = dm^.get_tensor[DType.float32, 2]()

    tensor[0, 0] = 1

    var anymemory = AnyMemory(tensor^)

    assert_equal(anymemory.get_rank(), 2)

    var dm_back = anymemory^.device_memory()
    var tensor2 = dm_back^.get_tensor[DType.float32, 2]()
    assert_equal(tensor2[0, 0], 1)


def _function_that_takes_anymemory(owned t1: AnyMemory, owned t2: AnyMemory):
    return t1.get_rank() + t2.get_rank()


def test_implicit_conversion():
    var dev = cpu_device()

    var dt1 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    var tensor = dt1^.get_tensor[DType.float32, 2]()

    var dt2 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    # FIXME (40568) should remove str
    assert_equal(str(_function_that_takes_anymemory(tensor^, dt2^)), str(4))


def main():
    test_from_device_memory()
    test_from_tensor()
    test_implicit_conversion()
