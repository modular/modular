# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s
from max._driver import AnyTensor, cpu_device, Tensor, AnyMemory, AnyMojoValue
from testing import assert_equal, assert_true, assert_false
from max.tensor import TensorSpec


def test_from_device_memory():
    dev = cpu_device()

    dm = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    anytensor = AnyTensor(dm^)

    assert_equal(anytensor.get_rank(), 2)


def test_from_tensor():
    dev = cpu_device()

    tensor = Tensor[DType.float32, 2]((2, 2))

    tensor[0, 0] = 1

    anytensor = AnyTensor(tensor^)

    assert_equal(anytensor.get_rank(), 2)

    dm_back = anytensor^.to_device_tensor()
    tensor2 = dm_back^.to_tensor[DType.float32, 2]()
    assert_equal(tensor2[0, 0], 1)


def _function_that_takes_anytensor(owned t1: AnyTensor, owned t2: AnyTensor):
    return t1.get_rank() + t2.get_rank()


def test_implicit_conversion():
    dev = cpu_device()

    tensor = Tensor[DType.float32, 2]((2, 2))

    dt2 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    # FIXME (40568) should remove str
    assert_equal(str(_function_that_takes_anytensor(tensor^, dt2^)), str(4))


@value
struct Foo:
    var val: String


@value
@register_passable
struct RegFoo:
    pass


def test_any_memory():
    dev = cpu_device()

    tensor = Tensor[DType.float32, 2]((2, 2))

    dt2 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    memory = AnyMemory(dt2^)

    assert_true(memory.is_tensor())

    foo = Foo("Hello")
    foo_memory = AnyMemory(AnyMojoValue(foo^))

    assert_false(foo_memory.is_tensor())

    foo_rt = foo_memory.to[Foo]()
    assert_equal(foo_rt.val, "Hello")

    reg_foo = RegFoo()
    reg_foo_memory = AnyMemory(AnyMojoValue(reg_foo))

    assert_false(reg_foo_memory.is_tensor())


def main():
    test_from_device_memory()
    test_from_tensor()
    test_implicit_conversion()
    test_any_memory()
