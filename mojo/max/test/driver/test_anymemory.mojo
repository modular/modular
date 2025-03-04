# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/test-anymemory
# RUN: %mojo-build %s -o %t/test-anymemory
# RUN: %t/test-anymemory

from max.driver import AnyMemory, AnyMojoValue, AnyTensor, Tensor, cpu
from max.tensor import TensorShape, TensorSpec
from testing import assert_equal, assert_false, assert_raises, assert_true


def test_from_device_memory():
    dev = cpu()

    dm = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    anytensor = AnyTensor(dm^)

    assert_equal(anytensor.get_rank(), 2)

    dm_back = anytensor^.to_device_tensor()
    assert_equal(dm_back.spec.rank(), 2)


def test_from_tensor():
    tensor = Tensor[DType.float32, 2](TensorShape(2, 2))

    tensor[0, 0] = 1

    anytensor = AnyTensor(tensor^)

    assert_equal(anytensor.get_rank(), 2)

    tensor2 = anytensor^.to_tensor[DType.float32, 2]()
    assert_equal(tensor2[0, 0], 1)


def test_from_tensor_incorrect():
    tensor = Tensor[DType.float32, 2](TensorShape(2, 2))

    tensor[0, 0] = 1

    anytensor = AnyTensor(tensor^)

    assert_equal(anytensor.get_rank(), 2)

    with assert_raises(contains="dtype does not match"):
        tensor2 = anytensor^.to_tensor[DType.int32, 2]()


def _function_that_takes_anytensor(owned t1: AnyTensor, owned t2: AnyTensor):
    return t1.get_rank() + t2.get_rank()


def test_implicit_conversion():
    dev = cpu()

    tensor = Tensor[DType.float32, 2](TensorShape(2, 2))

    dt2 = dev.allocate(
        TensorSpec(DType.float32, 2, 2),
    )

    # FIXME (40568) should remove str
    assert_equal(
        String(_function_that_takes_anytensor(tensor^, dt2^)), String(4)
    )


@value
struct Foo:
    var val: String


@value
@register_passable
struct RegFoo:
    pass


def test_any_memory():
    dev = cpu()

    tensor = Tensor[DType.float32, 2](TensorShape(2, 2))

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


def test_print_any_tensor():
    tensor = Tensor[DType.float32, 2](TensorShape(2, 2))

    tensor[0, 0] = 1
    tensor[0, 1] = 2
    tensor[1, 0] = 3
    tensor[1, 1] = 4

    anytensor = AnyTensor(tensor^)

    expected = """Tensor([[1.0, 2.0],
[3.0, 4.0]], dtype=float32, shape=2x2)"""
    assert_equal(expected, String(anytensor))


def main():
    test_from_device_memory()
    test_from_tensor()
    test_from_tensor_incorrect()
    test_implicit_conversion()
    test_any_memory()
    test_print_any_tensor()
