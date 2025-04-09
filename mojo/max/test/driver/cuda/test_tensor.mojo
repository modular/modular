# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo --debug-level full -D ENABLE_ASSERTIONS %s

# COM: Test with mojo build
# RUN: mkdir -p %t
# RUN: rm -rf %t/cuda-test-tensor
# RUN: %mojo-build -D ENABLE_ASSERTIONS %s -o %t/cuda-test-tensor
# RUN: %t/cuda-test-tensor

from max.driver import (
    Tensor,
    cpu,
    accelerator,
)
from max.tensor import TensorShape
from testing import assert_not_equal


def test_tensors_with_different_devices_are_not_equal():
    tensor1 = Tensor[DType.float32, 2](TensorShape(2, 2))
    tensor2 = Tensor[DType.float32, 2](TensorShape(2, 2))

    tensor1[0, 0] = 0
    tensor1[0, 1] = 1
    tensor1[1, 0] = 2
    tensor1[1, 1] = 3

    tensor2[0, 0] = 0
    tensor2[0, 1] = 1
    tensor2[1, 0] = 2
    tensor2[1, 1] = 3

    tensor2 = tensor2.move_to(accelerator())

    assert_not_equal(tensor1, tensor2)


def test_tensors_on_accelerator_devices_are_not_equal():
    tensor1 = Tensor[DType.float32, 2](TensorShape(2, 2))
    tensor2 = Tensor[DType.float32, 2](TensorShape(2, 2))

    tensor1[0, 0] = 0
    tensor1[0, 1] = 1
    tensor1[1, 0] = 2
    tensor1[1, 1] = 3

    tensor2[0, 0] = 0
    tensor2[0, 1] = 1
    tensor2[1, 0] = 2
    tensor2[1, 1] = 3

    tensor1 = tensor1.move_to(accelerator())
    tensor2 = tensor2.move_to(accelerator())

    assert_not_equal(tensor1, tensor2)


def main():
    test_tensors_with_different_devices_are_not_equal()
    test_tensors_on_accelerator_devices_are_not_equal()
