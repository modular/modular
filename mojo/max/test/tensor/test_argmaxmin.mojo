# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from max.tensor import Tensor, TensorShape

from testing import assert_equal


def test_argmaxmin():
    var tensor = Tensor[DType.float32](TensorShape(2, 2), 1, 2, 4, 3)
    var argmin = tensor.argmin(axis=-1)
    var argmax = tensor.argmax(axis=-1)

    assert_equal(argmin[0], 0)
    assert_equal(argmin[1], 1)

    assert_equal(argmax[0], 1)
    assert_equal(argmax[1], 0)

    assert_equal(argmin.rank(), 1)
    assert_equal(argmax.rank(), 1)


def main():
    test_argmaxmin()
