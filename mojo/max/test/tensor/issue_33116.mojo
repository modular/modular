# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from max.tensor import Tensor, TensorShape

from nn.argmaxmin import argmax
from testing import assert_equal


fn argmax_tensor(
    input: Tensor[DType.float32],
) raises -> Tensor[DType.float32]:
    var output = Tensor[DType.float32](TensorShape(2, 1))

    argmax(input._to_ndbuffer[2](), -1, output._to_ndbuffer[2]())

    return output.reshape(2)


def main():
    var tensor = Tensor[DType.float32](TensorShape(2, 2), 1, 2, 4, 3)
    var res = argmax_tensor(tensor)

    assert_equal(res[0], 1)
    assert_equal(res[1], 0)
