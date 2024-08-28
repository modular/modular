# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: Verify implict conversion between dtypes does not happen

# RUN: mkdir -p %t
# RUN: rm -rf %t/test-build-fail
# COM: Verify that this code does not compile successfully.
# RUN: not %mojo-build %s -o %t/test-build-fail

from max.driver import Tensor
from max.tensor import TensorShape


def main():
    var t: Tensor[DType.bool, 2]
    t = Tensor[DType.int64, 2](TensorShape(1, 2))
    print(t)
