# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import DimList
from internal_utils import TestTensor, assert_with_measure


def test_assert_with_custom_measure():
    var t0 = TestTensor[DType.float32, 1](DimList(100), List[Float32](1))
    var t1 = TestTensor[DType.float32, 1](DimList(100), List[Float32](1))

    @parameter
    fn always_true[
        type: DType
    ](
        lhs: UnsafePointer[Scalar[type]],
        rhs: UnsafePointer[Scalar[type]],
        n: Int,
    ) -> Bool:
        return True

    assert_with_measure[always_true](t0, t1)

    _ = t0^
    _ = t1^


def main():
    test_assert_with_custom_measure()
