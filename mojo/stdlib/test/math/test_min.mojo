# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from testing import assert_equal


def test_min():
    expected_result = SIMD[DType.bool, 4](False, True, False, False)
    actual_result = min(
        SIMD[DType.bool, 4](
            True,
            True,
            False,
            False,
        ),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(actual_result, expected_result)


def test_min_scalar():
    assert_equal(min(Bool(True), Bool(False)), Bool(False))
    assert_equal(min(Bool(False), Bool(True)), Bool(False))
    assert_equal(min(Bool(False), Bool(False)), Bool(False))
    assert_equal(min(Bool(True), Bool(True)), Bool(True))


def main():
    test_min()
    test_min_scalar()
