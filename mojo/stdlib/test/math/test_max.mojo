# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from testing import assert_equal


def test_max():
    expected_result = SIMD[DType.bool, 4](True, True, False, True)
    actual_result = max(
        SIMD[DType.bool, 4](
            True,
            True,
            False,
            False,
        ),
        SIMD[DType.bool, 4](False, True, False, True),
    )

    assert_equal(actual_result, expected_result)


def test_max_scalar():
    assert_equal(max(Bool(True), Bool(False)), Bool(True))
    assert_equal(max(Bool(False), Bool(True)), Bool(True))
    assert_equal(max(Bool(False), Bool(False)), Bool(False))
    assert_equal(max(Bool(True), Bool(True)), Bool(True))


def main():
    test_max()
