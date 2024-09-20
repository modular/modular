# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code_asm
from testing import assert_true


def test_abs():
    fn do_abs[
        type: DType, *, width: Int = 1
    ](val: SIMD[type, width]) -> __type_of(val):
        return abs(val)

    assert_true("abs.f16 " in _compile_code_asm[do_abs[DType.float16]]())
    assert_true("abs.bf16 " in _compile_code_asm[do_abs[DType.bfloat16]]())

    assert_true(
        "abs.f16x2 " in _compile_code_asm[do_abs[DType.float16, width=4]]()
    )
    assert_true(
        "abs.bf16x2 " in _compile_code_asm[do_abs[DType.bfloat16, width=4]]()
    )


def main():
    test_abs()
