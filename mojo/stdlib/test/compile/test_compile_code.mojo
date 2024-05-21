# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from compile import *
from testing import *


def test_compile_llvm():
    @parameter
    fn my_add_function[
        type: DType, size: Int
    ](x: SIMD[type, size], y: SIMD[type, size]) -> SIMD[type, size]:
        return x + y

    alias func = my_add_function[DType.float32, 4]
    var asm: String = compile_code[func, emission_kind="llvm"]()

    assert_true("fadd" in asm)


def test_compile_failure():
    fn always_fails():
        constrained[False, "always fails"]()

    alias compiled = compile_info[
        always_fails, is_failable=True, emission_kind="llvm"
    ]()
    alias is_error = compiled.is_error
    alias error_msg = compiled.error_msg
    assert_true(is_error)
    assert_true(error_msg)


def main():
    test_compile_llvm()
    test_compile_failure()
