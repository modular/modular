# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from compile import *
from testing import *


def main():
    @parameter
    fn my_add_function[
        type: DType, size: Int
    ](x: SIMD[type, size], y: SIMD[type, size]) -> SIMD[type, size]:
        return x + y

    alias func = my_add_function[DType.float32, 4]
    var asm: String = compile_code[
        __type_of(func), func, emission_kind="llvm"
    ]()

    assert_true("fadd" in asm)

    fn always_fails():
        constrained[False, "always fails"]()

    alias compiled = compile_info[
        __type_of(always_fails), always_fails, emission_kind="llvm"
    ]()
    alias is_error = compiled.is_error
    alias error_msg = compiled.error_msg
    assert_true(is_error)
    assert_true(error_msg)
