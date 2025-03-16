# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from compile import _internal_compile_code, compile_info
from testing import *


def test_compile_llvm():
    @parameter
    fn my_add_function[
        type: DType, size: Int
    ](x: SIMD[type, size], y: SIMD[type, size]) -> SIMD[type, size]:
        return x + y

    alias func = my_add_function[DType.float32, 4]
    var asm = _internal_compile_code[func, emission_kind="llvm"]()

    assert_true("fadd" in asm)


def main():
    test_compile_llvm()
