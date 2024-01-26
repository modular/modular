# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from kernel_utils import *
from testing import *


def main():
    @parameter
    fn my_add_function[
        type: DType, size: Int
    ](x: SIMD[type, size], y: SIMD[type, size]) -> SIMD[type, size]:
        return x + y

    let asm: String = compile_code[
        fn (
            SIMD[DType.float32, 4], SIMD[DType.float32, 4]
        ) capturing -> SIMD[DType.float32, 4], my_add_function[
            DType.float32, 4
        ], emission_kind="llvm"
    ]()

    assert_true(asm.__contains__("fadd"))
