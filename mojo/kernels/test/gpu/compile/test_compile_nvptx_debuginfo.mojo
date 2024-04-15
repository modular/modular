# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -debug-level full -O0 %s | FileCheck %s

from gpu import ThreadIdx
from gpu.host._compile import _compile_code


fn outer[y: Int]():
    @parameter
    fn param[x: Int](y: SIMD[DType.float32, y], /):
        pass

    print(
        _compile_code[
            fn (SIMD[DType.float32, y]) capturing -> None, param[y]
        ]().asm
    )


fn main():
    # CHECK: .debug_
    outer[2]()
