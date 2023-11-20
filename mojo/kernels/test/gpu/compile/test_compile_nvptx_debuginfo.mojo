# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -debug-level full -O0 --parsing-stdlib %s | FileCheck %s

from gpu import ThreadIdx
from gpu.host._compile import _compile_nvptx


fn outer[y: Int]():
    @parameter
    fn param[x: Int](y: SIMD[DType.float32, y], /):
        pass

    print(
        _compile_nvptx[
            fn (SIMD[DType.float32, y]) capturing -> None, param[y]
        ]().asm
    )


fn main():
    # CHECK: .debug_pubtypes
    outer[2]()
