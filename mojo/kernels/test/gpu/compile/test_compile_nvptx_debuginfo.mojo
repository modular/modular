# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -debug-level full -O0 %s | FileCheck %s

from gpu import thread_idx
from gpu.host._compile import _compile_code_asm


fn outer[y: Int]():
    @parameter
    fn param[x: Int](y: SIMD[DType.float32, y], /):
        pass

    print(_compile_code_asm[param[y]]())


fn main():
    # CHECK: .debug_
    outer[2]()
