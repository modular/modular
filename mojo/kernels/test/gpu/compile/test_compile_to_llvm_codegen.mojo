# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from gpu import ThreadIdx
from gpu.host._compile import _compile_code


# CHECK-LABEL: tese_cse_thread_id
fn tese_cse_thread_id():
    print("== tese_cse_thread_id")

    fn kernel() -> Int32:
        return ThreadIdx.x() + ThreadIdx.x() + ThreadIdx.x()

    # CHECK-COUNT-1: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    print(_compile_code[__type_of(kernel), kernel, emission_kind="llvm"]().asm)


fn main():
    tese_cse_thread_id()
