# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s


from gpu.host._compile import _compile_code_asm
from gpu.intrinsics import *


fn kernel[
    type: DType, memory: Bool = True
](ptr: UnsafePointer[Scalar[type]], val: Scalar[type]) -> Scalar[type]:
    store_release[memory](ptr, val)
    return load_acquire[memory](ptr)


# CHECK-LABEL: test_compile_code
def test_compile_code():
    print("== test_compile_code")

    # CHECK: st.release.sys.global.u32 [%r1], %rd1;
    # CHECK: ld.acquire.sys.global.u32 %r2, [%rd2];
    print(_compile_code_asm[kernel[DType.int32]]())

    # CHECK: st.release.sys.global.u16 [%rs1], %rd1;
    # CHECK: ld.acquire.sys.global.u16 %rs2, [%rd2];
    print(_compile_code_asm[kernel[DType.bfloat16]]())

    # CHECK: st.release.sys.global.u32 [%r1], %rd1;
    # CHECK: ld.acquire.sys.global.u32 %r2, [%rd2];
    print(_compile_code_asm[kernel[DType.int32, memory=False]]())

    # CHECK: st.release.sys.global.u16 [%rs1], %rd1;
    # CHECK: ld.acquire.sys.global.u16 %rs2, [%rd2];
    print(_compile_code_asm[kernel[DType.bfloat16, memory=False]]())

    # CHECK: tail call void asm sideeffect "st.release.sys.global.u16 [$0], $1;", "h,l,~{memory}"(ptr %0, bfloat %1)
    # CHECK: tail call bfloat asm sideeffect "ld.acquire.sys.global.u16 $0, [$1];", "=h,l,~{memory}"(ptr %0)
    print(
        _compile_code_asm[
            kernel[DType.bfloat16, memory=True], emission_kind="llvm-opt"
        ]()
    )

    # CHECK: tail call void asm sideeffect "st.release.sys.global.u16 [$0], $1;", "h,l"(ptr %0, bfloat %1)
    # CHECK: tail call bfloat asm sideeffect "ld.acquire.sys.global.u16 $0, [$1];", "=h,l"(ptr %0)
    print(
        _compile_code_asm[
            kernel[DType.bfloat16, memory=False], emission_kind="llvm-opt"
        ]()
    )


def main():
    test_compile_code()
