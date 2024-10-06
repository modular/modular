# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path
from sys._assembly import inlined_assembly

from gpu import ThreadIdx
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_nvptx_target
from memory import UnsafePointer

alias MI300X_TARGET = _get_nvptx_target["mi300x"]()


fn kernel(x: Int) -> Int:
    return ThreadIdx.x()


fn parametric[f: fn (Int) -> Int]() -> Int:
    return f(42)


# CHECK-LABEL: test_compile_code
def test_compile_code():
    print("== test_compile_code")

    # CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
    # CHECK: s_waitcnt lgkmcnt
    print(_compile_code_asm[kernel, target=MI300X_TARGET]())
    # CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
    # CHECK: s_waitcnt lgkmcnt
    print(_compile_code_asm[parametric[kernel], target=MI300X_TARGET]())
    # CHECK: ; ModuleID =
    # CHECK: llvm.amdgcn.workitem.id.x
    print(
        _compile_code_asm[
            parametric[kernel],
            target=MI300X_TARGET,
            emission_kind="llvm",
        ]()
    )


def main():
    test_compile_code()
