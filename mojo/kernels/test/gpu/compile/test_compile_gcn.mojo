# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path
from sys._assembly import inlined_assembly

from gpu import ThreadIdx, BlockDim, GridDim, barrier
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_nvptx_target
from memory import UnsafePointer

alias MI300X_TARGET = _get_nvptx_target["mi300x"]()


fn kernel(x: Int) -> Int:
    return ThreadIdx.x()


fn parametric[f: fn (Int) -> Int]() -> Int:
    return f(42)


# from https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html#naive-load-and-store
fn load_store(
    n: Int, input: UnsafePointer[Float32], output: UnsafePointer[Float32]
):
    var tid = ThreadIdx.x() + BlockDim.x() * GridDim.x()
    output[tid] = input[tid]


# CHECK-LABEL: test_barrier_compile
def test_barrier_compile():
    print("== test_barrier_compile")

    # CHECK: fence syncscope("workgroup") release
    # CHECK: tail call void @llvm.amdgcn.s.barrier()
    # CHECK: syncscope("workgroup") acquire
    print(
        _compile_code_asm[
            barrier, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_threadid_compile
def test_threadid_compile():
    print("== test_threadid_compile")

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

    # CHECK-LABEL: test_compile_gcn::load_store
    # CHECK: llvm.amdgcn.workitem.id.x
    # CHECK: %[[VAR:.*]] = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
    # CHECK: getelementptr i8, ptr addrspace(4) %[[VAR]], i64 12
    print(
        _compile_code_asm[
            load_store, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    print(_compile_code_asm[load_store, target=MI300X_TARGET]())


def main():
    test_barrier_compile()
    test_threadid_compile()
