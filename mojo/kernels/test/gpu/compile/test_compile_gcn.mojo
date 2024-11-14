# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from pathlib import Path
from sys._assembly import inlined_assembly

from gpu import ThreadIdx, BlockDim, GridDim, barrier, lane_id
from gpu.shuffle import shuffle_down, shuffle_up, shuffle_xor
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_nvptx_target
from memory import UnsafePointer

alias MI300X_TARGET = _get_nvptx_target["mi300x"]()


fn kernel(x: UnsafePointer[Int]):
    x[0] = ThreadIdx.x()


fn kernel_laneid(x: UnsafePointer[Int]):
    x[0] = lane_id()


fn kernel_shuffle_down(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(0xFFFFFFFF_FFFFFFFF)
    var offset = x[0]
    x[0] = shuffle_down(mask, val, offset)


fn kernel_shuffle_up(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(0xFFFFFFFF_FFFFFFFF)
    var offset = x[0]
    x[0] = shuffle_up(mask, val, offset)


fn kernel_shuffle_xor(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(0xFFFFFFFF_FFFFFFFF)
    var offset = x[0]
    x[0] = shuffle_xor(mask, val, offset)


fn parametric[f: fn (UnsafePointer[Int]) -> None](ptr: UnsafePointer[Int]):
    f(ptr)


# from https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html#naive-load-and-store
fn load_store(
    n: Int, input: UnsafePointer[Float32], output: UnsafePointer[Float32]
):
    var tid = ThreadIdx.x() + BlockDim.x() * GridDim.x()
    output[tid] = input[tid]


# CHECK-LABEL: test_shuffle_compile
def test_shuffle_compile():
    print("== test_shuffle_compile")
    # CHECK: %3 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %4 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %3)
    # CHECK: %5 = add i32 %4, %2
    # CHECK: %6 = icmp ugt i32 %5, 63
    # CHECK: %7 = select i1 %6, i32 0, i32 %2
    # CHECK: %8 = add i32 %7, %4
    # CHECK: %9 = shl i32 %8, 2
    # CHECK: %10 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %9, i32 %2)
    print(
        _compile_code_asm[
            kernel_shuffle_down, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    # CHECK: %3 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %4 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %3)
    # CHECK: %5 = sub i32 %4, %2
    # CHECK: %6 = and i32 %4, -64
    # CHECK: %7 = icmp slt i32 %5, %6
    # CHECK: %8 = select i1 %7, i32 %4, i32 %5
    # CHECK: %9 = shl i32 %8, 2
    # CHECK: %10 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %9, i32 %2)

    print(
        _compile_code_asm[
            kernel_shuffle_up, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    # CHECK: %3 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %4 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %3)
    # CHECK: %5 = xor i32 %4, %2
    # CHECK: %6 = and i32 %4, -64
    # CHECK: %7 = add i32 %6, 64
    # CHECK: %8 = icmp ult i32 %5, %7
    # CHECK: %9 = select i1 %8, i32 %5, i32 %4
    # CHECK: %10 = shl i32 %9, 2
    # CHECK: %11 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %10, i32 %2)

    print(
        _compile_code_asm[
            kernel_shuffle_xor, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_laneid_compile
def test_laneid_compile():
    print("== test_laneid_compile")

    # CHECK: tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %2)
    print(
        _compile_code_asm[
            kernel_laneid, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


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

    # CHECK-LABEL: @test_compile_gcn_load_store_
    # CHECK: llvm.amdgcn.workitem.id.x
    # CHECK: %[[VAR:.*]] = tail call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
    # CHECK: getelementptr inbounds i8, ptr addrspace(4) %[[VAR]], i64 12
    print(
        _compile_code_asm[
            load_store, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    print(_compile_code_asm[load_store, target=MI300X_TARGET]())


def main():
    test_shuffle_compile()
    test_laneid_compile()
    test_barrier_compile()
    test_threadid_compile()
