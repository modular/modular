# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s

from pathlib import Path
from sys._assembly import inlined_assembly

from gpu import (
    barrier,
    block_dim,
    grid_dim,
    lane_id,
    thread_idx,
    schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
)
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.shuffle import shuffle_down, shuffle_idx, shuffle_up, shuffle_xor
from memory import UnsafePointer

alias MI300X_TARGET = _get_gpu_target["mi300x"]()
alias FULL_MASK_AMD = 2**WARP_SIZE - 1


fn kernel(x: UnsafePointer[Int]):
    x[0] = thread_idx.x


fn kernel_laneid(x: UnsafePointer[Int]):
    x[0] = lane_id()


fn kernel_shuffle_down(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_down(mask, val, offset)


fn kernel_shuffle_up(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_up(mask, val, offset)


fn kernel_shuffle_xor(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_xor(mask, val, offset)


fn kernel_shuffle_idx(x: UnsafePointer[UInt32]):
    var val = x[0]
    var mask = UInt(FULL_MASK_AMD)
    var offset = x[0]
    x[0] = shuffle_idx(mask, val, offset)


fn parametric[f: fn (UnsafePointer[Int]) -> None](ptr: UnsafePointer[Int]):
    f(ptr)


# from https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html#naive-load-and-store
fn load_store(
    n: Int, input: UnsafePointer[Float32], output: UnsafePointer[Float32]
):
    var tid = thread_idx.x + block_dim.x * grid_dim.x
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

    # CHECK: %3 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
    # CHECK: %4 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %3)
    # CHECK: %5 = and i32 %4, 1073741760
    # CHECK: %6 = or i32 %5, %2
    # CHECK: %7 = shl i32 %6, 2
    # CHECK: %8 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %7, i32 %2)

    print(
        _compile_code_asm[
            kernel_shuffle_idx, target=MI300X_TARGET, emission_kind="llvm-opt"
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
    # CHECK: getelementptr inbounds nuw i8, ptr addrspace(4) %[[VAR]], i64 12
    print(
        _compile_code_asm[
            load_store, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )

    _ = _compile_code_asm[load_store, target=MI300X_TARGET]()


# CHECK-LABEL: test_schedule_barrier_compile
def test_schedule_barrier_compile():
    print("== test_schedule_barrier_compile")

    fn schedule_kernel():
        schedule_barrier(AMDScheduleBarrierMask.NONE)
        schedule_barrier(AMDScheduleBarrierMask.ALL_ALU)
        schedule_barrier(AMDScheduleBarrierMask.VALU)
        schedule_barrier(AMDScheduleBarrierMask.SALU)
        schedule_barrier(AMDScheduleBarrierMask.MFMA)
        schedule_barrier(AMDScheduleBarrierMask.ALL_VMEM)
        schedule_barrier(AMDScheduleBarrierMask.VMEM_READ)
        schedule_barrier(AMDScheduleBarrierMask.VMEM_WRITE)
        schedule_barrier(AMDScheduleBarrierMask.ALL_DS)
        schedule_barrier(AMDScheduleBarrierMask.DS_READ)
        schedule_barrier(AMDScheduleBarrierMask.DS_WRITE)
        schedule_barrier(AMDScheduleBarrierMask.TRANS)

    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 0)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 1)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 2)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 4)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 8)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 16)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 32)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 64)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 128)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 256)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 512)
    # CHECK: tail call void @llvm.amdgcn.sched.barrier(i32 1024)
    print(
        _compile_code_asm[
            schedule_kernel, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


# CHECK-LABEL: test_schedule_group_barrier_compile
def test_schedule_group_barrier_compile():
    print("== test_schedule_group_barrier_compile")

    fn schedule_kernel():
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 10, 0)
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 10, 1)
        schedule_group_barrier(AMDScheduleBarrierMask.MFMA, 11, 10)
        schedule_group_barrier(AMDScheduleBarrierMask.TRANS, 11, 10)

    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 10, i32 0)
    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 10, i32 1)
    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 11, i32 10)
    # CHECK: tail call void @llvm.amdgcn.sched.group.barrier(i32 1024, i32 11, i32 10)
    print(
        _compile_code_asm[
            schedule_kernel, target=MI300X_TARGET, emission_kind="llvm-opt"
        ]()
    )


def main():
    test_shuffle_compile()
    test_laneid_compile()
    test_barrier_compile()
    test_threadid_compile()
    test_schedule_barrier_compile()
    test_schedule_group_barrier_compile()
