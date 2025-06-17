# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import exp
from pathlib import Path
from sys._assembly import inlined_assembly

from gpu import (
    AMDScheduleBarrierMask,
    barrier,
    block_dim,
    grid_dim,
    lane_id,
    schedule_barrier,
    schedule_group_barrier,
    thread_idx,
)
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext
from gpu.host.compile import _compile_code_asm, get_gpu_target
from gpu.intrinsics import load_acquire, store_release
from gpu.warp import shuffle_down, shuffle_idx, shuffle_up, shuffle_xor

alias MI300X_TARGET = get_gpu_target["mi300x"]()
alias FULL_MASK_AMD = 2**WARP_SIZE - 1


fn kernel(x: UnsafePointer[Int]):
    x[0] = thread_idx.x


fn kernel_laneid(x: UnsafePointer[Int]):
    x[0] = lane_id()


fn kernel_exp[type: DType](x: UnsafePointer[Scalar[type]]):
    x[0] = exp(x[0])


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


fn kernel_cast[
    type: DType, target: DType
](x: UnsafePointer[Scalar[type]], y: UnsafePointer[Scalar[target]]):
    y[0] = x[0].cast[target]()


fn kernel_atomic[
    type: DType, memory: Bool = True
](
    output: UnsafePointer[Scalar[type]],
    ptr: UnsafePointer[Scalar[type]],
    val: Scalar[type],
):
    output[] = ptr[]
    store_release[memory=memory](ptr, val)
    output[] = load_acquire[memory=memory](ptr)


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


# CHECK-LABEL: test_cast_fp32_bf16_compile
def test_cast_fp32_bf16_compile():
    print("== test_cast_fp32_bf16_compile")

    # CHECK: tail call i64 asm "v_cmp_u_f32 $0, $1, $1"
    # CHECK: tail call i32 asm "v_bfe_u32 $0, $1, 16, 1"
    # CHECK: tail call i32 asm "v_add3_u32 $0, $1, $2, $3"
    # CHECK: tail call i32 asm "v_cndmask_b32 $0, $1, $2, $3"
    print(
        _compile_code_asm[
            kernel_cast[DType.float32, DType.bfloat16],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
        ]()
    )


# CHECK-LABEL: test_exp_f32_compile
def test_exp_f32_compile():
    print("== test_exp_f32_compile")

    # CHECK: tail call float @llvm.amdgcn.exp2.f32(float %3)
    print(
        _compile_code_asm[
            kernel_exp[DType.float32],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
        ]()
    )


# CHECK-LABEL: test_exp_f16_compile
def test_exp_f16_compile():
    print("== test_exp_f16_compile")

    # CHECK: tail call half @llvm.amdgcn.exp2.f16(half %3)
    print(
        _compile_code_asm[
            kernel_exp[DType.float16],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
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


def test_atomic_compile():
    print("== test_atomic_compile")

    # Memory model reference: https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942.

    # store atomic release system
    # CHECK: buffer_wbl2 sc0 sc1
    # CHECK: s_waitcnt vmcnt(0)
    # CHECK: global_store_dword {{.*}} sc0 sc1

    # load atomic acquire system
    # CHECK: global_load_dword {{.*}} sc0 sc1
    # CHECK: s_waitcnt vmcnt(0)
    # CHECK: buffer_inv sc0 sc1

    print(_compile_code_asm[kernel_atomic[DType.int32], target=MI300X_TARGET]())

    # CHECK: store atomic {{.*}} release
    # CHECK: load atomic {{.*}} acquire
    print(
        _compile_code_asm[
            kernel_atomic[DType.int32],
            target=MI300X_TARGET,
            emission_kind="llvm-opt",
        ]()
    )


def main():
    test_shuffle_compile()
    test_cast_fp32_bf16_compile()
    test_exp_f32_compile()
    test_exp_f16_compile()
    test_laneid_compile()
    test_barrier_compile()
    test_threadid_compile()
    test_schedule_barrier_compile()
    test_schedule_group_barrier_compile()
    test_atomic_compile()
