# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s


from gpu import thread_idx
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.memory import AddressSpace, external_memory
from memory import UnsafePointer


# CHECK-LABEL: test_array_offset
fn test_array_offset():
    print("== test_array_offset")

    fn kernel(
        output: UnsafePointer[Float32],
        p: UnsafePointer[Float32, address_space = AddressSpace.SHARED],
        idx: Int,
    ):
        output[] = p[idx]

    # CHECK: getelementptr inbounds float, ptr addrspace(3) %1, i32 %4
    print(_compile_code_asm[kernel, emission_kind="llvm"]())


# CHECK-LABEL: test_case_thread_id_nvidia
fn test_case_thread_id_nvidia():
    print("== test_case_thread_id_nvidia")

    fn kernel(output: UnsafePointer[Int32]):
        output[] = thread_idx.x + thread_idx.x + thread_idx.x

    # CHECK-COUNT-1: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    print(
        _compile_code_asm[
            kernel, emission_kind="llvm", target = _get_gpu_target["sm_80"]()
        ]()
    )


# CHECK-LABEL: test_case_thread_id_mi300x
fn test_case_thread_id_mi300x():
    print("== test_case_thread_id_mi300x")

    fn kernel(output: UnsafePointer[Int32]):
        output[] = thread_idx.x + thread_idx.x + thread_idx.x

    # CHECK-COUNT-1: call i32 @llvm.amdgcn.workitem.id.x()
    print(
        _compile_code_asm[
            kernel, emission_kind="llvm", target = _get_gpu_target["mi300x"]()
        ]()
    )


# CHECK-LABEL: test_dynamic_shared_mem
fn test_dynamic_shared_mem():
    print("== test_dynamic_shared_mem")

    # CHECK: @extern_ptr_syml = external dso_local addrspace(3) global [0 x float], align 4
    # CHECK: @extern_ptr_syml_0 = external dso_local addrspace(3) global [0 x float], align 4
    fn kernel(output: UnsafePointer[Float32]):
        # CHECK: %2 = load float, ptr addrspace(3) @extern_ptr_syml, align 4
        # CHECK: %3 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @extern_ptr_syml_0, i32 4), align 4
        # CHECK: fadd contract float %2, %3
        var dynamic_sram_ptr_1 = external_memory[
            Float32, address_space = AddressSpace.SHARED, alignment=4
        ]()
        var dynamic_sram_ptr_2 = external_memory[
            Float32, address_space = AddressSpace.SHARED, alignment=4
        ]()
        output[] = dynamic_sram_ptr_1[0] + dynamic_sram_ptr_2[1]

    print(_compile_code_asm[kernel, emission_kind="llvm"]())


fn main():
    test_array_offset()
    test_case_thread_id_nvidia()
    test_case_thread_id_mi300x()
    test_dynamic_shared_mem()
