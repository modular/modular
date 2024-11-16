# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s


from gpu import ThreadIdx
from gpu.host._compile import _compile_code_asm
from gpu.memory import AddressSpace, external_memory
from memory import UnsafePointer


# CHECK-LABEL: test_array_offset
fn test_array_offset():
    print("== test_array_offset")

    fn kernel(
        p: UnsafePointer[Float32, AddressSpace.SHARED], idx: Int
    ) -> Float32:
        return p[idx]

    # CHECK: getelementptr inbounds float, ptr addrspace(3) %0, i32 %3
    print(_compile_code_asm[kernel, emission_kind="llvm"]())


# CHECK-LABEL: tese_cse_thread_id
fn tese_cse_thread_id():
    print("== tese_cse_thread_id")

    fn kernel() -> Int32:
        return ThreadIdx.x() + ThreadIdx.x() + ThreadIdx.x()

    # CHECK-COUNT-1: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    print(_compile_code_asm[kernel, emission_kind="llvm"]())


# CHECK-LABEL: test_dynamic_shared_mem
fn test_dynamic_shared_mem():
    print("== test_dynamic_shared_mem")

    # CHECK: @extern_ptr_syml = external dso_local addrspace(3) global [0 x float], align 4
    # CHECK: @extern_ptr_syml_0 = external dso_local addrspace(3) global [0 x float], align 4
    fn kernel() -> Float32:
        # CHECK: %1 = load float, ptr addrspace(3) @extern_ptr_syml, align 4
        # CHECK: %2 = load float, ptr addrspace(3) getelementptr inbounds (float, ptr addrspace(3) @extern_ptr_syml_0, i32 1), align 4
        # CHECK: fadd contract float %1, %2
        var dynamic_sram_ptr_1 = external_memory[
            Float32, address_space = AddressSpace.SHARED, alignment=4
        ]()
        var dynamic_sram_ptr_2 = external_memory[
            Float32, address_space = AddressSpace.SHARED, alignment=4
        ]()
        return dynamic_sram_ptr_1[0] + dynamic_sram_ptr_2[1]

    print(_compile_code_asm[kernel, emission_kind="llvm"]())


fn main():
    test_array_offset()
    tese_cse_thread_id()
    test_dynamic_shared_mem()
