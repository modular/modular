# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from gpu import ThreadIdx
from gpu.host._compile import _compile_code
from gpu.memory import dynamic_shared_memory


# CHECK-LABEL: tese_cse_thread_id
fn tese_cse_thread_id():
    print("== tese_cse_thread_id")

    fn kernel() -> Int32:
        return ThreadIdx.x() + ThreadIdx.x() + ThreadIdx.x()

    # CHECK-COUNT-1: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    print(_compile_code[kernel, emission_kind="llvm"]().asm)


# CHECK-LABEL: test_dynamic_shared_mem
fn test_dynamic_shared_mem():
    print("== test_dynamic_shared_mem")

    # CHECK: @extern_ptr_syml = external dso_local addrspace(3) global float, align 4
    # CHECK: @extern_ptr_syml_0 = external dso_local addrspace(3) global float, align 4
    fn kernel() -> Float32:
        # CHECK: %1 = load float, ptr addrspacecast (ptr addrspace(3) @extern_ptr_syml to ptr), align 4
        # CHECK: %2 = load float, ptr addrspacecast (ptr addrspace(3) getelementptr (float, ptr addrspace(3) @extern_ptr_syml_0, i32 1) to ptr), align 4
        # CHECK: fadd contract float %1, %2
        var dynamic_sram_ptr_1 = dynamic_shared_memory[Float32, alignment=4]()
        var dynamic_sram_ptr_2 = dynamic_shared_memory[Float32, alignment=4]()
        return dynamic_sram_ptr_1.offset(0)[] + dynamic_sram_ptr_2.offset(1)[]

    print(_compile_code[kernel, emission_kind="llvm"]().asm)


fn main():
    tese_cse_thread_id()
    test_dynamic_shared_mem()
