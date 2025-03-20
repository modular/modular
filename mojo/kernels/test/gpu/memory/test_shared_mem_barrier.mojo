# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s | FileCheck %s


from gpu.host._compile import _get_gpu_target, _compile_code_asm
from gpu.id import thread_idx
from gpu.sync import mbarrier_init
from gpu.memory import AddressSpace
from memory import UnsafePointer, stack_allocation
from layout.tma_async import SharedMemBarrier


# CHECK-LABEL: test_shared_mem_barrier
# CHECK-NOT: ld.local
# CHECK-NOT: st.local
fn test_shared_mem_barrier():
    mbar = stack_allocation[
        10,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()

    @parameter
    for i in range(10):
        mbar[i].init()


def main():
    print("== test_shared_mem_barrier")
    alias kernel = test_shared_mem_barrier
    asm = _compile_code_asm[kernel, target = _get_gpu_target["sm_90a"]()]()
    print(asm)
