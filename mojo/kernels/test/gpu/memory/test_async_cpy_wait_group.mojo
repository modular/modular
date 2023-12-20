# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device

# RUN: %mojo %s | FileCheck %s

import gpu.host as gpu_host
from gpu import ThreadIdx
from gpu.memory import AddressSpace as _AddressSpace
from gpu import memory
import builtin
from memory import stack_allocation


fn copy_via_shared(
    src: DTypePointer[DType.float32],
    dst: DTypePointer[DType.float32],
):
    let thread_id = ThreadIdx.x()
    let mem_buff: Pointer[Float32, _AddressSpace.SHARED] = stack_allocation[
        16, Float32, address_space = _AddressSpace.SHARED
    ]()
    let src_global: Pointer[
        Float32, _AddressSpace.GLOBAL
    ] = src._as_scalar_pointer().address_space_cast[_AddressSpace.GLOBAL]()

    memory.async_copy[4](
        src_global.offset(thread_id),
        mem_buff.offset(thread_id),
    )

    memory.async_copy_commit_group()
    memory.async_copy_wait_group(0)

    dst.store(thread_id, mem_buff.load(thread_id))


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared() raises:
    print("== run_copy_via_shared")
    let in_data = gpu_host.memory._malloc_managed[DType.float32](16)
    let out_data = gpu_host.memory._malloc_managed[DType.float32](16)

    for i in range(16):
        in_data.store(i, i + 1)
        out_data.store(i, 0)

    let copy_via_shared_gpu = gpu_host.Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
        ) -> None, copy_via_shared
    ]()

    let stream = gpu_host.Stream()
    copy_via_shared_gpu(stream, (1,), (16), in_data, out_data)

    # CHECK: 1.0
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: 8.0
    # CHECK: 9.0
    # CHECK: 10.0
    # CHECK: 11.0
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    # CHECK: 16.0
    for i in range(16):
        print(out_data[i])


fn main():
    try:
        with gpu_host.Context() as ctx:
            run_copy_via_shared()
    except e:
        print("CUDA_ERROR:", e)
