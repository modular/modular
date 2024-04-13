# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device

# RUN: %mojo-no-debug %s | FileCheck %s

import builtin
import gpu.host as gpu_host
from gpu import ThreadIdx, memory
from gpu.memory import AddressSpace as _AddressSpace
from memory import stack_allocation


fn copy_via_shared(
    src: DTypePointer[DType.float32],
    dst: DTypePointer[DType.float32],
):
    var thread_id = ThreadIdx.x()
    var mem_buff: Pointer[Float32, _AddressSpace.SHARED] = stack_allocation[
        16, Float32, address_space = _AddressSpace.SHARED
    ]()
    var src_global: Pointer[
        Float32, _AddressSpace.GLOBAL
    ] = src._as_scalar_pointer().bitcast[address_space = _AddressSpace.GLOBAL]()

    memory.async_copy[4](
        src_global.offset(thread_id),
        mem_buff.offset(thread_id),
    )

    memory.async_copy_commit_group()
    memory.async_copy_wait_group(0)

    dst[thread_id] = mem_buff[thread_id]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared() raises:
    print("== run_copy_via_shared")
    var in_data = gpu_host.memory._malloc_managed[DType.float32](16)
    var out_data = gpu_host.memory._malloc_managed[DType.float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    var copy_via_shared_gpu = gpu_host.Function[
        __type_of(copy_via_shared), copy_via_shared
    ]()

    var stream = gpu_host.Stream()
    copy_via_shared_gpu(
        in_data, out_data, grid_dim=(1,), block_dim=(16), stream=stream
    )

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
