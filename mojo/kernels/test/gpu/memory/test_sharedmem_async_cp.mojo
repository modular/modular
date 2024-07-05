# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# Hangs with debug mode Issue #24921
# RUN: %mojo-no-debug %s | FileCheck %s
import builtin
import gpu.host as gpu_host
from gpu import AddressSpace, ThreadIdx, memory, sync
import time
from memory import stack_allocation


fn copy_via_shared(
    src: DTypePointer[DType.float32],
    dst: DTypePointer[DType.float32],
):
    var thId = ThreadIdx.x()
    var mem_buff: UnsafePointer[
        Float32, AddressSpace.SHARED
    ] = stack_allocation[16, Float32, address_space = AddressSpace.SHARED]()
    var src_global: UnsafePointer[
        Float32, AddressSpace.GLOBAL
    ] = src._as_scalar_pointer().bitcast[address_space = AddressSpace.GLOBAL]()

    memory.async_copy[4](
        src_global.offset(thId),
        mem_buff.offset(thId),
    )

    var m_barrier = stack_allocation[
        1, DType.int32, address_space = AddressSpace.SHARED
    ]()
    sync.mbarrier_init(m_barrier, 16)
    sync.mbarrier(m_barrier)
    var state = sync.mbarrier_arrive(m_barrier)
    var not_wait = False
    while not not_wait:
        time.sleep(100 * 1e-6)
        not_wait = sync.mbarrier_test_wait(m_barrier, state)

    dst[thId] = mem_buff[thId]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared() raises:
    print("== run_copy_via_shared")
    var in_data = gpu_host.memory._malloc_managed[DType.float32](16)
    var out_data = gpu_host.memory._malloc_managed[DType.float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    var copy_via_shared_gpu = gpu_host.Function[copy_via_shared]()

    var stream = gpu_host.Stream()
    copy_via_shared_gpu(
        in_data, out_data, grid_dim=(1,), block_dim=(16,), stream=stream
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
        print(Scalar.load(out_data, i))

    pass


fn main():
    try:
        with gpu_host.Context() as ctx:
            run_copy_via_shared()
    except e:
        print("CUDA_ERROR:", e)
