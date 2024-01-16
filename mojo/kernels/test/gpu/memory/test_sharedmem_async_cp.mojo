# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# Hangs with debug mode Issue #24921
# RUN: %mojo %s | FileCheck %s
import gpu.host as gpu_host
from gpu import ThreadIdx, AddressSpace
from gpu import memory, sync, time
import builtin
from memory import stack_allocation


fn copy_via_shared(
    src: DTypePointer[DType.float32],
    dst: DTypePointer[DType.float32],
):
    let thId = ThreadIdx.x()
    let mem_buff: Pointer[Float32, AddressSpace.SHARED] = stack_allocation[
        16, Float32, address_space = AddressSpace.SHARED
    ]()
    let src_global: Pointer[
        Float32, AddressSpace.GLOBAL
    ] = src._as_scalar_pointer().address_space_cast[AddressSpace.GLOBAL]()

    memory.async_copy[4](
        src_global.offset(thId),
        mem_buff.offset(thId),
    )

    let m_barrier = stack_allocation[
        1, DType.int32, address_space = AddressSpace.SHARED
    ]()
    sync.mbarrier_init(m_barrier, 16)
    sync.mbarrier(m_barrier)
    let state = sync.mbarrier_arrive(m_barrier)
    var not_wait = False
    while not not_wait:
        time.sleep(100 * 1e-6)
        not_wait = sync.mbarrier_test_wait(m_barrier, state)

    dst[thId] = mem_buff[thId]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared() raises:
    print("== run_copy_via_shared")
    let in_data = gpu_host.memory._malloc_managed[DType.float32](16)
    let out_data = gpu_host.memory._malloc_managed[DType.float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    let copy_via_shared_gpu = gpu_host.Function[
        fn (
            DTypePointer[DType.float32],
            DTypePointer[DType.float32],
        ) -> None, copy_via_shared
    ]()

    let stream = gpu_host.Stream()
    copy_via_shared_gpu(stream, (1,), (16,), in_data, out_data)

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
        print(out_data.load(i))

    pass


fn main():
    try:
        with gpu_host.Context() as ctx:
            run_copy_via_shared()
    except e:
        print("CUDA_ERROR:", e)
