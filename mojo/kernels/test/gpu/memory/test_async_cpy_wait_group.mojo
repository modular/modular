# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device

# RUN: %mojo-no-debug %s | FileCheck %s

from gpu import ThreadIdx
from gpu.host import Context, CudaInstance, Device, Function, Stream
from gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
)
from memory import stack_allocation
from testing import assert_equal


fn copy_via_shared(
    src: DTypePointer[DType.float32],
    dst: DTypePointer[DType.float32],
):
    var thread_id = ThreadIdx.x()
    var mem_buff: Pointer[Float32, AddressSpace.SHARED] = stack_allocation[
        16, Float32, address_space = AddressSpace.SHARED
    ]()
    var src_global: Pointer[
        Float32, AddressSpace.GLOBAL
    ] = src._as_scalar_pointer().bitcast[address_space = AddressSpace.GLOBAL]()

    async_copy[4](
        src_global.offset(thread_id),
        mem_buff.offset(thread_id),
    )

    async_copy_commit_group()
    async_copy_wait_group(0)

    dst[thread_id] = mem_buff[thread_id]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared(ctx: Context) raises:
    print("== run_copy_via_shared")
    var in_data = ctx.malloc_managed[DType.float32](16)
    var out_data = ctx.malloc_managed[DType.float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    var copy_via_shared_gpu = Function[copy_via_shared](ctx)

    var stream = Stream(ctx)
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


fn copy_with_src_size(
    src: DTypePointer[DType.float32, address_space = AddressSpace.GLOBAL],
    dst: DTypePointer[DType.float32, address_space = AddressSpace.GLOBAL],
    src_size: Int,
):
    var smem = stack_allocation[
        4, DType.float32, address_space = AddressSpace.SHARED
    ]()

    async_copy[16](src.address, smem.address, src_size)
    async_copy_wait_all()
    dst[0] = smem[0]
    dst[1] = smem[1]
    dst[2] = smem[2]
    dst[3] = smem[3]


fn test_copy_with_src_size(ctx: Context) raises:
    var stream = Stream(ctx)

    alias size = 4
    var a_host = DTypePointer[DType.float32].alloc(size)
    var b_host = DTypePointer[DType.float32].alloc(size)

    for i in range(size):
        a_host[i] = i + 1

    var a_device = ctx.malloc[Float32](size)
    var b_device = ctx.malloc[Float32](size)

    ctx.copy_host_to_device(a_device, a_host, size)

    alias kernel = copy_with_src_size
    var func = Function[kernel](ctx, threads_per_block=1)

    alias src_size = 3 * sizeof[DType.float32]()

    func(a_device, b_device, src_size, grid_dim=(1, 1, 1), block_dim=(1, 1, 1))

    ctx.synchronize()

    ctx.copy_device_to_host(b_host, b_device, size)

    assert_equal(b_host[0], 1)
    assert_equal(b_host[1], 2)
    assert_equal(b_host[2], 3)
    assert_equal(b_host[3], 0)

    ctx.free(a_device)
    ctx.free(b_device)
    a_host.free()
    b_host.free()

    _ = func^
    _ = stream^


fn main():
    try:
        with CudaInstance() as instance:
            with Context(Device(instance)) as ctx:
                run_copy_via_shared(ctx)
                test_copy_with_src_size(ctx)
    except e:
        print("CUDA_ERROR:", e)
