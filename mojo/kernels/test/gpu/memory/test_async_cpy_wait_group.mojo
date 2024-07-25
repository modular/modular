# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device

# RUN: %mojo-no-debug %s | FileCheck %s

from gpu import ThreadIdx
from gpu.host import DeviceContext
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
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
):
    var thread_id = Int(ThreadIdx.x())
    var mem_buff: UnsafePointer[
        Float32, AddressSpace.SHARED
    ] = stack_allocation[16, Float32, address_space = AddressSpace.SHARED]()
    var src_global: UnsafePointer[Float32, AddressSpace.GLOBAL] = src.bitcast[
        address_space = AddressSpace.GLOBAL
    ]()

    async_copy[4](
        src_global.offset(thread_id),
        mem_buff.offset(thread_id),
    )

    async_copy_commit_group()
    async_copy_wait_group(0)

    dst[thread_id] = mem_buff[thread_id]


# CHECK-LABEL: run_copy_via_shared
fn run_copy_via_shared(ctx: DeviceContext) raises:
    print("== run_copy_via_shared")
    var in_data = UnsafePointer[Float32].alloc(16)
    var out_data = UnsafePointer[Float32].alloc(16)
    var in_data_device = ctx.create_buffer[DType.float32](16)
    var out_data_device = ctx.create_buffer[DType.float32](16)

    for i in range(16):
        in_data[i] = i + 1
        out_data[i] = 0

    ctx.enqueue_copy_to_device(in_data_device, in_data)
    ctx.enqueue_copy_to_device(out_data_device, out_data)

    var copy_via_shared_gpu = ctx.compile_function[copy_via_shared]()

    ctx.enqueue_function(
        copy_via_shared_gpu,
        in_data_device,
        out_data_device,
        grid_dim=(1,),
        block_dim=(16),
    )

    ctx.enqueue_copy_from_device(out_data, out_data_device)

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

    _ = in_data_device
    _ = out_data_device
    in_data.free()
    out_data.free()
    _ = copy_via_shared_gpu^


fn copy_with_src_size(
    src: UnsafePointer[Float32, address_space = AddressSpace.GLOBAL],
    dst: UnsafePointer[Float32, address_space = AddressSpace.GLOBAL],
    src_size: Int,
):
    var smem = stack_allocation[
        4, DType.float32, address_space = AddressSpace.SHARED
    ]()

    async_copy[16](src, smem, src_size)
    async_copy_wait_all()
    dst[0] = smem[0]
    dst[1] = smem[1]
    dst[2] = smem[2]
    dst[3] = smem[3]


fn test_copy_with_src_size(ctx: DeviceContext) raises:
    alias size = 4
    var a_host = UnsafePointer[Float32].alloc(size)
    var b_host = UnsafePointer[Float32].alloc(size)

    for i in range(size):
        a_host[i] = i + 1

    var a_device = ctx.create_buffer[DType.float32](size)
    var b_device = ctx.create_buffer[DType.float32](size)

    ctx.enqueue_copy_to_device(a_device, a_host)

    alias kernel = copy_with_src_size
    var func = ctx.compile_function[kernel](threads_per_block=1)

    alias src_size = 3 * sizeof[DType.float32]()

    ctx.enqueue_function(
        func,
        a_device,
        b_device,
        src_size,
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )

    ctx.synchronize()

    ctx.enqueue_copy_from_device(b_host, b_device)

    assert_equal(b_host[0], 1)
    assert_equal(b_host[1], 2)
    assert_equal(b_host[2], 3)
    assert_equal(b_host[3], 0)

    _ = a_device
    _ = b_device
    a_host.free()
    b_host.free()

    _ = func^


fn main():
    try:
        with DeviceContext() as ctx:
            run_copy_via_shared(ctx)
            test_copy_with_src_size(ctx)
    except e:
        print("CUDA_ERROR:", e)
