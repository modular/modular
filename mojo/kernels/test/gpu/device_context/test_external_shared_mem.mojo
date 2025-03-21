# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from gpu.host import DeviceContext, FuncAttribute
from gpu.id import thread_idx
from gpu.memory import AddressSpace, external_memory
from gpu.sync import barrier
from memory import UnsafePointer


# CHECK-LABEL: test_external_shared_mem
fn test_external_shared_mem(ctx: DeviceContext) raises:
    print("== test_external_shared_mem")

    fn dynamic_smem_kernel(data: UnsafePointer[Float32]):
        var dynamic_sram = external_memory[
            Float32, address_space = AddressSpace.SHARED, alignment=4
        ]()
        dynamic_sram[thread_idx.x] = thread_idx.x
        barrier()
        data[thread_idx.x] = dynamic_sram[thread_idx.x]

    var res_host_ptr = UnsafePointer[Float32].alloc(16)
    var res_device = ctx.enqueue_create_buffer[DType.float32](16)

    for i in range(16):
        res_host_ptr[i] = 0

    ctx.enqueue_copy(res_device, res_host_ptr)

    ctx.enqueue_function[dynamic_smem_kernel](
        res_device,
        grid_dim=1,
        block_dim=16,
        shared_mem_bytes=64 * 1024,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(64 * 1024),
    )

    ctx.enqueue_copy(res_host_ptr, res_device)

    ctx.synchronize()

    # CHECK: 0.0
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
    for i in range(16):
        print(res_host_ptr[i])

    _ = res_device
    res_host_ptr.free()


def main():
    with DeviceContext() as ctx:
        test_external_shared_mem(ctx)
