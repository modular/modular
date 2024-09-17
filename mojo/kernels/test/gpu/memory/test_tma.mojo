# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# DISABLED: %mojo-no-debug %s | FileCheck %s
# RUN: true

from gpu.host import DeviceContext
from gpu.host.memory import _free, _malloc_managed, create_tma_descriptor
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
)
from gpu.sync import mbarrier_arrive, mbarrier_init, mbarrier_test_wait

from utils.index import Index


# CHECK-LABLE: test_tma_tile_copy
def test_tma_tile_copy(ctx: DeviceContext):
    print("== test_tma_tile_copy")
    var gmem_host = UnsafePointer[Float32].alloc(8 * 8)
    for i in range(16):
        gmem_host[i] = i

    var gmem_dev = ctx.create_buffer[DType.float32](8 * 8)

    ctx.enqueue_copy_to_device(gmem_dev, gmem_host)

    var descriptor = create_tma_descriptor[DType.float32, 2](
        gmem_dev.ptr, (8, 8), (8, 1), (4, 4), (1, 1)
    )

    @parameter
    fn kernel_copy_async_tma():
        var shmem = stack_allocation[
            16, DType.float32, address_space = _GPUAddressSpace.SHARED
        ]()
        var mbar = stack_allocation[
            1, Int64, address_space = _GPUAddressSpace.SHARED
        ]()
        mbarrier_init(mbar, 1)
        cp_async_bulk_tensor_shared_cluster_global(
            shmem, descriptor, mbar, Index(0, 0)
        )
        var state = mbarrier_arrive(mbar)
        while not mbarrier_test_wait(mbar, state):
            pass

        # CHECK: 0
        # CHECK: 1
        # CHECK: 2
        # CHECK: 3
        # CHECK: 8
        # CHECK: 9
        # CHECK: 10
        # CHECK: 11
        # CHECK: 16
        # CHECK: 17
        # CHECK: 18
        # CHECK: 19
        # CHECK: 24
        # CHECK: 25
        # CHECK: 26
        # CHECK: 27
        @parameter
        for i in range(16):
            var val = shmem[i]
            print(val)

    var kernel_copy_async = ctx.compile_function[
        kernel_copy_async_tma, dump_ptx=True
    ]()
    ctx.enqueue_function(kernel_copy_async, grid_dim=(1), block_dim=(1))
    _ = gmem_dev
    gmem_host.free()
    descriptor.free()
    _ = kernel_copy_async^


def main():
    with DeviceContext() as ctx:
        test_tma_tile_copy(ctx)
