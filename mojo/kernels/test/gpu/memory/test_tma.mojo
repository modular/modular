# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# DISABLED: %mojo-no-debug %s | FileCheck %s
# RUN: true

from gpu.host import Context, Function
from gpu.host.memory import create_tma_descriptor, _malloc_managed, _free
from gpu.memory import cp_async_bulk_tensor_shared_cluster_global
from gpu.sync import mbarrier_init, mbarrier_arrive, mbarrier_test_wait
from gpu.memory import _GPUAddressSpace

from utils.index import Index


# CHECK-LABLE: test_tma_tile_copy
def test_tma_tile_copy():
    print("== test_tma_tile_copy")
    var gmem_ptr = _malloc_managed[DType.float32](8 * 8)
    for i in range(16):
        gmem_ptr[i] = i

    var descriptor = create_tma_descriptor[DType.float32, 2](
        gmem_ptr, (8, 8), (8, 1), (4, 4), (1, 1)
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

    var kernel_copy_async = Function[kernel_copy_async_tma](dump_ptx=True)
    kernel_copy_async(grid_dim=(1), block_dim=(1))
    _free(gmem_ptr)
    descriptor.free()


def main():
    with Context():
        test_tma_tile_copy()
