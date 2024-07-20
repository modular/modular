# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.memory import cp_async_bulk_tensor_shared_cluser_global
from gpu.memory import _GPUAddressSpace
from utils.index import Index


# CHECK-LABEL: test_async_copy_asm
fn test_async_copy_asm():
    print("== test_async_copy_asm")

    fn test_async_copy_kernel(
        dst_mem: UnsafePointer[Float32, _GPUAddressSpace.SHARED],
        tma_descriptor: UnsafePointer[NoneType],
        mem_bar: UnsafePointer[Float32, _GPUAddressSpace.SHARED],
        *coords: Int32,
    ):
        # CHECK: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        cp_async_bulk_tensor_shared_cluser_global(
            dst_mem, tma_descriptor, mem_bar, Index(coords[0], coords[1])
        )
        # CHECK: cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes
        cp_async_bulk_tensor_shared_cluser_global(
            dst_mem, tma_descriptor, mem_bar, Index(coords[0])
        )

    print(
        str(
            _compile_code[
                test_async_copy_kernel,
                target = _get_nvptx_target(),
            ]().asm
        )
    )


fn main():
    test_async_copy_asm()
