# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host._compile import _compile_code, _get_nvptx_target
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
    fence_proxy_tensormap_generic_sys_acquire,
    fence_proxy_tensormap_generic_sys_release,
)

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
        cp_async_bulk_tensor_shared_cluster_global(
            dst_mem, tma_descriptor, mem_bar, Index(coords[0], coords[1])
        )
        # CHECK: cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes
        cp_async_bulk_tensor_shared_cluster_global(
            dst_mem, tma_descriptor, mem_bar, Index(coords[0])
        )

    print(
        str(
            _compile_code[
                test_async_copy_kernel,
                target = _get_nvptx_target["sm_90"](),
            ]().asm
        )
    )


# CHECK-LABEL: test_tma_fence_proxy
fn test_tma_fence_proxy():
    print("== test_tma_fence_proxy")

    fn test_tma_fence_proxy_kernel(descriptor_ptr: UnsafePointer[Int32]):
        # CHECK: fence.proxy.tensormap::generic.acquire.sys [%rd1], 128;
        fence_proxy_tensormap_generic_sys_acquire(descriptor_ptr, 128)
        # CHECK: fence.proxy.tensormap::generic.release.sys;
        fence_proxy_tensormap_generic_sys_release()

    print(
        str(
            _compile_code[
                test_tma_fence_proxy_kernel,
                target = _get_nvptx_target["sm_90"](),
            ]().asm
        )
    )


fn main():
    test_async_copy_asm()
    test_tma_fence_proxy()
