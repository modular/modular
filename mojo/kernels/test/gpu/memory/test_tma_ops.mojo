# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.memory import (
    _GPUAddressSpace,
    CacheEviction,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_reduce,
    fence_proxy_tensormap_generic_sys_acquire,
    fence_proxy_tensormap_generic_sys_release,
)
from gpu.sync import cp_async_bulk_wait_group, cp_async_bulk_commit_group
from memory import UnsafePointer

from utils.index import Index


# CHECK-LABEL: test_async_copy_asm
fn test_async_copy_asm():
    print("== test_async_copy_asm")

    fn test_async_copy_kernel(
        dst_mem: UnsafePointer[
            Float32, address_space = _GPUAddressSpace.SHARED
        ],
        tma_descriptor: UnsafePointer[NoneType],
        mem_bar: UnsafePointer[
            Float32, address_space = _GPUAddressSpace.SHARED
        ],
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
        _compile_code_asm[
            test_async_copy_kernel,
            target = _get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_async_store_asm
fn test_async_store_asm():
    print("== test_async_store_asm")

    fn test_async_store_kernel(
        src_mem: UnsafePointer[
            Float32, address_space = _GPUAddressSpace.SHARED
        ],
        tma_descriptor: UnsafePointer[NoneType],
        *coords: Int32,
    ):
        # CHECK: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%rd1, {%r2, %r3}], [%r1];
        cp_async_bulk_tensor_global_shared_cta(
            src_mem, tma_descriptor, Index(coords[0], coords[1])
        )
        # CHECK: cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd1, {%r2, %r3}], [%r1], %rd3;
        cp_async_bulk_tensor_global_shared_cta[
            eviction_policy = CacheEviction.EVICT_FIRST
        ](src_mem, tma_descriptor, Index(coords[0], coords[1]))
        # CHECK: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group [%rd1, {%r2}], [%r1];
        cp_async_bulk_tensor_global_shared_cta(
            src_mem, tma_descriptor, Index(coords[0])
        )
        # CHECK: cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.L2::cache_hint [%rd1, {%r2}], [%r1], %rd4;
        cp_async_bulk_tensor_global_shared_cta[
            eviction_policy = CacheEviction.EVICT_LAST
        ](src_mem, tma_descriptor, Index(coords[0]))

    print(
        _compile_code_asm[
            test_async_store_kernel,
            target = _get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_async_bulk_tensor_reduce_asm
fn test_async_bulk_tensor_reduce_asm():
    print("== test_async_bulk_tensor_reduce_asm")

    fn test_async_bulk_tensor_reduce_asm(
        src_mem: UnsafePointer[
            Float32, address_space = _GPUAddressSpace.SHARED
        ],
        tma_descriptor: UnsafePointer[NoneType],
        *coords: Int32,
    ):
        # CHECK:
        cp_async_bulk_tensor_reduce[reduction_kind="add"](
            src_mem, tma_descriptor, Index(coords[0], coords[1])
        )
        # CHECK:
        cp_async_bulk_tensor_reduce[
            reduction_kind="add", eviction_policy = CacheEviction.EVICT_FIRST
        ](src_mem, tma_descriptor, Index(coords[0], coords[1]))
        # CHECK:
        cp_async_bulk_tensor_reduce[reduction_kind="add"](
            src_mem, tma_descriptor, Index(coords[0])
        )
        # CHECK:
        cp_async_bulk_tensor_reduce[
            reduction_kind="add", eviction_policy = CacheEviction.EVICT_LAST
        ](src_mem, tma_descriptor, Index(coords[0]))

    print(
        _compile_code_asm[
            test_async_bulk_tensor_reduce_asm,
            target = _get_gpu_target["sm_90"](),
        ]()
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
        _compile_code_asm[
            test_tma_fence_proxy_kernel,
            target = _get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_cp_async_bulk_wait_group
fn test_cp_async_bulk_wait_group():
    print("== test_cp_async_bulk_wait_group")

    fn cp_async_bulk_wait_group_kernel[n: Int32]():
        # CHECK: cp.async.bulk.wait_group.read 0;
        cp_async_bulk_wait_group[0]()
        # CHECK: cp.async.bulk.wait_group 2;
        cp_async_bulk_wait_group[n, False]()

    print(
        _compile_code_asm[
            cp_async_bulk_wait_group_kernel[2],
            target = _get_gpu_target["sm_90"](),
        ]()
    )


# CHECK-LABEL: test_cp_async_bulk_commit_group
fn test_cp_async_bulk_commit_group():
    print("== test_cp_async_bulk_commit_group")

    fn cp_async_bulk_commit_group_kernel():
        # CHECK: cp.async.bulk.commit_group;
        cp_async_bulk_commit_group()

    print(
        _compile_code_asm[
            cp_async_bulk_commit_group_kernel,
            target = _get_gpu_target["sm_90"](),
        ]()
    )


fn main():
    test_async_copy_asm()
    test_async_store_asm()
    test_async_bulk_tensor_reduce_asm()
    test_tma_fence_proxy()
    test_cp_async_bulk_wait_group()
    test_cp_async_bulk_commit_group()
