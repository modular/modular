# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from gpu.cluster import elect_one_sync
from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.memory import (
    CacheEviction,
    ReduceOp,
    _GPUAddressSpace,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_reduce,
    cp_async_bulk_tensor_shared_cluster_global,
    fence_proxy_tensormap_generic_sys_acquire,
    fence_proxy_tensormap_generic_sys_release,
)

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
        cp_async_bulk_tensor_reduce[reduction_kind = ReduceOp.ADD](
            src_mem, tma_descriptor, Index(coords[0], coords[1])
        )
        # CHECK:
        cp_async_bulk_tensor_reduce[
            reduction_kind = ReduceOp.ADD,
            eviction_policy = CacheEviction.EVICT_FIRST,
        ](src_mem, tma_descriptor, Index(coords[0], coords[1]))
        # CHECK:
        cp_async_bulk_tensor_reduce[reduction_kind = ReduceOp.ADD](
            src_mem, tma_descriptor, Index(coords[0])
        )
        # CHECK:
        cp_async_bulk_tensor_reduce[
            reduction_kind = ReduceOp.ADD,
            eviction_policy = CacheEviction.EVICT_LAST,
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


# CHECK-LABEL: test_elect_one_sync
fn test_elect_one_sync():
    print("== test_elect_one_sync")

    fn test_elect_one_sync_kernel():
        # CHECK: elect.sync      %r1|%p1, -1;
        var _lane_predicate: Bool = elect_one_sync()

    print(
        _compile_code_asm[
            test_elect_one_sync_kernel,
            target = _get_gpu_target["sm_90"](),
        ]()
    )


fn main():
    test_async_copy_asm()
    test_async_store_asm()
    test_async_bulk_tensor_reduce_asm()
    test_tma_fence_proxy()
    test_elect_one_sync()
