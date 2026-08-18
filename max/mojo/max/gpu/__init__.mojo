# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
# TODO(MSTDL-2788): Placeholder package docstring. Without a package index here,
# the API reference publishes no page for this package and lists its subpackages
# as though they were top-level. Replace this docstring with the one from
# `std/gpu/__init__.mojo` once the rest of that package moves, repointing its
# `/docs/std/...` links and `from std.gpu import ...` example.
"""Provides GPU programming primitives for hardware-accelerated Mojo code."""

from .primitives import (
    block_rank_in_cluster,
    cluster_arrive,
    cluster_arrive_relaxed,
    cluster_sync,
    cluster_sync_relaxed,
    cluster_wait,
    elect_one_sync,
    PDL,
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)

from .memory import (
    AddressSpace,
    CacheEviction,
    CacheOperation,
    Consistency,
    Fill,
    ReduceOp,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
    async_copy_wait_group,
    cp_async_bulk_tensor_global_shared_cta,
    cp_async_bulk_tensor_global_shared_cta_elect,
    cp_async_bulk_tensor_reduce_global_shared_cta,
    cp_async_bulk_tensor_shared_cluster_global,
    cp_async_bulk_tensor_shared_cluster_global_multicast,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
    fence_proxy_tensormap_generic_sys_acquire,
    fence_proxy_tensormap_generic_sys_release,
    load,
    multimem_ld_reduce,
    multimem_st,
)

from .sync import (
    NamedBarrierSemaphore,
    Semaphore,
    AMDScheduleBarrierMask,
    async_copy_arrive,
    barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_test_wait,
    mbarrier_try_wait_parity_shared,
    named_barrier,
    schedule_barrier,
    schedule_group_barrier,
    syncwarp,
    s_waitcnt,
    s_waitcnt_barrier,
)
