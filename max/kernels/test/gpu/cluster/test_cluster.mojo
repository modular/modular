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

from sys import sizeof

from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.cluster import (
    cluster_sync_acquire,
    cluster_sync_release,
    clusterlaunchcontrol_try_cancel,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_query_cancel_get_first_ctaid,
    elect_one_sync_with_mask,
)
from layout.tma_async import SharedMemBarrier
from gpu.intrinsics import Scope
from memory import stack_allocation
from gpu.memory import _GPUAddressSpace as AddressSpace
from buffer import DimList, NDBuffer
from testing import assert_almost_equal


# Derived from https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernel-example-vector-scalar-multiplication
fn cluster_launch_control(data: UnsafePointer[Float32], n: UInt):
    result = stack_allocation[
        2,
        UInt64,
        address_space = AddressSpace.SHARED,
        alignment=16,
    ]()

    mbar = stack_allocation[
        1,
        SharedMemBarrier,
        address_space = AddressSpace.SHARED,
        alignment=8,
    ]()

    bx: UInt32 = block_idx.x
    tidx: UInt32 = bx * block_dim.x + thread_idx.x
    if tidx < n:
        data[tidx] = 1.0

    phase: UInt32 = 0

    # Single thread barrier.
    if thread_idx.x == 0:
        mbar[0].init(1)

    alpha: Int64 = thread_idx.x

    # Work-straling loop.
    while True:
        barrier()

        if thread_idx.x == 0:
            # Acquire write of result in the async proxy.
            # Matches `ptx::fence_proxy_async_generic_sync_restrict`.
            cluster_sync_acquire()

            # Matches `cg::invoke_one`. Selects the first thread in the block.
            if elect_one_sync_with_mask(mask=1):
                clusterlaunchcontrol_try_cancel(
                    result,
                    UnsafePointer[Int64, address_space = AddressSpace.SHARED](
                        to=mbar[0].mbar
                    ),
                )

            # Matches `ptx::mbarrier_arrive_expect_tx`.
            _ = mbar[0].expect_bytes_relaxed(2 * sizeof[UInt64]())

        if tidx < n:
            data[tidx] *= Float32(alpha)

        # Cancellation request synchronization:
        mbar[0].wait_acquire[Scope.BLOCK](phase)
        phase ^= 1

        # Cancellation request decoding.
        if not clusterlaunchcontrol_query_cancel_is_canceled(result):
            break

        bx = clusterlaunchcontrol_query_cancel_get_first_ctaid["x"](result)

        # Releases read result to the async proxy.
        # Matches `ptx::fence_proxy_async_generic_sync_restrict`.
        cluster_sync_release()


fn test_cluster_launch_control(ctx: DeviceContext) raises:
    alias n = 4000

    data = ctx.enqueue_create_buffer[DType.float32](n)

    ctx.enqueue_function[cluster_launch_control](
        data,
        n,
        grid_dim=((n + 1023) // 1024),
        block_dim=(1024),
    )

    var data_host_ptr = UnsafePointer[Float32].alloc(n)
    var data_host = NDBuffer[DType.float32, 1, _, DimList(n)](data_host_ptr)

    ctx.enqueue_copy(data_host_ptr, data)
    ctx.synchronize()

    for i in range(n):
        assert_almost_equal(data_host[i], Float32(i % 1024))

    _ = data
    _ = data_host


def main():
    with DeviceContext() as ctx:
        test_cluster_launch_control(ctx)
