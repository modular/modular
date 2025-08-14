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


from gpu import WARP_SIZE, lane_id
from gpu.host import DeviceContext
from gpu.host._nvidia_cuda import TMADescriptor, create_tma_descriptor
from gpu.id import block_idx, thread_idx, block_dim
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
)
from gpu.sync import (
    mbarrier_arrive_expect_tx_shared,
    mbarrier_init,
    mbarrier_try_wait_parity_shared,
    barrier,
)
import gpu.warp as warp
from math import ceildiv
from memory import stack_allocation
from random import rand
from sys.info import sizeof
from testing import assert_almost_equal
from utils.index import Index
from utils.numerics import get_accum_type


@always_inline
fn block_reduce[
    dtype: DType, max_warps_per_block: Int = 32
](val: Scalar[dtype]) -> Scalar[dtype]:
    var m2_shared = stack_allocation[
        max_warps_per_block, dtype, address_space = _GPUAddressSpace.SHARED
    ]()
    var m2_broadcast = stack_allocation[
        1, dtype, address_space = _GPUAddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    for i in range(tid, max_warps_per_block, block_dim.x):
        m2_shared[i] = 0

    if tid == 0:
        m2_broadcast[0] = 0

    barrier()

    var warp_m2 = warp.sum(val)

    var warp_id = warp.broadcast(tid // WARP_SIZE)
    var lane_idx = lane_id()

    if lane_idx == 0:
        m2_shared[warp_id] = warp_m2
    barrier()

    if warp_id == 0 and lane_idx < max_warps_per_block:
        var block_m2 = warp.lane_group_sum[num_lanes=max_warps_per_block](
            m2_shared[lane_idx]
        )
        if lane_idx == 0:
            m2_broadcast[0] = block_m2
    barrier()
    return m2_broadcast[0]


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
fn tma_reduction_kernel[
    dtype: DType,
    accum_type: DType,
    block_size: Int,
    items_per_thread: Int,
](
    descriptor: TMADescriptor,
    n: Int,
    d_data: UnsafePointer[Scalar[dtype]],
    d_out: UnsafePointer[Scalar[accum_type]],
):
    alias shmem_size = block_size * items_per_thread
    var shmem = stack_allocation[
        shmem_size,
        dtype,
        alignment=16,
        address_space = _GPUAddressSpace.SHARED,
    ]()

    # Calculate elements for this block
    var block_offset = block_idx.x * shmem_size

    # Create barrier for TMA transfer from GMEM to SMEM.
    var mbar = stack_allocation[
        1, Int64, address_space = _GPUAddressSpace.SHARED
    ]()

    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    if thread_idx.x == 0:
        # Add expected_bytes requirement to barrier.
        var expected_bytes = shmem_size * sizeof[dtype]()
        mbarrier_arrive_expect_tx_shared(mbar, expected_bytes)
        cp_async_bulk_tensor_shared_cluster_global(
            shmem,
            descriptor_ptr,
            mbar,
            Index(block_offset),
        )

    # Wait for TMA to complete (expected_bytes transferred).
    mbarrier_try_wait_parity_shared(mbar, 0, 10_000_000)

    # Local thread reduction of loaded data.
    var local_sum = Scalar[accum_type](0)

    @parameter
    for i in range(items_per_thread):
        var idx = thread_idx.x * items_per_thread + i
        if idx < shmem_size:
            local_sum += shmem[idx].cast[accum_type]()

    # Block reduction of local sums.
    local_sum = block_reduce(local_sum)

    # Write block result to output buffer for result checking.
    if thread_idx.x == 0:
        d_out[block_idx.x] = local_sum.cast[accum_type]()


def test_tma_block_reduce[dtype: DType](ctx: DeviceContext, n: Int):
    alias block_size = 32
    alias items_per_thread = 8
    alias accum_type = get_accum_type[dtype]()

    var h_data = UnsafePointer[Scalar[dtype]].alloc(n)
    var expected_sum = Scalar[accum_type](0)
    rand[dtype](h_data, n)
    for i in range(n):
        expected_sum += h_data[i].cast[accum_type]()

    var d_data = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(d_data, h_data)

    var tma_desc = create_tma_descriptor[dtype, 1](
        d_data,
        (n),
        (1),
        (block_size * items_per_thread),
    )

    var num_blocks = min(
        65536,
        ceildiv(n, block_size * items_per_thread),
    )

    var result_host = UnsafePointer[Scalar[accum_type]].alloc(num_blocks)
    var d_out = ctx.enqueue_create_buffer[accum_type](num_blocks)
    ctx.enqueue_memset(d_out, 0)

    ctx.enqueue_function[
        tma_reduction_kernel[dtype, accum_type, block_size, items_per_thread]
    ](tma_desc, n, d_data, d_out, grid_dim=num_blocks, block_dim=block_size)

    ctx.enqueue_copy(result_host, d_out)

    var total_sum = Scalar[accum_type](0)
    for i in range(num_blocks):
        total_sum += result_host[i]

    assert_almost_equal(
        total_sum,
        expected_sum,
        rtol=1e-2,
        atol=1e-2,
    )

    h_data.free()
    result_host.free()
    _ = d_data
    _ = d_out


def main():
    var test_sizes = [2048, 4096, 8192, 16384]
    alias dtype = DType.bfloat16

    with DeviceContext() as ctx:
        for test_size in test_sizes:
            test_tma_block_reduce[dtype](ctx, test_size)
