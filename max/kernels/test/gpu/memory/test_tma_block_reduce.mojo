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

from buffer import NDBuffer
from gpu import WARP_SIZE, lane_id
from gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from gpu.host._nvidia_cuda import TMADescriptor, create_tma_descriptor
from gpu.id import block_idx, thread_idx, block_dim
from gpu.memory import (
    _GPUAddressSpace,
    cp_async_bulk_tensor_shared_cluster_global,
    external_memory,
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
from sys.info import size_of, simd_width_of
from testing import assert_almost_equal
from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from sys import argv


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


fn global_reduction_kernel[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
    max_warps_per_block: Int,
    input_fn: fn[width: Int, _rank: Int] (
        idx: IndexList[_rank]
    ) capturing -> SIMD[dtype, width],
](d_out: UnsafePointer[Scalar[accum_type]], num_cols: Int):
    var tid = thread_idx.x
    var row = block_idx.x
    var idx = tid * simd_width
    var vec_data = SIMD[accum_type, simd_width](0)

    if idx < num_cols:
        vec_data = input_fn[simd_width, 2](IndexList[2](row, idx)).cast[
            accum_type
        ]()

    var thread_sum = vec_data.reduce_add()

    var block_sum = block_reduce[max_warps_per_block=max_warps_per_block](
        thread_sum
    )

    if thread_idx.x == 0:
        d_out[row] = block_sum


@__llvm_arg_metadata(descriptor, `nvvm.grid_constant`)
fn tma_reduction_kernel[
    dtype: DType,
    accum_type: DType,
    simd_width: Int,
](
    descriptor: TMADescriptor,
    rows: Int,
    cols: Int,
    d_data: UnsafePointer[Scalar[dtype]],
    d_out: UnsafePointer[Scalar[accum_type]],
):
    var shmem = external_memory[
        Scalar[dtype], address_space = _GPUAddressSpace.SHARED, alignment=128
    ]()
    # Calculate elements offset for this block (row).
    var block_offset = block_idx.x

    # Create barrier for TMA transfer from GMEM to SMEM.
    var mbar = stack_allocation[
        1, Int64, address_space = _GPUAddressSpace.SHARED
    ]()

    var descriptor_ptr = UnsafePointer(to=descriptor).bitcast[NoneType]()
    mbarrier_init(mbar, 1)

    if thread_idx.x == 0:
        # Add expected_bytes requirement to barrier.
        var expected_bytes = cols * size_of[dtype]()
        mbarrier_arrive_expect_tx_shared(mbar, expected_bytes)
        cp_async_bulk_tensor_shared_cluster_global(
            shmem,
            descriptor_ptr,
            mbar,
            Index(0, block_offset),
        )

    # Wait for TMA to complete (expected_bytes transferred).
    mbarrier_try_wait_parity_shared(mbar, 0, 10_000_000)

    # Local thread reduction of loaded data.
    var vec_data = SIMD[accum_type, simd_width](0)
    var idx = thread_idx.x * simd_width
    if idx < cols:
        vec_data = shmem.load[width=simd_width](idx).cast[accum_type]()
    var local_sum = vec_data.reduce_add()

    # Block reduction of local sums.
    local_sum = block_reduce(local_sum)

    # Write block result to output buffer for result checking.
    if thread_idx.x == 0:
        d_out[block_idx.x] = local_sum.cast[accum_type]()


def test_tma_block_reduce[
    dtype: DType, use_tma: Bool
](ctx: DeviceContext, rows: Int, cols: Int, benchmark: Bool = False,):
    var n = rows * cols
    alias simd_width = simd_width_of[dtype, target = get_gpu_target()]()
    alias max_warps_per_block = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    alias accum_type = get_accum_type[dtype]()

    var h_data = UnsafePointer[Scalar[dtype]].alloc(n)
    var expected_sum = Scalar[accum_type](0)
    rand[dtype](h_data, n)
    for i in range(n):
        expected_sum += h_data[i].cast[accum_type]()

    var d_data = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(d_data, h_data)

    var grid_dim = rows
    var block_dim = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps_per_block,
    )

    var result_host = UnsafePointer[Scalar[accum_type]].alloc(grid_dim)
    var d_out = ctx.enqueue_create_buffer[accum_type](grid_dim)
    ctx.enqueue_memset(d_out, 0)

    # Define the kernel launch function for benchmarking
    @parameter
    @always_inline
    fn kernel_launch(ctx: DeviceContext) raises -> None:
        @parameter
        if use_tma:
            var tma_desc = create_tma_descriptor[dtype, 2](
                d_data,
                (rows, cols),
                (cols, 1),
                (1, cols),
            )
            # Calculate shared memory size needed per row.
            var shared_mem_bytes = cols * size_of[dtype]()
            ctx.enqueue_function[
                tma_reduction_kernel[dtype, accum_type, simd_width]
            ](
                tma_desc,
                rows,
                cols,
                d_data,
                d_out,
                grid_dim=grid_dim,
                block_dim=block_dim,
                shared_mem_bytes=shared_mem_bytes,
            )
        else:
            var shape = Index(rows, cols)
            var data_buf = NDBuffer[dtype, 2](d_data.unsafe_ptr(), shape)

            # Change the input function to match RMS norm pattern
            @__copy_capture(data_buf)
            @always_inline
            @parameter
            fn input_fn_2d[
                width: Int, _rank: Int
            ](idx: IndexList[_rank]) -> SIMD[dtype, width]:
                return data_buf.load[width=width](rebind[IndexList[2]](idx))

            ctx.enqueue_function[
                global_reduction_kernel[
                    dtype,
                    accum_type,
                    simd_width,
                    max_warps_per_block,
                    input_fn_2d,
                ]
            ](
                d_out,
                cols,  # num_cols
                grid_dim=grid_dim,
                block_dim=block_dim,
            )

    if benchmark:
        # Run kernel multiple times for benchmarking.
        alias num_warmup = 5
        alias num_iters = 100

        # Warmup runs.
        for _ in range(num_warmup):
            kernel_launch(ctx)

        # Timed runs
        var total_time = ctx.execution_time[kernel_launch](num_iters)
        var avg_time_ms = Float64(total_time) / (num_iters * 1e6)

        print(
            "  Average kernel time for:",
            rows,
            "x",
            cols,
            ":",
            avg_time_ms,
            "ms",
        )
    else:
        # Single run for correctness testing.
        kernel_launch(ctx)

    ctx.enqueue_copy(result_host, d_out)

    var total_sum = Scalar[accum_type](0)
    for i in range(grid_dim):
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
    var test_sizes = [128, 256, 512, 1024]
    var depths = [64, 128, 256]
    alias dtype = DType.bfloat16

    # Parse command line arguments.
    var benchmark = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == "--benchmark" or args[i] == "--benchmark=yes":
            benchmark = True

    alias use_tma = True

    with DeviceContext() as ctx:
        for test_size in test_sizes:
            for depth in depths:
                test_tma_block_reduce[dtype, use_tma](
                    ctx, test_size, depth, benchmark=benchmark
                )
