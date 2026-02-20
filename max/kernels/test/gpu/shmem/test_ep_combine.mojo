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
# REQUIRES: NVIDIA-GPU

# RUN: %mojo-build %s -o %t
# RUN: %mpirun-gpu-per-process %t

from collections import OptionalReg

import time
from io.io import _printf
from math import sqrt
from os.path import dirname
from pathlib import Path
from random import randint, randn, seed
from sys import align_of, argv, simd_width_of, size_of
from sys.param_env import env_get_string

from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from memory import UnsafePointer
from shmem import *
from shmem._mpi import MPI_Finalize
from shmem.ep_comm import (
    BF16TokenFormat,
    EPLocalSyncCounters,
    combine_wait_kernel,
    combine_async_kernel,
    dispatch_wait_kernel,
    dispatch_async_kernel,
)
from testing import assert_equal

from utils import IndexList


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


fn is_pressure_test() -> Bool:
    for arg in argv():
        if arg == "--pressure-test":
            return True
    return False


@always_inline
fn welford_update(
    mut mean: Float64, mut m2: Float64, count: Int, new_value: Float64
):
    var delta: Float64
    var delta2: Float64
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2


fn legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: UnsafePointer[mut=True, Int32], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids + tok_id * top_k

        # The top-k ids for a token should be unique. If not, we will assign a
        # random id to the duplicate id.
        fn is_duplicate() -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if topk_ids_for_token[i] == topk_ids_for_token[j]:
                        return i
            return -1

        var duplicate_idx = is_duplicate()
        while duplicate_idx != -1:
            randint(topk_ids_for_token + duplicate_idx, 1, 0, n_experts - 1)
            duplicate_idx = is_duplicate()


fn test_combine[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
](ctx: DeviceContext, my_rank: Int) raises:
    comptime input_type = DType.bfloat16
    comptime gpu_target = get_gpu_target()
    comptime gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
    comptime gpu_alignment = align_of[
        SIMD[DType.uint8, gpu_simd_width], target=gpu_target
    ]()
    comptime token_fmt_type = BF16TokenFormat[
        output_layout = Layout(), hidden_size, top_k, gpu_alignment
    ]
    comptime msg_bytes = token_fmt_type.msg_size()
    comptime combine_msg_bytes = size_of[input_type]() * hidden_size
    comptime n_local_experts = n_experts // n_ranks

    if my_rank == 0:
        print(
            "Running ep_combine test: input_type:",
            input_type,
            "hidden_size:",
            hidden_size,
            "top_k:",
            top_k,
            "n_experts:",
            n_experts,
            "n_ranks:",
            n_ranks,
            "n_tokens_per_rank:",
            n_tokens_per_rank,
        )

    var send_buf = shmem_malloc[DType.uint8](
        UInt(top_k * n_tokens_per_rank * msg_bytes)
    )
    var recv_buf = shmem_malloc[DType.uint8](
        UInt(n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes)
    )
    var recv_count = shmem_malloc[DType.uint64](UInt(n_local_experts * n_ranks))
    var recv_count_buf = DeviceBuffer(
        ctx, recv_count, n_local_experts * n_ranks, owning=False
    )
    var atomic_counter = ctx.enqueue_create_buffer[DType.int32](
        EPLocalSyncCounters[n_experts].total_size()
    )

    ctx.enqueue_memset(recv_count_buf, UInt64.MAX_FINITE)
    ctx.enqueue_memset(atomic_counter, Int32(0))

    var host_topk_ids = alloc[Int32](n_tokens_per_rank * top_k)
    var host_input_tokens = alloc[Scalar[input_type]](
        n_tokens_per_rank * hidden_size
    )

    var device_topk_buf = ctx.enqueue_create_buffer[DType.int32](
        n_tokens_per_rank * top_k
    )
    var device_input_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * hidden_size
    )
    var device_output_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * n_ranks * n_local_experts * hidden_size
    )
    var device_row_offsets_buf = ctx.enqueue_create_buffer[DType.uint32](
        n_local_experts + 1
    )
    var device_expert_ids_buf = ctx.enqueue_create_buffer[DType.int32](
        n_local_experts
    )
    var device_src_token_info_buf = ctx.enqueue_create_buffer[DType.int32](
        n_tokens_per_rank * n_ranks * n_local_experts * 2
    )

    device_output_2_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * top_k * hidden_size
    )

    comptime topk_ids_layout = Layout.row_major(UNKNOWN_VALUE, top_k)
    comptime input_tokens_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    comptime output_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, hidden_size
    )
    comptime row_offsets_layout = Layout.row_major(n_local_experts + 1)
    comptime expert_ids_layout = Layout.row_major(n_local_experts)
    comptime src_token_info_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, 2
    )
    comptime output_2_layout = Layout.row_major(
        UNKNOWN_VALUE, top_k, hidden_size
    )

    var topk_ids_tensor = LayoutTensor[DType.int32, topk_ids_layout](
        device_topk_buf,
        RuntimeLayout[topk_ids_layout].row_major(
            IndexList[2](n_tokens_per_rank, top_k)
        ),
    )
    var input_tokens_tensor = LayoutTensor[input_type, input_tokens_layout](
        device_input_buf,
        RuntimeLayout[input_tokens_layout].row_major(
            IndexList[2](n_tokens_per_rank, hidden_size)
        ),
    )
    var output_tensor = LayoutTensor[input_type, output_layout](
        device_output_buf,
        RuntimeLayout[output_layout].row_major(
            IndexList[2](
                n_tokens_per_rank * n_ranks * n_local_experts, hidden_size
            )
        ),
    )
    var row_offsets_tensor = LayoutTensor[DType.uint32, row_offsets_layout](
        device_row_offsets_buf,
        RuntimeLayout[row_offsets_layout].row_major(
            IndexList[1](n_local_experts + 1)
        ),
    )
    var expert_ids_tensor = LayoutTensor[DType.int32, expert_ids_layout](
        device_expert_ids_buf,
        RuntimeLayout[expert_ids_layout].row_major(
            IndexList[1](n_local_experts)
        ),
    )
    var src_token_info_tensor = LayoutTensor[
        DType.int32, src_token_info_layout
    ](
        device_src_token_info_buf,
        RuntimeLayout[src_token_info_layout].row_major(
            IndexList[2](n_tokens_per_rank * n_ranks * n_local_experts, 2)
        ),
    )
    var output_2_tensor = LayoutTensor[input_type, output_2_layout](
        device_output_2_buf,
        RuntimeLayout[output_2_layout].row_major(
            IndexList[3](n_tokens_per_rank, top_k, hidden_size)
        ),
    )

    var format_handler = BF16TokenFormat[hidden_size, top_k, gpu_alignment](
        output_tensor.as_any_origin()
    )

    comptime hw_info = ctx.default_device_info

    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        input_tokens_layout,
        topk_ids_layout,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        1,  # p2p_world_size
        token_fmt_type,
    ]
    var func = ctx.compile_function_experimental[dispatch_async]()
    shmem_module_init(func)

    comptime dispatch_wait = dispatch_wait_kernel[
        hw_info.max_thread_block_size,
        row_offsets_layout,
        expert_ids_layout,
        src_token_info_layout,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        type_of(format_handler),
    ]
    var func_dispatch_wait = ctx.compile_function_experimental[dispatch_wait]()

    comptime combine_async = combine_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        output_layout,
        src_token_info_layout,
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_bytes,
        n_tokens_per_rank,
        1,  # p2p_world_size
    ]
    var func_combine_async = ctx.compile_function_experimental[combine_async]()
    shmem_module_init(func_combine_async)

    comptime combine_wait = combine_wait_kernel[
        input_type,
        hw_info.max_thread_block_size,
        output_2_layout,
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_bytes,
        n_tokens_per_rank,
    ]
    var func_combine_async_wait = ctx.compile_function_experimental[
        combine_wait
    ]()

    var num_iters: Int = 100 if is_benchmark() or is_pressure_test() else 3
    var combine_async_stat_m: Float64 = 0
    var combine_async_stat_m2: Float64 = 0
    var combine_wait_stat_m: Float64 = 0
    var combine_wait_stat_m2: Float64 = 0
    var e2e_stat_m: Float64 = 0
    var e2e_stat_m2: Float64 = 0

    @always_inline
    @parameter
    fn run_full_dispatch(ctx: DeviceContext) raises:
        # the recv_buf ptrs and recv_count ptrs need to be passed in a InlinedArray
        var recv_buf_ptrs = InlineArray[UnsafePointer[UInt8, MutAnyOrigin], 1](
            fill={}
        )
        var recv_count_ptrs = InlineArray[
            UnsafePointer[UInt64, MutAnyOrigin], 1
        ](fill={})
        recv_buf_ptrs[0] = recv_buf
        recv_count_ptrs[0] = recv_count

        ctx.enqueue_function(
            func,
            input_tokens_tensor,
            topk_ids_tensor,
            send_buf,
            recv_buf_ptrs,
            recv_count_ptrs,
            EPLocalSyncCounters[n_experts](atomic_counter.unsafe_ptr()),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )
        ctx.enqueue_function(
            func_dispatch_wait,
            format_handler,
            row_offsets_tensor,
            expert_ids_tensor,
            src_token_info_tensor,
            recv_buf,
            recv_count,
            EPLocalSyncCounters[n_experts](atomic_counter.unsafe_ptr()),
            Int32(my_rank),
            OptionalReg[
                LayoutTensor[input_type, Layout.row_major[2](), ImmutAnyOrigin]
            ](),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )
        shmem_barrier_all_on_stream(ctx.stream())

    @always_inline
    @parameter
    fn run_combine_async(ctx: DeviceContext) raises:
        # the recv_buf ptrs and recv_count ptrs need to be passed in a InlinedArray
        var combine_recv_buf_ptrs = InlineArray[
            UnsafePointer[UInt8, MutAnyOrigin], 1
        ](fill={})
        var combine_recv_count_ptrs = InlineArray[
            UnsafePointer[UInt64, MutAnyOrigin], 1
        ](fill={})
        combine_recv_buf_ptrs[0] = send_buf
        combine_recv_count_ptrs[0] = recv_count

        ctx.enqueue_function(
            func_combine_async,
            output_tensor,
            src_token_info_tensor,
            recv_buf,
            combine_recv_buf_ptrs,
            combine_recv_count_ptrs,
            EPLocalSyncCounters[n_experts](atomic_counter.unsafe_ptr()),
            Int32(my_rank),
            OptionalReg[
                LayoutTensor[input_type, Layout.row_major[2](), MutAnyOrigin]
            ](),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_combine_async_wait(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func_combine_async_wait,
            output_2_tensor,
            send_buf,
            recv_count,
            EPLocalSyncCounters[n_experts](atomic_counter.unsafe_ptr()),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_e2e(ctx: DeviceContext) raises:
        run_combine_async(ctx)
        run_combine_async_wait(ctx)

    for i in range(num_iters):
        # Initialize the topk ids and input tokens using fixed seed,
        # so that we can reproduce the results later on other ranks.
        seed(Int(my_rank) + i * n_ranks)
        randint(host_topk_ids, n_tokens_per_rank * top_k, 0, n_experts - 1)
        legalize_topk_ids[n_experts, top_k](host_topk_ids, n_tokens_per_rank)

        seed(Int(my_rank) + i * n_ranks)
        randn(host_input_tokens, n_tokens_per_rank * hidden_size)

        ctx.enqueue_copy(device_topk_buf, host_topk_ids)
        ctx.enqueue_copy(device_input_buf, host_input_tokens)

        # warm-up
        shmem_barrier_all_on_stream(ctx.stream())
        run_full_dispatch(ctx)
        run_e2e(ctx)

        shmem_barrier_all_on_stream(ctx.stream())

        var new_value: Float64

        # First, bench kernel overhead
        run_full_dispatch(ctx)
        new_value = ctx.execution_time[run_combine_async](1) * 1e-3
        welford_update(
            combine_async_stat_m, combine_async_stat_m2, i + 1, new_value
        )

        # sleep 10 ms to make sure transfer is finished
        time.sleep(1e-2)

        new_value = ctx.execution_time[run_combine_async_wait](1) * 1e-3
        welford_update(
            combine_wait_stat_m, combine_wait_stat_m2, i + 1, new_value
        )

        # run one more time to measure bandwidth
        shmem_barrier_all_on_stream(ctx.stream())
        run_full_dispatch(ctx)
        new_value = ctx.execution_time[run_e2e](1) * 1e-3
        welford_update(e2e_stat_m, e2e_stat_m2, i + 1, new_value)
        # this time we do the clean up after we verify the results

        if not is_benchmark():
            var host_output_2 = alloc[Scalar[input_type]](
                n_tokens_per_rank * top_k * hidden_size
            )
            ctx.enqueue_copy(host_output_2, device_output_2_buf)

            ctx.synchronize()

            # Check the results
            for token_idx in range(n_tokens_per_rank):
                var ref_token = host_input_tokens + token_idx * hidden_size
                for topk_idx in range(top_k):
                    var received_token = (
                        host_output_2
                        + token_idx * top_k * hidden_size
                        + topk_idx * hidden_size
                    )
                    for i in range(hidden_size):
                        assert_equal(
                            received_token[i],
                            ref_token[i],
                            String(token_idx)
                            + ", "
                            + String(topk_idx)
                            + ", "
                            + String(i),
                        )

    _printf[
        "Rank #%d:  combine_async latency: %4.2fus ± %1.2fus  combine_wait"
        " latency: %4.2fus ± %1.2fus  E2E latency: %4.2fus ± %1.2fus\n"
    ](
        my_rank,
        combine_async_stat_m,
        sqrt(combine_async_stat_m2 / num_iters),
        combine_wait_stat_m,
        sqrt(combine_wait_stat_m2 / num_iters),
        e2e_stat_m,
        sqrt(e2e_stat_m2 / num_iters),
    )

    shmem_free(send_buf)
    shmem_free(recv_buf)
    shmem_free(recv_count)


def main():
    comptime test_gpu_counts = (8,)

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        with SHMEMContext() as shmem_ctx:
            var mype_node = shmem_team_my_pe(SHMEM_TEAM_NODE)
            test_combine[
                hidden_size=7168,
                top_k=8,
                n_experts = min(num_gpus * 32, 256),
                n_ranks=num_gpus,
                n_tokens_per_rank=128,
            ](shmem_ctx.get_device_context(), Int(mype_node))
