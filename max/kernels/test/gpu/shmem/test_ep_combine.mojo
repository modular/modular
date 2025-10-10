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
# REQUIRES: NVIDIA-GPU

# RUN: NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# RUN: %mojo-build %s -o %t
# RUN: %mpirun -n $NUM_GPUS %t

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
from shmem import *
from shmem._mpi import MPI_Finalize
from shmem.ep_comm import (
    BF16TokenFormat,
    combine_cb_kernel,
    combine_kernel,
    dispatch_cb_kernel,
    dispatch_kernel,
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
](topk_ids: UnsafePointer[Int32], n_tokens: Int):
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
    alias input_type = DType.bfloat16
    alias gpu_target = get_gpu_target()
    alias gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
    alias gpu_alignment = align_of[
        SIMD[DType.uint8, gpu_simd_width], target=gpu_target
    ]()
    alias token_fmt_type = BF16TokenFormat[
        output_layout = Layout(), hidden_size, top_k, gpu_alignment
    ]
    alias msg_bytes = token_fmt_type.msg_size()
    alias combine_msg_bytes = size_of[input_type]() * hidden_size
    alias n_local_experts = n_experts // n_ranks

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
    var atomic_counter = ctx.enqueue_create_buffer[DType.int32](2 * n_experts)

    ctx.enqueue_memset(recv_count_buf, UInt64.MAX_FINITE)
    ctx.enqueue_memset(atomic_counter, Int32(0))

    var host_topk_ids = UnsafePointer[Int32].alloc(n_tokens_per_rank * top_k)
    var host_input_tokens = UnsafePointer[Scalar[input_type]].alloc(
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
    var device_row_offsets_buf = ctx.enqueue_create_buffer[DType.int32](
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

    alias topk_ids_layout = Layout.row_major(UNKNOWN_VALUE, top_k)
    alias input_tokens_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    alias output_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, hidden_size
    )
    alias row_offsets_layout = Layout.row_major(n_local_experts + 1)
    alias expert_ids_layout = Layout.row_major(n_local_experts)
    alias src_token_info_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, 2
    )
    alias output_2_layout = Layout.row_major(UNKNOWN_VALUE, top_k, hidden_size)

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
    var row_offsets_tensor = LayoutTensor[DType.int32, row_offsets_layout](
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
        output_tensor.origin_cast[True, MutableAnyOrigin]()
    )

    alias hw_info = ctx.default_device_info

    alias dispatch = dispatch_kernel[
        input_type,
        hw_info.max_thread_block_size,
        input_tokens_layout,
        topk_ids_layout,
        hw_info.sm_count,
        n_experts // (hw_info.max_thread_block_size // hw_info.warp_size),
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        token_fmt_type,
    ]
    var func = ctx.compile_function[dispatch]()
    shmem_module_init(func)

    alias dispatch_cb = dispatch_cb_kernel[
        hw_info.max_thread_block_size,
        output_layout,
        row_offsets_layout,
        expert_ids_layout,
        src_token_info_layout,
        hw_info.sm_count,
        1,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        __type_of(format_handler),
    ]
    var func_cb = ctx.compile_function[dispatch_cb]()

    alias combine = combine_kernel[
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
    ]
    var func_combine = ctx.compile_function[combine]()
    shmem_module_init(func_combine)

    alias combine_cb = combine_cb_kernel[
        input_type,
        hw_info.max_thread_block_size,
        output_2_layout,
        hw_info.sm_count,
        1,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_bytes,
        n_tokens_per_rank,
    ]
    var func_combine_cb = ctx.compile_function[combine_cb]()

    var num_iters: Int = 100 if is_benchmark() or is_pressure_test() else 3
    var combine_stat_m: Float64 = 0
    var combine_stat_m2: Float64 = 0
    var combine_cb_stat_m: Float64 = 0
    var combine_cb_stat_m2: Float64 = 0
    var e2e_stat_m: Float64 = 0
    var e2e_stat_m2: Float64 = 0

    @always_inline
    @parameter
    fn run_full_dispatch(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func,
            input_tokens_tensor,
            topk_ids_tensor,
            send_buf,
            recv_buf,
            recv_count,
            atomic_counter.unsafe_ptr(),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )
        ctx.enqueue_function(
            func_cb,
            format_handler,
            row_offsets_tensor,
            expert_ids_tensor,
            src_token_info_tensor,
            recv_buf,
            recv_count,
            atomic_counter.unsafe_ptr(),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )
        shmem_barrier_all_on_stream(ctx.stream())

    @always_inline
    @parameter
    fn run_combine(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func_combine,
            output_tensor,
            src_token_info_tensor,
            recv_buf,
            send_buf,
            recv_count,
            atomic_counter.unsafe_ptr(),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_combine_cb(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func_combine_cb,
            output_2_tensor,
            send_buf,
            recv_count,
            atomic_counter.unsafe_ptr(),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_e2e(ctx: DeviceContext) raises:
        run_combine(ctx)
        run_combine_cb(ctx)

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
        new_value = ctx.execution_time[run_combine](1) * 1e-3
        welford_update(combine_stat_m, combine_stat_m2, i + 1, new_value)

        # sleep 10 ms to make sure transfer is finished
        time.sleep(1e-2)

        new_value = ctx.execution_time[run_combine_cb](1) * 1e-3
        welford_update(combine_cb_stat_m, combine_cb_stat_m2, i + 1, new_value)

        # run one more time to measure bandwidth
        shmem_barrier_all_on_stream(ctx.stream())
        run_full_dispatch(ctx)
        new_value = ctx.execution_time[run_e2e](1) * 1e-3
        welford_update(e2e_stat_m, e2e_stat_m2, i + 1, new_value)
        # this time we do the clean up after we verify the results

        if not is_benchmark():
            var host_output_2 = UnsafePointer[Scalar[input_type]].alloc(
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
        "Rank #%d:  Combine latency: %4.2fus ± %1.2fus  Combine_cb latency:"
        " %4.2fus ± %1.2fus  E2E latency: %4.2fus ± %1.2fus\n"
    ](
        my_rank,
        combine_stat_m,
        sqrt(combine_stat_m2 / num_iters),
        combine_cb_stat_m,
        sqrt(combine_cb_stat_m2 / num_iters),
        e2e_stat_m,
        sqrt(e2e_stat_m2 / num_iters),
    )

    shmem_free(send_buf)
    shmem_free(recv_buf)
    shmem_free(recv_count)


def main():
    alias test_gpu_counts = (8,)

    @parameter
    for gpu_idx in range(len(test_gpu_counts)):
        alias num_gpus = test_gpu_counts[gpu_idx]
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
