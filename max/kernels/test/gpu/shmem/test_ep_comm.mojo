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

from gpu.host import DeviceContext, DeviceBuffer
from io.io import _printf
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from math import sqrt
from shmem import *
from testing import assert_equal
from pathlib import Path
from os.path import dirname
from random import randint, randn, seed
from sys import argv, size_of
from sys.param_env import env_get_string
from utils import IndexList

from shmem.ep_comm import dispatch_kernel, dispatch_cb_kernel

import time


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


fn test_dispatch[
    input_type: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
](ctx: DeviceContext, my_rank: Int) raises:
    alias msg_bytes = size_of[input_type]() * hidden_size + 4 * size_of[Int32]()
    alias n_local_experts = n_experts // n_ranks

    if my_rank == 0:
        print(
            "Running ep_dispatch test: input_type:",
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

    var send_buf = shmem_malloc[DType.uint8](n_tokens_per_rank * msg_bytes)
    var recv_buf = shmem_malloc[DType.uint8](
        n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes
    )
    var recv_count = shmem_malloc[DType.uint64](n_local_experts * n_ranks)
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

    alias topk_ids_layout = Layout.row_major(UNKNOWN_VALUE, top_k)
    alias input_tokens_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    alias output_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, hidden_size
    )
    alias row_offsets_layout = Layout.row_major(n_local_experts + 1)
    alias expert_ids_layout = Layout.row_major(n_local_experts)
    alias src_token_info_layout = Layout.row_major(
        2, n_tokens_per_rank * n_ranks * n_local_experts
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
            IndexList[2](2, n_tokens_per_rank * n_ranks * n_local_experts)
        ),
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
        msg_bytes,
        n_tokens_per_rank,
    ]

    var func = ctx.compile_function[dispatch]()
    shmem_module_init(func)

    alias dispatch_cb = dispatch_cb_kernel[
        input_type,
        hw_info.max_thread_block_size,
        output_layout,
        row_offsets_layout,
        expert_ids_layout,
        src_token_info_layout,
        hw_info.sm_count,
        1,
        n_experts,
        n_ranks,
        msg_bytes,
        n_tokens_per_rank,
    ]

    var func_cb = ctx.compile_function[dispatch_cb]()

    var num_iters: Int = 100 if is_benchmark() or is_pressure_test() else 3
    var dispatch_stat_m: Float64 = 0
    var dispatch_stat_m2: Float64 = 0
    var dispatch_cb_stat_m: Float64 = 0
    var dispatch_cb_stat_m2: Float64 = 0
    var e2e_stat_m: Float64 = 0
    var e2e_stat_m2: Float64 = 0

    @always_inline
    @parameter
    fn run_dispatch(ctx: DeviceContext) raises:
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

    @always_inline
    @parameter
    fn run_dispatch_cb(ctx: DeviceContext) raises:
        ctx.enqueue_function(
            func_cb,
            output_tensor,
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

    @always_inline
    @parameter
    fn run_e2e(ctx: DeviceContext) raises:
        run_dispatch(ctx)
        run_dispatch_cb(ctx)

    for i in range(num_iters):
        # Initialize the topk ids and input tokens using fixed seed,
        # so that we can reproduce the results later on other ranks.
        seed(Int(my_rank) + i * n_ranks)
        randint(host_topk_ids, n_tokens_per_rank * top_k, 0, n_experts - 1)

        seed(Int(my_rank) + i * n_ranks)
        randn(host_input_tokens, n_tokens_per_rank * hidden_size)

        ctx.enqueue_copy(device_topk_buf, host_topk_ids)
        ctx.enqueue_copy(device_input_buf, host_input_tokens)

        shmem_barrier_all_on_stream(ctx.stream())

        var new_value: Float64

        # First, bench kernel overhead
        new_value = ctx.execution_time[run_dispatch](1) * 1e-3
        welford_update(dispatch_stat_m, dispatch_stat_m2, i + 1, new_value)

        # sleep 10 ms to make sure transfer is finished
        time.sleep(1e-2)

        new_value = ctx.execution_time[run_dispatch_cb](1) * 1e-3
        welford_update(
            dispatch_cb_stat_m, dispatch_cb_stat_m2, i + 1, new_value
        )

        # run one more time to measure bandwidth
        shmem_barrier_all_on_stream(ctx.stream())
        new_value = ctx.execution_time[run_e2e](1) * 1e-3
        welford_update(e2e_stat_m, e2e_stat_m2, i + 1, new_value)

        if not is_benchmark():
            var host_output = UnsafePointer[Scalar[input_type]].alloc(
                n_tokens_per_rank * n_ranks * n_local_experts * hidden_size
            )
            ctx.enqueue_copy(host_output, device_output_buf)

            var host_row_offsets = UnsafePointer[Int32].alloc(
                n_local_experts + 1
            )
            ctx.enqueue_copy(host_row_offsets, device_row_offsets_buf)

            var host_expert_ids = UnsafePointer[Int32].alloc(n_tokens_per_rank)
            ctx.enqueue_copy(host_expert_ids, device_expert_ids_buf)

            var host_src_token_info = UnsafePointer[Int32].alloc(
                n_tokens_per_rank * n_ranks * n_local_experts * 2
            )
            ctx.enqueue_copy(host_src_token_info, device_src_token_info_buf)

            ctx.synchronize()

            # Check the results

            # Fisrt, reproduce the input tokens and topk ids
            var all_ranks_input_tokens = UnsafePointer[
                Scalar[input_type]
            ].alloc(n_tokens_per_rank * n_ranks * hidden_size)
            var all_ranks_topk_ids = UnsafePointer[Int32].alloc(
                n_tokens_per_rank * n_ranks * top_k
            )

            for rank in range(n_ranks):
                seed(Int(rank) + i * n_ranks)
                randn(
                    all_ranks_input_tokens
                    + rank * n_tokens_per_rank * hidden_size,
                    n_tokens_per_rank * hidden_size,
                )
                seed(Int(rank) + i * n_ranks)
                randint(
                    all_ranks_topk_ids + rank * n_tokens_per_rank * top_k,
                    n_tokens_per_rank * top_k,
                    0,
                    n_experts - 1,
                )

            # Check if we have received the correct number of tokens
            var expert_start_idx = n_local_experts * my_rank
            var expert_end_idx = expert_start_idx + n_local_experts
            var count = 0
            for i in range(n_tokens_per_rank * n_ranks * top_k):
                if (
                    expert_start_idx
                    <= Int(all_ranks_topk_ids[i])
                    < expert_end_idx
                ):
                    count += 1
            assert_equal(count, Int(host_row_offsets[n_local_experts]))

            # Then, check the output
            for expert_idx in range(n_local_experts):
                var curr_local_expert = host_expert_ids[expert_idx]
                var curr_expert = n_local_experts * my_rank + curr_local_expert

                for token_idx in range(
                    host_row_offsets[expert_idx],
                    host_row_offsets[expert_idx + 1],
                ):
                    var remote_loc = host_src_token_info[token_idx]
                    var remote_rank = host_src_token_info[
                        token_idx
                        + n_tokens_per_rank * n_ranks * n_local_experts
                    ]

                    # check if curr_expert is in remote rank's topk_ids
                    var remote_rank_top_k_ids = (
                        all_ranks_topk_ids
                        + remote_rank * n_tokens_per_rank * top_k
                    )

                    var found = False
                    for i in range(top_k):
                        if (
                            remote_rank_top_k_ids[remote_loc * top_k + i]
                            == curr_expert
                        ):
                            found = True
                            break
                    assert_equal(found, True)

                    # check if the received token matches the remote rank's token

                    var remote_rank_input_tokens = (
                        all_ranks_input_tokens
                        + remote_rank * n_tokens_per_rank * hidden_size
                    )
                    for i in range(hidden_size):
                        var remote_token_val = remote_rank_input_tokens[
                            remote_loc * hidden_size + i
                        ]
                        var curr_token_val = host_output[
                            token_idx * hidden_size + i
                        ]
                        assert_equal(
                            remote_token_val,
                            curr_token_val,
                            String(token_idx) + ", " + String(i),
                        )

    _printf[
        "Rank #%d:  Dispatch latency: %4.2fus ± %1.2fus  Dispatch_cb latency:"
        " %4.2fus ± %1.2fus  E2E latency: %4.2fus ± %1.2fus\n"
    ](
        my_rank,
        dispatch_stat_m,
        sqrt(dispatch_stat_m2 / num_iters),
        dispatch_cb_stat_m,
        sqrt(dispatch_cb_stat_m2 / num_iters),
        e2e_stat_m,
        sqrt(e2e_stat_m2 / num_iters),
    )

    shmem_free(send_buf)
    shmem_free(recv_buf)
    shmem_free(recv_count)


fn main() raises:
    alias test_gpu_counts = (2,)

    @parameter
    for gpu_idx in range(len(test_gpu_counts)):
        alias num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        shmem_init()
        var mype_node = shmem_team_my_pe(SHMEM_TEAM_NODE)

        with DeviceContext(device_id=Int(mype_node)) as ctx:
            test_dispatch[
                input_type = DType.bfloat16,
                hidden_size=3584,  # equivalent to send 7168 FP8s.
                top_k=8,
                n_experts = min(num_gpus * 32, 256),
                n_ranks=num_gpus,
                n_tokens_per_rank=128,
            ](ctx, Int(mype_node))

        shmem_finalize()
