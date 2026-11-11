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

# Checks the expert-load tally that dispatch wait keeps for EPLB: each step
# adds every local expert's routed-token count to the row of the step's layer,
# the totals run across steps without a reset, and the other layer's row is
# left alone.

from std.random import randint, seed

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx
from layout.layout import row_major
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from shmem import *
from shmem.ep_comm import (
    EP_NO_INPUT_SCALES,
    BF16TokenFormat,
    EPLocalSyncCounters,
    ExpertLoadTally,
    dispatch_wait_kernel,
    dispatch_async_kernel,
)
from std.testing import assert_equal


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: Pointer[mut=True, Int32, _], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids.unsafe_offset(tok_id * top_k)

        def is_duplicate() {imm} -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if (
                        topk_ids_for_token[unsafe_offset=i]
                        == topk_ids_for_token[unsafe_offset=j]
                    ):
                        return i
            return -1

        var duplicate_idx = is_duplicate()
        while duplicate_idx != -1:
            randint(
                topk_ids_for_token.unsafe_offset(duplicate_idx),
                1,
                0,
                n_experts - 1,
            )
            duplicate_idx = is_duplicate()


def fill_topk_ids[
    n_experts: Int, top_k: Int, n_tokens_per_rank: Int
](topk_ids: Pointer[mut=True, Int32, _], rank: Int, step: Int, n_ranks: Int):
    seed(rank + step * n_ranks)
    randint(topk_ids, n_tokens_per_rank * top_k, 0, n_experts - 1)
    legalize_topk_ids[n_experts, top_k](topk_ids, n_tokens_per_rank)


def test_expert_load[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
    n_layers: Int,
    n_steps: Int,
](ctx: DeviceContext, my_rank: Int) raises:
    comptime input_type = DType.bfloat16
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank
    # Above 2^32, so a 32-bit tally or a reset would show.
    comptime tally_start = UInt64(1) << 33

    comptime output_tt_layout = row_major(
        (Idx[max_recv_num_tokens], Idx[hidden_size])
    )
    comptime token_fmt_type = BF16TokenFormat[
        output_layout=type_of(output_tt_layout), hidden_size, top_k
    ]
    comptime msg_bytes = token_fmt_type.msg_size()

    if my_rank == 0:
        print(
            "Running ep_dispatch expert-load test: n_experts:",
            n_experts,
            "n_ranks:",
            n_ranks,
            "n_layers:",
            n_layers,
            "n_steps:",
            n_steps,
        )

    var send_buf = shmem_malloc[.uint8](n_tokens_per_rank * msg_bytes)
    var recv_buf = shmem_malloc[.uint8](
        n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes
    )
    var recv_count = shmem_malloc[.uint64](n_local_experts * n_ranks)
    var recv_count_buf = DeviceBuffer(
        ctx, recv_count, n_local_experts * n_ranks, owning=False
    )
    var atomic_counter = ctx.enqueue_create_buffer[.int32](
        EPLocalSyncCounters[n_experts].total_size()
    )
    var tally_buf = ctx.enqueue_create_buffer[.uint64](
        n_layers * n_local_experts
    )
    var layer_idx_buf = ctx.enqueue_create_buffer[.int32](1)

    ctx.enqueue_memset(recv_count_buf, UInt64.MAX_FINITE)
    ctx.enqueue_memset(atomic_counter, Int32(0))
    ctx.enqueue_memset(tally_buf, tally_start)

    var host_topk_ids = unsafe_alloc[Int32](n_tokens_per_rank * top_k)
    var all_ranks_topk_ids = unsafe_alloc[Int32](
        n_tokens_per_rank * n_ranks * top_k
    )
    var host_layer_idx = unsafe_alloc[Int32](1)
    var host_tally = unsafe_alloc[UInt64](n_layers * n_local_experts)
    var host_row_offsets = unsafe_alloc[UInt32](n_local_experts + 1)
    var expected = unsafe_alloc[UInt64](n_layers * n_local_experts)
    for j in range(n_layers * n_local_experts):
        expected[unsafe_offset=j] = tally_start

    var device_topk_buf = ctx.enqueue_create_buffer[.int32](
        n_tokens_per_rank * top_k
    )
    var device_input_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * hidden_size
    )
    var device_output_buf = ctx.enqueue_create_buffer[input_type](
        max_recv_num_tokens * hidden_size
    )
    var device_row_offsets_buf = ctx.enqueue_create_buffer[.uint32](
        n_local_experts + 1
    )
    var device_expert_ids_buf = ctx.enqueue_create_buffer[.int32](
        n_local_experts
    )
    var device_src_token_info_buf = ctx.enqueue_create_buffer[.int32](
        max_recv_num_tokens * 2
    )
    ctx.enqueue_memset(device_input_buf, Scalar[input_type](0))

    var topk_ids_tensor = TileTensor[origin=ImmutAnyOrigin](
        device_topk_buf, row_major(n_tokens_per_rank, Idx[top_k])
    )
    var input_tokens_tensor = TileTensor[origin=ImmutAnyOrigin](
        device_input_buf,
        row_major(n_tokens_per_rank, Idx[hidden_size]),
    )
    var output_tensor = TileTensor[origin=MutAnyOrigin](
        device_output_buf,
        row_major(Idx[max_recv_num_tokens], Idx[hidden_size]),
    )
    var row_offsets_tensor = TileTensor[origin=MutAnyOrigin](
        device_row_offsets_buf, row_major[n_local_experts + 1]()
    )
    var expert_ids_tensor = TileTensor[origin=MutAnyOrigin](
        device_expert_ids_buf, row_major[n_local_experts]()
    )
    var src_token_info_tensor = TileTensor[origin=MutAnyOrigin](
        device_src_token_info_buf,
        row_major(Idx[max_recv_num_tokens], Idx[2]),
    )

    var format_handler = token_fmt_type(output_tensor)
    var expert_load = ExpertLoadTally(
        MutPointer[UInt64, MutUntrackedOrigin](
            unsafe_from_address=Int(tally_buf.unsafe_ptr())
        ),
        ImmPointer[Int32, ImmUntrackedOrigin](
            unsafe_from_address=Int(layer_idx_buf.unsafe_ptr())
        ),
    )

    comptime hw_info = ctx.default_device_info

    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        input_tokens_tensor.LayoutType,
        topk_ids_tensor.LayoutType,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        1,  # p2p world size
        token_fmt_type,
        InputScalesFnType=type_of(EP_NO_INPUT_SCALES),
    ]
    var func = ctx.compile_function[dispatch_async]()
    shmem_module_init(func)

    comptime dispatch_wait = dispatch_wait_kernel[
        hw_info.max_thread_block_size,
        row_offsets_tensor.LayoutType,
        expert_ids_tensor.LayoutType,
        src_token_info_tensor.LayoutType,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        type_of(format_handler),
        ExpertLoadT=ExpertLoadTally,
    ]
    var func_wait = ctx.compile_function[dispatch_wait]()

    var recv_buf_ptrs: Array[Pointer[UInt8, MutAnyOrigin], 1] = [
        recv_buf.as_unsafe_any_origin()
    ]
    var recv_count_ptrs: Array[Pointer[UInt64, MutAnyOrigin], 1] = [
        recv_count.as_unsafe_any_origin()
    ]

    for step in range(n_steps):
        var layer = step % n_layers
        fill_topk_ids[n_experts, top_k, n_tokens_per_rank](
            host_topk_ids, my_rank, step, n_ranks
        )
        host_layer_idx[unsafe_offset=0] = Int32(layer)
        ctx.enqueue_copy(device_topk_buf, host_topk_ids)
        ctx.enqueue_copy(layer_idx_buf, host_layer_idx)

        shmem_barrier_all_on_stream(ctx.stream())
        func._call_with_pack_checked(
            ctx,
            input_tokens_tensor,
            topk_ids_tensor,
            send_buf,
            recv_buf_ptrs,
            recv_count_ptrs,
            EPLocalSyncCounters[n_experts](atomic_counter),
            Int32(my_rank),
            host_args=(EP_NO_INPUT_SCALES,),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )
        ctx.enqueue_function(
            func_wait,
            format_handler,
            row_offsets_tensor,
            expert_ids_tensor,
            src_token_info_tensor,
            recv_buf,
            recv_count,
            EPLocalSyncCounters[n_experts](atomic_counter),
            Int32(my_rank),
            expert_load,
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )
        ctx.enqueue_copy(host_tally, tally_buf)
        ctx.enqueue_copy(host_row_offsets, device_row_offsets_buf)
        ctx.enqueue_memset(atomic_counter, Int32(0))
        ctx.synchronize()

        for rank in range(n_ranks):
            fill_topk_ids[n_experts, top_k, n_tokens_per_rank](
                all_ranks_topk_ids.unsafe_offset(
                    rank * n_tokens_per_rank * top_k
                ),
                rank,
                step,
                n_ranks,
            )
        var first_expert = n_local_experts * my_rank
        for e in range(n_local_experts):
            var step_count = 0
            for t in range(n_tokens_per_rank * n_ranks * top_k):
                if Int(all_ranks_topk_ids[unsafe_offset=t]) == first_expert + e:
                    step_count += 1
            expected[unsafe_offset=layer * n_local_experts + e] += UInt64(
                step_count
            )
            # BF16 adds no grouped-matmul padding, so the offsets carry the
            # same per-expert counts the tally received.
            assert_equal(
                step_count,
                Int(
                    host_row_offsets[unsafe_offset=e + 1]
                    - host_row_offsets[unsafe_offset=e]
                ),
                String("row offsets, step ", step, " expert ", e),
            )

        for j in range(n_layers * n_local_experts):
            assert_equal(
                host_tally[unsafe_offset=j],
                expected[unsafe_offset=j],
                String(
                    "tally, step ",
                    step,
                    " layer ",
                    j // n_local_experts,
                    " expert ",
                    j % n_local_experts,
                ),
            )

    if my_rank == 0:
        print("expert-load tally matches after", n_steps, "steps")

    shmem_free(send_buf)
    shmem_free(recv_buf)
    shmem_free(recv_count)


def main() raises:
    comptime test_gpu_counts = (2, 4, 8)

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        with SHMEMContext() as shmem_ctx:
            var mype_node = shmem_team_my_pe(SHMEM_TEAM_NODE)
            test_expert_load[
                hidden_size=3584,
                top_k=8,
                n_experts=min(num_gpus * 32, 256),
                n_ranks=num_gpus,
                n_tokens_per_rank=128,
                n_layers=2,
                n_steps=5,
            ](shmem_ctx.get_device_context(), Int(mype_node))
