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

# RUN: ./bazelw build @nvshmem_prebuilt//:device
# RUN: BITCODE_PATH=$(./bazelw cquery '@nvshmem_prebuilt//:device' --output=files 2>/dev/null | head -1)
# RUN: mojo build --bitcode-libs $BITCODE_PATH  <path_to>/modular/max/kernels/benchmarks/gpu/bench_ep_dispatch.mojo -o ./test
# RUN: %mpirun-gpu-per-process %t
#
# Alternatively, run manually with:
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# br --run_under="mpirun -n $NUM_GPUS --allow-run-as-root --bind-to none" //max/kernels/benchmarks:gpu/bench_ep_dispatch

from collections import OptionalReg

from random import randint, randn, seed
from sys import align_of, env_get_int, env_get_dtype, simd_width_of, size_of

from benchmark import Bench, Bencher, BenchId, BenchMetric, ThroughputMeasure
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from memory import UnsafePointer
from shmem import *
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    TokenFormat,
    dispatch_wait_kernel,
    dispatch_async_kernel,
    combine_wait_kernel,
    combine_async_kernel,
)
from shmem._mpi import MPI_Finalize

from utils import IndexList


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


fn bench_dispatch[
    token_dtype: DType,
    scales_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
](ctx: DeviceContext, mut b: Bench, my_rank: Int) raises:
    comptime input_type = token_dtype
    comptime combine_msg_bytes = size_of[input_type]() * hidden_size

    comptime group_size = 128
    comptime gpu_target = get_gpu_target()
    comptime gpu_simd_width = simd_width_of[DType.uint8, target=gpu_target]()
    comptime gpu_alignment = align_of[
        SIMD[DType.uint8, gpu_simd_width], target=gpu_target
    ]()

    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_tokens = n_experts * n_tokens_per_rank

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
    var device_output_scales_buf = ctx.enqueue_create_buffer[scales_dtype](
        max_recv_tokens * hidden_size // group_size
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
    var device_output_2_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * top_k * hidden_size
    )

    comptime topk_ids_layout = Layout.row_major(UNKNOWN_VALUE, top_k)
    comptime input_tokens_layout = Layout.row_major(UNKNOWN_VALUE, hidden_size)
    comptime output_layout = Layout.row_major(
        n_tokens_per_rank * n_ranks * n_local_experts, hidden_size
    )

    comptime output_scales_layout = Layout.row_major(
        hidden_size // group_size, max_recv_tokens
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

    var output_scales_tensor = LayoutTensor[scales_dtype, output_scales_layout](
        device_output_scales_buf,
        RuntimeLayout[output_scales_layout].row_major(
            IndexList[2](hidden_size // group_size, max_recv_tokens)
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

    comptime hw_info = ctx.default_device_info

    # Initialize the topk ids and input tokens using fixed seed,
    # so that we can reproduce the results later on other ranks.
    seed(Int(my_rank) * n_ranks)
    randint(host_topk_ids, n_tokens_per_rank * top_k, 0, n_experts - 1)

    # The topk ids for a token is the expert id it needs to be sent to.
    # Since a token won't be sent to the same expert multiple times, we
    # need to legalize the topk ids to make sure they are unique for
    # each token.
    legalize_topk_ids[n_experts, top_k](host_topk_ids, n_tokens_per_rank)

    seed(Int(my_rank) * n_ranks)
    randn(host_input_tokens, n_tokens_per_rank * hidden_size)

    ctx.enqueue_copy(device_topk_buf, host_topk_ids)
    ctx.enqueue_copy(device_input_buf, host_input_tokens)

    @always_inline
    @parameter
    fn clean_up(ctx: DeviceContext) raises:
        ctx.enqueue_memset(atomic_counter, Int32(0))

    @always_inline
    @parameter
    fn setup_and_run_benchmark[
        TokenFmtType: TokenFormat,
        FormatHandlerType: TokenFormat,
        ThroughputDtype: DType,
    ](
        ctx: DeviceContext,
        mut b: Bench,
        format_handler: FormatHandlerType,
        throughput_dtype: DType,
    ) raises:
        var msg_bytes = TokenFmtType.msg_size()
        var send_buf = shmem_malloc[DType.uint8](
            UInt(n_tokens_per_rank * msg_bytes)
        )
        var recv_buf = shmem_malloc[DType.uint8](
            UInt(n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes)
        )

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
            TokenFmtType,
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
            FormatHandlerType,
        ]

        var func_dispatch_wait = ctx.compile_function_experimental[
            dispatch_wait
        ]()

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
        var func_combine_async = ctx.compile_function_experimental[
            combine_async
        ]()
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

        @always_inline
        @parameter
        fn run_dispatch_async(ctx: DeviceContext) raises:
            # the recv_buf ptrs and recv_count ptrs need to be passed in a InlinedArray
            var recv_buf_ptrs = InlineArray[
                UnsafePointer[UInt8, MutAnyOrigin], 1
            ](fill={})
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

        @always_inline
        @parameter
        fn run_dispatch_async_wait(ctx: DeviceContext) raises:
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
                    LayoutTensor[
                        DType.bfloat16, Layout.row_major[2](), ImmutAnyOrigin
                    ]
                ](),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

        @always_inline
        @parameter
        fn run_dispatch_async_e2e(ctx: DeviceContext) raises:
            run_dispatch_async(ctx)
            run_dispatch_async_wait(ctx)

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
                    LayoutTensor[
                        input_type, Layout.row_major[2](), MutAnyOrigin
                    ]
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
        fn run_combine_async_e2e(ctx: DeviceContext) raises:
            run_combine_async(ctx)
            run_combine_async_wait(ctx)

        shmem_barrier_all_on_stream(ctx.stream())

        @always_inline
        @parameter
        fn run_dispatch_async_func() raises:
            run_dispatch_async_e2e(ctx)

        @always_inline
        @parameter
        fn run_combine_async_func() raises:
            run_combine_async_e2e(ctx)

        @always_inline
        @parameter
        fn run_clean_up_func() raises:
            clean_up(ctx)

        @parameter
        @always_inline
        fn bench_dispatch_func(mut b: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext) raises:
                run_dispatch_async_func()

            b.iter_custom[kernel_launch](ctx)

        @parameter
        @always_inline
        fn bench_combine_func(mut b: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext) raises:
                run_combine_async_func()

            b.iter_custom[kernel_launch](ctx)

        @parameter
        @always_inline
        fn bench_clean_up_func(mut b: Bencher):
            @parameter
            @always_inline
            fn kernel_launch(ctx: DeviceContext) raises:
                run_clean_up_func()

            b.iter_custom[kernel_launch](ctx)

        var input_id_parts = String(
            "n_tokens_per_rank=",
            n_tokens_per_rank,
            " top_k=",
            top_k,
            " hidden_size=",
            hidden_size,
            " n_experts=",
            n_experts,
            " n_ranks=",
            n_ranks,
            " my_rank=",
            my_rank,
            " token_dtype=",
            String(throughput_dtype),
        )

        b.bench_function[bench_dispatch_func](
            BenchId("ep_dispatch", input_id=input_id_parts),
            [
                ThroughputMeasure(
                    BenchMetric.bytes,
                    size_of[ThroughputDtype]()
                    * n_tokens_per_rank
                    * hidden_size,
                )
            ],
            fixed_iterations=1,
        )
        b.bench_function[bench_combine_func](
            BenchId("ep_combine", input_id=input_id_parts),
            [
                ThroughputMeasure(
                    BenchMetric.bytes,
                    size_of[ThroughputDtype]()
                    * n_tokens_per_rank
                    * hidden_size,
                )
            ],
            fixed_iterations=1,
        )
        b.bench_function[bench_clean_up_func](
            BenchId("ep_clean_up", input_id=input_id_parts),
            [
                ThroughputMeasure(
                    BenchMetric.bytes,
                    size_of[ThroughputDtype]()
                    * n_tokens_per_rank
                    * hidden_size,
                )
            ],
            fixed_iterations=1,
        )

        shmem_free(send_buf)
        shmem_free(recv_buf)

    comptime if token_dtype == DType.bfloat16:
        comptime token_fmt_type = BF16TokenFormat[
            output_layout = Layout(), hidden_size, top_k, gpu_alignment
        ]

        comptime msg_bytes = token_fmt_type.msg_size()

        var send_buf = shmem_malloc[DType.uint8](
            UInt(n_tokens_per_rank * msg_bytes)
        )
        var recv_buf = shmem_malloc[DType.uint8](
            UInt(n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes)
        )

        var format_handler = BF16TokenFormat[hidden_size, top_k, gpu_alignment](
            output_tensor.bitcast[DType.bfloat16]().as_any_origin()
        )

        setup_and_run_benchmark[
            token_fmt_type,
            type_of(format_handler),
            token_dtype,
        ](
            ctx,
            b,
            format_handler,
            token_dtype,
        )

    else:
        comptime token_fmt_type = BlockwiseFP8TokenFormat[
            fp8_dtype=token_dtype,
            scales_dtype=scales_dtype,
            output_layout = Layout(),
            scales_layout = Layout(),
            hidden_size,
            top_k,
            gpu_alignment,
        ]

        var format_handler = BlockwiseFP8TokenFormat[
            hidden_size, top_k, gpu_alignment
        ](
            output_tensor.bitcast[DType.float8_e4m3fn]().as_any_origin(),
            output_scales_tensor.as_any_origin(),
        )

        setup_and_run_benchmark[
            token_fmt_type,
            type_of(format_handler),
            token_dtype,
        ](
            ctx,
            b,
            format_handler,
            token_dtype,
        )

    shmem_free(recv_count)


def main():
    comptime hidden_size = env_get_int["hidden_size", 3584]()
    comptime top_k = env_get_int["top_k", 8]()
    comptime n_experts = env_get_int["n_experts", 256]()
    comptime n_ranks = env_get_int["n_ranks", 8]()
    comptime n_tokens_per_rank = env_get_int["n_tokens_per_rank", 128]()
    comptime num_gpus = env_get_int["num_gpus", 8]()
    comptime token_dtype = env_get_dtype["token_dtype", DType.float8_e4m3fn]()
    comptime scales_dtype = env_get_dtype["scales_dtype", DType.float32]()

    var m = Bench()
    var bencher_rank = m.check_mpirun()
    with SHMEMContext() as shmem_ctx:
        var mype_node = shmem_team_my_pe(SHMEM_TEAM_NODE)
        if bencher_rank != Int(mype_node):
            raise Error("bencher_rank does not match mype_node")

        bench_dispatch[
            token_dtype=token_dtype,
            scales_dtype=scales_dtype,
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts = min(num_gpus * 32, n_experts),
            n_ranks=n_ranks,
            n_tokens_per_rank=n_tokens_per_rank,
        ](shmem_ctx.get_device_context(), m, Int(mype_node))

    m.dump_report()
