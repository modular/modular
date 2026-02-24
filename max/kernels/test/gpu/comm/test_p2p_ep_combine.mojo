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

from collections import OptionalReg

from io.io import _printf
from random import randint, randn, seed
from sys import (
    align_of,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    simd_width_of,
    size_of,
)

from algorithm import sync_parallelize
from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchmarkInfo,
    BenchId,
    BenchMetric,
    Report,
    ThroughputMeasure,
)
from comm.sync import enable_p2p
from gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from layout import UNKNOWN_VALUE, Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from shmem.ep_comm import (
    BF16TokenFormat,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    combine_wait_kernel,
    combine_async_kernel,
    dispatch_wait_kernel,
    dispatch_async_kernel,
)
from testing import assert_equal
from utils import IndexList


fn legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: UnsafePointer[Int32, MutExternalOrigin], n_tokens: Int):
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
    n_slots: Int,
    n_tokens_per_rank: Int,
](list_of_ctx: List[DeviceContext]) raises:
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
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank

    comptime num_bytes = combine_msg_bytes * top_k * n_tokens_per_rank

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

    # fmt: off
    # Buffers for dispatch phase
    var dispatch_send_bufs_list = List[DeviceBuffer[DType.uint8]](capacity=n_ranks)
    var dispatch_recv_bufs_list = List[DeviceBuffer[DType.uint8]](capacity=n_ranks)
    var dispatch_recv_count_bufs_list = List[DeviceBuffer[DType.uint64]](capacity=n_ranks)

    # Buffers for combine phase
    var combine_send_bufs_list = List[DeviceBuffer[DType.uint8]](capacity=n_ranks)
    var combine_recv_bufs_list = List[DeviceBuffer[DType.uint8]](capacity=n_ranks)
    var combine_recv_count_bufs_list = List[DeviceBuffer[DType.uint64]](capacity=n_ranks)

    # Shared atomic counter buffer for dispatch and combine
    var atomic_counters_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)

    var host_topk_ids_list = InlineArray[UnsafePointer[Int32, MutExternalOrigin], n_ranks](fill={})
    var host_input_tokens_list = InlineArray[UnsafePointer[Scalar[input_type], MutExternalOrigin], n_ranks](fill={})

    var device_topk_bufs_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)
    var device_input_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_output_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_row_offsets_bufs_list = List[DeviceBuffer[DType.uint32]](capacity=n_ranks)
    var device_expert_ids_bufs_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)
    var device_src_token_info_bufs_list = List[DeviceBuffer[DType.int32]](capacity=n_ranks)

    # Output buffer for combine_wait
    var device_output_2_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)

    for i in range(n_ranks):
        var ctx = list_of_ctx[i]
        # Dispatch buffers
        dispatch_send_bufs_list.append(ctx.enqueue_create_buffer[DType.uint8](n_slots * n_tokens_per_rank * msg_bytes))
        dispatch_recv_bufs_list.append(ctx.enqueue_create_buffer[DType.uint8](n_slots * max_recv_num_tokens * msg_bytes))
        dispatch_recv_count_bufs_list.append(ctx.enqueue_create_buffer[DType.uint64](n_slots * n_experts))
        ctx.enqueue_memset(dispatch_recv_count_bufs_list[i], UInt64.MAX_FINITE)

        # Combine buffers
        combine_send_bufs_list.append(ctx.enqueue_create_buffer[DType.uint8](n_slots * max_recv_num_tokens * combine_msg_bytes))
        combine_recv_bufs_list.append(ctx.enqueue_create_buffer[DType.uint8](n_slots * n_tokens_per_rank * top_k * combine_msg_bytes))
        combine_recv_count_bufs_list.append(ctx.enqueue_create_buffer[DType.uint64](n_slots * n_experts))
        ctx.enqueue_memset(combine_recv_count_bufs_list[i], UInt64.MAX_FINITE)

        # Shared atomic counter
        atomic_counters_list.append(ctx.enqueue_create_buffer[DType.int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        ))
        ctx.enqueue_memset(atomic_counters_list[i], Int32(0))

        host_topk_ids_list[i] = alloc[Int32](n_slots * n_tokens_per_rank * top_k)
        host_input_tokens_list[i] = alloc[Scalar[input_type]](n_slots * n_tokens_per_rank * hidden_size)

        device_topk_bufs_list.append(ctx.enqueue_create_buffer[DType.int32](n_slots * n_tokens_per_rank * top_k))
        device_input_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * hidden_size))
        device_output_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * max_recv_num_tokens * hidden_size))
        device_row_offsets_bufs_list.append(ctx.enqueue_create_buffer[DType.uint32](n_slots * (n_local_experts + 1)))
        device_expert_ids_bufs_list.append(ctx.enqueue_create_buffer[DType.int32](n_slots * n_local_experts))
        device_src_token_info_bufs_list.append(ctx.enqueue_create_buffer[DType.int32](n_slots * max_recv_num_tokens * 2))

        device_output_2_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * top_k * hidden_size))
    # fmt: on

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

    # Initialize the inputs
    for dev_idx in range(n_ranks):
        var ctx = list_of_ctx[dev_idx]
        # Initialize the topk ids and input tokens using fixed seed,
        seed(dev_idx)
        randint(
            host_topk_ids_list[dev_idx],
            n_slots * n_tokens_per_rank * top_k,
            0,
            n_experts - 1,
        )

        # The topk ids for a token is the expert id it needs to be sent to.
        # Since a token won't be sent to the same expert multiple times, we
        # need to legalize the topk ids to make sure they are unique for
        # each token.
        legalize_topk_ids[n_experts, top_k](
            host_topk_ids_list[dev_idx], n_slots * n_tokens_per_rank
        )

        randn(
            host_input_tokens_list[dev_idx],
            n_slots * n_tokens_per_rank * hidden_size,
        )

        ctx.enqueue_copy(
            device_topk_bufs_list[dev_idx], host_topk_ids_list[dev_idx]
        )
        ctx.enqueue_copy(
            device_input_bufs_list[dev_idx], host_input_tokens_list[dev_idx]
        )

    # fmt: off
    # Dispatch buffers
    var dispatch_recv_bufs_inputs = InlineArray[InlineArray[UnsafePointer[UInt8, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)
    var dispatch_recv_count_bufs_inputs = InlineArray[InlineArray[UnsafePointer[UInt64, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)

    # Combine buffers
    var combine_recv_bufs_inputs = InlineArray[InlineArray[UnsafePointer[UInt8, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)
    var combine_recv_count_bufs_inputs = InlineArray[InlineArray[UnsafePointer[UInt64, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)

    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            dispatch_recv_bufs_inputs[slot_idx][dev_idx] = dispatch_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * msg_bytes
            dispatch_recv_count_bufs_inputs[slot_idx][dev_idx] = dispatch_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts
            combine_recv_bufs_inputs[slot_idx][dev_idx] = combine_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes
            combine_recv_count_bufs_inputs[slot_idx][dev_idx] = combine_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts

    # Dispatch helpers
    @always_inline
    @parameter
    fn get_dispatch_send_buf_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt8, MutExternalOrigin]) raises:
        return type_of(result)(dispatch_send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * msg_bytes)

    # Combine helpers
    @always_inline
    @parameter
    fn get_combine_send_buf_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt8, MutExternalOrigin]) raises:
        return type_of(result)(combine_send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * combine_msg_bytes)

    @always_inline
    @parameter
    fn get_combine_recv_buf_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt8, MutExternalOrigin]) raises:
        return type_of(result)(combine_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes)

    @always_inline
    @parameter
    fn get_combine_recv_count_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt64, MutExternalOrigin]) raises:
        return type_of(result)(combine_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts)

    @always_inline
    @parameter
    fn get_atomic_counters(dev_idx: Int, slot_idx: Int, out result: EPLocalSyncCounters[n_experts]) raises:
        return EPLocalSyncCounters[n_experts](atomic_counters_list[dev_idx].unsafe_ptr() + slot_idx * EPLocalSyncCounters[n_experts].total_size())

    @always_inline
    @parameter
    fn get_topk_ids_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, topk_ids_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(
            device_topk_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k,
            RuntimeLayout[topk_ids_layout].row_major(
                IndexList[2](n_tokens_per_rank, top_k)
            )
        )

    @always_inline
    @parameter
    fn get_input_tokens_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[input_type, input_tokens_layout, ImmutAnyOrigin]) raises:
        return type_of(result)(
            device_input_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * hidden_size,
            RuntimeLayout[input_tokens_layout].row_major(
                IndexList[2](n_tokens_per_rank, hidden_size)
            )
        )

    @always_inline
    @parameter
    fn get_output_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[input_type, output_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_output_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * hidden_size,
            RuntimeLayout[output_layout].row_major(
                IndexList[2](max_recv_num_tokens, hidden_size)
            )
        )

    @always_inline
    @parameter
    fn get_row_offsets_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.uint32, row_offsets_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_row_offsets_bufs_list[dev_idx].unsafe_ptr() + slot_idx * (n_local_experts + 1),
            RuntimeLayout[row_offsets_layout].row_major(
                IndexList[1](n_local_experts + 1)
            )
        )

    @always_inline
    @parameter
    fn get_expert_ids_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_expert_ids_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_local_experts,
            RuntimeLayout[expert_ids_layout].row_major(
                IndexList[1](n_local_experts)
            )
        )

    @always_inline
    @parameter
    fn get_src_token_info_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[DType.int32, src_token_info_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_src_token_info_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * 2,
            RuntimeLayout[src_token_info_layout].row_major(
                IndexList[2](max_recv_num_tokens, 2)
            )
        )

    @always_inline
    @parameter
    fn get_output_2_tensor(dev_idx: Int, slot_idx: Int, out result: LayoutTensor[input_type, output_2_layout, MutAnyOrigin]) raises:
        return type_of(result)(
            device_output_2_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k * hidden_size,
            RuntimeLayout[output_2_layout].row_major(
                IndexList[3](n_tokens_per_rank, top_k, hidden_size)
            )
        )
    # fmt: on

    comptime hw_info = type_of(list_of_ctx[0]).default_device_info
    var format_handler = BF16TokenFormat[hidden_size, top_k, gpu_alignment](
        get_output_tensor(0, 0)
    )

    # Dispatch kernel
    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        input_tokens_layout,
        topk_ids_layout,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        n_ranks,  # p2p world size
        token_fmt_type,
        use_shmem=False,
    ]

    # Dispatch callback kernel
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

    # Combine kernel
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
        n_ranks,  # p2p world size
        use_shmem=False,
    ]

    # Combine callback kernel
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

    @always_inline
    @parameter
    fn run_dispatch_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[dispatch_async, dispatch_async](
            get_input_tokens_tensor(dev_idx, slot_idx),
            get_topk_ids_tensor(dev_idx, slot_idx),
            get_dispatch_send_buf_ptr(dev_idx, slot_idx),
            dispatch_recv_bufs_inputs[slot_idx],
            dispatch_recv_count_bufs_inputs[slot_idx],
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_dispatch_async_wait(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[dispatch_wait, dispatch_wait](
            type_of(format_handler)(get_output_tensor(dev_idx, slot_idx)),
            get_row_offsets_tensor(dev_idx, slot_idx),
            get_expert_ids_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            dispatch_recv_bufs_inputs[slot_idx][dev_idx],
            dispatch_recv_count_bufs_inputs[slot_idx][dev_idx],
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            OptionalReg[
                LayoutTensor[input_type, Layout.row_major[2](), ImmutAnyOrigin]
            ](),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_full_dispatch(dev_idx: Int, slot_idx: Int) raises:
        run_dispatch_async(dev_idx, slot_idx)
        run_dispatch_async_wait(dev_idx, slot_idx)

    @always_inline
    @parameter
    fn run_combine_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[combine_async, combine_async](
            get_output_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            get_combine_send_buf_ptr(dev_idx, slot_idx),
            combine_recv_bufs_inputs[slot_idx],
            combine_recv_count_bufs_inputs[slot_idx],
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            OptionalReg[
                LayoutTensor[input_type, Layout.row_major[2](), MutAnyOrigin]
            ](),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_combine_async_wait(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[combine_wait, combine_wait](
            get_output_2_tensor(dev_idx, slot_idx),
            get_combine_recv_buf_ptr(dev_idx, slot_idx),
            get_combine_recv_count_ptr(dev_idx, slot_idx),
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    @parameter
    fn run_e2e(dev_idx: Int, slot_idx: Int) raises:
        run_combine_async(dev_idx, slot_idx)
        run_combine_async_wait(dev_idx, slot_idx)

    @always_inline
    @parameter
    fn clean_up(dev_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_memset(atomic_counters_list[dev_idx], Int32(0))
        ctx.enqueue_memset(
            dispatch_recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE
        )
        ctx.enqueue_memset(
            combine_recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE
        )

    # warm up by running once
    for dev_i in range(n_ranks):
        run_full_dispatch(dev_i, 0)

    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    for dev_i in range(n_ranks):
        run_e2e(dev_i, 0)

    for dev_i in range(n_ranks):
        clean_up(dev_i)
        list_of_ctx[dev_i].synchronize()

    # Necessary to fill this InlineArray w/ default BenchmarkInfo
    # otherwise each thread attempts to free uninitialized BenchmarkInfo
    # when copying below
    var default_info = BenchmarkInfo(
        name="",
        result=Report(),
        measures=List[ThroughputMeasure](),
    )
    var results_b = InlineArray[BenchmarkInfo, n_ranks](fill=default_info)

    # First, prepare the data for the combine kernel
    for dev_i in range(n_ranks):
        for slot_idx in range(n_slots):
            run_full_dispatch(dev_i, slot_idx)

    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @parameter
    fn per_gpu_combine(i: Int) raises:
        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn call_fn(ctx: DeviceContext, cache_iter: Int) raises:
                var dev_id = Int(ctx.id())
                run_combine_async(dev_id, cache_iter)

            b.iter_custom[call_fn](list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function[bench_iter](
            BenchId("bench combine"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize[per_gpu_combine](n_ranks)

    var max_time = 0.0
    var max_loc = 0

    for i in range(n_ranks):
        var val = results_b[i].result.mean(unit="ms")
        if val > max_time:
            max_time = val
            max_loc = i

    var b_final = Bench()
    b_final.info_vec.append(results_b[max_loc].copy())
    b_final.dump_report()

    # Then, bench the combine_wait kernel overhead
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @parameter
    fn per_gpu_combine_wait(i: Int) raises:
        @parameter
        @always_inline
        fn bench_iter(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn call_fn(ctx: DeviceContext, cache_iter: Int) raises:
                var dev_id = Int(ctx.id())
                run_combine_async_wait(dev_id, cache_iter)

            b.iter_custom[call_fn](list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function[bench_iter](
            BenchId("bench combine_wait"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize[per_gpu_combine_wait](n_ranks)

    max_time = 0.0
    max_loc = 0

    for i in range(n_ranks):
        var val = results_b[i].result.mean(unit="ms")
        if val > max_time:
            max_time = val
            max_loc = i

    b_final = Bench()
    b_final.info_vec.append(results_b[max_loc].copy())
    b_final.dump_report()

    # Verify the results for each device and each slot
    print("Verifying results...")

    @parameter
    @always_inline
    fn verify_results(dev_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]

        # Allocate host buffers for copying device outputs
        var host_output_2 = alloc[Scalar[input_type]](
            n_slots * n_tokens_per_rank * top_k * hidden_size
        )

        # Copy device outputs to host
        ctx.enqueue_copy(host_output_2, device_output_2_bufs_list[dev_idx])
        ctx.synchronize()

        # Check results for each slot
        for slot_idx in range(n_slots):
            # Get pointers to this slot's data
            var slot_output_2 = (
                host_output_2
                + slot_idx * n_tokens_per_rank * top_k * hidden_size
            )
            var slot_input_tokens = (
                host_input_tokens_list[dev_idx]
                + slot_idx * n_tokens_per_rank * hidden_size
            )

            # The combine kernel should send tokens back to the original rank.
            # Check that the received tokens match the original input tokens.
            for token_idx in range(n_tokens_per_rank):
                var ref_token = slot_input_tokens + token_idx * hidden_size
                for topk_idx in range(top_k):
                    var received_token = (
                        slot_output_2
                        + token_idx * top_k * hidden_size
                        + topk_idx * hidden_size
                    )
                    for i in range(hidden_size):
                        assert_equal(
                            received_token[i],
                            ref_token[i],
                            "Token mismatch for dev "
                            + String(dev_idx)
                            + " slot "
                            + String(slot_idx)
                            + " token "
                            + String(token_idx)
                            + " topk "
                            + String(topk_idx)
                            + " hidden "
                            + String(i),
                        )

        # Free host buffers
        host_output_2.free()

    sync_parallelize[verify_results](n_ranks)
    print("All results verified successfully!")

    for dev_idx in range(n_ranks):
        host_topk_ids_list[dev_idx].free()
        host_input_tokens_list[dev_idx].free()


def main():
    comptime test_gpu_counts = (2, 4, 8)

    if enable_p2p():
        print("Enabled P2P Mem Access on all GPUs.")
    else:
        raise Error("Cannot enable P2P Mem Access!")

    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "Only NVIDIA and AMD GPUs are supported"
    comptime n_local_experts = 32 if has_nvidia_gpu_accelerator() else 16

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        test_combine[
            hidden_size=7168,
            top_k=8,
            n_experts = num_gpus * n_local_experts,
            n_ranks=num_gpus,
            n_slots=3,
            n_tokens_per_rank=128,
        ](ctx)
