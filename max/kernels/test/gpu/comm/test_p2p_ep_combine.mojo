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

from std.atomic import Atomic, Ordering
from std.random import randint, randn, seed
from std.sys import size_of
from std.sys.intrinsics import llvm_intrinsic

from max.algorithm import sync_parallelize
from max.benchmark import bencher_iter_custom
from std.benchmark import (
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
from max.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    _SUPPORT_PDL_LAUNCH,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from layout import Coord, TileTensor, Idx
from layout.tile_layout import row_major
from shmem.ep_comm import (
    EP_NO_INPUT_SCALES,
    EP_NO_ROUTER_WEIGHTS,
    EP_PDL_MAX_RANKS,
    BF16TokenFormat,
    EPLocalSyncCounters,
    combine_wait_kernel,
    combine_async_kernel,
    dispatch_wait_kernel,
    dispatch_async_kernel,
)
from std.testing import assert_equal
from std.math import ceildiv
from std.memory.alloc import unsafe_alloc


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: MutPointer[Int32, MutAnyOrigin], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids.unsafe_offset(tok_id * top_k)

        # The top-k ids for a token should be unique. If not, we will assign a
        # random id to the duplicate id.
        def is_duplicate() {imm} -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if topk_ids_for_token[i] == topk_ids_for_token[j]:
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


@inline(.always)
def _sleep_us(us: Int32):
    comptime if _SUPPORT_PDL_LAUNCH:
        # nanosleep takes nanoseconds and caps near 1 ms per call.
        var left = Int(us)
        while left > 0:
            llvm_intrinsic["llvm.nvvm.nanosleep", NoneType](Int32(1_000_000))
            left -= 1000


def _late_fill[
    dtype: DType
](
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    n: Int32,
    delay_us: Int32,
):
    """Releases the dependent grid at entry, spins, then writes `dst`.

    Stands in for the router or the grouped matmul. A PDL-launched consumer
    that reads `dst` before its own `wait_on_dependent_grids` sees whatever
    `dst` held when this kernel started, which the test sets to zeros.
    """
    launch_dependent_grids()
    _sleep_us(delay_us)
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(n), stride):
        dst[i] = src[i]


def _probe_output[
    dtype: DType
](
    buf: MutPointer[Scalar[dtype], MutAnyOrigin],
    n: Int32,
    sentinel: Scalar[dtype],
    changed: MutPointer[Scalar[DType.uint32], MutAnyOrigin],
    sleep_us: Int32,
):
    """Sets `changed[0]` if `buf` stops holding `sentinel` while this grid runs.

    Stands in for a kernel that still reads a tensor the planner aliased with
    `combine_wait`'s output. It waits for its predecessor, releases the
    dependent at entry, sleeps, and then reads `buf`. The dependent's own PDL
    wait is what keeps it from writing `buf` in the meantime.
    """
    wait_on_dependent_grids()
    launch_dependent_grids()
    _sleep_us(sleep_us)
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = grid_dim.x * block_dim.x
    for i in range(tid, Int(n), stride):
        if buf[i] != sentinel:
            Atomic[Scalar[DType.uint32]].store[ordering=Ordering.RELAXED](
                changed, UInt32(1)
            )


def test_combine[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    skewed: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime input_type = DType.bfloat16
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank
    # The same decision `ep.mojo` makes at launch: wider worlds run PDL-off.
    comptime split_pdl_level = (
        PDLLevel.ON if n_ranks <= EP_PDL_MAX_RANKS else PDLLevel.OFF
    )

    comptime output_layout = row_major(
        (Idx[max_recv_num_tokens], Idx[hidden_size])
    )
    comptime token_fmt_type = BF16TokenFormat[
        output_layout=type_of(output_layout), hidden_size, top_k
    ]
    comptime msg_bytes = token_fmt_type.msg_size()
    comptime combine_msg_bytes = size_of[input_type]() * hidden_size

    comptime num_bytes = combine_msg_bytes * top_k * n_tokens_per_rank

    print(
        t"Running ep_combine test: input_type: {input_type} hidden_size:"
        t" {hidden_size} top_k: {top_k} n_experts: {n_experts} n_ranks:"
        t" {n_ranks} n_tokens_per_rank: {n_tokens_per_rank}"
    )

    # fmt: off
    # Buffers for dispatch phase
    var dispatch_send_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var dispatch_recv_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var dispatch_recv_count_bufs_list = List[DeviceBuffer[.uint64]](capacity=n_ranks)

    # Buffers for combine phase
    var combine_send_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var combine_recv_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var combine_recv_count_bufs_list = List[DeviceBuffer[.uint64]](capacity=n_ranks)

    # Shared atomic counter buffer for dispatch and combine
    var atomic_counters_list = List[DeviceBuffer[.int32]](capacity=n_ranks)

    var host_topk_ids_list = Array[MutPointer[Int32, MutAnyOrigin], n_ranks](uninitialized=True)
    var host_input_tokens_list = Array[MutPointer[Scalar[input_type], MutAnyOrigin], n_ranks](uninitialized=True)

    var device_topk_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var device_input_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_output_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_row_offsets_bufs_list = List[DeviceBuffer[.uint32]](capacity=n_ranks)
    var device_expert_ids_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var device_src_token_info_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)

    # Output buffer for combine_wait
    var device_output_2_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)

    for i in range(n_ranks):
        var ctx = list_of_ctx[i]
        # Dispatch buffers
        dispatch_send_bufs_list.append(ctx.enqueue_create_buffer[.uint8](n_slots * n_tokens_per_rank * msg_bytes))
        dispatch_recv_bufs_list.append(ctx.enqueue_create_buffer[.uint8](n_slots * max_recv_num_tokens * msg_bytes))
        dispatch_recv_count_bufs_list.append(ctx.enqueue_create_buffer[.uint64](n_slots * n_experts))
        ctx.enqueue_memset(dispatch_recv_count_bufs_list[i], UInt64.MAX_FINITE)

        # Combine buffers
        combine_send_bufs_list.append(ctx.enqueue_create_buffer[.uint8](n_slots * max_recv_num_tokens * combine_msg_bytes))
        combine_recv_bufs_list.append(ctx.enqueue_create_buffer[.uint8](n_slots * n_tokens_per_rank * top_k * combine_msg_bytes))
        combine_recv_count_bufs_list.append(ctx.enqueue_create_buffer[.uint64](n_slots * n_experts))
        ctx.enqueue_memset(combine_recv_count_bufs_list[i], UInt64.MAX_FINITE)

        # Shared atomic counter
        atomic_counters_list.append(ctx.enqueue_create_buffer[.int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        ))
        ctx.enqueue_memset(atomic_counters_list[i], Int32(0))

        host_topk_ids_list[i] = unsafe_alloc[Int32](n_slots * n_tokens_per_rank * top_k).as_unsafe_any_origin()
        host_input_tokens_list[i] = unsafe_alloc[Scalar[input_type]](n_slots * n_tokens_per_rank * hidden_size).as_unsafe_any_origin()

        device_topk_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * n_tokens_per_rank * top_k))
        device_input_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * hidden_size))
        device_output_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * max_recv_num_tokens * hidden_size))
        device_row_offsets_bufs_list.append(ctx.enqueue_create_buffer[.uint32](n_slots * (n_local_experts + 1)))
        device_expert_ids_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * n_local_experts))
        device_src_token_info_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * max_recv_num_tokens * 2))

        device_output_2_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * top_k * hidden_size))
    # fmt: on

    var topk_ids_layout = row_major(n_tokens_per_rank, Idx[top_k])
    var input_tokens_layout = row_major((n_tokens_per_rank, Idx[hidden_size]))
    var output_tt_layout = row_major(
        (Idx[max_recv_num_tokens], Idx[hidden_size])
    )
    var row_offsets_layout = row_major[n_local_experts + 1]()
    var expert_ids_layout = row_major[n_local_experts]()
    var src_token_info_layout = row_major((Idx[max_recv_num_tokens], Idx[2]))
    var output_2_layout = row_major(
        (n_tokens_per_rank, Idx[top_k], Idx[hidden_size])
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

        comptime if skewed:
            # Pile the tokens onto three local experts per rank in a 5:2:1
            # split, leaving the rest empty. `combine_async`'s balanced send
            # spreads one (expert, rank) pair's tokens over many warps and
            # lets whichever warp lands the last one signal the pair, so it
            # needs a pair hot enough to be shared and pairs that are empty.
            # Uniform routing gives every pair about the same handful of
            # tokens and reaches neither. A token's top_k walks the ranks, and
            # each further lap shifts to the next triple of local experts, so
            # its experts stay distinct however top_k compares to n_ranks.
            comptime n_local = n_experts // n_ranks
            comptime n_laps = ceildiv(top_k, n_ranks)
            comptime assert 3 * n_laps <= n_local
            for t in range(n_slots * n_tokens_per_rank):
                var phase = t % 8
                var hot = 0 if phase < 5 else (1 if phase < 7 else 2)
                for k in range(top_k):
                    var r = (t + k) % n_ranks
                    host_topk_ids_list[dev_idx][t * top_k + k] = Int32(
                        r * n_local + hot + 3 * (k // n_ranks)
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
    var dispatch_recv_bufs_inputs = Array[Array[MutPointer[UInt8, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)
    var dispatch_recv_count_bufs_inputs = Array[Array[MutPointer[UInt64, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)

    # Combine buffers
    var combine_recv_bufs_inputs = Array[Array[MutPointer[UInt8, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)
    var combine_recv_count_bufs_inputs = Array[Array[MutPointer[UInt64, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)

    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            dispatch_recv_bufs_inputs[slot_idx][dev_idx] = dispatch_recv_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * max_recv_num_tokens * msg_bytes).as_unsafe_any_origin()
            dispatch_recv_count_bufs_inputs[slot_idx][dev_idx] = dispatch_recv_count_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_experts).as_unsafe_any_origin()
            combine_recv_bufs_inputs[slot_idx][dev_idx] = combine_recv_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes).as_unsafe_any_origin()
            combine_recv_count_bufs_inputs[slot_idx][dev_idx] = combine_recv_count_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_experts).as_unsafe_any_origin()

    # Dispatch helpers
    @inline(.always)
    def get_dispatch_send_buf_ptr(dev_idx: Int, slot_idx: Int, out result: MutPointer[UInt8, MutAnyOrigin]) raises {mut dispatch_send_bufs_list, imm}:
        result = dispatch_send_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_tokens_per_rank * msg_bytes).as_unsafe_any_origin()

    # Combine helpers
    @inline(.always)
    def get_combine_send_buf_ptr(dev_idx: Int, slot_idx: Int, out result: MutPointer[UInt8, MutAnyOrigin]) raises {mut combine_send_bufs_list, imm}:
        result = combine_send_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * max_recv_num_tokens * combine_msg_bytes).as_unsafe_any_origin()

    @inline(.always)
    def get_combine_recv_buf_ptr(dev_idx: Int, slot_idx: Int, out result: MutPointer[UInt8, MutAnyOrigin]) raises {mut combine_recv_bufs_list, imm}:
        result = combine_recv_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes).as_unsafe_any_origin()

    @inline(.always)
    def get_combine_recv_count_ptr(dev_idx: Int, slot_idx: Int, out result: MutPointer[UInt64, MutAnyOrigin]) raises {mut combine_recv_count_bufs_list, imm}:
        result = combine_recv_count_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_experts).as_unsafe_any_origin()

    @inline(.always)
    def get_atomic_counters(dev_idx: Int, slot_idx: Int, out result: EPLocalSyncCounters[n_experts]) raises {mut atomic_counters_list, imm}:
        return EPLocalSyncCounters[n_experts](atomic_counters_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * EPLocalSyncCounters[n_experts].total_size()))

    comptime get_topk_ids_tensor_result_type = TileTensor[.int32, type_of(topk_ids_layout), ImmutAnyOrigin]
    @inline(.always)
    def get_topk_ids_tensor(dev_idx: Int, slot_idx: Int, out result: get_topk_ids_tensor_result_type) raises {imm}:
        return type_of(result)(
            ptr=device_topk_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_tokens_per_rank * top_k).as_unsafe_any_origin(),
            layout=topk_ids_layout
        )

    comptime get_input_tokens_tensor_result_type = TileTensor[input_type, type_of(input_tokens_layout), ImmutAnyOrigin]
    @inline(.always)
    def get_input_tokens_tensor(dev_idx: Int, slot_idx: Int, out result: get_input_tokens_tensor_result_type) raises {imm}:
        return type_of(result)(
            ptr=device_input_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_tokens_per_rank * hidden_size).as_unsafe_any_origin(),
            layout=input_tokens_layout
        )

    comptime get_output_tensor_result_type = TileTensor[input_type, type_of(output_tt_layout), MutAnyOrigin]
    @inline(.always)
    def get_output_tensor(dev_idx: Int, slot_idx: Int, out result: get_output_tensor_result_type) raises {mut device_output_bufs_list, imm}:
        return type_of(result)(
            ptr=device_output_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * max_recv_num_tokens * hidden_size).as_unsafe_any_origin(),
            layout=output_tt_layout
        )

    comptime get_row_offsets_tensor_result_type = TileTensor[.uint32, type_of(row_offsets_layout), MutAnyOrigin]
    @inline(.always)
    def get_row_offsets_tensor(dev_idx: Int, slot_idx: Int, out result: get_row_offsets_tensor_result_type) raises {mut device_row_offsets_bufs_list, imm}:
        return type_of(result)(
            ptr=device_row_offsets_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * (n_local_experts + 1)).as_unsafe_any_origin(),
            layout=row_offsets_layout
        )

    comptime get_expert_ids_tensor_result_type = TileTensor[.int32, type_of(expert_ids_layout), MutAnyOrigin]
    @inline(.always)
    def get_expert_ids_tensor(dev_idx: Int, slot_idx: Int, out result: get_expert_ids_tensor_result_type) raises {mut device_expert_ids_bufs_list, imm}:
        return type_of(result)(
            ptr=device_expert_ids_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_local_experts).as_unsafe_any_origin(),
            layout=expert_ids_layout
        )

    comptime get_src_token_info_tensor_result_type = TileTensor[.int32, type_of(src_token_info_layout), MutAnyOrigin]
    @inline(.always)
    def get_src_token_info_tensor(dev_idx: Int, slot_idx: Int, out result: get_src_token_info_tensor_result_type) raises {mut device_src_token_info_bufs_list, imm}:
        return type_of(result)(
            ptr=device_src_token_info_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * max_recv_num_tokens * 2).as_unsafe_any_origin(),
            layout=src_token_info_layout
        )

    comptime get_output_2_tensor_result_type = TileTensor[input_type, type_of(output_2_layout), MutAnyOrigin]
    @inline(.always)
    def get_output_2_tensor(dev_idx: Int, slot_idx: Int, out result: get_output_2_tensor_result_type) raises {mut device_output_2_bufs_list, imm}:
        return type_of(result)(
            ptr=device_output_2_bufs_list[dev_idx].unsafe_ptr().unsafe_offset(slot_idx * n_tokens_per_rank * top_k * hidden_size).as_unsafe_any_origin(),
            layout=output_2_layout
        )
    # fmt: on

    comptime hw_info = type_of(list_of_ctx[0]).default_device_info
    var format_handler = token_fmt_type(get_output_tensor(0, 0))

    # Dispatch kernel
    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        type_of(input_tokens_layout),
        type_of(topk_ids_layout),
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        n_ranks,  # p2p world size
        token_fmt_type,
        InputScalesFnType=type_of(EP_NO_INPUT_SCALES),
        use_shmem=False,
    ]

    # Dispatch callback kernel
    comptime dispatch_wait = dispatch_wait_kernel[
        hw_info.max_thread_block_size,
        type_of(row_offsets_layout),
        type_of(expert_ids_layout),
        type_of(src_token_info_layout),
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
        type_of(output_tt_layout),
        type_of(src_token_info_layout),
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_bytes,
        n_tokens_per_rank,
        n_ranks,  # p2p world size
        use_shmem=False,
    ]

    @inline(.always)
    def run_dispatch_async(dev_idx: Int, slot_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[dispatch_async](
            get_input_tokens_tensor(dev_idx, slot_idx),
            get_topk_ids_tensor(dev_idx, slot_idx),
            get_dispatch_send_buf_ptr(dev_idx, slot_idx),
            dispatch_recv_bufs_inputs[slot_idx],
            dispatch_recv_count_bufs_inputs[slot_idx],
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            host_args=(EP_NO_INPUT_SCALES,),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
            attributes=pdl_launch_attributes(split_pdl_level),
        )

    @inline(.always)
    def run_dispatch_async_wait(dev_idx: Int, slot_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[dispatch_wait](
            type_of(format_handler)(get_output_tensor(dev_idx, slot_idx)),
            get_row_offsets_tensor(dev_idx, slot_idx),
            get_expert_ids_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            dispatch_recv_bufs_inputs[slot_idx][dev_idx],
            dispatch_recv_count_bufs_inputs[slot_idx][dev_idx],
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @inline(.always)
    def run_full_dispatch(dev_idx: Int, slot_idx: Int) raises {imm}:
        run_dispatch_async(dev_idx, slot_idx)
        run_dispatch_async_wait(dev_idx, slot_idx)

    @inline(.always)
    def run_combine_async(dev_idx: Int, slot_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[combine_async](
            get_output_tensor(dev_idx, slot_idx).as_imm(),
            get_src_token_info_tensor(dev_idx, slot_idx).as_imm(),
            get_combine_send_buf_ptr(dev_idx, slot_idx),
            combine_recv_bufs_inputs[slot_idx],
            combine_recv_count_bufs_inputs[slot_idx],
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
            attributes=pdl_launch_attributes(split_pdl_level),
        )

    @inline(.always)
    def run_combine_async_wait(dev_idx: Int, slot_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]
        var output_2_tensor = get_output_2_tensor(dev_idx, slot_idx)

        @inline(.always)
        def store_output_2[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: Coord, val: SIMD[dtype, width]) {var output_2_tensor}:
            output_2_tensor.store[
                width=width, alignment=alignment * size_of[dtype]()
            ](coords, rebind[SIMD[input_type, width]](val))

        comptime combine_wait = combine_wait_kernel[
            input_type,
            hw_info.max_thread_block_size,
            hw_info.sm_count,
            top_k,
            n_experts,
            n_ranks,
            combine_msg_bytes,
            n_tokens_per_rank,
            type_of(store_output_2),
            type_of(EP_NO_ROUTER_WEIGHTS),
            has_router_weights=False,
        ]
        ctx.enqueue_function[combine_wait](
            Int32(n_tokens_per_rank),
            get_combine_recv_buf_ptr(dev_idx, slot_idx),
            get_combine_recv_count_ptr(dev_idx, slot_idx),
            get_atomic_counters(dev_idx, slot_idx),
            Int32(dev_idx),
            host_args=(store_output_2, EP_NO_ROUTER_WEIGHTS),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
            attributes=pdl_launch_attributes(split_pdl_level),
        )

    @inline(.always)
    def run_e2e(dev_idx: Int, slot_idx: Int) raises {imm}:
        run_combine_async(dev_idx, slot_idx)
        run_combine_async_wait(dev_idx, slot_idx)

    @inline(.always)
    def clean_up(dev_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_memset(atomic_counters_list[dev_idx], Int32(0))
        ctx.enqueue_memset(
            dispatch_recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE
        )
        ctx.enqueue_memset(
            combine_recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE
        )

    # Each device resets its own recv counts on its own stream, but peers
    # write signals into them. Wait for all resets, or a late reset can erase
    # a peer's signal and the wait kernel spins forever.
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

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

    # Necessary to fill this Array w/ default BenchmarkInfo
    # otherwise each thread attempts to free uninitialized BenchmarkInfo
    # when copying below
    var default_info = BenchmarkInfo(
        name="",
        result=Report(),
        measures=List[ThroughputMeasure](),
    )
    var results_b = Array[BenchmarkInfo, n_ranks](fill=default_info)

    # First, prepare the data for the combine kernel
    for dev_i in range(n_ranks):
        for slot_idx in range(n_slots):
            run_full_dispatch(dev_i, slot_idx)

    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @inline(.always)
    def call_fn_combine(ctx: DeviceContext, cache_iter: Int) raises {imm}:
        var dev_id = Int(ctx.device().id)
        run_combine_async(dev_id, cache_iter)

    def per_gpu_combine(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_combine, list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench combine"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_combine, n_ranks)

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

    @inline(.always)
    def call_fn_combine_wait(ctx: DeviceContext, cache_iter: Int) raises {imm}:
        var dev_id = Int(ctx.device().id)
        run_combine_async_wait(dev_id, cache_iter)

    def per_gpu_combine_wait(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_combine_wait, list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench combine_wait"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_combine_wait, n_ranks)

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

    @inline(.always)
    def verify_results(dev_idx: Int) raises {imm}:
        var ctx = list_of_ctx[dev_idx]

        # Allocate host buffers for copying device outputs
        var host_output_2 = unsafe_alloc[Scalar[input_type]](
            n_slots * n_tokens_per_rank * top_k * hidden_size
        )

        # Copy device outputs to host
        ctx.enqueue_copy(host_output_2, device_output_2_bufs_list[dev_idx])
        ctx.synchronize()

        # Check results for each slot
        for slot_idx in range(n_slots):
            # Get pointers to this slot's data
            var slot_output_2 = host_output_2.unsafe_offset(
                slot_idx * n_tokens_per_rank * top_k * hidden_size
            )
            var slot_input_tokens = host_input_tokens_list[
                dev_idx
            ].unsafe_offset(slot_idx * n_tokens_per_rank * hidden_size)

            # The combine kernel should send tokens back to the original rank.
            # Check that the received tokens match the original input tokens.
            for token_idx in range(n_tokens_per_rank):
                var ref_token = slot_input_tokens.unsafe_offset(
                    token_idx * hidden_size
                )
                for topk_idx in range(top_k):
                    var received_token = slot_output_2.unsafe_offset(
                        token_idx * top_k * hidden_size
                    ).unsafe_offset(topk_idx * hidden_size)
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
        host_output_2.unsafe_free()

    sync_parallelize(verify_results, n_ranks)
    print("All results verified successfully!")

    # PDL ordering. The runs above sync the host between kernels and prepare
    # their inputs up front, so they would pass even if the entry waits were
    # missing. Here a late producer releases `dispatch_async` and
    # `combine_async` at entry and only then publishes their inputs (the
    # tokens, and the dispatched tokens the grouped matmul would produce), and
    # the whole dispatch -> combine chain runs on one stream per GPU with no
    # host sync. A kernel that reads an input before its own PDL wait sends
    # zeros, which the same check as above reports. Above `EP_PDL_MAX_RANKS`
    # ranks the kernels have no wait to test and launch without the attribute.
    comptime if _SUPPORT_PDL_LAUNCH and n_ranks <= EP_PDL_MAX_RANKS:
        print("Verifying PDL ordering...")
        comptime input_elems = n_slots * n_tokens_per_rank * hidden_size
        comptime output_elems = n_slots * max_recv_num_tokens * hidden_size

        var ref_input_bufs = List[DeviceBuffer[input_type]](capacity=n_ranks)
        var ref_output_bufs = List[DeviceBuffer[input_type]](capacity=n_ranks)
        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            ref_input_bufs.append(
                ctx.enqueue_create_buffer[input_type](input_elems)
            )
            ref_output_bufs.append(
                ctx.enqueue_create_buffer[input_type](output_elems)
            )
            ctx.enqueue_copy(
                ref_input_bufs[dev_idx], device_input_bufs_list[dev_idx]
            )
            # Load the producer on every device before a collective spins: a
            # module load behind a spinning kernel would block the host and
            # strand the peers.
            ctx.enqueue_function[_late_fill[input_type]](
                ref_output_bufs[dev_idx],
                ref_output_bufs[dev_idx],
                Int32(0),
                Int32(0),
                grid_dim=1,
                block_dim=32,
            )
        for dev_idx in range(n_ranks):
            list_of_ctx[dev_idx].synchronize()

        @inline(.always)
        def run_ordered_chain(dev_idx: Int, delay_us: Int) raises {imm}:
            var ctx = list_of_ctx[dev_idx]
            ctx.enqueue_memset(
                device_input_bufs_list[dev_idx], Scalar[input_type](0)
            )
            ctx.enqueue_function[_late_fill[input_type]](
                device_input_bufs_list[dev_idx],
                ref_input_bufs[dev_idx],
                Int32(input_elems),
                Int32(delay_us + 1000 * dev_idx),
                grid_dim=64,
                block_dim=256,
            )
            run_dispatch_async(dev_idx, 0)
            run_dispatch_async_wait(dev_idx, 0)
            # Snapshot what dispatch delivered, blank it, and republish it from
            # a late producer, as the grouped matmul would.
            ctx.enqueue_copy(
                ref_output_bufs[dev_idx], device_output_bufs_list[dev_idx]
            )
            ctx.enqueue_memset(
                device_output_bufs_list[dev_idx], Scalar[input_type](0)
            )
            ctx.enqueue_function[_late_fill[input_type]](
                device_output_bufs_list[dev_idx],
                ref_output_bufs[dev_idx],
                Int32(output_elems),
                Int32(delay_us + 1000 * dev_idx),
                grid_dim=64,
                block_dim=256,
            )
            run_combine_async(dev_idx, 0)
            run_combine_async_wait(dev_idx, 0)

        # Rank `round % n_ranks` runs its producers a millisecond later than
        # its peers, and the host enqueues the chains starting from rank
        # `(round // n_ranks) % n_ranks`, so four rounds on two ranks cover
        # each rank being late with each rank enqueued first.
        for round in range(4):
            var late_rank = round % n_ranks
            var first_rank = (round // n_ranks) % n_ranks
            for dev_idx in range(n_ranks):
                clean_up(dev_idx)
                list_of_ctx[dev_idx].enqueue_memset(
                    device_output_2_bufs_list[dev_idx], Scalar[input_type](0)
                )
            for dev_idx in range(n_ranks):
                list_of_ctx[dev_idx].synchronize()
            for i in range(n_ranks):
                var dev_idx = (first_rank + i) % n_ranks
                run_ordered_chain(
                    dev_idx, 2000 + (1000 if dev_idx == late_rank else 0)
                )
            for dev_idx in range(n_ranks):
                list_of_ctx[dev_idx].synchronize()
            sync_parallelize(verify_results, n_ranks)

        # `combine_wait` writes an output tensor the planner may alias with a
        # tensor its predecessor still reads, so its PDL wait must hold it off
        # until that predecessor completes. A stand-in predecessor releases
        # `combine_wait` at entry, sleeps, and then reads the output tensor,
        # which is filled with a sentinel. It sees the sentinel unless
        # `combine_wait` started writing before the predecessor completed.
        var changed_bufs = List[DeviceBuffer[DType.uint32]](capacity=n_ranks)
        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            changed_bufs.append(ctx.enqueue_create_buffer[DType.uint32](1))
            # Load the probe on every device before a collective spins.
            ctx.enqueue_function[_probe_output[input_type]](
                device_output_2_bufs_list[dev_idx],
                Int32(0),
                Scalar[input_type](0),
                changed_bufs[dev_idx],
                Int32(0),
                grid_dim=1,
                block_dim=32,
            )
        for dev_idx in range(n_ranks):
            list_of_ctx[dev_idx].synchronize()

        comptime output_2_elems = (
            n_slots * n_tokens_per_rank * top_k * hidden_size
        )
        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            clean_up(dev_idx)
            ctx.enqueue_memset(
                device_output_2_bufs_list[dev_idx], Scalar[input_type](1000)
            )
            ctx.enqueue_memset(changed_bufs[dev_idx], UInt32(0))
        for dev_idx in range(n_ranks):
            list_of_ctx[dev_idx].synchronize()
        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            run_dispatch_async(dev_idx, 0)
            run_dispatch_async_wait(dev_idx, 0)
            run_combine_async(dev_idx, 0)
            ctx.enqueue_function[_probe_output[input_type]](
                device_output_2_bufs_list[dev_idx],
                Int32(output_2_elems),
                Scalar[input_type](1000),
                changed_bufs[dev_idx],
                Int32(2000),
                grid_dim=64,
                block_dim=256,
                attributes=pdl_launch_attributes(split_pdl_level),
            )
            run_combine_async_wait(dev_idx, 0)
        for dev_idx in range(n_ranks):
            list_of_ctx[dev_idx].synchronize()
        for dev_idx in range(n_ranks):
            var ctx = list_of_ctx[dev_idx]
            var host_changed = unsafe_alloc[Scalar[DType.uint32]](1)
            ctx.enqueue_copy(host_changed, changed_bufs[dev_idx])
            ctx.synchronize()
            assert_equal(
                host_changed[0],
                0,
                "combine_wait wrote its output before its predecessor"
                " completed on dev "
                + String(dev_idx),
            )
            host_changed.unsafe_free()
        sync_parallelize(verify_results, n_ranks)

        print("PDL ordering verified successfully!")
        _ = ref_input_bufs^
        _ = ref_output_bufs^

    for dev_idx in range(n_ranks):
        host_topk_ids_list[dev_idx].unsafe_free()
        host_input_tokens_list[dev_idx].unsafe_free()


def main() raises:
    comptime test_gpu_counts = (2, 4, 8)

    if enable_p2p():
        print("Enabled P2P Mem Access on all GPUs.")
    else:
        raise Error("Cannot enable P2P Mem Access!")

    comptime assert (
        default_accelerator().is_nvidia_gpu()
        or default_accelerator().is_amd_gpu()
    ), "Only NVIDIA and AMD GPUs are supported"

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        comptime for n_local_experts in [32, 64]:
            comptime if num_gpus * n_local_experts > 256:
                continue

            test_combine[
                hidden_size=7168,
                top_k=8,
                n_experts=num_gpus * n_local_experts,
                n_ranks=num_gpus,
                n_slots=1,
                n_tokens_per_rank=128,
            ](ctx)

            # Same shape, routing piled onto three local experts per rank.
            test_combine[
                hidden_size=7168,
                top_k=8,
                n_experts=num_gpus * n_local_experts,
                n_ranks=num_gpus,
                n_slots=1,
                n_tokens_per_rank=128,
                skewed=True,
            ](ctx)

            # A decode-sized batch: few tokens, most (expert, rank) pairs empty.
            # This is below the threshold where `combine_async` spreads tokens
            # across all warps, so it covers the per-pair assignment the
            # prefill-sized cases above no longer reach.
            test_combine[
                hidden_size=7168,
                top_k=8,
                n_experts=num_gpus * n_local_experts,
                n_ranks=num_gpus,
                n_slots=1,
                n_tokens_per_rank=8,
            ](ctx)
