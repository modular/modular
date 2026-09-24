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
"""Benchmarks the three expert-parallelism combine kernels over P2P.

Covers `combine_async`, `combine_wait`, and the fused kernel that replaces
both, on 8 GPUs sharing an NVLink or XGMI fabric. Multi-node SHMEM is out of
scope; the single-node path is the one these kernels are tuned for.

Combine has no token format -- it moves whatever dtype the experts produced --
so the dtype axis here is message size: BF16 is the real case, FP8 halves the
bytes. Two shapes run per dtype: a decode-sized point, where cost is per
launch, and a prefill-sized point, where it is per byte.

Routing comes in two modes, selected at run time so one binary serves both:

    EP_SKEW=0 (default)  draw each token's experts uniformly
    EP_SKEW=1            draw from a Zipf popularity distribution
    EP_SKEW_S            Zipf exponent in thousandths (default 1000 = 1.0)

`combine_async` gives each (expert, rank) pair its own work, so its cost
tracks the busiest pair rather than the average -- which is why the skewed
mode exists, and why a speedup measured under `EP_SKEW=0` says nothing about
it. Every run prints the imbalance it actually generated.

Correctness is not checked here; `test_p2p_ep_combine.mojo` owns that.
"""

from std.random import random_float64, seed
from std.os import getenv
from std.sys import get_defined_int
from std.time import perf_counter_ns
from std.sys import (
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    size_of,
)

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
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx
from layout.tile_layout import row_major
from shmem.ep import (
    ep_combine_async_kernel_api,
    ep_combine_wait_kernel_api,
    ep_fused_combine_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    EPLocalSyncCounters,
    combine_wait_kernel,
    combine_async_kernel,
    dispatch_wait_kernel,
    dispatch_async_kernel,
)
from std.testing import assert_equal


# Routing mode is read at run time, not baked in: measuring an optimization
# means running both modes over the same binary, and a compile-time switch
# would double every rebuild in that sweep.
def skew_enabled() -> Bool:
    return getenv("EP_SKEW") == "1"


def skew_exponent() raises -> Float64:
    var raw = getenv("EP_SKEW_S")
    if raw.byte_length() == 0:
        return 1.0
    return Float64(Int(raw)) / 1000.0


# Each timed iteration consumes one buffer slot, so slots == iterations.
comptime N_TOK_DECODE = get_defined_int["n_tok_decode", 16]()
comptime N_TOK_PREFILL = get_defined_int["n_tok_prefill", 2048]()
comptime SLOTS_DECODE = get_defined_int["slots_decode", 400]()
comptime SLOTS_PREFILL = get_defined_int["slots_prefill", 10]()

comptime HIDDEN = get_defined_int["hidden_size", 6144]()
comptime TOP_K = get_defined_int["top_k", 4]()
comptime N_EXPERTS = get_defined_int["n_experts", 128]()


def zipf_expert_weights[
    n_experts: Int, n_ranks: Int
](s: Float64, out result: List[Float64]):
    """Per-expert sampling weight: Zipf popularity, balanced across GPUs.

    Walks popularity ranks in decreasing order and gives each to the lightest
    GPU that still has a free expert slot, so every GPU stays near its equal
    share of the tokens while individual (expert, rank) pairs get very hot.
    Dealing ranks round-robin instead leaves the Zipf head on one GPU, which
    turns this into a data-placement problem and hides what combine is
    actually governed by.

    Parameters:
        n_experts: Total experts across all ranks.
        n_ranks: Number of ranks.

    Args:
        s: Zipf exponent; larger concentrates more on the hottest experts.

    Returns:
        Sampling weight indexed by global expert id.
    """
    comptime n_local = n_experts // n_ranks

    var load = List[Float64](capacity=n_ranks)
    var filled = List[Int](capacity=n_ranks)
    for _ in range(n_ranks):
        load.append(0.0)
        filled.append(0)

    result = List[Float64](capacity=n_experts)
    for _ in range(n_experts):
        result.append(0.0)

    for pos in range(n_experts):
        var w = 1.0 / (Float64(pos + 1) ** s)
        var best = -1
        for g in range(n_ranks):
            if filled[g] >= n_local:
                continue
            if best < 0 or load[g] < load[best]:
                best = g
        result[best * n_local + filled[best]] = w
        load[best] += w
        filled[best] += 1


def routing_cdf[n_experts: Int, n_ranks: Int](out result: List[Float64]) raises:
    """Cumulative sampling distribution, empty for the uniform mode."""
    result = List[Float64]()
    if not skew_enabled():
        return
    var w = zipf_expert_weights[n_experts, n_ranks](skew_exponent())
    var total = 0.0
    for e in range(n_experts):
        total += w[e]
    var acc = 0.0
    result = List[Float64](capacity=n_experts)
    for e in range(n_experts):
        acc += w[e] / total
        result.append(acc)


def fill_topk_ids[
    origin: MutOrigin, //, n_experts: Int, top_k: Int
](
    topk_ids: UnsafePointer[Int32, origin],
    n_tokens: Int,
    cdf: List[Float64],
) -> None:
    """Draws `top_k` distinct experts for each of `n_tokens` tokens.

    An empty `cdf` means the uniform draw. Either way the ids for one token
    must be distinct: a token is never routed to the same expert twice.
    """
    var uniform = len(cdf) == 0

    @inline(.always)
    def draw() {imm} -> Int32:
        if uniform:
            return Int32(Int(random_float64(0.0, Float64(n_experts))))
        var u = random_float64(0.0, 1.0)
        var lo = 0
        var hi = n_experts - 1
        while lo < hi:
            var mid = (lo + hi) // 2
            if u <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        return Int32(lo)

    for tok in range(n_tokens):
        var row = topk_ids + tok * top_k
        for k in range(top_k):
            var pick = draw()
            var clash = True
            while clash:
                clash = False
                for j in range(k):
                    if row[j] == pick:
                        clash = True
                        break
                if clash:
                    pick = draw()
            row[k] = pick


def bench_combine[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
](list_of_ctx: List[DeviceContext]) raises:
    comptime input_type = DType.bfloat16
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank

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
        "Running ep_combine bench: input_type:",
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
    var dispatch_send_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var dispatch_recv_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var dispatch_recv_count_bufs_list = List[DeviceBuffer[.uint64]](capacity=n_ranks)

    # Buffers for combine phase
    var combine_send_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var combine_recv_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var combine_recv_count_bufs_list = List[DeviceBuffer[.uint64]](capacity=n_ranks)

    # Shared atomic counter buffer for dispatch and combine
    var atomic_counters_list = List[DeviceBuffer[.int32]](capacity=n_ranks)

    var host_topk_ids_list = Array[UnsafePointer[Int32, MutAnyOrigin], n_ranks](uninitialized=True)

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

        host_topk_ids_list[i] = alloc[Int32](n_slots * n_tokens_per_rank * top_k).as_unsafe_any_origin()

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
    # The kernel APIs take the reduced output, `(tokens, hidden)`, not the
    # per-top-k buffer the raw kernels write into.
    var output_2_layout = row_major((n_tokens_per_rank, Idx[hidden_size]))

    # One distribution shared by every rank and slot. Each slot is an
    # independent buffer set, so reusing it repeats the distribution without
    # aliasing state between iterations.
    var cdf = routing_cdf[n_experts, n_ranks]()

    for dev_idx in range(n_ranks):
        var ctx = list_of_ctx[dev_idx]
        seed(dev_idx)
        fill_topk_ids[n_experts, top_k](
            host_topk_ids_list[dev_idx], n_slots * n_tokens_per_rank, cdf
        )
        # Perf-only: token content is never verified, so skip the slow randn
        # fill and the input upload.
        ctx.enqueue_copy(
            device_topk_bufs_list[dev_idx], host_topk_ids_list[dev_idx]
        )

    # Report the imbalance this run actually produced, since that -- not the
    # byte count -- is what sets combine_async's cost. A pair is
    # (destination expert, source rank), the unit of work one thread block
    # owns, so `max/mean` over pairs is the factor by which the slowest block
    # outlasts the average one.
    var pair_counts = alloc[Int32](n_experts * n_ranks)
    var rank_counts = alloc[Int32](n_ranks)
    for i in range(n_experts * n_ranks):
        pair_counts[i] = 0
    for i in range(n_ranks):
        rank_counts[i] = 0
    for src_rank in range(n_ranks):
        for i in range(n_slots * n_tokens_per_rank * top_k):
            var e = Int(host_topk_ids_list[src_rank][i])
            pair_counts[e * n_ranks + src_rank] += 1
            rank_counts[e // n_local_experts] += 1
    var max_pair: Int = 0
    var total: Int = 0
    for i in range(n_experts * n_ranks):
        total += Int(pair_counts[i])
        if Int(pair_counts[i]) > max_pair:
            max_pair = Int(pair_counts[i])
    var mean_pair = Float64(total) / Float64(n_experts * n_ranks)
    var max_rank: Int = 0
    for i in range(n_ranks):
        if Int(rank_counts[i]) > max_rank:
            max_rank = Int(rank_counts[i])
    print(
        "ROUTING mode:",
        "skewed" if skew_enabled() else "uniform",
        ",max_pair:",
        max_pair,
        ",mean_pair:",
        mean_pair,
        ",pair_imbalance:",
        Float64(max_pair) / mean_pair,
        ",busiest_gpu_share:",
        Float64(max_rank) / Float64(total),
    )
    pair_counts.free()
    rank_counts.free()

    # fmt: off
    # Dispatch buffers
    var dispatch_recv_bufs_inputs = Array[Array[UnsafePointer[UInt8, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)
    var dispatch_recv_count_bufs_inputs = Array[Array[UnsafePointer[UInt64, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)

    # Combine buffers
    var combine_recv_bufs_inputs = Array[Array[UnsafePointer[UInt8, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)
    var combine_recv_count_bufs_inputs = Array[Array[UnsafePointer[UInt64, MutAnyOrigin], n_ranks], n_slots](uninitialized=True)

    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            dispatch_recv_bufs_inputs[slot_idx][dev_idx] = (dispatch_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * msg_bytes).as_unsafe_any_origin()
            dispatch_recv_count_bufs_inputs[slot_idx][dev_idx] = (dispatch_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts).as_unsafe_any_origin()
            combine_recv_bufs_inputs[slot_idx][dev_idx] = (combine_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes).as_unsafe_any_origin()
            combine_recv_count_bufs_inputs[slot_idx][dev_idx] = (combine_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts).as_unsafe_any_origin()

    # Dispatch helpers
    @inline(.always)
    @__parameter
    def get_dispatch_send_buf_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt8, MutAnyOrigin]) raises:
        result = (dispatch_send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * msg_bytes).as_unsafe_any_origin()

    # Combine helpers
    @inline(.always)
    @__parameter
    def get_combine_send_buf_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt8, MutAnyOrigin]) raises:
        result = (combine_send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * combine_msg_bytes).as_unsafe_any_origin()

    @inline(.always)
    @__parameter
    def get_combine_recv_buf_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt8, MutAnyOrigin]) raises:
        result = (combine_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes).as_unsafe_any_origin()

    @inline(.always)
    @__parameter
    def get_combine_recv_count_ptr(dev_idx: Int, slot_idx: Int, out result: UnsafePointer[UInt64, MutAnyOrigin]) raises:
        result = (combine_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts).as_unsafe_any_origin()

    @inline(.always)
    @__parameter
    def get_atomic_counters(dev_idx: Int, slot_idx: Int, out result: EPLocalSyncCounters[n_experts]) raises:
        return EPLocalSyncCounters[n_experts](atomic_counters_list[dev_idx].unsafe_ptr() + slot_idx * EPLocalSyncCounters[n_experts].total_size())

    # fmt: off
    comptime counters_size = EPLocalSyncCounters[n_experts].total_size()
    var counters_layout = row_major[counters_size]()

    @inline(.always)
    @__parameter
    def get_atomic_counters_tt(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(counters_layout), MutAnyOrigin]) raises:
        return type_of(result)(
            ptr=(atomic_counters_list[dev_idx].unsafe_ptr() + slot_idx * counters_size).as_unsafe_any_origin(), layout=counters_layout
        )
    # fmt: on

    @inline(.always)
    @__parameter
    def get_topk_ids_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            .int32, type_of(topk_ids_layout), ImmutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_topk_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * n_tokens_per_rank * top_k
            ).as_unsafe_any_origin(),
            layout=topk_ids_layout,
        )

    @inline(.always)
    @__parameter
    def get_input_tokens_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            input_type, type_of(input_tokens_layout), ImmutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_input_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * n_tokens_per_rank * hidden_size
            ).as_unsafe_any_origin(),
            layout=input_tokens_layout,
        )

    @inline(.always)
    @__parameter
    def get_output_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            input_type, type_of(output_tt_layout), MutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_output_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * max_recv_num_tokens * hidden_size
            ).as_unsafe_any_origin(),
            layout=output_tt_layout,
        )

    @inline(.always)
    @__parameter
    def get_row_offsets_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            .uint32, type_of(row_offsets_layout), MutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_row_offsets_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * (n_local_experts + 1)
            ).as_unsafe_any_origin(),
            layout=row_offsets_layout,
        )

    @inline(.always)
    @__parameter
    def get_expert_ids_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            .int32, type_of(expert_ids_layout), MutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_expert_ids_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * n_local_experts
            ).as_unsafe_any_origin(),
            layout=expert_ids_layout,
        )

    @inline(.always)
    @__parameter
    def get_src_token_info_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            .int32, type_of(src_token_info_layout), MutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_src_token_info_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * max_recv_num_tokens * 2
            ).as_unsafe_any_origin(),
            layout=src_token_info_layout,
        )

    @inline(.always)
    @__parameter
    def get_output_2_tensor(
        dev_idx: Int,
        slot_idx: Int,
        out result: TileTensor[
            input_type, type_of(output_2_layout), MutAnyOrigin
        ],
    ) raises:
        return type_of(result)(
            ptr=(
                device_output_2_bufs_list[dev_idx].unsafe_ptr()
                + slot_idx * n_tokens_per_rank * top_k * hidden_size
            ).as_unsafe_any_origin(),
            layout=output_2_layout,
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

    # Combine callback kernel
    comptime combine_wait = combine_wait_kernel[
        input_type,
        hw_info.max_thread_block_size,
        type_of(output_2_layout),
        hw_info.sm_count,
        top_k,
        n_experts,
        n_ranks,
        combine_msg_bytes,
        n_tokens_per_rank,
    ]

    @inline(.always)
    @__parameter
    def run_dispatch_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_function[dispatch_async](
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

    @inline(.always)
    @__parameter
    def run_dispatch_async_wait(dev_idx: Int, slot_idx: Int) raises:
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
    @__parameter
    def run_full_dispatch(dev_idx: Int, slot_idx: Int) raises:
        run_dispatch_async(dev_idx, slot_idx)
        run_dispatch_async_wait(dev_idx, slot_idx)

    # The kernel APIs take peer pointers as uint64 tensors, one entry per
    # rank, rather than the raw arrays the kernels themselves take.
    # fmt: off
    var c_send_ptrs = alloc[UInt64](n_slots * n_ranks)
    var c_recv_ptrs = alloc[UInt64](n_slots * n_ranks)
    var c_recv_count_ptrs = alloc[UInt64](n_slots * n_ranks)
    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            var ptr_idx = slot_idx * n_ranks + dev_idx
            c_send_ptrs[ptr_idx] = UInt64(Int(combine_send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * combine_msg_bytes))
            c_recv_ptrs[ptr_idx] = UInt64(Int(combine_recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k * combine_msg_bytes))
            c_recv_count_ptrs[ptr_idx] = UInt64(Int(combine_recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts))

    var c_ptrs_layout = row_major[n_ranks]()

    @inline(.always)
    @__parameter
    def get_c_send_ptrs(slot_idx: Int, out result: TileTensor[.uint64, type_of(c_ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(c_send_ptrs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=c_ptrs_layout)

    @inline(.always)
    @__parameter
    def get_c_recv_ptrs(slot_idx: Int, out result: TileTensor[.uint64, type_of(c_ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(c_recv_ptrs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=c_ptrs_layout)

    @inline(.always)
    @__parameter
    def get_c_recv_count_ptrs(slot_idx: Int, out result: TileTensor[.uint64, type_of(c_ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(c_recv_count_ptrs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=c_ptrs_layout)
    # fmt: on

    # combine_wait reduces over top_k only when it is given router weights;
    # without them it writes the unreduced (tokens, top_k, hidden) buffer,
    # which is not the shape the kernel APIs accept. Production always passes
    # weights, so the benchmark does too. Unit weights keep the arithmetic
    # identical in cost to the real thing.
    @inline(.always)
    @__parameter
    def unit_router_weight[
        width: Int
    ](token_idx: Int, topk_id: Int) capturing -> SIMD[.float32, width]:
        return SIMD[.float32, width](1.0)

    @inline(.always)
    @__parameter
    def run_combine_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ep_combine_async_kernel_api[
            input_type,
            hidden_size,
            top_k,
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
            use_shmem=False,
        ](
            get_atomic_counters_tt(dev_idx, slot_idx),
            get_output_tensor(dev_idx, slot_idx).as_imm(),
            get_src_token_info_tensor(dev_idx, slot_idx).as_imm(),
            get_c_send_ptrs(slot_idx),
            get_c_recv_ptrs(slot_idx),
            get_c_recv_count_ptrs(slot_idx),
            ctx,
        )

    @inline(.always)
    @__parameter
    def run_combine_async_wait(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ep_combine_wait_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
            router_weights_wrapper=unit_router_weight,
        ](
            get_output_2_tensor(dev_idx, slot_idx),
            get_atomic_counters_tt(dev_idx, slot_idx),
            get_c_recv_ptrs(slot_idx),
            get_c_recv_count_ptrs(slot_idx),
            ctx,
        )

    @inline(.always)
    @__parameter
    def run_e2e(dev_idx: Int, slot_idx: Int) raises:
        run_combine_async(dev_idx, slot_idx)
        run_combine_async_wait(dev_idx, slot_idx)

    @inline(.always)
    @__parameter
    def run_fused_combine(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ep_fused_combine_kernel_api[
            hidden_size,
            top_k,
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
            router_weights_wrapper=unit_router_weight,
            use_shmem=False,
        ](
            get_output_2_tensor(dev_idx, slot_idx),
            get_atomic_counters_tt(dev_idx, slot_idx),
            get_output_tensor(dev_idx, slot_idx).as_imm(),
            get_src_token_info_tensor(dev_idx, slot_idx).as_imm(),
            get_c_send_ptrs(slot_idx),
            get_c_recv_ptrs(slot_idx),
            get_c_recv_count_ptrs(slot_idx),
            ctx,
        )

    @inline(.always)
    @__parameter
    def clean_up(dev_idx: Int) raises:
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

    # Wall-clock cross-check, one op per slot.
    #
    # `combine_async` consumes the per-(expert, rank) counters that
    # `dispatch_wait` produced and zeroes them on the way out, so a second
    # run against the same slot finds zero tokens to send and copies
    # nothing. Any measurement that replays a slot therefore times an empty
    # kernel. This section runs each slot exactly once, against state
    # freshly produced above, and times the batch by the clock.
    var t0 = perf_counter_ns()
    for slot_idx in range(n_slots):
        for dev_i in range(n_ranks):
            run_combine_async(dev_i, slot_idx)
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()
    var t1 = perf_counter_ns()
    print(
        "WALLCLOCK,combine,",
        Float64(t1 - t0) / 1e6 / Float64(n_slots),
        "ms/op",
    )

    t0 = perf_counter_ns()
    for slot_idx in range(n_slots):
        for dev_i in range(n_ranks):
            run_combine_async_wait(dev_i, slot_idx)
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()
    t1 = perf_counter_ns()
    print(
        "WALLCLOCK,combine_wait,",
        Float64(t1 - t0) / 1e6 / Float64(n_slots),
        "ms/op",
    )

    # Restore per-slot state for the sections below.
    for dev_i in range(n_ranks):
        for slot_idx in range(n_slots):
            run_full_dispatch(dev_i, slot_idx)
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @inline(.always)
    def call_fn_combine(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
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
    def call_fn_combine_wait(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
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

    # Split vs fused, timed identically: one host thread per rank, each
    # timing its own device with `execution_time_iter`, then take the
    # slowest rank. Driving every device from a single host thread instead
    # serializes the launches, and the fused kernel, which spins for peer
    # arrivals, charges that serialization to itself.
    #
    # Like `combine_wait` above, this needs at least `n_ranks` AsyncRT
    # worker threads: every rank waits on its peers, so a rank whose host
    # thread never runs hangs the others.
    #
    # Each section re-runs the full dispatch first because `combine_async`
    # zeroes the counters it consumes, so a slot only carries work once.
    for dev_i in range(n_ranks):
        clean_up(dev_i)
        list_of_ctx[dev_i].synchronize()
    for dev_i in range(n_ranks):
        for slot_idx in range(n_slots):
            run_full_dispatch(dev_i, slot_idx)
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @inline(.always)
    def call_fn_e2e_split(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_e2e(dev_id, cache_iter + 1)

    def per_gpu_e2e_split(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_e2e_split, list_of_ctx[i])

        # Slot 0 warms this rank up; the timed batch takes the rest, one
        # fresh slot per iteration, which is also what keeps each slot's
        # counters unconsumed until its own iteration.
        run_e2e(i, 0)
        list_of_ctx[i].synchronize()

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench e2e_split"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots - 1,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_e2e_split, n_ranks)

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

    for dev_i in range(n_ranks):
        clean_up(dev_i)
        list_of_ctx[dev_i].synchronize()
    for dev_i in range(n_ranks):
        for slot_idx in range(n_slots):
            run_full_dispatch(dev_i, slot_idx)
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @inline(.always)
    def call_fn_e2e_fused(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_fused_combine(dev_id, cache_iter + 1)

    def per_gpu_e2e_fused(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_e2e_fused, list_of_ctx[i])

        run_fused_combine(i, 0)
        list_of_ctx[i].synchronize()

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench e2e_fused"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots - 1,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_e2e_fused, n_ranks)

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

    for dev_idx in range(n_ranks):
        host_topk_ids_list[dev_idx].free()


def bench_all_dtypes[
    n_ranks: Int, n_tokens_per_rank: Int, n_slots: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Runs combine at one shape.

    Only BF16: combine moves the experts\' outputs, which the dispatch setup
    in this benchmark produces, so its dtype is not independently selectable
    the way a dispatch token format is. BF16 is what MoE experts emit.
    """
    print(
        "\n===== combine shape: tokens_per_rank=",
        n_tokens_per_rank,
        " slots=",
        n_slots,
        " hidden=",
        HIDDEN,
        " top_k=",
        TOP_K,
        " n_experts=",
        N_EXPERTS,
        " =====",
    )

    bench_combine[
        hidden_size=HIDDEN,
        top_k=TOP_K,
        n_experts=N_EXPERTS,
        n_ranks=n_ranks,
        n_slots=n_slots,
        n_tokens_per_rank=n_tokens_per_rank,
    ](list_of_ctx)


def main() raises:
    comptime n_ranks = 8

    if enable_p2p():
        print("Enabled P2P Mem Access on all GPUs.")
    else:
        raise Error("Cannot enable P2P Mem Access!")

    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "Only NVIDIA and AMD GPUs are supported"

    if DeviceContext.number_of_devices() != n_ranks:
        print("This benchmark requires exactly 8 GPUs; skipping.")
        return

    var ctx = List[DeviceContext]()
    for i in range(n_ranks):
        ctx.append(DeviceContext(device_id=i))

    # Decode first: it is the cheap one, and running it before the prefill
    # allocations keeps peak device memory down.
    bench_all_dtypes[
        n_ranks=n_ranks,
        n_tokens_per_rank=N_TOK_DECODE,
        n_slots=SLOTS_DECODE,
    ](ctx)

    bench_all_dtypes[
        n_ranks=n_ranks,
        n_tokens_per_rank=N_TOK_PREFILL,
        n_slots=SLOTS_PREFILL,
    ](ctx)
