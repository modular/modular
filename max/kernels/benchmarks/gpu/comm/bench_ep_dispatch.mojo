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
"""Benchmarks the three expert-parallelism dispatch kernels over P2P.

Covers `dispatch_async`, `dispatch_wait`, and the fused kernel that replaces
both, on 8 GPUs sharing an NVLink or XGMI fabric. Multi-node SHMEM is out of
scope: it is a different transport with different costs, and the single-node
path is the one these kernels are tuned for.

Each vendor is driven through the token formats it actually dispatches with --
BF16 and MXFP8 on both, NVFP4 on NVIDIA, MXFP4 on AMD. Two shapes run per
format: a decode-sized point, where cost is per launch, and a prefill-sized
point, where it is per byte. They are different regimes, and an optimization
can help one while doing nothing for the other.

Routing comes in two modes, because they measure different things:

- `-D skew=0` draws each token's experts uniformly, so every expert and every
  (expert, rank) pair carries the same load. This measures bandwidth and
  launch overhead.
- `-D skew=1` draws them from a Zipf popularity distribution, reproducing the
  shape of routing captured from a served MoE: a few very hot experts while
  each GPU still holds close to its equal share of the bytes. That separation
  is the point -- it makes the imbalance a scheduling problem inside a GPU
  rather than a data-placement one. `dispatch_wait` is governed by it,
  `dispatch_async` is not.

Every run prints the imbalance it actually generated, because a speedup
figure from this benchmark means nothing without it.

Correctness is not checked here; `test_p2p_ep_dispatch.mojo` owns that.
"""

from std.os import getenv
from std.random import random_float64, seed
from std.sys import (
    get_defined_int,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)


# Routing mode is read at run time, not baked in: measuring an optimization
# means running both modes over the same binary, and a compile-time switch
# would double every rebuild in that sweep.
#   EP_SKEW=0 (default)  draw experts uniformly
#   EP_SKEW=1            draw from a Zipf popularity distribution
#   EP_SKEW_S            Zipf exponent in thousandths (default 1000 = 1.0),
#                        which reproduces the imbalance of routing captured
#                        from a served MoE at 128 experts
def skew_enabled() -> Bool:
    return getenv("EP_SKEW") == "1"


def skew_exponent() raises -> Float64:
    var raw = getenv("EP_SKEW_S")
    if raw.byte_length() == 0:
        return 1.0
    return Float64(Int(raw)) / 1000.0


# Each timed iteration consumes one buffer slot, so slots == iterations.
# Decode launches are tiny, so timing them takes more of them.
comptime N_TOK_DECODE = get_defined_int["n_tok_decode", 16]()
comptime N_TOK_PREFILL = get_defined_int["n_tok_prefill", 2048]()
comptime SLOTS_DECODE = get_defined_int["slots_decode", 400]()
comptime SLOTS_PREFILL = get_defined_int["slots_prefill", 10]()

comptime HIDDEN = get_defined_int["hidden_size", 6144]()
comptime TOP_K = get_defined_int["top_k", 4]()
comptime N_EXPERTS = get_defined_int["n_experts", 128]()

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
from layout import TileTensor, Idx, row_major
from std.math import ceildiv
from shmem.ep import (
    ep_dispatch_async_kernel_api,
    ep_dispatch_wait_kernel_api,
    ep_fused_dispatch_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    MXTokenFormat,
    NVBlockScaledTokenFormat,
    TokenFormat,
)
from std.testing import assert_almost_equal, assert_equal

from std.math import align_up
from linalg.fp4_utils import (
    E2M1_TO_FLOAT32,
    MXFP4_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    get_scale_factor,
)
from max.gpu.host.info import _is_sm10x_gpu, MI355X


def zipf_expert_weights[
    n_experts: Int, n_ranks: Int
](s: Float64, out result: List[Float64]):
    """Per-expert sampling weight: Zipf popularity, balanced across GPUs.

    Walks popularity ranks in decreasing order and gives each to the lightest
    GPU that still has a free expert slot. That keeps every GPU near its equal
    share of the tokens while individual experts stay very hot. Dealing ranks
    round-robin instead leaves the Zipf head on one GPU, which turns the
    benchmark into a data-placement problem and hides the imbalance worth
    measuring.

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


def fill_topk_ids[
    n_experts: Int, top_k: Int
](
    topk_ids: UnsafePointer[Int32, MutUntrackedOrigin],
    n_tokens: Int,
    cdf: List[Float64],
) -> None:
    """Draws `top_k` distinct experts for each of `n_tokens` tokens.

    An empty `cdf` means the uniform draw. Either way the ids for one token
    must be distinct: a token is never routed to the same expert twice, and
    the kernels depend on it.
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


def report_routing[
    n_experts: Int, n_ranks: Int, top_k: Int
](topk_ids: UnsafePointer[Int32, MutUntrackedOrigin], n_tokens: Int) -> None:
    """Prints the imbalance this run will actually see.

    The two rebalancing optimizations are within noise on even routing by
    construction, so a speedup from this benchmark only means something next
    to the skew it was measured at.
    """
    comptime n_local = n_experts // n_ranks
    var per_expert = List[Int](capacity=n_experts)
    for _ in range(n_experts):
        per_expert.append(0)
    for tok in range(n_tokens):
        for k in range(top_k):
            per_expert[Int(topk_ids[tok * top_k + k])] += 1

    var total = 0
    var max_expert = 0
    for e in range(n_experts):
        total += per_expert[e]
        max_expert = max(max_expert, per_expert[e])

    var per_gpu = List[Int](capacity=n_ranks)
    for _ in range(n_ranks):
        per_gpu.append(0)
    for e in range(n_experts):
        per_gpu[e // n_local] += per_expert[e]
    var max_gpu = 0
    for g in range(n_ranks):
        max_gpu = max(max_gpu, per_gpu[g])

    print(
        "ROUTING mode:",
        "skewed" if skew_enabled() else "uniform",
        "expert_imbalance:",
        Float64(max_expert) * Float64(n_experts) / Float64(total),
        "busiest_gpu_share:",
        Float64(max_gpu) / Float64(total),
        "even_share:",
        1.0 / Float64(n_ranks),
    )


trait DispatchTestT(Deinitable):
    """Trait to unify the test dispatch logic for different token formats."""

    comptime hidden_size: Int
    comptime top_k: Int
    comptime n_experts: Int
    comptime n_ranks: Int
    comptime n_slots: Int
    comptime n_tokens_per_rank: Int
    comptime TokenFormatType: TokenFormat

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        ...

    @inline(.always)
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        ...


struct NVFP4DispatchTest[
    fp4_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank
    comptime n_local_experts = Self.n_experts // Self.n_ranks

    comptime scales_padded_size = Self.max_recv_num_tokens + Self.n_local_experts * SF_MN_GROUP_SIZE

    comptime uint8_last_dim = Self.hidden_size // 2

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.uint8_last_dim])
    )
    comptime output_scales_layout = row_major(
        (
            Self.scales_padded_size // SF_MN_GROUP_SIZE,
            Idx[ceildiv(Self.hidden_size, SF_ATOM_K * NVFP4_SF_VECTOR_SIZE)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    comptime output_scales_offset_layout = row_major[
        Self.n_experts // Self.n_ranks
    ]()
    comptime TokenFormatType = NVBlockScaledTokenFormat[
        quant_dtype=Self.fp4_dtype,
        scales_dtype=Self.scales_dtype,
        output_layout=type_of(Self.output_layout),
        scales_offset_layout=type_of(Self.output_scales_offset_layout),
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.fp4_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]
    var device_output_scales_offset_bufs_list: List[DeviceBuffer[.uint32]]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.fp4_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        self.device_output_scales_offset_bufs_list = List[
            DeviceBuffer[.uint32]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.fp4_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.uint8_last_dim
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots
                    * Self.scales_padded_size
                    * Self.hidden_size
                    // NVFP4_SF_VECTOR_SIZE
                )
            )
            self.device_output_scales_offset_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[.uint32](
                    Self.n_slots * (Self.n_experts // Self.n_ranks)
                )
            )

    @inline(.always)
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.uint8_last_dim,
            layout=Self.output_layout,
        )
        var output_scales_tensor = TileTensor(
            ptr=self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx
            * Self.scales_padded_size
            * Self.hidden_size
            // NVFP4_SF_VECTOR_SIZE,
            layout=Self.output_scales_layout,
        )
        var output_scales_offset_tensor = TileTensor(
            ptr=self.device_output_scales_offset_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * (Self.n_experts // Self.n_ranks),
            layout=Self.output_scales_offset_layout,
        )

        result = Self.TokenFormatType(
            output_tensor,
            output_scales_tensor,
            output_scales_offset_tensor,
            ctx,
        )


struct BF16DispatchTest[
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    """Unquantized dispatch: the widest message, so the bandwidth-bound end."""

    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.hidden_size])
    )
    comptime TokenFormatType = BF16TokenFormat[
        output_layout=type_of(Self.output_layout),
        Self.hidden_size,
        Self.top_k,
    ]

    var device_output_bufs_list: List[DeviceBuffer[.bfloat16]]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[.bfloat16]](
            capacity=Self.n_ranks
        )
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[.bfloat16](
                    Self.n_slots * Self.max_recv_num_tokens * Self.hidden_size
                )
            )

    @inline(.always)
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.hidden_size,
            layout=Self.output_layout,
        )
        return Self.TokenFormatType(output_tensor.as_unsafe_any_origin())


struct MXDispatchTest[
    # `uint8` selects MXFP4 (two E2M1 nibbles per byte); an FP8 dtype
    # (e.g. `float8_e4m3fn`) selects MXFP8 (one element per byte). Both use
    # one E8M0 scale per MXFP4_SF_VECTOR_SIZE elements.
    quant_dtype: DType,
    scales_dtype: DType,
    _hidden_size: Int,
    _top_k: Int,
    _n_experts: Int,
    _n_ranks: Int,
    _n_slots: Int,
    _n_tokens_per_rank: Int,
](DispatchTestT):
    comptime hidden_size = Self._hidden_size
    comptime top_k = Self._top_k
    comptime n_experts = Self._n_experts
    comptime n_ranks = Self._n_ranks
    comptime n_slots = Self._n_slots
    comptime n_tokens_per_rank = Self._n_tokens_per_rank
    comptime max_recv_num_tokens = min(
        Self.n_experts, Self.n_ranks * Self.top_k
    ) * Self.n_tokens_per_rank
    comptime n_local_experts = Self.n_experts // Self.n_ranks
    comptime k_scales = Self.hidden_size // MXFP4_SF_VECTOR_SIZE

    comptime scales_padded_size = Self.max_recv_num_tokens + Self.n_local_experts * SF_MN_GROUP_SIZE

    # Measure the KS224 up-proj fold, which is what MiniMax-M3 runs on CDNA4:
    # the scale store lands in the grouped matmul's `scale_4d` slot rather
    # than a row-major scale row, and the two have different store patterns.
    # The slot layout is a CDNA4 MFMA layout, so NVIDIA keeps the row-major
    # path it actually dispatches with.
    comptime fuse_a_scale_preshuffle = has_amd_gpu_accelerator()
    comptime max_padded_m = align_up(
        Self.n_tokens_per_rank * Self.n_ranks, 32
    ) if Self.fuse_a_scale_preshuffle else 0
    comptime scales_rows = (
        Self.n_local_experts * Self.max_padded_m
    ) if Self.fuse_a_scale_preshuffle else Self.max_recv_num_tokens
    comptime scales_per_slot = (
        Self.n_local_experts * Self.max_padded_m * Self.k_scales
    ) if Self.fuse_a_scale_preshuffle else (
        Self.scales_padded_size * Self.k_scales
    )

    comptime is_fp4 = Self.quant_dtype == DType.uint8
    comptime quant_last_dim = (
        Self.hidden_size // 2 if Self.is_fp4 else Self.hidden_size
    )

    comptime output_layout = row_major(
        (Self.max_recv_num_tokens, Idx[Self.quant_last_dim])
    )
    comptime output_scales_layout = row_major(
        (Idx[Self.scales_rows], Self.k_scales)
    )
    comptime TokenFormatType = MXTokenFormat[
        quant_dtype=Self.quant_dtype,
        scales_dtype=Self.scales_dtype,
        output_layout=type_of(Self.output_layout),
        scales_layout=type_of(Self.output_scales_layout),
        Self.hidden_size,
        Self.top_k,
        fuse_a_scale_preshuffle=Self.fuse_a_scale_preshuffle,
    ]

    var device_output_bufs_list: List[DeviceBuffer[Self.quant_dtype]]
    var device_output_scales_bufs_list: List[DeviceBuffer[Self.scales_dtype]]

    def __init__(out self, list_of_ctx: List[DeviceContext]) raises:
        self.device_output_bufs_list = List[DeviceBuffer[Self.quant_dtype]](
            capacity=Self.n_ranks
        )
        self.device_output_scales_bufs_list = List[
            DeviceBuffer[Self.scales_dtype]
        ](capacity=Self.n_ranks)
        for i in range(Self.n_ranks):
            self.device_output_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.quant_dtype](
                    Self.n_slots
                    * Self.max_recv_num_tokens
                    * Self.quant_last_dim
                )
            )
            self.device_output_scales_bufs_list.append(
                list_of_ctx[i].enqueue_create_buffer[Self.scales_dtype](
                    Self.n_slots * Self.scales_per_slot
                )
            )

    @inline(.always)
    def get_token_handler(
        self,
        dev_idx: Int,
        slot_idx: Int,
        ctx: DeviceContext,
        out result: Self.TokenFormatType,
    ):
        var output_tensor = TileTensor(
            ptr=self.device_output_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.max_recv_num_tokens * Self.quant_last_dim,
            layout=Self.output_layout,
        )
        var output_scales_tensor = TileTensor(
            ptr=self.device_output_scales_bufs_list[dev_idx].unsafe_ptr()
            + slot_idx * Self.scales_per_slot,
            layout=Self.output_scales_layout,
        )

        result = Self.TokenFormatType(
            output_tensor, output_scales_tensor, Self.max_padded_m
        )


def bench_dispatch_common[
    DispatchTestType: DispatchTestT,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime input_type = DType.bfloat16
    comptime hidden_size = DispatchTestType.hidden_size
    comptime top_k = DispatchTestType.top_k
    comptime n_experts = DispatchTestType.n_experts
    comptime n_ranks = DispatchTestType.n_ranks
    comptime n_slots = DispatchTestType.n_slots
    comptime n_tokens_per_rank = DispatchTestType.n_tokens_per_rank
    comptime token_fmt_type = DispatchTestType.TokenFormatType

    comptime msg_bytes = token_fmt_type.msg_size()
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_num_tokens = n_experts * n_tokens_per_rank

    comptime num_bytes = msg_bytes * top_k * n_tokens_per_rank

    var dispatch_test = DispatchTestType(list_of_ctx)

    print(
        "Running ep_dispatch bench:",
        token_fmt_type.get_type_name(),
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
        "msg_bytes:",
        msg_bytes,
        "send_bytes_per_rank:",
        num_bytes,
    )

    # fmt: off
    var send_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var recv_bufs_list = List[DeviceBuffer[.uint8]](capacity=n_ranks)
    var recv_count_bufs_list = List[DeviceBuffer[.uint64]](capacity=n_ranks)
    var atomic_counters_list = List[DeviceBuffer[.int32]](capacity=n_ranks)

    var host_topk_ids_list = Array[UnsafePointer[Int32, MutUntrackedOrigin], n_ranks](uninitialized=True)

    var device_topk_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var device_input_bufs_list = List[DeviceBuffer[input_type]](capacity=n_ranks)
    var device_row_offsets_bufs_list = List[DeviceBuffer[.uint32]](capacity=n_ranks)
    var device_expert_ids_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)
    var device_src_token_info_bufs_list = List[DeviceBuffer[.int32]](capacity=n_ranks)


    for i in range(n_ranks):
        var ctx = list_of_ctx[i]
        send_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[.uint8](n_slots * n_tokens_per_rank * msg_bytes))
        recv_bufs_list.append(ctx.enqueue_create_buffer[.uint8](n_slots * max_recv_num_tokens * msg_bytes))
        recv_count_bufs_list.append(ctx.enqueue_create_buffer[.uint64](n_slots * n_experts))
        atomic_counters_list.append(ctx.enqueue_create_buffer[.int32](
            n_slots * EPLocalSyncCounters[n_experts].total_size()
        ))
        ctx.enqueue_memset(atomic_counters_list[i], Int32(0))
        ctx.enqueue_memset(recv_count_bufs_list[i], UInt64.MAX_FINITE)

        host_topk_ids_list[i] = alloc[Int32](n_slots * n_tokens_per_rank * top_k)

        device_topk_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * n_tokens_per_rank * top_k))
        device_input_bufs_list.append(ctx.enqueue_create_buffer[input_type](n_slots * n_tokens_per_rank * hidden_size))
        device_row_offsets_bufs_list.append(ctx.enqueue_create_buffer[.uint32](n_slots * (n_local_experts + 1)))
        device_expert_ids_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * n_local_experts))
        device_src_token_info_bufs_list.append(ctx.enqueue_create_buffer[.int32](n_slots * max_recv_num_tokens * 2))
    # fmt: on

    var topk_ids_layout = row_major(n_tokens_per_rank, Idx[top_k])
    var input_tokens_layout = row_major((n_tokens_per_rank, Idx[hidden_size]))
    var row_offsets_layout = row_major[n_local_experts + 1]()
    var expert_ids_layout = row_major[n_local_experts]()
    var src_token_info_layout = row_major((Idx[max_recv_num_tokens], Idx[2]))
    var ptrs_layout = row_major[n_ranks]()
    comptime counters_size = EPLocalSyncCounters[n_experts].total_size()
    var counters_layout = row_major[counters_size]()

    # One distribution shared by every rank and slot. Each slot is an
    # independent buffer set, so reusing it repeats the distribution without
    # aliasing any state between iterations.
    var cdf = routing_cdf[n_experts, n_ranks]()

    for dev_idx in range(n_ranks):
        var ctx = list_of_ctx[dev_idx]
        seed(dev_idx)
        fill_topk_ids[n_experts, top_k](
            host_topk_ids_list[dev_idx], n_slots * n_tokens_per_rank, cdf
        )
        if dev_idx == 0:
            # Over every slot, not one: at the decode shape a single slot is
            # a few dozen draws over n_experts, and its imbalance is sampling
            # noise rather than the distribution being generated.
            report_routing[n_experts, n_ranks, top_k](
                host_topk_ids_list[dev_idx], n_slots * n_tokens_per_rank
            )
        # Perf-only: token content is never verified, so skip the slow randn
        # fill and the input upload; the kernels read whatever the device
        # buffer already holds.
        ctx.enqueue_copy(
            device_topk_bufs_list[dev_idx], host_topk_ids_list[dev_idx]
        )

    # fmt: off
    var send_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)
    var recv_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)
    var recv_count_ptrs_inputs = alloc[UInt64](n_slots * n_ranks)

    for slot_idx in range(n_slots):
        for dev_idx in range(n_ranks):
            var ptr_idx = slot_idx * n_ranks + dev_idx
            send_ptrs_inputs[ptr_idx] = UInt64(
                Int(send_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * msg_bytes)
            )
            recv_ptrs_inputs[ptr_idx] = UInt64(
                Int(recv_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * msg_bytes)
            )
            recv_count_ptrs_inputs[ptr_idx] = UInt64(
                Int(recv_count_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_experts)
            )

    @inline(.always)
    @__parameter
    def get_send_ptrs_tensor(slot_idx: Int, out result: TileTensor[.uint64, type_of(ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(send_ptrs_inputs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=ptrs_layout)

    @inline(.always)
    @__parameter
    def get_recv_ptrs_tensor(slot_idx: Int, out result: TileTensor[.uint64, type_of(ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)( ptr=(recv_ptrs_inputs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=ptrs_layout)

    @inline(.always)
    @__parameter
    def get_recv_count_ptrs_tensor(slot_idx: Int, out result: TileTensor[.uint64, type_of(ptrs_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(recv_count_ptrs_inputs + slot_idx * n_ranks).as_unsafe_any_origin(), layout=ptrs_layout)

    @inline(.always)
    @__parameter
    def get_atomic_counters_tensor( dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(counters_layout), MutAnyOrigin]) raises:
        return type_of(result)(
            ptr=(atomic_counters_list[dev_idx].unsafe_ptr() + slot_idx * counters_size).as_unsafe_any_origin(), layout=counters_layout
        )

    @inline(.always)
    @__parameter
    def get_topk_ids_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(topk_ids_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_topk_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * top_k).as_unsafe_any_origin(), layout=topk_ids_layout)

    @inline(.always)
    @__parameter
    def get_input_tokens_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[input_type, type_of(input_tokens_layout), ImmutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_input_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_tokens_per_rank * hidden_size).as_unsafe_any_origin(), layout=input_tokens_layout)

    @inline(.always)
    @__parameter
    def get_row_offsets_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.uint32, type_of(row_offsets_layout), MutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_row_offsets_bufs_list[dev_idx].unsafe_ptr() + slot_idx * (n_local_experts + 1)).as_unsafe_any_origin(), layout=row_offsets_layout)

    @inline(.always)
    @__parameter
    def get_expert_ids_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(expert_ids_layout), MutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_expert_ids_bufs_list[dev_idx].unsafe_ptr() + slot_idx * n_local_experts).as_unsafe_any_origin(), layout=expert_ids_layout)

    @inline(.always)
    @__parameter
    def get_src_token_info_tensor(dev_idx: Int, slot_idx: Int, out result: TileTensor[.int32, type_of(src_token_info_layout), MutAnyOrigin]) raises:
        return type_of(result)(ptr=(device_src_token_info_bufs_list[dev_idx].unsafe_ptr() + slot_idx * max_recv_num_tokens * 2).as_unsafe_any_origin(), layout=src_token_info_layout)
    # fmt: on

    @inline(.always)
    @__parameter
    def run_dispatch_async(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ep_dispatch_async_kernel_api[
            token_fmt_type,
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
            use_shmem=False,
        ](
            get_atomic_counters_tensor(dev_idx, slot_idx),
            get_input_tokens_tensor(dev_idx, slot_idx),
            get_topk_ids_tensor(dev_idx, slot_idx),
            get_send_ptrs_tensor(slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            ctx,
        )

    @inline(.always)
    @__parameter
    def run_dispatch_async_wait(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        var format_handler = dispatch_test.get_token_handler(
            dev_idx, slot_idx, ctx
        )
        ep_dispatch_wait_kernel_api[
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            "gpu",
        ](
            format_handler,
            get_row_offsets_tensor(dev_idx, slot_idx),
            get_expert_ids_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            get_atomic_counters_tensor(dev_idx, slot_idx),
            ctx,
        )

    @inline(.always)
    @__parameter
    def run_e2e(dev_idx: Int, slot_idx: Int) raises:
        run_dispatch_async(dev_idx, slot_idx)
        run_dispatch_async_wait(dev_idx, slot_idx)

    @inline(.always)
    @__parameter
    def run_fused_dispatch(dev_idx: Int, slot_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        var format_handler = dispatch_test.get_token_handler(
            dev_idx, slot_idx, ctx
        )
        ep_fused_dispatch_kernel_api[
            n_experts,
            n_tokens_per_rank,
            n_ranks,
            1,
            False,
            "gpu",
            use_shmem=False,
        ](
            format_handler,
            get_row_offsets_tensor(dev_idx, slot_idx),
            get_expert_ids_tensor(dev_idx, slot_idx),
            get_src_token_info_tensor(dev_idx, slot_idx),
            get_atomic_counters_tensor(dev_idx, slot_idx),
            get_input_tokens_tensor(dev_idx, slot_idx),
            get_topk_ids_tensor(dev_idx, slot_idx),
            get_send_ptrs_tensor(slot_idx),
            get_recv_ptrs_tensor(slot_idx),
            get_recv_count_ptrs_tensor(slot_idx),
            ctx,
        )

    @inline(.always)
    @__parameter
    def clean_up(dev_idx: Int) raises:
        var ctx = list_of_ctx[dev_idx]
        ctx.enqueue_memset(atomic_counters_list[dev_idx], Int32(0))
        ctx.enqueue_memset(recv_count_bufs_list[dev_idx], UInt64.MAX_FINITE)

    # warm up by running once
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

    # First, bench the dispatch kernel overhead

    @inline(.always)
    def call_fn_dispatch(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_dispatch_async(dev_id, cache_iter)

    def per_gpu_dispatch(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_dispatch, list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench dispatch"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_dispatch, n_ranks)

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

    # Then, bench the dispatch_wait kernel overhead
    for dev_i in range(n_ranks):
        list_of_ctx[dev_i].synchronize()

    @inline(.always)
    def call_fn_dispatch_wait(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_dispatch_async_wait(dev_id, cache_iter)

    def per_gpu_dispatch_wait(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_dispatch_wait, list_of_ctx[i])

        var bench_config = BenchConfig()
        bench_config.show_progress = False
        var b = Bench(bench_config^)
        b.bench_function(
            bench_iter,
            BenchId("bench dispatch_wait"),
            [ThroughputMeasure(BenchMetric.bytes, 0)],
            fixed_iterations=n_slots,
        )
        results_b[i] = b.info_vec[0].copy()

    sync_parallelize(per_gpu_dispatch_wait, n_ranks)

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
    # Like `dispatch_wait` above, this needs at least `n_ranks` AsyncRT
    # worker threads: every rank waits on its peers, so a rank whose host
    # thread never runs hangs the others.
    for dev_i in range(n_ranks):
        clean_up(dev_i)
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
        # fresh slot per iteration so it needs no counter reset inside.
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

    @inline(.always)
    def call_fn_e2e_fused(ctx: DeviceContext, cache_iter: Int) raises {}:
        var dev_id = Int(ctx.id())
        run_fused_dispatch(dev_id, cache_iter + 1)

    def per_gpu_e2e_fused(i: Int) raises {mut results_b, imm}:
        @inline(.always)
        def bench_iter(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, call_fn_e2e_fused, list_of_ctx[i])

        run_fused_dispatch(i, 0)
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


def bench_dispatch_nvfp4[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = NVFP4DispatchTest[
        fp4_dtype=DType.uint8,
        scales_dtype=DType.float8_e4m3fn,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    bench_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def bench_dispatch_bf16[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = BF16DispatchTest[
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    bench_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def bench_dispatch_mxfp4[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    comptime dispatch_test_type = MXDispatchTest[
        quant_dtype=DType.uint8,
        scales_dtype=DType.float8_e8m0fnu,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    bench_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def bench_dispatch_mxfp8[
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_slots: Int,
    n_tokens_per_rank: Int,
    bench_e2e: Bool = False,
](list_of_ctx: List[DeviceContext]) raises:
    # OCP E4M3 (`float8_e4m3fn`) is the MXFP8 element encoding CDNA4's
    # f8f6f4 MFMA consumes and the only FP8 flavor `MXFormat` names; the
    # MI300-era `float8_e4m3fnuz` is not an MX wire format.
    comptime dispatch_test_type = MXDispatchTest[
        quant_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float8_e8m0fnu,
        _hidden_size=hidden_size,
        _top_k=top_k,
        _n_experts=n_experts,
        _n_ranks=n_ranks,
        _n_slots=n_slots,
        _n_tokens_per_rank=n_tokens_per_rank,
    ]
    bench_dispatch_common[
        DispatchTestType=dispatch_test_type, bench_e2e=bench_e2e
    ](list_of_ctx)


def bench_all_formats[
    n_ranks: Int, n_tokens_per_rank: Int, n_slots: Int
](list_of_ctx: List[DeviceContext]) raises:
    """Runs every token format this vendor dispatches with, at one shape."""
    print(
        "\n===== dispatch shape: tokens_per_rank=",
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

    bench_dispatch_bf16[
        hidden_size=HIDDEN,
        top_k=TOP_K,
        n_experts=N_EXPERTS,
        n_ranks=n_ranks,
        n_slots=n_slots,
        n_tokens_per_rank=n_tokens_per_rank,
    ](list_of_ctx)

    bench_dispatch_mxfp8[
        hidden_size=HIDDEN,
        top_k=TOP_K,
        n_experts=N_EXPERTS,
        n_ranks=n_ranks,
        n_slots=n_slots,
        n_tokens_per_rank=n_tokens_per_rank,
    ](list_of_ctx)

    comptime if has_amd_gpu_accelerator():
        bench_dispatch_mxfp4[
            hidden_size=HIDDEN,
            top_k=TOP_K,
            n_experts=N_EXPERTS,
            n_ranks=n_ranks,
            n_slots=n_slots,
            n_tokens_per_rank=n_tokens_per_rank,
        ](list_of_ctx)
    else:
        # NVIDIA dispatches FP4 through the block-scaled format, not the MX
        # one; the two differ in scale layout and in dispatch_wait tile shape.
        bench_dispatch_nvfp4[
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
    bench_all_formats[
        n_ranks=n_ranks,
        n_tokens_per_rank=N_TOK_DECODE,
        n_slots=SLOTS_DECODE,
    ](ctx)

    bench_all_formats[
        n_ranks=n_ranks,
        n_tokens_per_rank=N_TOK_PREFILL,
        n_slots=SLOTS_PREFILL,
    ](ctx)
