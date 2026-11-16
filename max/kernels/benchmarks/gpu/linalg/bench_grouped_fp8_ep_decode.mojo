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
"""Times the SM100 blockwise-FP8 grouped GEMM at GLM-5.3-Flash EP4 shapes.

The operands are one rank's under TP4 attention and EP4 experts: 72 routed
experts (288 over 4 ranks) plus the fused shared expert in slot 0, the EP
receive buffer of `TOTAL_ROWS` FP8 rows with token-minor A scales, and slot
offsets padded to 4 rows a slot, as the dispatch writes them. A step of `T`
tokens over all ranks routes every token to 8 distinct experts drawn
uniformly from 288; this rank keeps the rows of its 72 experts and its shared
slot holds its own `T / 4` tokens. gate/up is N 4096, K 4096; down is N 4096,
K 2048.

A timed graph replays `REPS` launches, each on its own routing draw and on
one of `LAYERS` weight copies in turn, so the weights stream from DRAM as a
decode step's layers do. Arms run interleaved, in alternating order each
round, and the median round is reported with the weight bytes a launch reads
(non-empty slots x N x K). Arm 0 is the op the graph calls; every other arm's
output is compared with it before timing.

`--tokens=T1,T2,...` picks the steps: at MTP 5, 24/96/144/192 are the verify
steps of 4/16/24/32 requests, 4-32 their draft steps, and 8192 a prefill
chunk.
"""

from std.math import align_up
from std.sys import get_defined_int

from internal_utils import arg_parse
from layout import Idx, TileTensor, row_major
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
)
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

comptime NUM_SLOTS = 73
comptime NUM_EXPERTS = 288
comptime LOCAL_EXPERTS = 72
comptime TOP_K = 8
comptime N_RANKS = 4
comptime ROW_PAD = 4
comptime TOTAL_ROWS = 67584
comptime MAX_ROWS_PER_SLOT = 8192
comptime SCALE_BLOCK = 128
comptime LAYERS = 2
comptime REPS = 8
comptime REPLAYS = 5
comptime ROUNDS = get_defined_int["ROUNDS", 5]()
comptime NARMS = 5


def _hash(i: Int, stream: UInt32) -> UInt32:
    var x = UInt32(i & 0xFFFFFFFF) * 0x9E3779B9 + stream * 0x85EBCA6B
    x ^= UInt32((i >> 32) & 0xFFFFFFFF) * 0xC2B2AE35
    x ^= x >> 16
    x *= 0x7FEB352D
    x ^= x >> 15
    return x


def fill_e4m3_kernel(
    ptr: MutPointer[UInt8, MutUntrackedOrigin], n: Int64, stream: UInt32
):
    """Fills E4M3 bytes with finite values of magnitude 2**-3 to 2**1."""
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        var x = _hash(i, stream)
        ptr[unsafe_offset=i] = (
            UInt8((x >> 31) & 1) << 7
            | UInt8(4 + ((x >> 8) % 4)) << 3
            | UInt8((x >> 16) & 7)
        )
        i += stride


def fill_scale_kernel(
    ptr: MutPointer[Float32, MutUntrackedOrigin], n: Int64, stream: UInt32
):
    """Fills scales in [0.5, 1.5)."""
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    while i < Int(n):
        ptr[unsafe_offset=i] = Float32(0.5) + Float32(
            _hash(i, stream) >> 8
        ) / Float32(1 << 24)
        i += stride


def read_kernel(
    ptr: ImmPointer[SIMD[DType.uint32, 4], ImmUntrackedOrigin],
    n: Int64,
    sink: MutPointer[UInt32, MutUntrackedOrigin],
):
    var i = global_idx.x
    var stride = grid_dim.x * block_dim.x
    var acc = SIMD[DType.uint32, 4](0)
    while i < Int(n):
        acc ^= ptr[unsafe_offset=i]
        i += stride
    if acc[0] ^ acc[1] ^ acc[2] ^ acc[3] == 0x12345678:
        sink[unsafe_offset=global_idx.x] = acc[0]


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019

    def below(mut self, n: Int) -> Int:
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        return Int(((self.state * 0x2545F4914F6CDD1D) >> 33) % UInt64(n))


def slot_rows(tokens: Int, seed: Int) -> List[Int]:
    """Returns one routing draw's real rows a slot on rank 0."""
    var rows = List[Int](length=NUM_SLOTS, fill=0)
    rows[0] = tokens // N_RANKS
    var rng = Rng(seed)
    for _ in range(tokens):
        var picked = List[Int](capacity=TOP_K)
        while len(picked) < TOP_K:
            var e = rng.below(NUM_EXPERTS)
            if e not in picked:
                picked.append(e)
        for e in picked:
            if e < LOCAL_EXPERTS:
                rows[1 + e] += 1
    return rows^


def parse_list(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        out.append(Int(part))
    return out^


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def graph_us[
    F: def(DeviceContext, Int) raises -> None
](ctx: DeviceContext, f: F) raises -> Float64:
    """Microseconds a launch, `REPS` launches replayed back to back."""

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        var rec = builder.recording_context()
        for r in range(REPS):
            f(rec, r)

    var graph = DeviceGraph.create(ctx, build)

    def replay() raises {imm}:
        graph.replay()

    for _ in range(2):
        replay()
    ctx.synchronize()
    return (
        Float64(ctx.execution_time(replay, REPLAYS))
        / Float64(REPLAYS * REPS)
        / 1e3
    )


def read_bandwidth(ctx: DeviceContext) raises:
    """Prints a plain streaming read's rate over 4 GiB, as a reference."""
    comptime n_vec = (4 << 30) // 16
    var src = ctx.enqueue_create_buffer[DType.uint32](n_vec * 4)
    var sink = ctx.enqueue_create_buffer[DType.uint32](148 * 4 * 1024)
    ctx.enqueue_memset(src, UInt32(1))

    def read(cx: DeviceContext, r: Int) raises {imm}:
        cx.enqueue_function[read_kernel](
            src.unsafe_ptr()
            .unsafe_bitcast[SIMD[DType.uint32, 4]]()
            .as_imm()
            .unsafe_origin_cast[ImmUntrackedOrigin](),
            Int64(n_vec),
            sink.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
            grid_dim=148 * 4,
            block_dim=1024,
        )

    var times = List[Float64]()
    for _ in range(ROUNDS):
        times.append(graph_us(ctx, read))
    var us = median(times^)
    print(
        t"BW read bytes={n_vec * 16} us={us}"
        t" TBps={Float64(n_vec * 16) / us / 1e6}"
    )
    _ = src^
    _ = sink^


@inline(.always)
def arm_name[idx: Int]() -> String:
    comptime if idx == 0:
        return "entry"
    comptime cfg = arm_cfg[idx]()
    comptime if cfg[3]:
        return String(t"swap_bm{cfg[0]}a{cfg[2]}")
    return String(t"m{cfg[0]}n{cfg[1]}a{cfg[2]}")


def arm_cfg[idx: Int]() -> Tuple[Int, Int, Int, Bool]:
    """(token rows, weight rows, accumulator stages, swapped AB) of an arm.

    Arm 1 is the 64-row tile the op ran at decode before the swapped tile,
    arm 4 the 128-row tile it runs for prefill.
    """
    comptime if idx == 1:
        return (64, 128, 2, False)
    elif idx == 2:
        return (32, 128, 8, True)
    elif idx == 3:
        return (16, 128, 8, True)
    else:
        return (128, 128, 2, False)


def bench_shape[N: Int, K: Int](ctx: DeviceContext, steps: List[Int]) raises:
    comptime k_blocks = K // SCALE_BLOCK
    comptime n_blocks = N // SCALE_BLOCK
    comptime weight_elems = NUM_SLOTS * N * K
    comptime scale_elems = NUM_SLOTS * n_blocks * k_blocks

    var a_buf = ctx.enqueue_create_buffer[DType.float8_e4m3fn](TOTAL_ROWS * K)
    var c_buf = ctx.enqueue_create_buffer[DType.bfloat16](TOTAL_ROWS * N)
    var a_scales_buf = ctx.enqueue_create_buffer[DType.float32](
        k_blocks * TOTAL_ROWS
    )
    var b_buf = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        LAYERS * weight_elems
    )
    var bs_buf = ctx.enqueue_create_buffer[DType.float32](LAYERS * scale_elems)

    ctx.enqueue_function[fill_e4m3_kernel](
        a_buf.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(TOTAL_ROWS * K),
        UInt32(1),
        grid_dim=4096,
        block_dim=256,
    )
    ctx.enqueue_function[fill_e4m3_kernel](
        b_buf.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(LAYERS * weight_elems),
        UInt32(2),
        grid_dim=4096,
        block_dim=256,
    )
    ctx.enqueue_function[fill_scale_kernel](
        a_scales_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(k_blocks * TOTAL_ROWS),
        UInt32(3),
        grid_dim=1024,
        block_dim=256,
    )
    ctx.enqueue_function[fill_scale_kernel](
        bs_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(LAYERS * scale_elems),
        UInt32(4),
        grid_dim=1024,
        block_dim=256,
    )

    var ids_host = List[Int32](capacity=NUM_SLOTS)
    for i in range(NUM_SLOTS):
        ids_host.append(Int32(i))
    var ids_buf = ctx.enqueue_create_buffer[DType.int32](NUM_SLOTS)
    ctx.enqueue_copy(ids_buf, ids_host)
    ctx.synchronize()

    var a = TileTensor(a_buf, row_major(TOTAL_ROWS, Idx[K])).as_imm()
    var c = TileTensor(c_buf, row_major(TOTAL_ROWS, Idx[N]))
    var a_scales = TileTensor(
        a_scales_buf, row_major(Idx[k_blocks], TOTAL_ROWS)
    ).as_imm()
    var ids = TileTensor(ids_buf, row_major(Idx[NUM_SLOTS])).as_imm()

    for tokens in steps:
        var est_m = tokens * TOP_K // N_RANKS
        var offsets_host = List[UInt32](capacity=REPS * (NUM_SLOTS + 1))
        var active_sum = 0
        var real_sum = 0
        var max_slot = 0
        for r in range(REPS):
            var rows = slot_rows(tokens, tokens * 1000 + r)
            var off = 0
            offsets_host.append(UInt32(0))
            for s in range(NUM_SLOTS):
                if rows[s] > 0:
                    active_sum += 1
                real_sum += rows[s]
                max_slot = max(max_slot, rows[s])
                off += align_up(rows[s], ROW_PAD)
                offsets_host.append(UInt32(off))
            if off > TOTAL_ROWS:
                raise Error("routing exceeds the receive buffer")
        var offsets_buf = ctx.enqueue_create_buffer[DType.uint32](
            REPS * (NUM_SLOTS + 1)
        )
        ctx.enqueue_copy(offsets_buf, offsets_host)
        ctx.synchronize()
        var mean_active = Float64(active_sum) / Float64(REPS)
        var bytes = mean_active * Float64(N * K)
        var rows0 = Int(offsets_host[NUM_SLOTS])
        print(
            t"STEP N={N} K={K} tokens={tokens} est_m={est_m}"
            t" mean_active_slots={mean_active}"
            t" mean_real_rows={Float64(real_sum) / Float64(REPS)}"
            t" max_slot_rows={max_slot} padded_rows_draw0={rows0}"
        )

        @inline(.always)
        def launch[idx: Int](cx: DeviceContext, r: Int) raises {imm}:
            var offsets = TileTensor(
                offsets_buf.unsafe_ptr().unsafe_offset(r * (NUM_SLOTS + 1)),
                row_major(Idx[NUM_SLOTS + 1]),
            ).as_imm()
            var layer = r % LAYERS
            var b = TileTensor(
                b_buf.unsafe_ptr().unsafe_offset(layer * weight_elems),
                row_major(Idx[NUM_SLOTS], Idx[N], Idx[K]),
            ).as_imm()
            var bs = TileTensor(
                bs_buf.unsafe_ptr().unsafe_offset(layer * scale_elems),
                row_major(Idx[NUM_SLOTS], Idx[n_blocks], Idx[k_blocks]),
            ).as_imm()
            comptime if idx == 0:
                grouped_matmul_dynamic_scaled_fp8[
                    input_scale_granularity="block",
                    weight_scale_granularity="block",
                    m_scale_granularity=1,
                    n_scale_granularity=SCALE_BLOCK,
                    k_scale_granularity=SCALE_BLOCK,
                    transpose_b=True,
                ](
                    c,
                    a,
                    b,
                    a_scales,
                    bs,
                    offsets,
                    ids,
                    max_num_tokens_per_expert=MAX_ROWS_PER_SLOT,
                    num_active_experts=NUM_SLOTS,
                    ctx=cx,
                    estimated_total_m=est_m,
                )
            else:
                comptime cfg = arm_cfg[idx]()
                comptime config = MatmulConfig[
                    DType.float8_e4m3fn,
                    DType.float8_e4m3fn,
                    DType.bfloat16,
                    True,
                ](
                    cluster_shape=(1, 1, 1),
                    mma_shape=(cfg[0], cfg[1], 32),
                    cta_group=1,
                    AB_swapped=False,
                    k_group_size=1,
                    num_accum_pipeline_stages=cfg[2],
                )
                grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
                    config=config, swap_ab=cfg[3]
                ](
                    c,
                    a,
                    b,
                    a_scales,
                    bs,
                    offsets,
                    ids,
                    MAX_ROWS_PER_SLOT,
                    NUM_SLOTS,
                    cx,
                )

        # Every arm's rows must match the op's: same arithmetic, other
        # schedule.
        var base = ctx.enqueue_create_host_buffer[DType.bfloat16](
            max(rows0, 1) * N
        )
        var got = ctx.enqueue_create_host_buffer[DType.bfloat16](
            max(rows0, 1) * N
        )
        comptime for v in range(NARMS):
            if rows0 > 0:
                ctx.enqueue_memset(c_buf, BFloat16(0))
                launch[v](ctx, 0)
                var rows_c = c_buf.create_sub_buffer[DType.bfloat16](
                    0, rows0 * N
                )
                if v > 0:
                    ctx.enqueue_copy(got, rows_c)
                else:
                    ctx.enqueue_copy(base, rows_c)
                ctx.synchronize()
                if v > 0:
                    var differ = 0
                    var max_diff: Float32 = 0
                    var max_ref: Float32 = 0
                    for i in range(rows0 * N):
                        var x = got[i].cast[DType.float32]()
                        var y = base[i].cast[DType.float32]()
                        max_ref = max(max_ref, abs(y))
                        if x != y:
                            differ += 1
                            max_diff = max(max_diff, abs(x - y))
                    print(
                        t"CHECK N={N} K={K} tokens={tokens}"
                        t" arm={arm_name[v]()}"
                        t" differ={differ} max_abs_diff={max_diff}"
                        t" max_abs_ref={max_ref}"
                    )

        var times = List[List[Float64]]()
        for _ in range(NARMS):
            times.append(List[Float64]())
        for rnd in range(ROUNDS):
            for j in range(NARMS):
                var a_idx = j if rnd % 2 == 0 else NARMS - 1 - j
                comptime for v in range(NARMS):
                    if a_idx == v:

                        def timed(cx: DeviceContext, r: Int) raises {imm}:
                            launch[v](cx, r)

                        times[v].append(graph_us(ctx, timed))

        comptime for v in range(NARMS):
            var ts = times[v].copy()
            sort(ts)
            var us = ts[len(ts) // 2]
            print(
                t"TIME N={N} K={K} tokens={tokens} arm={arm_name[v]()}"
                t" us={us}"
                t" min={ts[0]} max={ts[len(ts) - 1]}"
                t" TBps={bytes / us / 1e6}"
            )
        _ = offsets_buf^
        _ = base^
        _ = got^

    _ = a_buf^
    _ = c_buf^
    _ = a_scales_buf^
    _ = b_buf^
    _ = bs_buf^
    _ = ids_buf^


def main() raises:
    var steps = parse_list(arg_parse("tokens", "24,96,144,192,8192"))
    var which = arg_parse("shapes", "both")
    with DeviceContext() as ctx:
        read_bandwidth(ctx)
        if which == "both" or which == "all" or which == "gateup":
            bench_shape[4096, 4096](ctx, steps)
        if which == "both" or which == "all" or which == "down":
            bench_shape[4096, 2048](ctx, steps)
        # Hidden-7168 expert shapes (gate/up K 7168, down N 7168), to check
        # that a decode configuration does not regress them.
        if which == "h7168" or which == "all":
            bench_shape[4096, 7168](ctx, steps)
            bench_shape[7168, 2048](ctx, steps)
