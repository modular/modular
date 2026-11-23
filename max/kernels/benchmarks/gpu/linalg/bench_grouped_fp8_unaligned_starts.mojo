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
"""Times the SM100 blockwise-FP8 grouped GEMM at unpadded group starts.

The operands are one rank's of an expert-sharded (not expert-parallel) MoE:
the rank holds 32 of 256 routed experts, a step of `T` tokens routes each
token to 8 distinct experts drawn uniformly, and the `T * 8`-row activation
buffer holds the rank's rows from row 0 in 32 groups, as
`moe_create_indices` lays them out. Most group starts are then not a
multiple of 4 rows, the alignment of the fp32 A scales' TMA. gate/up is
N 4096, K 6144; down is N 6144, K 2048 (one rank of a 256-expert MoE at TP8).

Arms, all through the op the graph calls:
  unpadded      group starts as `moe_create_indices` writes them
  unpadded_est  the same, with `estimated_total_m` = `T * 8` routed rows
  pad4          each group start padded to 4 rows (the aligned bound)
  pad4_est      the same, with `estimated_total_m` = `T * 8`

Before timing, both unpadded arms' outputs on the first draw are compared
with the naive reference. `--tokens=T1,T2,...` picks the steps: 8192 is a
prefill chunk, 1 a decode step. One weight copy serves every launch, so a
decode step's few experts stay in L2: decode times are L2-warm.
"""

from std.math import align_up

from internal_utils import arg_parse
from layout import Idx, TileTensor, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_dynamic_scaled_fp8,
)
from max.gpu import block_dim, global_idx, grid_dim
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder
from std.utils.coord import Coord

comptime NUM_GROUPS = 32
comptime NUM_EXPERTS = 256
comptime TOP_K = 8
comptime SCALE_BLOCK = 128
comptime REPS = 2
comptime REPLAYS = 20
comptime ROUNDS = 5
comptime NARMS = 4


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


struct Rng(Movable):
    var state: UInt64

    def __init__(out self, seed: Int):
        self.state = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019

    def below(mut self, n: Int) -> Int:
        self.state ^= self.state >> 12
        self.state ^= self.state << 25
        self.state ^= self.state >> 27
        return Int(((self.state * 0x2545F4914F6CDD1D) >> 33) % UInt64(n))


def group_rows(tokens: Int, seed: Int) -> List[Int]:
    """Rows a local expert gets on the rank holding experts 0-31."""
    var rows = List[Int](length=NUM_GROUPS, fill=0)
    var rng = Rng(seed)
    for _ in range(tokens):
        var picked = List[Int](capacity=TOP_K)
        while len(picked) < TOP_K:
            var e = rng.below(NUM_EXPERTS)
            if e not in picked:
                picked.append(e)
        for e in picked:
            if e < NUM_GROUPS:
                rows[e] += 1
    return rows^


def total(xs: List[Int]) -> Int:
    var n = 0
    for x in xs:
        n += x
    return n


def parse_list(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        out.append(Int(part))
    return out^


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


def arm_name(idx: Int) -> String:
    if idx == 0:
        return "unpadded"
    if idx == 1:
        return "unpadded_est"
    if idx == 2:
        return "pad4"
    return "pad4_est"


def bench_shape[N: Int, K: Int](ctx: DeviceContext, tokens: Int) raises:
    comptime k_blocks = K // SCALE_BLOCK
    comptime n_blocks = N // SCALE_BLOCK
    comptime weight_elems = NUM_GROUPS * N * K
    comptime scale_elems = NUM_GROUPS * n_blocks * k_blocks
    var total_rows = tokens * TOP_K

    # Draws with at least one local row; a decode step can route none here.
    var draws = List[List[Int]]()
    var seed = 0
    while len(draws) < REPS:
        var rows = group_rows(tokens, seed)
        seed += 1
        if total(rows) > 0:
            draws.append(rows^)

    # Padding adds up to 3 rows a group past the unpadded buffer.
    var buf_rows = total_rows + 3 * NUM_GROUPS
    var a_buf = ctx.enqueue_create_buffer[DType.float8_e4m3fn](buf_rows * K)
    var c_buf = ctx.enqueue_create_buffer[DType.bfloat16](buf_rows * N)
    var c_ref_buf = ctx.enqueue_create_buffer[DType.bfloat16](total_rows * N)
    var a_scales_buf = ctx.enqueue_create_buffer[DType.float32](
        k_blocks * buf_rows
    )
    var b_buf = ctx.enqueue_create_buffer[DType.float8_e4m3fn](weight_elems)
    var bs_buf = ctx.enqueue_create_buffer[DType.float32](scale_elems)

    ctx.enqueue_function[fill_e4m3_kernel](
        a_buf.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(buf_rows * K),
        UInt32(1),
        grid_dim=4096,
        block_dim=256,
    )
    ctx.enqueue_function[fill_e4m3_kernel](
        b_buf.unsafe_ptr()
        .unsafe_bitcast[UInt8]()
        .unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(weight_elems),
        UInt32(2),
        grid_dim=4096,
        block_dim=256,
    )
    ctx.enqueue_function[fill_scale_kernel](
        a_scales_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(k_blocks * buf_rows),
        UInt32(3),
        grid_dim=1024,
        block_dim=256,
    )
    ctx.enqueue_function[fill_scale_kernel](
        bs_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        Int64(scale_elems),
        UInt32(4),
        grid_dim=1024,
        block_dim=256,
    )

    var ids_host = List[Int32](capacity=NUM_GROUPS)
    for i in range(NUM_GROUPS):
        ids_host.append(Int32(i))
    var ids_buf = ctx.enqueue_create_buffer[DType.int32](NUM_GROUPS)
    ctx.enqueue_copy(ids_buf, ids_host)

    # Offsets per draw: unpadded, then padded to 4 rows.
    var offsets_host = List[UInt32](capacity=2 * REPS * (NUM_GROUPS + 1))
    var local_rows = 0
    var max_group = 0
    var unaligned = 0
    for pad in range(2):
        for r in range(REPS):
            var off = 0
            offsets_host.append(UInt32(0))
            for g in range(NUM_GROUPS):
                var rows = draws[r][g]
                if pad == 0:
                    local_rows += rows
                    max_group = max(max_group, rows)
                    if off % 4 != 0 and rows > 0:
                        unaligned += 1
                off += align_up(rows, 4) if pad == 1 else rows
                offsets_host.append(UInt32(off))
    var offsets_buf = ctx.enqueue_create_buffer[DType.uint32](
        2 * REPS * (NUM_GROUPS + 1)
    )
    ctx.enqueue_copy(offsets_buf, offsets_host)
    ctx.synchronize()
    print(
        t"STEP N={N} K={K} tokens={tokens} buffer_rows={total_rows}"
        t" mean_local_rows={local_rows // REPS} max_group_rows={max_group}"
        t" unaligned_group_starts={Float64(unaligned) / Float64(REPS)}"
        t" of {NUM_GROUPS}"
    )

    var ids = TileTensor(ids_buf, row_major(Idx[NUM_GROUPS])).as_imm()
    var b = TileTensor(
        b_buf, row_major(Idx[NUM_GROUPS], Idx[N], Idx[K])
    ).as_imm()
    var bs = TileTensor(
        bs_buf, row_major(Idx[NUM_GROUPS], Idx[n_blocks], Idx[k_blocks])
    ).as_imm()

    # The padded layout's buffer is longer by the padding, rounded so its
    # scales' K-row stride stays 16-byte aligned.
    var pad_rows = align_up(buf_rows, 4)
    var c = TileTensor(c_buf, row_major(total_rows, Idx[N]))
    var a = TileTensor(a_buf, row_major(total_rows, Idx[K])).as_imm()
    var a_scales = TileTensor(
        a_scales_buf, row_major(Idx[k_blocks], total_rows)
    ).as_imm()
    var c_pad = TileTensor(c_buf, row_major(pad_rows, Idx[N]))
    var a_pad = TileTensor(a_buf, row_major(pad_rows, Idx[K])).as_imm()
    var a_scales_pad = TileTensor(
        a_scales_buf, row_major(Idx[k_blocks], pad_rows)
    ).as_imm()

    def launch(cx: DeviceContext, r: Int, arm: Int) raises {imm}:
        var pad = 1 if arm >= 2 else 0
        var offsets = TileTensor(
            offsets_buf.unsafe_ptr().unsafe_offset(
                (pad * REPS + r) * (NUM_GROUPS + 1)
            ),
            row_major(Idx[NUM_GROUPS + 1]),
        ).as_imm()
        grouped_matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=1,
            n_scale_granularity=SCALE_BLOCK,
            k_scale_granularity=SCALE_BLOCK,
            transpose_b=True,
        ](
            c_pad if pad else c,
            a_pad if pad else a,
            b,
            a_scales_pad if pad else a_scales,
            bs,
            offsets,
            ids,
            max_num_tokens_per_expert=total_rows,
            num_active_experts=NUM_GROUPS,
            ctx=cx,
            estimated_total_m=total_rows if arm % 2 == 1 else 0,
        )

    # Check both unpadded arms against the naive kernel on draw 0's rows.
    var offsets0 = TileTensor(
        offsets_buf, row_major(Idx[NUM_GROUPS + 1])
    ).as_imm()
    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=True,
        scales_granularity_mnk=Coord[Int, Int, Int](
            1, SCALE_BLOCK, SCALE_BLOCK
        ),
    ](
        TileTensor(c_ref_buf, row_major(total_rows, Idx[N])),
        a,
        b,
        a_scales,
        bs,
        offsets0,
        ids,
        max_group,
        NUM_GROUPS,
        ctx,
    )
    var checked = total(draws[0]) * N
    var c_host = ctx.enqueue_create_host_buffer[DType.bfloat16](checked)
    var ref_host = ctx.enqueue_create_host_buffer[DType.bfloat16](checked)
    ctx.enqueue_copy(
        ref_host, c_ref_buf.create_sub_buffer[DType.bfloat16](0, checked)
    )
    for arm in range(2):
        launch(ctx, 0, arm)
        ctx.enqueue_copy(
            c_host, c_buf.create_sub_buffer[DType.bfloat16](0, checked)
        )
        ctx.synchronize()
        var bad = 0
        for i in range(checked):
            var got = c_host[i].cast[DType.float32]()
            var want = ref_host[i].cast[DType.float32]()
            # Written so a NaN counts as a mismatch.
            if not (abs(got - want) <= 1e-2 + 1e-2 * abs(want)):
                if bad < 5:
                    print(t"MISMATCH idx={i} got={got} want={want}")
                bad += 1
        print(
            t"CHECK N={N} K={K} tokens={tokens} arm={arm_name(arm)}"
            t" elems={checked} mismatches={bad}"
        )
        if bad:
            raise Error("unpadded output differs from the naive reference")

    var flops = 2.0 * Float64(local_rows // REPS) * Float64(N) * Float64(K)
    var times = List[Float64](length=NARMS * ROUNDS, fill=0.0)
    for round in range(ROUNDS):
        for j in range(NARMS):
            # Alternate the arm order every round.
            var arm = j if round % 2 == 0 else NARMS - 1 - j

            def timed(cx: DeviceContext, r: Int) raises {imm}:
                launch(cx, r, arm)

            times[arm * ROUNDS + round] = graph_us(ctx, timed)
    for arm in range(NARMS):
        var ts = List[Float64](capacity=ROUNDS)
        for i in range(ROUNDS):
            ts.append(times[arm * ROUNDS + i])
        sort(ts)
        var us = ts[ROUNDS // 2]
        print(
            t"TIME N={N} K={K} tokens={tokens} arm={arm_name(arm)} us={us}"
            t" min={ts[0]} max={ts[ROUNDS - 1]}"
            t" TFLOPs={flops / us / 1e6}"
        )

    _ = a_buf^
    _ = c_buf^
    _ = c_ref_buf^
    _ = a_scales_buf^
    _ = b_buf^
    _ = bs_buf^
    _ = ids_buf^
    _ = offsets_buf^


def main() raises:
    var steps = parse_list(arg_parse("tokens", "8192,1"))
    with DeviceContext() as ctx:
        for t in steps:
            bench_shape[4096, 6144](ctx, t)
            bench_shape[6144, 2048](ctx, t)
