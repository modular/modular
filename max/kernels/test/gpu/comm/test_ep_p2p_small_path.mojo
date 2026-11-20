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

"""Checks the single-node P2P EP paths on 2, 4 or 8 AMD GPUs.

Runs the protocol dispatch and combine (`p2p_small_max_tokens=0`) and the
P2P paths on the same tokens, over a sequence of steps that reuses one set
of buffers per path and crosses the small/prefill decision both ways
(including a step where only some devices sit at the bound). Steps the small
path declines take the prefill dispatch and the partial-row combine. The
small path's epoch starts just short of its wrap, and every step first
scribbles over the dispatch receive data, as an exchange sharing the buffers
would. For each step it checks that:
- the row offsets and expert ids match exactly;
- every expert group holds the same rows (as a multiset of row bytes and
  scales);
- each local expert's routed-token tally (EPLB expert load) matches;
- with each row's expert output a function of its bytes and group, the two
  combines agree: exactly on small steps, within one BF16 rounding of each
  partial row on the others.
`-D max_tok=2048` adds an 8K-token step. With `-D P2P_SMALL_TIME=true` it
then times dispatch + combine of both paths.
"""

from std.memory.alloc import unsafe_alloc
from std.random import randn, random_si64, seed
from std.sys import get_defined_bool, get_defined_int, size_of
from std.time import perf_counter_ns

from max.algorithm import sync_parallelize
from comm.sync import enable_p2p
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, TileTensor, Idx, row_major
from shmem.ep import ep_fused_combine_kernel_api, ep_fused_dispatch_kernel_api
from shmem.ep_comm import (
    BlockwiseFP8TokenFormat,
    EP_P2P_EPOCH_MAX,
    EP_P2P_SMALL_WORDS,
    EPLocalSyncCounters,
    ExpertLoadTally,
    ep_dispatch_recv_size,
    ep_p2p_combine_header_bytes,
    ep_p2p_dispatch_header_bytes,
)
from std.testing import assert_equal, assert_true

comptime HIDDEN = 4096
comptime TOP_K = 8
comptime MAX_WORLD = 8
comptime N_LOCAL = 72
comptime MAX_TOK = get_defined_int["max_tok", 64]()
comptime SMALL = 32
comptime N_SCALES = HIDDEN // 128
comptime MAX_RECV = MAX_WORLD * TOP_K * MAX_TOK + MAX_TOK + 4 * (N_LOCAL + 1)


def _header[combine: Bool, world: Int]() -> Int:
    comptime if combine:
        return ep_p2p_combine_header_bytes[
            N_LOCAL * world, world, MAX_TOK, TOP_K
        ]()
    return ep_p2p_dispatch_header_bytes[
        N_LOCAL * world, world, MAX_TOK, TOP_K
    ]()


# Headers before the published receive buffers, as `ep.init` lays them out,
# sized for any device count; a kernel reaches only the part it needs.
comptime D_HDR = max(
    max(_header[False, 2](), _header[False, 4]()), _header[False, 8]()
)
comptime C_HDR = max(
    max(_header[True, 2](), _header[True, 4]()), _header[True, 8]()
)
comptime SRC_ROWS = N_LOCAL * MAX_WORLD * MAX_TOK


struct Rank:
    """One device's buffers for one path."""

    var tokens: DeviceBuffer[DType.uint8]
    var scales: DeviceBuffer[DType.float32]
    var row_offsets: DeviceBuffer[DType.uint32]
    var expert_ids: DeviceBuffer[DType.int32]
    var src_info: DeviceBuffer[DType.int32]
    var expert_out: DeviceBuffer[DType.bfloat16]
    var combined: DeviceBuffer[DType.bfloat16]
    var counters: DeviceBuffer[DType.int32]
    var d_send: DeviceBuffer[DType.uint8]
    var d_recv: DeviceBuffer[DType.uint8]
    var d_recv_data: Int
    var d_count: DeviceBuffer[DType.uint64]
    var c_send: DeviceBuffer[DType.uint8]
    var c_recv: DeviceBuffer[DType.uint8]
    var c_count: DeviceBuffer[DType.uint64]
    var load: DeviceBuffer[DType.uint64]
    var layer: DeviceBuffer[DType.int32]

    def __init__(
        out self, ctx: DeviceContext, msg_bytes: Int, world: Int
    ) raises:
        comptime n_groups = N_LOCAL + 1
        var n_experts = N_LOCAL * world
        self.tokens = ctx.enqueue_create_buffer[DType.uint8](MAX_RECV * HIDDEN)
        self.tokens.enqueue_fill(0)
        self.scales = ctx.enqueue_create_buffer[DType.float32](
            MAX_RECV * N_SCALES
        )
        self.scales.enqueue_fill(0)
        self.row_offsets = ctx.enqueue_create_buffer[DType.uint32](n_groups + 1)
        self.expert_ids = ctx.enqueue_create_buffer[DType.int32](n_groups)
        self.src_info = ctx.enqueue_create_buffer[DType.int32](SRC_ROWS * 2)
        self.expert_out = ctx.enqueue_create_buffer[DType.bfloat16](
            MAX_RECV * HIDDEN
        )
        self.combined = ctx.enqueue_create_buffer[DType.bfloat16](
            MAX_TOK * HIDDEN
        )
        self.counters = ctx.enqueue_create_buffer[DType.int32](
            EPLocalSyncCounters[N_LOCAL * MAX_WORLD].total_size()
            + EP_P2P_SMALL_WORDS
        )
        ctx.enqueue_memset(self.counters, Int32(0))
        self.d_send = ctx.enqueue_create_buffer[DType.uint8](
            MAX_TOK * msg_bytes
        )
        # Like `ep.init`: the small path's flags sit just before the
        # published receive buffers.
        self.d_recv_data = ep_dispatch_recv_size(
            n_experts=n_experts,
            n_ranks=world,
            max_tokens_per_rank=MAX_TOK,
            msg_bytes=msg_bytes,
            token_recv=True,
        )
        self.d_recv = ctx.enqueue_create_buffer[DType.uint8](
            D_HDR + self.d_recv_data
        )
        ctx.enqueue_memset(self.d_recv, UInt8(0))
        self.d_count = ctx.enqueue_create_buffer[DType.uint64](n_experts)
        ctx.enqueue_memset(self.d_count, UInt64.MAX_FINITE)
        self.c_send = ctx.enqueue_create_buffer[DType.uint8](
            MAX_RECV * HIDDEN * 2
        )
        self.c_recv = ctx.enqueue_create_buffer[DType.uint8](
            C_HDR + TOP_K * MAX_TOK * HIDDEN * 2
        )
        ctx.enqueue_memset(self.c_recv, UInt8(0))
        self.c_count = ctx.enqueue_create_buffer[DType.uint64](n_experts)
        ctx.enqueue_memset(self.c_count, UInt64.MAX_FINITE)
        self.load = ctx.enqueue_create_buffer[DType.uint64](N_LOCAL)
        ctx.enqueue_memset(self.load, UInt64(0))
        self.layer = ctx.enqueue_create_buffer[DType.int32](1)
        ctx.enqueue_memset(self.layer, Int32(0))


def ptr_table(
    ranks: List[Rank], which: Int
) -> MutPointer[UInt64, MutAnyOrigin]:
    var table = unsafe_alloc[UInt64](len(ranks))
    for r in range(len(ranks)):
        var p: Int
        if which == 0:
            p = Int(ranks[r].d_send.unsafe_ptr())
        elif which == 1:
            p = Int(ranks[r].d_recv.unsafe_ptr()) + D_HDR
        elif which == 2:
            p = Int(ranks[r].d_count.unsafe_ptr())
        elif which == 3:
            p = Int(ranks[r].c_send.unsafe_ptr())
        elif which == 4:
            p = Int(ranks[r].c_recv.unsafe_ptr()) + C_HDR
        else:
            p = Int(ranks[r].c_count.unsafe_ptr())
        table[unsafe_offset=r] = UInt64(p)
    return table.unsafe_origin_cast[MutAnyOrigin]()


def run_dispatch[
    small: Int, world: Int
](
    ctx: DeviceContext,
    rank: Rank,
    tables: List[MutPointer[UInt64, MutAnyOrigin]],
    input_buf: DeviceBuffer[DType.bfloat16],
    topk_buf: DeviceBuffer[DType.int32],
    n: Int,
) raises:
    comptime n_groups = N_LOCAL + 1
    comptime n_experts = N_LOCAL * world
    comptime ptrs_layout = row_major[world]()
    var out_tokens = TileTensor(
        rank.tokens.unsafe_ptr().unsafe_bitcast[Float8_e4m3fn](),
        row_major((Idx[MAX_RECV], Idx[HIDDEN])),
    )
    var out_scales = TileTensor(
        rank.scales.unsafe_ptr(),
        row_major((Idx[N_SCALES], Idx[MAX_RECV])),
    )
    var handler = BlockwiseFP8TokenFormat[HIDDEN, TOP_K](out_tokens, out_scales)
    ep_fused_dispatch_kernel_api[
        n_experts,
        MAX_TOK,
        world,
        1,
        True,
        use_shmem=False,
        p2p_small_max_tokens=small,
    ](
        handler,
        TileTensor(rank.row_offsets, row_major[n_groups + 1]()),
        TileTensor(rank.expert_ids, row_major[n_groups]()),
        TileTensor(rank.src_info, row_major((Idx[SRC_ROWS], Idx[2]))),
        TileTensor(
            rank.counters,
            row_major[
                EPLocalSyncCounters[n_experts].total_size() + EP_P2P_SMALL_WORDS
            ](),
        ),
        TileTensor(input_buf, row_major((n, Idx[HIDDEN]))).as_imm(),
        TileTensor(topk_buf, row_major(n, Idx[TOP_K])).as_imm(),
        TileTensor(tables[0], ptrs_layout),
        TileTensor(tables[1], ptrs_layout),
        TileTensor(tables[2], ptrs_layout),
        ctx,
        ExpertLoadTally(
            MutPointer[UInt64, MutUntrackedOrigin](
                unsafe_from_address=Int(rank.load.unsafe_ptr())
            ),
            ImmPointer[Int32, ImmUntrackedOrigin](
                unsafe_from_address=Int(rank.layer.unsafe_ptr())
            ),
        ),
    )


def run_combine[
    small: Int, world: Int
](
    ctx: DeviceContext,
    rank: Rank,
    tables: List[MutPointer[UInt64, MutAnyOrigin]],
    weights_buf: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    comptime n_experts = N_LOCAL * world
    comptime ptrs_layout = row_major[world]()
    var combined_tt = TileTensor(
        MutPointer[BFloat16, MutUntrackedOrigin](
            unsafe_from_address=Int(rank.combined.unsafe_ptr())
        ),
        row_major((n, Idx[HIDDEN])),
    )
    var weights_tt = TileTensor(weights_buf, row_major((n, Idx[TOP_K])))

    @inline(.always)
    def router_weights_fn[
        width: Int
    ](token_idx: Int, topk_id: Int) {var weights_tt} -> SIMD[.float32, width]:
        return SIMD[.float32, width](weights_tt[token_idx, topk_id])

    @inline(.always)
    def store_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: Coord, val: SIMD[dtype, width]) {var combined_tt}:
        combined_tt.store[alignment=alignment * size_of[dtype]()](
            coords, rebind[SIMD[.bfloat16, width]](val)
        )

    ep_fused_combine_kernel_api[
        HIDDEN,
        TOP_K,
        n_experts,
        MAX_TOK,
        world,
        1,
        fused_shared_expert=True,
        p2p_small_max_tokens=small,
    ](
        n,
        TileTensor(
            rank.counters,
            row_major[
                EPLocalSyncCounters[n_experts].total_size() + EP_P2P_SMALL_WORDS
            ](),
        ),
        TileTensor(
            rank.expert_out, row_major((Idx[MAX_RECV], Idx[HIDDEN]))
        ).as_imm(),
        TileTensor(rank.src_info, row_major((Idx[SRC_ROWS], Idx[2]))).as_imm(),
        TileTensor(tables[3], ptrs_layout),
        TileTensor(tables[4], ptrs_layout),
        TileTensor(tables[5], ptrs_layout),
        ctx,
        store_fn,
        router_weights_fn,
    )


def row_key(
    tok: MutPointer[UInt8, _], scl: MutPointer[Float32, _], r: Int
) -> UInt64:
    """A hash of row `r`'s bytes and scales."""
    var h = UInt64(1469598103934665603)
    for i in range(HIDDEN):
        h = (h ^ UInt64(tok[unsafe_offset=r * HIDDEN + i])) * 1099511628211
    for g in range(N_SCALES):
        var bits = scl.unsafe_bitcast[UInt32]()[unsafe_offset=g * MAX_RECV + r]
        h = (h ^ UInt64(bits)) * 1099511628211
    return h


def fill_expert_out(
    tok: MutPointer[UInt8, _],
    off: MutPointer[UInt32, _],
    dst: MutPointer[BFloat16, _],
):
    """Expert output of each row: a function of its bytes and its group."""
    for g in range(N_LOCAL + 1):
        for r in range(
            Int(off[unsafe_offset=g]), Int(off[unsafe_offset=g + 1])
        ):
            for i in range(HIDDEN):
                var b = Int(tok[unsafe_offset=r * HIDDEN + i])
                dst[unsafe_offset=r * HIDDEN + i] = BFloat16(
                    Float32(b - 128) * Float32(0.01) * Float32(g % 7 + 1)
                )


def scribble(ctx: DeviceContext, rank: Rank) raises:
    """Overwrites `rank`'s dispatch receive data past the small path's flags."""
    ctx.enqueue_memset(
        rank.d_recv.create_sub_buffer[DType.uint8](D_HDR, rank.d_recv_data),
        UInt8(0xA5),
    )


def step[
    world: Int
](
    ctxs: List[DeviceContext[]],
    slow: List[Rank],
    fast: List[Rank],
    st: List[MutPointer[UInt64, MutAnyOrigin]],
    ft: List[MutPointer[UInt64, MutAnyOrigin]],
    counts: List[Int],
    step_seed: Int,
) raises -> Int:
    comptime n_experts = N_LOCAL * world
    var line = String("step counts")
    for r in range(world):
        line += String(" ", counts[r])
    print(line)
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    var topks = List[DeviceBuffer[DType.int32]]()
    var weights = List[DeviceBuffer[DType.float32]]()
    for r in range(world):
        var n = counts[r]
        var ctx = ctxs[r]
        var h_topk = unsafe_alloc[Int32](max(n, 1) * TOP_K)
        var h_input = unsafe_alloc[BFloat16](max(n, 1) * HIDDEN)
        var h_w = unsafe_alloc[Float32](max(n, 1) * TOP_K)
        seed(step_seed * 17 + r)
        for t in range(n):
            # Distinct experts per token.
            var used = List[Bool](length=n_experts, fill=False)
            for k in range(TOP_K):
                var e = Int32(random_si64(0, Int64(n_experts - 1)))
                while used[Int(e)]:
                    e = Int32((Int(e) + 1) % n_experts)
                used[Int(e)] = True
                h_topk[unsafe_offset=t * TOP_K + k] = e
        randn(h_input, max(n, 1) * HIDDEN)
        randn(h_w, max(n, 1) * TOP_K)
        inputs.append(
            ctx.enqueue_create_buffer[DType.bfloat16](max(n, 1) * HIDDEN)
        )
        topks.append(ctx.enqueue_create_buffer[DType.int32](max(n, 1) * TOP_K))
        weights.append(
            ctx.enqueue_create_buffer[DType.float32](max(n, 1) * TOP_K)
        )
        ctx.enqueue_copy(inputs[r], h_input)
        ctx.enqueue_copy(topks[r], h_topk)
        ctx.enqueue_copy(weights[r], h_w)
        ctx.synchronize()

    # Padding rows are never written, so clear what earlier steps left there.
    # Scribble over the dispatch receive data too, as another exchange
    # sharing the buffers would: no flag may live there.
    for r in range(world):
        scribble(ctxs[r], slow[r])
        scribble(ctxs[r], fast[r])
    for r in range(world):
        slow[r].tokens.enqueue_fill(0)
        slow[r].scales.enqueue_fill(0)
        fast[r].tokens.enqueue_fill(0)
        fast[r].scales.enqueue_fill(0)
    # Peers write these buffers from their own streams.
    for r in range(world):
        ctxs[r].synchronize()
    for r in range(world):
        run_dispatch[0, world](
            ctxs[r], slow[r], st, inputs[r], topks[r], counts[r]
        )
    for r in range(world):
        ctxs[r].synchronize()
    for r in range(world):
        run_dispatch[SMALL, world](
            ctxs[r], fast[r], ft, inputs[r], topks[r], counts[r]
        )
    for r in range(world):
        ctxs[r].synchronize()

    # The counter buffer carries the small path's words, so a step with at
    # most SMALL tokens on every device must take the small path.
    var all_small = True
    for r in range(world):
        all_small = all_small and counts[r] <= SMALL
    for r in range(world):
        var took = unsafe_alloc[Int32](1)
        ctxs[r].enqueue_copy(
            took,
            fast[r].counters.create_sub_buffer[DType.int32](
                EPLocalSyncCounters[n_experts].total_size() + 2, 1
            ),
        )
        ctxs[r].synchronize()
        assert_equal(
            took[unsafe_offset=0] == 1,
            all_small,
            String("rank ", r, " small path taken"),
        )

    comptime n_groups = N_LOCAL + 1
    var step_bad = 0
    for r in range(world):
        var ctx = ctxs[r]
        # fmt: off
        var s_off = unsafe_alloc[UInt32](n_groups + 1)
        var f_off = unsafe_alloc[UInt32](n_groups + 1)
        var s_ids = unsafe_alloc[Int32](n_groups)
        var f_ids = unsafe_alloc[Int32](n_groups)
        var s_tok = unsafe_alloc[UInt8](MAX_RECV * HIDDEN)
        var f_tok = unsafe_alloc[UInt8](MAX_RECV * HIDDEN)
        var s_scl = unsafe_alloc[Float32](MAX_RECV * N_SCALES)
        var f_scl = unsafe_alloc[Float32](MAX_RECV * N_SCALES)
        ctx.enqueue_copy(s_off, slow[r].row_offsets); ctx.enqueue_copy(f_off, fast[r].row_offsets)
        ctx.enqueue_copy(s_ids, slow[r].expert_ids); ctx.enqueue_copy(f_ids, fast[r].expert_ids)
        ctx.enqueue_copy(s_tok, slow[r].tokens); ctx.enqueue_copy(f_tok, fast[r].tokens)
        ctx.enqueue_copy(s_scl, slow[r].scales); ctx.enqueue_copy(f_scl, fast[r].scales)
        # fmt: on
        ctx.synchronize()
        for g in range(n_groups + 1):
            assert_equal(
                f_off[unsafe_offset=g],
                s_off[unsafe_offset=g],
                String("rank ", r, " row offset ", g),
            )
        for g in range(n_groups):
            assert_equal(f_ids[unsafe_offset=g], s_ids[unsafe_offset=g])
        var s_load = unsafe_alloc[UInt64](N_LOCAL)
        var f_load = unsafe_alloc[UInt64](N_LOCAL)
        ctx.enqueue_copy(s_load, slow[r].load)
        ctx.enqueue_copy(f_load, fast[r].load)
        ctx.synchronize()
        for e in range(N_LOCAL):
            assert_equal(
                f_load[unsafe_offset=e],
                s_load[unsafe_offset=e],
                String("rank ", r, " expert load ", e),
            )
        var n_rows = 0
        var n_bad_total = 0
        for g in range(n_groups):
            var s_keys = List[UInt64]()
            var f_keys = List[UInt64]()
            for row in range(
                Int(s_off[unsafe_offset=g]), Int(s_off[unsafe_offset=g + 1])
            ):
                s_keys.append(row_key(s_tok, s_scl, row))
                f_keys.append(row_key(f_tok, f_scl, row))
            sort(s_keys)
            sort(f_keys)
            var bad = 0
            for i in range(len(s_keys)):
                if f_keys[i] != s_keys[i]:
                    bad += 1
            if bad > 0:
                # Which half differs: rows matched on their FP8 bytes alone.
                var s_b = List[UInt64]()
                var f_b = List[UInt64]()
                for row in range(
                    Int(s_off[unsafe_offset=g]), Int(s_off[unsafe_offset=g + 1])
                ):
                    var hs = UInt64(1469598103934665603)
                    var hf = UInt64(1469598103934665603)
                    for i in range(HIDDEN):
                        hs = (
                            hs ^ UInt64(s_tok[unsafe_offset=row * HIDDEN + i])
                        ) * 1099511628211
                        hf = (
                            hf ^ UInt64(f_tok[unsafe_offset=row * HIDDEN + i])
                        ) * 1099511628211
                    s_b.append(hs)
                    f_b.append(hf)
                sort(s_b)
                sort(f_b)
                var bad_bytes = 0
                for i in range(len(s_b)):
                    if f_b[i] != s_b[i]:
                        bad_bytes += 1
                var zero_rows = 0
                for row in range(
                    Int(f_off[unsafe_offset=g]), Int(f_off[unsafe_offset=g + 1])
                ):
                    var nz = 0
                    for i in range(HIDDEN):
                        if f_tok[unsafe_offset=row * HIDDEN + i] != 0:
                            nz += 1
                    if nz == 0:
                        zero_rows += 1
                print(
                    t"  MISMATCH rank {r} group {g}: {bad}/{len(s_keys)} rows"
                    t" differ, {bad_bytes} on FP8 bytes, {zero_rows} all-zero"
                    t" small-path rows"
                )
                n_bad_total += bad
            n_rows += len(s_keys)
        var h_out_s = unsafe_alloc[BFloat16](MAX_RECV * HIDDEN)
        var h_out_f = unsafe_alloc[BFloat16](MAX_RECV * HIDDEN)
        fill_expert_out(s_tok, s_off, h_out_s)
        fill_expert_out(f_tok, f_off, h_out_f)
        ctx.enqueue_copy(slow[r].expert_out, h_out_s)
        ctx.enqueue_copy(fast[r].expert_out, h_out_f)
        ctx.synchronize()
        for p in [s_tok, f_tok]:
            p.unsafe_free()
        for p in [s_scl, f_scl]:
            p.unsafe_free()
        for p in [h_out_s, h_out_f]:
            p.unsafe_free()
        print(t"  rank {r}: {n_rows} rows, {n_bad_total} differ")
        step_bad += n_bad_total

    for r in range(world):
        run_combine[0, world](ctxs[r], slow[r], st, weights[r], counts[r])
    for r in range(world):
        ctxs[r].synchronize()
    for r in range(world):
        run_combine[SMALL, world](ctxs[r], fast[r], ft, weights[r], counts[r])
    for r in range(world):
        ctxs[r].synchronize()

    for r in range(world):
        var n = counts[r]
        var s_c = unsafe_alloc[BFloat16](MAX_TOK * HIDDEN)
        var f_c = unsafe_alloc[BFloat16](MAX_TOK * HIDDEN)
        ctxs[r].enqueue_copy(s_c, slow[r].combined)
        ctxs[r].enqueue_copy(f_c, fast[r].combined)
        ctxs[r].synchronize()
        # Relative to the output scale: the partial-row combine rounds each
        # device's partial to BF16, so near-cancelling sums differ by more
        # than their own magnitude allows.
        var n_exact = 0
        var max_abs = Float32(0)
        var max_diff = Float32(0)
        for i in range(n * HIDDEN):
            var a = Float32(f_c[unsafe_offset=i])
            var b = Float32(s_c[unsafe_offset=i])
            if a == b:
                n_exact += 1
            max_abs = max(max_abs, abs(b))
            max_diff = max(max_diff, abs(a - b))
        var max_err = max_diff / max(max_abs, Float32(1e-6))
        print(
            t"  rank {r}: combine {n_exact}/{n * HIDDEN} exact, max rel err"
            t" {max_err}"
        )
        s_c.unsafe_free()
        f_c.unsafe_free()
        # The small combine sums the same values in the same order.
        if max_err >= 0.02 or (all_small and n_exact != n * HIDDEN):
            print("  COMBINE MISMATCH")
            step_bad += 1
    return step_bad


def time_path[
    small: Int, world: Int
](
    ctxs: List[DeviceContext[]],
    ranks: List[Rank],
    tables: List[MutPointer[UInt64, MutAnyOrigin]],
    n: Int,
    iters: Int,
) raises:
    var inputs = List[DeviceBuffer[DType.bfloat16]]()
    var topks = List[DeviceBuffer[DType.int32]]()
    var weights = List[DeviceBuffer[DType.float32]]()
    for r in range(world):
        var ctx = ctxs[r]
        var h_topk = unsafe_alloc[Int32](n * TOP_K)
        seed(1000 + r)
        for t in range(n):
            for k in range(TOP_K):
                # Distinct experts: one per eighth of the expert range.
                var e = Int32(
                    random_si64(0, Int64(N_LOCAL * world // TOP_K - 1))
                )
                h_topk[unsafe_offset=t * TOP_K + k] = e + Int32(
                    k * (N_LOCAL * world // TOP_K)
                )
        inputs.append(ctx.enqueue_create_buffer[DType.bfloat16](n * HIDDEN))
        inputs[r].enqueue_fill(1)
        topks.append(ctx.enqueue_create_buffer[DType.int32](n * TOP_K))
        ctx.enqueue_copy(topks[r], h_topk)
        weights.append(ctx.enqueue_create_buffer[DType.float32](n * TOP_K))
        weights[r].enqueue_fill(0.125)
        ranks[r].expert_out.enqueue_fill(0.5)
        ctx.synchronize()

    var times = List[Int](length=world, fill=0)

    def per_rank(r: Int) raises {mut times, imm}:
        var ctx = ctxs[r]
        for _ in range(20):
            run_dispatch[small, world](
                ctx, ranks[r], tables, inputs[r], topks[r], n
            )
            run_combine[small, world](ctx, ranks[r], tables, weights[r], n)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(iters):
            run_dispatch[small, world](
                ctx, ranks[r], tables, inputs[r], topks[r], n
            )
            run_combine[small, world](ctx, ranks[r], tables, weights[r], n)
        ctx.synchronize()
        times[r] = Int(perf_counter_ns() - t0)

    sync_parallelize(per_rank, world)
    var worst = 0
    for r in range(world):
        worst = max(worst, times[r])
    print(
        t"time small={small} n={n}: dispatch+combine"
        t" {Float64(worst) / Float64(iters) / 1000.0} us/iter (slowest rank,"
        t" host-timed)"
    )


def run[world: Int]() raises:
    print(t"world {world}")
    comptime msg_bytes = BlockwiseFP8TokenFormat[
        fp8_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float32,
        output_layout=type_of(row_major((Idx[MAX_RECV], Idx[HIDDEN]))),
        scales_layout=type_of(row_major((Idx[N_SCALES], Idx[MAX_RECV]))),
        HIDDEN,
        TOP_K,
    ].msg_size()
    var ctxs = List[DeviceContext[]]()
    for i in range(world):
        ctxs.append(DeviceContext(device_id=i))
    var slow = List[Rank]()
    var fast = List[Rank]()
    for i in range(world):
        slow.append(Rank(ctxs[i], msg_bytes, world))
        fast.append(Rank(ctxs[i], msg_bytes, world))
        ctxs[i].synchronize()
    var st = List[MutPointer[UInt64, MutAnyOrigin]]()
    var ft = List[MutPointer[UInt64, MutAnyOrigin]]()
    for w in range(6):
        st.append(ptr_table(slow, w))
        ft.append(ptr_table(fast, w))

    # Plans are written for 4 devices; device r takes entry r * 4 // world,
    # which keeps counts within one of each other at any device count.
    def per_rank(plan: List[Int]) {} -> List[Int]:
        var counts = List[Int]()
        for r in range(world):
            counts.append(plan[r * 4 // world])
        return counts^

    # Decode-sized, then the bound from both sides (only some devices at it),
    # protocol-dispatched steps (combined by the partial-row combine), a
    # zero-token device, and back to small. A prefill-sized build adds
    # Flash's 8K-token step.
    var plans: List[List[Int]] = [
        [12, 12, 12, 12],
        [2, 2, 1, 1],
        [32, 32, 31, 31],
        [33, 32, 32, 32],
        [40, 40, 40, 40],
        [1, 0, 0, 0],
        [24, 24, 24, 24],
        [32, 32, 32, 32],
        [12, 12, 12, 12],
    ]
    comptime if MAX_TOK >= 2028:
        plans.append([2028, 2028, 2028, 2028])
        plans.append([12, 12, 12, 12])
        plans.append([MAX_TOK, MAX_TOK, MAX_TOK - 1, MAX_TOK - 1])
    # Start the small path's epoch just short of its wrap, so the steps
    # cross it.
    comptime n_experts = N_LOCAL * world
    var seed_epoch = unsafe_alloc[Int32](1)
    seed_epoch[unsafe_offset=0] = Int32(EP_P2P_EPOCH_MAX - 2)
    for r in range(world):
        ctxs[r].enqueue_copy(
            fast[r].counters.create_sub_buffer[DType.int32](
                EPLocalSyncCounters[n_experts].total_size(), 1
            ),
            seed_epoch,
        )
        ctxs[r].synchronize()
    var total_bad = 0
    for i in range(len(plans)):
        total_bad += step[world](
            ctxs, slow, fast, st, ft, per_rank(plans[i]), i + 1
        )
    assert_equal(total_bad, 0, "small path differs from the protocol")

    comptime if not get_defined_bool["P2P_SMALL_TIME", False]():
        return
    var sizes: List[Int] = [2, 12, 24]
    comptime if MAX_TOK >= 2028:
        sizes = [2028, 512]
    for n in sizes:
        time_path[0, world](ctxs, slow, st, n, 200)
        time_path[SMALL, world](ctxs, fast, ft, n, 200)
        time_path[0, world](ctxs, slow, st, n, 200)
        time_path[SMALL, world](ctxs, fast, ft, n, 200)


def main() raises:
    if not enable_p2p():
        raise Error("Cannot enable P2P Mem Access!")
    var n_devices = DeviceContext.number_of_devices()
    comptime for world in [2, 4, 8]:
        if n_devices == world:
            run[world]()
            return
    print(t"needs 2, 4 or 8 GPUs, found {n_devices}; skipping")
