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
"""Checks the fused EP dispatch's prefill send against the stock kernel.

Runs on every visible GPU (1, 2, 4 or 8) at DeepSeek-V4.1's shape: hidden
5120, top-k 6, 96 experts a rank, for both wire formats that take the warp
send: NVFP4 and MXFP8 (the rows DeepSeek-V4.1's MXFP4 experts read).
`dispatch_warp_min_tokens` picks the kernel: -1 is stock, and any other value
counts experts with the histogram monitor and sends one token per warp from
that many tokens a rank on. Each variant (the
histogram with the per-CTA send, the default `EP_WARP_TOKENS_MIN_TOKENS`, and
the warp send at every count) dispatches the same tokens as stock, and its
result must equal stock's as sets keyed by (destination rank, local expert,
source rank, source token): the receive buffer's messages, and the copy-out
the grouped GEMM reads (codes, the 5D scale-factor tiles, src_info), plus
the per (expert, source rank) row counts, which must also match the routing.
Each (local expert, source rank) of the receive buffer must list exactly the
source tokens the routing sends it, and each listed token's message must
carry that token's index. Only the order of rows inside a segment may
differ. Every checked run starts from a poisoned receive buffer and scale
tiles, so a row a variant never writes cannot pass on the previous run's
copy.

Token counts: 0, 1, 13, 509 and 8168 a rank, and with more than one GPU an
empty rank 1 beside full ranks. Planted faults must fail the comparison: a
changed input element, a corrupted copy-out byte, row offsets that drop one
expert's rows (what a monitor that loses a count would publish), and a
receive-buffer token index pointing at another token.

`EP_DISPATCH_AB=1` also times, with a stock control arm, arm order rotated
over 3 rounds and 5 samples an arm a round: the variants at 8,168 tokens,
decode at 144 tokens (24 requests x 6), and both sends, with a second per-CTA
arm as the control, over the token sweep that sets
`EP_WARP_TOKENS_MIN_TOKENS`. Each timed launch sits behind a busy kernel so
its timer excludes the host's launch work; the slowest rank's time is
reported.
"""

from std.memory.alloc import unsafe_alloc
from std.os import getenv
from std.random import randint, randn, seed
from std.time import global_perf_counter_ns

from max.algorithm import sync_parallelize
from comm.sync import enable_p2p
from max.gpu.host import DeviceBuffer, DeviceContext, default_accelerator
from layout import TileTensor, Idx, row_major
from linalg.fp4_utils import (
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from shmem.ep import ep_fused_dispatch_kernel_api
from shmem.ep_comm import (
    EP_TOKEN_INDEX_BYTES,
    EP_WARP_TOKENS_MIN_TOKENS,
    EPLocalSyncCounters,
    NVBlockScaledTokenFormat,
    ep_dispatch_recv_size,
    ep_token_index_offset,
)
from std.testing import assert_equal, assert_true

comptime HIDDEN = 5120
comptime TOP_K = 6
comptime N_LOCAL = 96

# `dispatch_warp_min_tokens` values besides the default: the stock kernel, the
# histogram monitor with the per-CTA send (a threshold above every count), and
# the warp send at every count.
comptime STOCK = -1
comptime PER_CTA_SEND = 1 << 30
comptime WARP_SEND = 0

# Poisoned bytes past the receive buffer's end, which no dispatch may write.
comptime RECV_GUARD = 4096
# Int32 words in one receive-buffer token index sector.
comptime INDEX_WORDS = EP_TOKEN_INDEX_BYTES // 4


def _mut[
    dtype: DType
](buf: DeviceBuffer[dtype]) -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    """Returns the buffer's address as a mutable pointer; the caller keeps the
    buffer alive."""
    return (
        buf.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )


def spin_kernel(ns: UInt64):
    """Keeps the GPU busy so a timed launch queued behind it starts at once."""
    var t0 = global_perf_counter_ns()
    while global_perf_counter_ns() - t0 < ns:
        pass


def legalize_topk_ids(ids: MutPointer[Int32, _], n_tokens: Int, n_experts: Int):
    """Makes each token's top-k experts distinct."""
    for t in range(n_tokens):
        var row = ids.unsafe_offset(t * TOP_K)
        for i in range(TOP_K):
            var dup = True
            while dup:
                dup = False
                for j in range(i):
                    if row[i] == row[j]:
                        dup = True
                if dup:
                    randint(row.unsafe_offset(i), 1, 0, n_experts - 1)


struct DispatchCase[n_ranks: Int, max_tokens: Int, mxfp8: Bool]:
    """Buffers and launches of the NVFP4 (or, with `mxfp8`, MXFP8) fused
    dispatch on `n_ranks` GPUs."""

    comptime quant_dtype = DType.float8_e4m3fn if Self.mxfp8 else DType.uint8
    comptime scales_dtype = (
        DType.float8_e8m0fnu if Self.mxfp8 else DType.float8_e4m3fn
    )
    comptime sf_vector = (
        MXFP8_SF_VECTOR_SIZE if Self.mxfp8 else NVFP4_SF_VECTOR_SIZE
    )
    comptime code_bytes = HIDDEN if Self.mxfp8 else HIDDEN // 2
    comptime n_scales = HIDDEN // Self.sf_vector
    comptime k_tiles = HIDDEN // (Self.sf_vector * SF_ATOM_K)
    comptime n_experts = N_LOCAL * Self.n_ranks
    comptime max_recv = Self.n_ranks * Self.max_tokens * TOP_K
    comptime scales_rows = Self.max_recv + N_LOCAL * SF_MN_GROUP_SIZE
    comptime out_layout = row_major((Idx[Self.max_recv], Idx[Self.code_bytes]))
    comptime scales_layout = row_major(
        (
            Self.scales_rows // SF_MN_GROUP_SIZE,
            Idx[Self.k_tiles],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    comptime offsets_layout = row_major[N_LOCAL]()
    comptime Fmt = NVBlockScaledTokenFormat[
        quant_dtype=Self.quant_dtype,
        scales_dtype=Self.scales_dtype,
        output_layout=type_of(Self.out_layout),
        scales_offset_layout=type_of(Self.offsets_layout),
        HIDDEN,
        TOP_K,
    ]
    comptime msg = Self.Fmt.msg_size()
    comptime recv_bytes = ep_dispatch_recv_size(
        n_experts=Self.n_experts,
        n_ranks=Self.n_ranks,
        max_tokens_per_rank=Self.max_tokens,
        msg_bytes=Self.msg,
        token_recv=True,
    )
    # The receive buffer's (local expert, source rank, slot) source token
    # indices follow its (source rank, source token) messages.
    comptime index_offset = ep_token_index_offset(
        n_ranks=Self.n_ranks,
        max_tokens_per_rank=Self.max_tokens,
        msg_bytes=Self.msg,
    )
    comptime n_index = N_LOCAL * Self.n_ranks * Self.max_tokens
    comptime n_counters = EPLocalSyncCounters[Self.n_experts].allocation_size()

    var ctxs: List[DeviceContext[]]
    var send: List[DeviceBuffer[.uint8]]
    var recv: List[DeviceBuffer[.uint8]]
    var recv_count: List[DeviceBuffer[.uint64]]
    var counters: List[DeviceBuffer[.int32]]
    var tokens: List[DeviceBuffer[.bfloat16]]
    var topk: List[DeviceBuffer[.int32]]
    var codes: List[DeviceBuffer[Self.quant_dtype]]
    var scales: List[DeviceBuffer[Self.scales_dtype]]
    var scales_offsets: List[DeviceBuffer[.uint32]]
    var row_offsets: List[DeviceBuffer[.uint32]]
    var expert_ids: List[DeviceBuffer[.int32]]
    var src_info: List[DeviceBuffer[.int32]]
    var handlers: List[Self.Fmt]
    # One uint64 address per rank, the form the kernel API takes.
    var p_send: MutPointer[UInt64, MutUntrackedOrigin]
    var p_recv: MutPointer[UInt64, MutUntrackedOrigin]
    var p_recv_count: MutPointer[UInt64, MutUntrackedOrigin]

    def __init__(out self, ctxs: List[DeviceContext[]]) raises:
        self.ctxs = ctxs.copy()
        self.send = []
        self.recv = []
        self.recv_count = []
        self.counters = []
        self.tokens = []
        self.topk = []
        self.codes = []
        self.scales = []
        self.scales_offsets = []
        self.row_offsets = []
        self.expert_ids = []
        self.src_info = []
        self.handlers = []
        self.p_send = unsafe_alloc[UInt64](Self.n_ranks).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.p_recv = unsafe_alloc[UInt64](Self.n_ranks).unsafe_origin_cast[
            MutUntrackedOrigin
        ]()
        self.p_recv_count = unsafe_alloc[UInt64](
            Self.n_ranks
        ).unsafe_origin_cast[MutUntrackedOrigin]()
        for r in range(Self.n_ranks):
            var ctx = self.ctxs[r]
            # fmt: off
            self.send.append(ctx.enqueue_create_buffer[.uint8](Self.max_tokens * Self.msg))
            self.recv.append(ctx.enqueue_create_buffer[.uint8](Self.recv_bytes + RECV_GUARD))
            self.recv_count.append(ctx.enqueue_create_buffer[.uint64](Self.n_experts))
            self.counters.append(ctx.enqueue_create_buffer[.int32](Self.n_counters))
            self.tokens.append(ctx.enqueue_create_buffer[.bfloat16](max(Self.max_tokens, 1) * HIDDEN))
            self.topk.append(ctx.enqueue_create_buffer[.int32](max(Self.max_tokens, 1) * TOP_K))
            self.codes.append(ctx.enqueue_create_buffer[Self.quant_dtype](Self.max_recv * Self.code_bytes))
            self.scales.append(ctx.enqueue_create_buffer[Self.scales_dtype](Self.scales_rows * Self.n_scales))
            self.scales_offsets.append(ctx.enqueue_create_buffer[.uint32](N_LOCAL))
            self.row_offsets.append(ctx.enqueue_create_buffer[.uint32](N_LOCAL + 1))
            self.expert_ids.append(ctx.enqueue_create_buffer[.int32](N_LOCAL))
            self.src_info.append(ctx.enqueue_create_buffer[.int32](Self.max_recv * 2))
            # fmt: on
            self.p_send[r] = UInt64(Int(self.send[r].unsafe_ptr()))
            self.p_recv[r] = UInt64(Int(self.recv[r].unsafe_ptr()))
            self.p_recv_count[r] = UInt64(Int(self.recv_count[r].unsafe_ptr()))
        for r in range(Self.n_ranks):
            # Built once: the format's TMA descriptor is host work a timer
            # must not see.
            self.handlers.append(
                Self.Fmt(
                    TileTensor(_mut(self.codes[r]), Self.out_layout),
                    TileTensor(_mut(self.scales[r]), Self.scales_layout),
                    TileTensor(
                        _mut(self.scales_offsets[r]), Self.offsets_layout
                    ),
                    self.ctxs[r],
                )
            )
        self.sync()

    def __deinit__(deinit self):
        self.p_send.unsafe_free()
        self.p_recv.unsafe_free()
        self.p_recv_count.unsafe_free()

    def sync(self) raises:
        for r in range(Self.n_ranks):
            self.ctxs[r].synchronize()

    def upload(self, host: HostInputs) raises:
        for r in range(Self.n_ranks):
            var n = host.n_tokens[r]
            if n == 0:
                continue
            var ctx = self.ctxs[r]
            ctx.enqueue_copy(
                self.tokens[r].create_sub_buffer[.bfloat16](0, n * HIDDEN),
                host.x[r].unsafe_ptr(),
            )
            ctx.enqueue_copy(
                self.topk[r].create_sub_buffer[.int32](0, n * TOP_K),
                host.topk[r].unsafe_ptr(),
            )
        self.sync()

    def poke(self, r: Int, i: Int, value: BFloat16) raises:
        """Overwrites input element `i` of rank `r`."""
        var v: List[BFloat16] = [value]
        self.ctxs[r].enqueue_copy(
            self.tokens[r].create_sub_buffer[.bfloat16](i, 1), v.unsafe_ptr()
        )
        self.ctxs[r].synchronize()

    def reset(self, poison: Bool) raises:
        """Zeroes the counters and arms the receive counts on every rank;
        `poison` also fills the receive buffer and the scale tiles with 0xFF.

        Peers write into each other's receive counts, so every reset lands
        before any rank launches.
        """
        for r in range(Self.n_ranks):
            var ctx = self.ctxs[r]
            ctx.enqueue_memset(self.counters[r], Int32(0))
            ctx.enqueue_memset(self.recv_count[r], UInt64.MAX_FINITE)
            ctx.enqueue_memset(
                self.codes[r].create_sub_buffer[.uint8](
                    0, Self.max_recv * Self.code_bytes
                ),
                UInt8(0),
            )
            ctx.enqueue_memset(self.src_info[r], Int32(-1))
            if poison:
                ctx.enqueue_memset(self.recv[r], UInt8(0xFF))
                ctx.enqueue_memset(
                    self.scales[r].create_sub_buffer[.uint8](
                        0, Self.scales_rows * Self.n_scales
                    ),
                    UInt8(0xFF),
                )
        self.sync()

    def launch[
        min_tokens: Int
    ](self, r: Int, n_tokens: Int, ctx: DeviceContext[]) raises:
        var ptrs_layout = row_major[Self.n_ranks]()
        ep_fused_dispatch_kernel_api[
            Self.n_experts,
            Self.max_tokens,
            Self.n_ranks,
            1,
            False,
            dispatch_warp_min_tokens=min_tokens,
        ](
            self.handlers[r],
            TileTensor(_mut(self.row_offsets[r]), row_major[N_LOCAL + 1]()),
            TileTensor(_mut(self.expert_ids[r]), row_major[N_LOCAL]()),
            TileTensor(
                _mut(self.src_info[r]), row_major((Idx[Self.max_recv], Idx[2]))
            ),
            TileTensor(_mut(self.counters[r]), row_major[Self.n_counters]()),
            TileTensor(
                _mut(self.tokens[r]), row_major((n_tokens, Idx[HIDDEN]))
            ).as_imm(),
            TileTensor(
                _mut(self.topk[r]), row_major((n_tokens, Idx[TOP_K]))
            ).as_imm(),
            TileTensor(self.p_send, ptrs_layout),
            TileTensor(self.p_recv, ptrs_layout),
            TileTensor(self.p_recv_count, ptrs_layout),
            ctx,
        )

    def run[min_tokens: Int](self, host: HostInputs) raises:
        self.reset(poison=True)
        for r in range(Self.n_ranks):
            self.launch[min_tokens](r, host.n_tokens[r], self.ctxs[r])
        self.sync()

    def time_arm(self, n_tokens: List[Int], min_tokens: Int) raises -> Float64:
        """Times one dispatch at a `dispatch_warp_min_tokens` the test
        instantiates, over each rank's first `n_tokens[r]` uploaded tokens."""
        if min_tokens == STOCK:
            return self.timed[STOCK](n_tokens)
        if min_tokens == PER_CTA_SEND:
            return self.timed[PER_CTA_SEND](n_tokens)
        if min_tokens == EP_WARP_TOKENS_MIN_TOKENS:
            return self.timed[EP_WARP_TOKENS_MIN_TOKENS](n_tokens)
        if min_tokens == WARP_SEND:
            return self.timed[WARP_SEND](n_tokens)
        raise Error("no timed arm for this dispatch_warp_min_tokens")

    def timed[min_tokens: Int](self, n_tokens: List[Int]) raises -> Float64:
        """Times one dispatch on every rank at once; returns the slowest
        rank's microseconds."""
        self.reset(poison=False)
        for r in range(Self.n_ranks):
            self.ctxs[r].enqueue_function[spin_kernel](
                UInt64(200_000), grid_dim=1, block_dim=1
            )
        var ns = List[Int](length=Self.n_ranks, fill=0)

        def per_rank(r: Int) raises {mut ns, imm}:
            @inline(.always)
            def body(ctx: DeviceContext[]) raises {imm}:
                self.launch[min_tokens](r, n_tokens[r], ctx)

            ns[r] = self.ctxs[r].execution_time(body, 1)

        sync_parallelize(per_rank, Self.n_ranks)
        var worst = 0
        for r in range(Self.n_ranks):
            worst = max(worst, ns[r])
        return Float64(worst) / 1e3


struct HostInputs(Movable):
    """Each rank's tokens and top-k ids, and the rows the routing sends."""

    var n_tokens: List[Int]
    var x: List[List[BFloat16]]
    var topk: List[List[Int32]]
    var expected: List[Int]
    """Rows per (destination rank, local expert, source rank)."""

    def __init__(
        out self, n_tokens: List[Int], n_ranks: Int, case_seed: Int
    ) raises:
        self.n_tokens = n_tokens.copy()
        self.x = []
        self.topk = []
        self.expected = List[Int](length=n_ranks * N_LOCAL * n_ranks, fill=0)
        var n_experts = N_LOCAL * n_ranks
        for r in range(n_ranks):
            seed(case_seed * 977 + r)
            var n = n_tokens[r]
            var x = List[BFloat16](length=max(n, 1) * HIDDEN, fill=0)
            var t = List[Int32](length=max(n, 1) * TOP_K, fill=0)
            if n > 0:
                randn(x.unsafe_ptr(), n * HIDDEN)
                randint(t.unsafe_ptr(), n * TOP_K, 0, n_experts - 1)
                legalize_topk_ids(t.unsafe_ptr(), n, n_experts)
            for i in range(n * TOP_K):
                var e = Int(t[i])
                var d = e // N_LOCAL
                self.expected[(d * N_LOCAL + e % N_LOCAL) * n_ranks + r] += 1
            self.x.append(x^)
            self.topk.append(t^)


struct Snapshot(Movable):
    """One rank's dispatch result, as host copies."""

    var row_offsets: List[UInt32]
    var scales_offsets: List[UInt32]
    var src_info: List[Int32]
    var codes: List[UInt8]
    var scales: List[UInt8]
    var recv: List[UInt8]
    """The receive buffer's messages, by (source rank, source token)."""
    var index: List[Int32]
    """The receive buffer's token index sectors, by (local expert, source
    rank, slot); a sector's first word is the source token."""

    def __init__(out self):
        self.row_offsets = []
        self.scales_offsets = []
        self.src_info = []
        self.codes = []
        self.scales = []
        self.recv = []
        self.index = []


def snapshot[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](
    dc: DispatchCase[n_ranks, max_tokens, mxfp8], host: HostInputs, r: Int
) raises -> Snapshot:
    comptime code_bytes = type_of(dc).code_bytes
    var s = Snapshot()
    var ctx = dc.ctxs[r]
    s.row_offsets = List[UInt32](length=N_LOCAL + 1, fill=0)
    s.scales_offsets = List[UInt32](length=N_LOCAL, fill=0)
    ctx.enqueue_copy(s.row_offsets.unsafe_ptr(), dc.row_offsets[r])
    ctx.enqueue_copy(s.scales_offsets.unsafe_ptr(), dc.scales_offsets[r])
    ctx.synchronize()
    var rows = Int(s.row_offsets[N_LOCAL])
    s.src_info = List[Int32](length=max(rows, 1) * 2, fill=0)
    s.codes = List[UInt8](length=max(rows, 1) * code_bytes, fill=0)
    if rows > 0:
        ctx.enqueue_copy(
            s.src_info.unsafe_ptr(),
            dc.src_info[r].create_sub_buffer[.int32](0, rows * 2),
        )
        ctx.enqueue_copy(
            s.codes.unsafe_ptr(),
            dc.codes[r].create_sub_buffer[.uint8](0, rows * code_bytes),
        )
    var n_scale_bytes = dc.scales_rows * dc.n_scales
    s.scales = List[UInt8](length=n_scale_bytes, fill=0)
    ctx.enqueue_copy(
        s.scales.unsafe_ptr(),
        dc.scales[r].create_sub_buffer[.uint8](0, n_scale_bytes),
    )
    s.recv = List[UInt8](length=dc.index_offset, fill=0)
    s.index = List[Int32](length=dc.n_index * INDEX_WORDS, fill=0)
    ctx.enqueue_copy(
        s.recv.unsafe_ptr(),
        dc.recv[r].create_sub_buffer[.uint8](0, dc.index_offset),
    )
    ctx.enqueue_copy(
        s.index.unsafe_ptr().unsafe_bitcast[UInt8](),
        dc.recv[r].create_sub_buffer[.uint8](
            dc.index_offset, dc.n_index * EP_TOKEN_INDEX_BYTES
        ),
    )
    var guard = List[UInt8](length=RECV_GUARD, fill=0)
    ctx.enqueue_copy(
        guard.unsafe_ptr(),
        dc.recv[r].create_sub_buffer[.uint8](dc.recv_bytes, RECV_GUARD),
    )
    ctx.synchronize()
    for b in guard:
        assert_equal(b, UInt8(0xFF), "a dispatch wrote past its receive buffer")
    return s^


def scale_index(
    s: Snapshot, expert: Int, row: Int, k: Int, k_tiles: Int
) -> Int:
    """Where the copy-out put scale `k` of output row `row` in the 5D
    scale-factor tiles: tile `t` of an expert's rows takes block
    `start // 128 + scales_offset[expert] + t`, row `u` of the tile."""
    var start = Int(s.row_offsets[expert])
    var local = row - start
    var block = (
        start // SF_MN_GROUP_SIZE
        + Int(s.scales_offsets[expert])
        + local // SF_MN_GROUP_SIZE
    )
    var u = local % SF_MN_GROUP_SIZE
    return (
        ((block * k_tiles + k // SF_ATOM_K) * SF_ATOM_M[0] + u % SF_ATOM_M[0])
        * SF_ATOM_M[1]
        + u // SF_ATOM_M[0]
    ) * SF_ATOM_K + k % SF_ATOM_K


def rows_by_key(
    s: Snapshot, expert: Int, d_counts: List[Int], n_ranks: Int, max_tokens: Int
) -> List[Int]:
    """Maps (source rank, source token) to the expert's output row, or -1.

    The copy-out places an expert's rows source rank by source rank, so a
    row's source rank follows from its position and the per-rank counts.
    """
    var out = List[Int](length=n_ranks * max(max_tokens, 1), fill=-1)
    var row = Int(s.row_offsets[expert])
    for src in range(n_ranks):
        for _ in range(d_counts[src]):
            var tok = Int(s.src_info[2 * row])
            if tok >= 0 and tok < max_tokens:
                out[src * max_tokens + tok] = row
            row += 1
    return out^


def compare[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](
    host: HostInputs,
    ref_snaps: List[Snapshot],
    new_snaps: List[Snapshot],
    msg_bytes: Int,
) -> Int:
    """Counts mismatches between two dispatch results, compared as sets.

    Covers per (destination, expert, source) row counts against the
    routing, the receive buffer's messages, and the copy-out's codes,
    scales and src_info.
    """
    comptime DC = DispatchCase[n_ranks, max_tokens, mxfp8]
    comptime code_bytes = DC.code_bytes
    var bad = 0
    for d in range(n_ranks):
        ref s0 = ref_snaps[d]
        ref s1 = new_snaps[d]
        for e in range(N_LOCAL + 1):
            if s0.row_offsets[e] != s1.row_offsets[e]:
                bad += 1
        for e in range(N_LOCAL):
            var counts = List[Int](length=n_ranks, fill=0)
            var total = 0
            for src in range(n_ranks):
                counts[src] = host.expected[(d * N_LOCAL + e) * n_ranks + src]
                total += counts[src]
            var n0 = Int(s0.row_offsets[e + 1]) - Int(s0.row_offsets[e])
            var n1 = Int(s1.row_offsets[e + 1]) - Int(s1.row_offsets[e])
            if n0 != total or n1 != total:
                # Rows are missing or extra, so the keyed walks below would
                # read the wrong ranges.
                bad += 1
                continue
            var k0 = rows_by_key(s0, e, counts, n_ranks, max_tokens)
            var k1 = rows_by_key(s1, e, counts, n_ranks, max_tokens)
            for key in range(len(k0)):
                var r0 = k0[key]
                var r1 = k1[key]
                if (r0 < 0) != (r1 < 0):
                    bad += 1
                    continue
                if r0 < 0:
                    continue
                if s0.src_info[2 * r0 + 1] != s1.src_info[2 * r1 + 1]:
                    bad += 1
                for b in range(code_bytes):
                    if (
                        s0.codes[r0 * code_bytes + b]
                        != s1.codes[r1 * code_bytes + b]
                    ):
                        bad += 1
                        break
                for k in range(DC.n_scales):
                    if (
                        s0.scales[scale_index(s0, e, r0, k, DC.k_tiles)]
                        != s1.scales[scale_index(s1, e, r1, k, DC.k_tiles)]
                    ):
                        bad += 1
                        break
            # Receive buffer: each (expert, source rank) lists exactly the
            # tokens the routing sends it, and their messages are the same
            # and name their own token.
            var expert = Int32(d * N_LOCAL + e)
            for src in range(n_ranks):
                var n_src = host.n_tokens[src]
                var seg = (e * n_ranks + src) * max_tokens

                def listed(s: Snapshot) {mut bad, imm} -> List[Bool]:
                    var out = List[Bool](length=max(n_src, 1), fill=False)
                    for slot in range(counts[src]):
                        var t = Int(s.index[(seg + slot) * INDEX_WORDS])
                        if t < 0 or t >= n_src or out[t]:
                            bad += 1
                        else:
                            out[t] = True
                    return out^

                var listed0 = listed(s0)
                var listed1 = listed(s1)
                for t in range(n_src):
                    var routed = False
                    for k in range(TOP_K):
                        if host.topk[src][t * TOP_K + k] == expert:
                            routed = True
                    if listed0[t] != routed or listed1[t] != routed:
                        bad += 1
                        continue
                    if not routed:
                        continue
                    var row = (src * max_tokens + t) * msg_bytes
                    var off = row + DC.Fmt.src_info_offset()
                    var named = Int(
                        Int32(s1.recv[off])
                        | (Int32(s1.recv[off + 1]) << 8)
                        | (Int32(s1.recv[off + 2]) << 16)
                        | (Int32(s1.recv[off + 3]) << 24)
                    )
                    if named != t:
                        bad += 1
                    for x in range(msg_bytes):
                        if s0.recv[row + x] != s1.recv[row + x]:
                            bad += 1
                            break
    return bad


def snapshots[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](
    dc: DispatchCase[n_ranks, max_tokens, mxfp8], host: HostInputs
) raises -> List[Snapshot]:
    var out = List[Snapshot]()
    for r in range(n_ranks):
        out.append(snapshot(dc, host, r))
    return out^


def run_variant[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool, min_tokens: Int
](
    dc: DispatchCase[n_ranks, max_tokens, mxfp8],
    host: HostInputs,
    stock: List[Snapshot],
    name: StaticString,
    expect_fail: Bool = False,
) raises -> List[Snapshot]:
    dc.run[min_tokens](host)
    var snaps = snapshots(dc, host)
    var bad = compare[n_ranks, max_tokens, mxfp8](host, stock, snaps, dc.msg)
    print("   ", name, "mismatches", bad)
    if expect_fail:
        assert_true(bad > 0, "a planted fault passed the check")
    else:
        assert_equal(bad, 0, "variant differs from the stock dispatch")
    return snaps^


def check_tokens[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](dc: DispatchCase[n_ranks, max_tokens, mxfp8], n_tokens: List[Int]) raises:
    print(t"== mxfp8 {mxfp8} ranks {n_ranks} tokens", end="")
    for r in range(n_ranks):
        print(" ", n_tokens[r], end="")
    print()
    var host = HostInputs(n_tokens, n_ranks, n_tokens[0] + 3)
    dc.upload(host)
    dc.run[STOCK](host)
    var stock = snapshots(dc, host)
    # The stock result must itself match the routing.
    assert_equal(
        compare[n_ranks, max_tokens, mxfp8](host, stock, stock, dc.msg),
        0,
        "stock dispatch disagrees with the routing",
    )
    _ = run_variant[n_ranks, max_tokens, mxfp8, PER_CTA_SEND](
        dc, host, stock, "histogram, per-CTA send"
    )
    _ = run_variant[n_ranks, max_tokens, mxfp8, EP_WARP_TOKENS_MIN_TOKENS](
        dc, host, stock, "final (warp send from the threshold, histogram)"
    )
    var snaps = run_variant[n_ranks, max_tokens, mxfp8, WARP_SEND](
        dc, host, stock, "warp send at every count, histogram"
    )

    # Planted faults on the check itself.
    var d = 0
    while d < n_ranks and snaps[d].row_offsets[N_LOCAL] == 0:
        d += 1
    if d == n_ranks:
        return
    # A corrupted copy-out byte.
    snaps[d].codes[0] ^= 0x5A
    var bad = compare[n_ranks, max_tokens, mxfp8](host, stock, snaps, dc.msg)
    print("    planted: corrupted code byte mismatches", bad)
    assert_true(bad > 0, "a corrupted code byte passed the check")
    snaps[d].codes[0] ^= 0x5A
    # Row offsets without the first non-empty expert's rows, as a monitor
    # that lost that expert's count would publish them.
    var e = 0
    while snaps[d].row_offsets[e + 1] == snaps[d].row_offsets[e]:
        e += 1
    var dropped = snaps[d].row_offsets[e + 1] - snaps[d].row_offsets[e]
    for i in range(e + 1, N_LOCAL + 1):
        snaps[d].row_offsets[i] -= dropped
    bad = compare[n_ranks, max_tokens, mxfp8](host, stock, snaps, dc.msg)
    print("    planted: dropped expert count mismatches", bad)
    assert_true(bad > 0, "a dropped expert count passed the check")
    for i in range(e + 1, N_LOCAL + 1):
        snaps[d].row_offsets[i] += dropped
    # A receive-buffer token index pointing at another token of its source.
    var src = 0
    while host.expected[(d * N_LOCAL + e) * n_ranks + src] == 0:
        src += 1
    ref listed = snaps[d].index[(e * n_ranks + src) * max_tokens * INDEX_WORDS]
    listed = Int32((Int(listed) + 1) % max(n_tokens[src], 2))
    bad = compare[n_ranks, max_tokens, mxfp8](host, stock, snaps, dc.msg)
    print("    planted: wrong receive token index mismatches", bad)
    assert_true(bad > 0, "a wrong receive token index passed the check")
    # A changed input element changes that token's codes or scales.
    var r0 = 0
    while n_tokens[r0] == 0:
        r0 += 1
    dc.poke(r0, 0, BFloat16(1000.0))
    _ = run_variant[n_ranks, max_tokens, mxfp8, WARP_SEND](
        dc, host, stock, "planted: changed input", expect_fail=True
    )
    dc.poke(r0, 0, host.x[r0][0])


def rotated_medians[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](
    dc: DispatchCase[n_ranks, max_tokens, mxfp8],
    n_tokens: List[Int],
    arms: List[Int],
) raises -> List[List[Float64]]:
    """Times each arm (a `dispatch_warp_min_tokens`) over 3 rounds of 5
    samples, rotating the arm order each round. Returns each arm's sorted
    round medians, so `[arm][1]` is its median over rounds."""
    var med = List[List[Float64]]()
    for _ in range(len(arms)):
        med.append(List[Float64]())
    for rnd in range(3):
        for j in range(len(arms)):
            var arm = (rnd + j) % len(arms)
            var ts = List[Float64]()
            for _ in range(5):
                ts.append(dc.time_arm(n_tokens, arms[arm]))
            sort(Span(ts))
            med[arm].append(ts[2])
    for arm in range(len(arms)):
        sort(Span(med[arm]))
    return med^


def time_arms[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](
    dc: DispatchCase[n_ranks, max_tokens, mxfp8], n: Int, label: StaticString
) raises:
    """Times stock, the histogram alone, the final config and a stock control
    at `n` uploaded tokens a rank."""
    var names: List[StaticString] = [
        "stock",
        "histogram",
        "final",
        "stock_control",
    ]
    var arms: List[Int] = [
        STOCK,
        PER_CTA_SEND,
        EP_WARP_TOKENS_MIN_TOKENS,
        STOCK,
    ]
    var med = rotated_medians(dc, List[Int](length=n_ranks, fill=n), arms)
    for arm in range(len(arms)):
        print(
            t"{label} mxfp8 {mxfp8} summary {names[arm]} median over rounds us"
            t" {med[arm][1]} rounds {med[arm][0]} {med[arm][1]} {med[arm][2]}"
        )


def sweep[
    n_ranks: Int, max_tokens: Int, mxfp8: Bool
](dc: DispatchCase[n_ranks, max_tokens, mxfp8]) raises:
    """Stock, both sends with the histogram, and a second per-CTA arm as the
    control, over uploaded-token prefixes."""
    var counts: List[Int] = [
        144,
        290,
        580,
        768,
        896,
        1024,
        1152,
        1280,
        1536,
        2320,
        4640,
        8168,
    ]
    var arms: List[Int] = [STOCK, PER_CTA_SEND, WARP_SEND, PER_CTA_SEND]
    for c in counts:
        var n = min(c, max_tokens)
        var med = rotated_medians(dc, List[Int](length=n_ranks, fill=n), arms)
        print(
            t"SWEEP mxfp8 {mxfp8} tokens {n} stock us {med[0][1]} per-CTA"
            t" send+histogram us {med[1][1]} warp send+histogram us"
            t" {med[2][1]} per-CTA control us {med[3][1]} warp rounds"
            t" {med[2][0]} {med[2][2]} per-CTA rounds {med[1][0]} {med[1][2]}"
        )


def run_all[n_ranks: Int](ctxs: List[DeviceContext[]]) raises:
    run_format[n_ranks, False](ctxs)
    run_format[n_ranks, True](ctxs)


def run_format[n_ranks: Int, mxfp8: Bool](ctxs: List[DeviceContext[]]) raises:
    comptime big = 8168
    var small_case = DispatchCase[n_ranks, 512, mxfp8](ctxs)
    for n in [0, 1, 13, 509]:
        check_tokens(small_case, List[Int](length=n_ranks, fill=n))
    comptime if n_ranks > 1:
        var ragged = List[Int](length=n_ranks, fill=509)
        ragged[1] = 0
        check_tokens(small_case, ragged)
    var big_case = DispatchCase[n_ranks, big, mxfp8](ctxs)
    check_tokens(big_case, List[Int](length=n_ranks, fill=big))
    if getenv("EP_DISPATCH_AB") == "1":
        big_case.upload(
            HostInputs(List[Int](length=n_ranks, fill=big), n_ranks, 11)
        )
        time_arms(big_case, big, "PREFILL")
        time_arms(big_case, 144, "DECODE")
        sweep(big_case)


def main() raises:
    comptime assert (
        default_accelerator().is_nvidia_gpu()
    ), "the NVFP4 and MXFP8 dispatch formats are tested on NVIDIA GPUs"
    var n = DeviceContext.number_of_devices()
    if n > 1 and not enable_p2p():
        raise Error("Cannot enable P2P Mem Access!")
    var ctxs = List[DeviceContext[]]()
    for i in range(n):
        ctxs.append(DeviceContext(device_id=i))
    if n == 1:
        run_all[1](ctxs)
    elif n == 2:
        run_all[2](ctxs)
    elif n == 4:
        run_all[4](ctxs)
    elif n == 8:
        run_all[8](ctxs)
    else:
        raise Error("needs 1, 2, 4 or 8 visible GPUs")
    print("All dispatch send checks passed")
