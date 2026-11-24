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

"""Runs the all-reduce mode's EP dispatch and combine within one device.

In the all-reduce mode (`skip_a2a`) every device holds every row, so the
dispatch only sorts this device's (token, top-k) slots by expert and the
combine only gathers them back. The protocol kernels do that through the
handshakes they run between devices: one CTA counts, another computes
offsets, and the rest wait on its flags before they copy. Here each CTA
derives the expert-sorted layout from the top-k ids itself, so no CTA waits
on another. `ep_fused_dispatch_kernel_api` and `ep_fused_combine_kernel_api`
run these kernels where `ep_local_on` allows them.

The outputs match the protocol kernels' bit for bit: the same bytes in each
row, the same padded expert offsets, the same row ranges in the counter
buffer for the combine, and the same weighted sum. Inside an expert, rows run
in (token, top-k) order, which the protocol kernels leave to their atomics.
"""

from std.atomic import Atomic, Ordering
from std.bit import pop_count
from std.math import align_up, ceildiv, max, min
from std.math.uutils import ufloordiv
from std.memory import unsafe_stack_allocation
from std.sys.info import align_of, simd_width_of, size_of
from std.utils.index import StaticTuple

from layout import (
    Coord,
    DefaultEngine,
    Idx,
    TensorEngine,
    TensorLayout,
    TileTensor,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
import max.gpu.primitives.warp as warp
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.sync import barrier

from shmem.ep_comm import (
    BLOCK_SCOPE,
    EP_DATA_READY_FLAG,
    EPCombineStoreFn,
    EPLocalSyncCounters,
    ExpertLoadSink,
    InputScalesFn,
    NullExpertLoad,
    RouterWeightsFn,
    TokenFormat,
    block_prefix_sum,
    ep_shared_expert_rows,
)

comptime _EP_SELF_THREADS = 256
"""Threads per CTA of every kernel here."""

comptime _EP_SELF_CHUNK = _EP_SELF_THREADS
"""Tokens one dispatch CTA ranks in shared memory, one per thread. A step
with more tokens adds a launch that counts each chunk's slots per expert."""

comptime _EP_SELF_MAX_TOKENS_PER_CTA = 32
"""Most tokens one dispatch or combine CTA takes."""

comptime _EP_SELF_COMBINE_SCAN_MAX_TOKENS = 256
"""Most tokens in a step whose combine CTAs find their own rows by scanning
the source info of every routed row. Such a scan covers at most this many
tokens' top-k rows, a few loads per thread, and it saves the launch that
indexes every row once; larger steps take that launch instead, since every
CTA's scan would grow with the step."""

comptime _SharedPtr[dtype: DType] = MutPointer[
    Scalar[dtype], MutUntrackedOrigin, address_space=.SHARED
]


@always_inline
def _local_expert(expert: Int32, first_expert: Int32, n_local: Int) -> Int32:
    """Returns `expert`'s index among this device's experts, or -1."""
    var e = expert - first_expert
    if e >= 0 and e < Int32(n_local):
        return e
    return -1


@always_inline
def _slots_below(mask: _SharedPtr[.uint32], slot: Int) -> Int32:
    """Returns how many set bits of `mask` lie below bit `slot`.

    It reads only the words up to `slot`'s: a decode step's slots fill a few
    of a chunk's words, and each read is a step of a serial chain.
    """
    var word = slot // 32
    var below = (UInt32(1) << UInt32(slot % 32)) - 1
    var rank = pop_count(mask[unsafe_offset=word] & below)
    for w in range(word):
        rank += pop_count(mask[unsafe_offset=w])
    return Int32(rank)


@always_inline
def _load_row[
    dtype: DType, width: Int, n_vecs: Int, per_thread: Int
](src: ImmPointer[Scalar[dtype], ...]) -> Array[SIMD[dtype, width], per_thread]:
    """Loads this thread's vectors of a row, all in flight at once."""
    var row = Array[SIMD[dtype, width], per_thread](fill=0)
    comptime for v in range(per_thread):
        var i = thread_idx.x + v * _EP_SELF_THREADS
        if i < n_vecs:
            row[v] = src.unsafe_load[
                width=width, alignment=align_of[SIMD[dtype, width]]()
            ](i * width)
    return row^


comptime _EP_SELF_SCALE_CTAS = 32
"""Fewest CTAs a dispatch with a blocked scale layout launches. A decode
step's few CTAs would otherwise each store whole groups' scale tails (128
rows of `hidden / 16` bytes a group), at one SM's store rate."""


@always_inline
def local_dispatch_grid[
    sf_block: Int
](num_tokens: Int, tokens_per_cta: Int) -> Int:
    """Returns the CTAs of a local dispatch of `num_tokens` tokens; those
    past the tokens build the layout and take a share of the tails."""
    var grid = max(ceildiv(num_tokens, tokens_per_cta), 1)
    comptime if sf_block > 0:
        grid = max(grid, _EP_SELF_SCALE_CTAS)
    # A one-chunk step's `total` counts only its own chunk's mask, so no
    # padding CTA may start past that chunk.
    debug_assert(
        num_tokens > _EP_SELF_CHUNK or grid * tokens_per_cta <= _EP_SELF_CHUNK,
        "a one-chunk dispatch's CTAs must stay in its chunk",
    )
    return grid


@always_inline
def _tokens_per_cta(num_tokens: Int, sm_count: Int) -> Int:
    """Returns the tokens a CTA takes, for a grid of about eight CTAs an SM.

    A power of two, so a dispatch CTA's tokens never straddle two chunks.
    """
    var per_cta = 1
    while (
        per_cta < _EP_SELF_MAX_TOKENS_PER_CTA
        and ceildiv(num_tokens, per_cta) > 8 * sm_count
    ):
        per_cta *= 2
    return per_cta


# ===-----------------------------------------------------------------------===#
# Dispatch
# ===-----------------------------------------------------------------------===#


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_EP_SELF_THREADS)
    )
)
@__name(t"ep_local_count_{top_k}_{n_local}")
def _count_kernel[
    topk_ids_layout: TensorLayout,
    top_k: Int,
    n_local: Int,
    *,
    TopkIdsEngine: TensorEngine = DefaultEngine[element_width=1],
](
    topk_ids: TileTensor[
        .int32, topk_ids_layout, ImmUntrackedOrigin, Engine=TopkIdsEngine
    ],
    chunk_counts: MutPointer[Int32, MutUntrackedOrigin],
    my_rank: Int32,
):
    """Counts each chunk's slots per local expert.

    CTA `c` writes `chunk_counts[c * n_local + e]`, the slots of the tokens
    `[c * _EP_SELF_CHUNK, (c + 1) * _EP_SELF_CHUNK)` routed to local expert
    `e`, which `_dispatch_kernel` offsets each chunk's rows by.

    Parameters:
        topk_ids_layout: The layout of the top-k ids.
        top_k: Experts each token is routed to.
        n_local: Experts on this device.
        TopkIdsEngine: Storage engine backing `topk_ids`.

    Args:
        topk_ids: Every token's expert ids.
        chunk_counts: The per-chunk counts this writes.
        my_rank: The rank of this device.
    """
    comptime assert topk_ids.flat_rank == 2
    var hist = unsafe_stack_allocation[n_local, Int32, address_space=.SHARED]()
    var tid = thread_idx.x
    for e in range(tid, n_local, _EP_SELF_THREADS):
        hist[unsafe_offset=e] = 0
    var first_expert = my_rank * Int32(n_local)

    with PDL():
        barrier()
        var t = block_idx.x * _EP_SELF_CHUNK + tid
        if t < Int(topk_ids.dim(0)):
            comptime for k in range(top_k):
                var e = _local_expert(
                    rebind[Int32](topk_ids[t, k]), first_expert, n_local
                )
                if e >= 0:
                    _ = Atomic[Int32, scope=BLOCK_SCOPE].fetch_add[
                        ordering=.relaxed
                    ](hist.unsafe_offset(Int(e)), 1)
        barrier()
        for e in range(tid, n_local, _EP_SELF_THREADS):
            chunk_counts.unsafe_store(
                block_idx.x * n_local + e, hist[unsafe_offset=e]
            )


struct LocalDispatchTables[
    n_local: Int, top_k: Int, fused_shared_expert: Bool, sf_block: Int
](TrivialRegisterPassable):
    """The shared-memory tables one local dispatch CTA builds its rows from.

    Each method is one phase of the dispatch, run by every thread of a CTA of
    `_EP_SELF_THREADS`; the caller meets the CTA between phases. No method
    waits on another thread. `ep_local_dispatch`'s kernel and the dispatch's
    Moka component (`Kernels/test/gpu/Moka/kernels/ep/`) call these phases.
    The kernel writes the scale offsets through its format's
    `pad_expert_offsets` and the component through `scale_offsets`; the
    component's test holds the two bit for bit.

    Parameters:
        n_local: Experts on this device.
        top_k: Experts each token is routed to.
        fused_shared_expert: Whether the shared expert's rows lead the
            output, as its group 0.
        sf_block: The token format's `scale_block_rows`.
    """

    comptime shared_off = 1 if Self.fused_shared_expert else 0
    comptime n_groups = Self.n_local + Self.shared_off
    comptime n_words = _EP_SELF_CHUNK * Self.top_k // 32
    # One spare word per expert puts each expert's words on different banks.
    comptime mask_stride = Self.n_words + 1
    comptime n_slots = _EP_SELF_MAX_TOKENS_PER_CTA * Self.top_k
    # Each group's first block of the format's blocked scale layout, and each
    # slot's row in it; one word each when the format has no such layout.
    comptime sf_groups = Self.n_groups + 1 if Self.sf_block > 0 else 1
    """Also the words of the tail list: its length, then the groups whose
    last scale block their rows do not fill."""
    comptime sf_slots = Self.n_slots if Self.sf_block > 0 else 1

    comptime _mask_words = Self.n_local * Self.mask_stride
    comptime layout_words = (
        Self._mask_words
        + 2 * Self.n_local
        + Self.n_groups
        + 1
        + 2 * Self.sf_groups
        + Self.n_slots
    )
    """The 4-byte words of the tables `place` reads: the masks, counts,
    earlier chunks' counts, group and scale-block starts, the tail list and
    this CTA's slots' experts."""
    comptime slot_words = Self.n_slots + Self.sf_slots
    """The 4-byte words of the tables `place` writes: each slot's row and
    scale row."""

    var mask: _SharedPtr[.uint32]
    var counts: _SharedPtr[.int32]
    var before: _SharedPtr[.int32]
    var group_start: _SharedPtr[.uint32]
    var sf_start: _SharedPtr[.uint32]
    var tails: _SharedPtr[.uint32]
    var mine_experts: _SharedPtr[.int32]
    var slot_row: _SharedPtr[.int32]
    var slot_sf_row: _SharedPtr[.int32]

    @always_inline
    def __init__(
        out self, base: _SharedPtr[.uint32], slots: _SharedPtr[.uint32]
    ):
        """Lays the tables out from `base`, which holds `layout_words`
        words, and `slots`, which holds `slot_words`."""
        var at = 0
        self.mask = base
        at += Self._mask_words
        self.counts = base.unsafe_offset(at).unsafe_bitcast[Int32]()
        at += Self.n_local
        self.before = base.unsafe_offset(at).unsafe_bitcast[Int32]()
        at += Self.n_local
        self.group_start = base.unsafe_offset(at)
        at += Self.n_groups + 1
        self.sf_start = base.unsafe_offset(at)
        at += Self.sf_groups
        self.tails = base.unsafe_offset(at)
        at += Self.sf_groups
        self.mine_experts = base.unsafe_offset(at).unsafe_bitcast[Int32]()
        self.slot_row = slots.unsafe_bitcast[Int32]()
        self.slot_sf_row = slots.unsafe_offset(Self.n_slots).unsafe_bitcast[
            Int32
        ]()

    @always_inline
    def clear(self):
        """Zeros the slot masks and the per-expert counts."""
        var tid = thread_idx.x
        for i in range(tid, Self._mask_words, _EP_SELF_THREADS):
            self.mask[unsafe_offset=i] = 0
        for e in range(tid, Self.n_local, _EP_SELF_THREADS):
            self.counts[unsafe_offset=e] = 0
            self.before[unsafe_offset=e] = 0

    @always_inline
    @staticmethod
    def chunk_ids(
        topk_ids: TileTensor[mut=False, .int32, ...], first_tok: Int
    ) -> Array[Int32, Self.top_k]:
        """Returns the expert ids of this thread's token of the chunk that
        holds `first_tok`; -1 past the step's tokens. A caller issues the
        loads early and hands the ids to `mark`."""
        comptime assert topk_ids.flat_rank == 2
        var ids = Array[Int32, Self.top_k](fill=-1)
        var t = first_tok // _EP_SELF_CHUNK * _EP_SELF_CHUNK + thread_idx.x
        if t < Int(topk_ids.dim(0)):
            comptime for k in range(Self.top_k):
                ids[k] = rebind[Int32](topk_ids[t, k])
        return ids^

    @always_inline
    def mark(
        self,
        ids: Array[Int32, Self.top_k],
        chunk_counts: ImmPointer[Int32, _],
        num_tokens: Int,
        first_tok: Int,
        n_mine: Int,
        first_expert: Int32,
    ):
        """Marks every local slot of this CTA's chunk in its expert's mask,
        keeps the experts of this CTA's own tokens, and adds the earlier
        chunks' counts from `chunk_counts` when the step has more than one.

        `ids` are this thread's token's expert ids (`chunk_ids`).
        """
        comptime assert (
            _EP_SELF_CHUNK * Self.top_k % 32 == 0
        ), "a chunk's slots must fill whole words"
        var tid = thread_idx.x
        var chunk = first_tok // _EP_SELF_CHUNK
        var chunk_start = chunk * _EP_SELF_CHUNK
        var t = chunk_start + tid
        if t < num_tokens:
            comptime for k in range(Self.top_k):
                var e = _local_expert(ids[k], first_expert, Self.n_local)
                if e >= 0:
                    var slot = tid * Self.top_k + k
                    _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                        ordering=.relaxed
                    ](
                        self.mask.unsafe_offset(
                            Int(e) * Self.mask_stride + slot // 32
                        ),
                        UInt32(1) << UInt32(slot % 32),
                    )
                if t >= first_tok and t < first_tok + n_mine:
                    self.mine_experts[
                        unsafe_offset=(t - first_tok) * Self.top_k + k
                    ] = e

        # TODO(KERN-4016): every CTA reads every chunk's counts. Past
        # 8 * _EP_SELF_MAX_TOKENS_PER_CTA * sm_count tokens a step (37,888 on
        # B200) the grid grows with the step as well, so the reads of all CTAs
        # grow with its square. Before a configuration dispatches that many
        # tokens a step, compute each chunk's exclusive prefix and the totals
        # once, right after the count launch.
        var n_chunks = ceildiv(num_tokens, _EP_SELF_CHUNK)
        if n_chunks > 1:
            # All of a batch's loads are in flight before any is summed.
            comptime batch = 8
            var n_entries = n_chunks * Self.n_local
            for base in range(tid, n_entries, _EP_SELF_THREADS * batch):
                var vals = Array[Int32, batch](fill=0)
                comptime for i in range(batch):
                    var w = base + i * _EP_SELF_THREADS
                    if w < n_entries:
                        vals[i] = chunk_counts.unsafe_load(w)
                comptime for i in range(batch):
                    var w = base + i * _EP_SELF_THREADS
                    if w < n_entries and vals[i] != 0:
                        var c, e = divmod(w, Self.n_local)
                        _ = Atomic[Int32, scope=BLOCK_SCOPE].fetch_add[
                            ordering=.relaxed
                        ](self.counts.unsafe_offset(e), vals[i])
                        if c < chunk:
                            _ = Atomic[Int32, scope=BLOCK_SCOPE].fetch_add[
                                ordering=.relaxed
                            ](self.before.unsafe_offset(e), vals[i])

    @always_inline
    def total(self, num_tokens: Int):
        """Counts each expert's slots from its mask, for a one-chunk step.

        It reads only the words the step's slots fill.
        """
        var tid = thread_idx.x
        if num_tokens <= _EP_SELF_CHUNK and tid < Self.n_local:
            var total = UInt32(0)
            var used = ceildiv(num_tokens * Self.top_k, 32)
            for w in range(used):
                total += pop_count(
                    self.mask[unsafe_offset=tid * Self.mask_stride + w]
                )
            self.counts[unsafe_offset=tid] = Int32(total)

    @always_inline
    def starts[m_padding: Int](self, shared_count: UInt32):
        """Warp 0 scans the groups' padded rows into their first rows, and
        their scale blocks into their first blocks, and lists the groups
        whose last block their rows do not fill.

        Parameters:
            m_padding: Rows each group rounds up to.
        """
        if warp_id() != 0:
            return
        var lane = lane_id()
        # One scan a round of 32 groups. With a blocked scale layout it adds
        # the rows (bits 0-31), the scale blocks (32-47) and the tail flags
        # (48-63) at once: a step holds far fewer than 2^23 rows, so the
        # blocks stay below 2^16.
        var run = UInt64(0)
        if lane == 0:
            self.group_start[unsafe_offset=0] = 0
            comptime if Self.sf_block > 0:
                self.sf_start[unsafe_offset=0] = 0
        comptime for base in range(0, Self.n_groups, WARP_SIZE):
            var g = base + lane
            var rows = UInt32(0)
            if g < Self.n_groups:
                comptime if Self.fused_shared_expert:
                    rows = shared_count if g == 0 else UInt32(
                        self.counts[unsafe_offset=g - 1]
                    )
                else:
                    rows = UInt32(self.counts[unsafe_offset=g])
                rows = UInt32(align_up(Int(rows), m_padding))
            var item = UInt64(rows)
            var has = UInt64(0)
            comptime if Self.sf_block > 0:
                has = UInt64(1) if rows % UInt32(Self.sf_block) else 0
                var blocks = UInt64(ceildiv(Int(rows), Self.sf_block))
                item |= (blocks << 32) | (has << 48)
            var end = run + warp.prefix_sum[exclusive=False](item)
            run = warp.shuffle_idx(end, UInt32(WARP_SIZE - 1))
            if g < Self.n_groups:
                self.group_start[unsafe_offset=g + 1] = UInt32(
                    end & 0xFFFF_FFFF
                )
                comptime if Self.sf_block > 0:
                    self.sf_start[unsafe_offset=g + 1] = UInt32(
                        (end >> 32) & 0xFFFF
                    )
                    if has != 0:
                        var at = Int(end >> 48) - 1
                        self.tails[unsafe_offset=1 + at] = UInt32(g)
        comptime if Self.sf_block > 0:
            if lane == 0:
                self.tails[unsafe_offset=0] = UInt32(run >> 48)

    @always_inline
    def publish[
        ExpertLoadT: ExpertLoadSink
    ](
        self,
        row_offsets: TileTensor[mut=True, .uint32, ...],
        expert_ids: TileTensor[mut=True, .int32, ...],
        ep_counters: EPLocalSyncCounters[Self.n_local],
        expert_load: ExpertLoadT,
    ):
        """Writes the grouped matmul's offsets and expert ids, and the row
        ranges both all-reduce-mode combines read; CTA 0 alone calls it."""
        var tid = thread_idx.x
        if tid <= Self.n_groups:
            row_offsets[tid] = self.group_start[unsafe_offset=tid]
        if tid < Self.n_groups:
            expert_ids[tid] = Int32(tid)
        if tid < Self.n_local:
            # The protocol dispatch's Region A: (flag + end, count).
            var count = self.counts[unsafe_offset=tid]
            var region_a = ep_counters.get_dispatch_wait_ptr()
            region_a.unsafe_store(
                2 * tid,
                EP_DATA_READY_FLAG
                + Int32(self.group_start[unsafe_offset=tid + Self.shared_off])
                + count,
            )
            region_a.unsafe_store(2 * tid + 1, count)
            expert_load.add(tid, Self.n_local, UInt32(count))

    @always_inline
    def scale_offsets(self, offsets: MutPointer[UInt32, _]):
        """Writes each group's first scale block less its first row's block,
        which the grouped matmul adds to a row's block; CTA 0 alone calls it.
        """
        comptime assert Self.sf_block > 0
        var g = thread_idx.x
        if g < Self.n_groups:
            offsets[g] = self.sf_start[unsafe_offset=g] - self.group_start[
                unsafe_offset=g
            ] // UInt32(Self.sf_block)

    @always_inline
    def tail_items(self, atoms: Int) -> Int:
        """Returns the tail items, one atom of one listed group each. Every
        warp of the grid takes its share, so the tails' stores spread over
        the grid."""
        comptime assert Self.sf_block > 0
        return Int(self.tails[unsafe_offset=0]) * atoms

    @always_inline
    def tail_item(self, item: Int, atoms: Int) -> Tuple[Int, Int, Int]:
        """Returns the tail rows `[begin, end)` and the atom of tail item
        `item`."""
        var k, atom = divmod(item, atoms)
        var begin, end = self.tail(Int(self.tails[unsafe_offset=1 + k]))
        return (begin, end, atom)

    @always_inline
    def tail(self, g: Int) -> Tuple[Int, Int]:
        """Returns the scale rows `[begin, end)` of group `g`'s last block
        that hold no row; empty when its rows fill the block."""
        comptime assert Self.sf_block > 0
        var rows = Int(self.group_start[unsafe_offset=g + 1]) - Int(
            self.group_start[unsafe_offset=g]
        )
        var first = Int(self.sf_start[unsafe_offset=g]) * Self.sf_block
        return (first + rows, first + align_up(rows, Self.sf_block))

    @always_inline
    def place(self, first_tok: Int, n_mine: Int):
        """Gives each slot of this CTA's tokens its row, and its row in the
        scale blocks; -1 for a slot routed elsewhere."""
        var chunk_start = first_tok // _EP_SELF_CHUNK * _EP_SELF_CHUNK
        for j in range(thread_idx.x, n_mine * Self.top_k, _EP_SELF_THREADS):
            var dst = Int32(-1)
            var e = Int(self.mine_experts[unsafe_offset=j])
            if e >= 0:
                dst = (
                    Int32(self.group_start[unsafe_offset=e + Self.shared_off])
                    + self.before[unsafe_offset=e]
                    + _slots_below(
                        self.mask.unsafe_offset(e * Self.mask_stride),
                        (first_tok - chunk_start) * Self.top_k + j,
                    )
                )
            self.slot_row[unsafe_offset=j] = dst
            comptime if Self.sf_block > 0:
                var g = e + Self.shared_off
                self.slot_sf_row[unsafe_offset=j] = (
                    Int32(self.sf_start[unsafe_offset=g]) * Int32(Self.sf_block)
                    + dst
                    - Int32(self.group_start[unsafe_offset=g])
                ) if e >= 0 else Int32(-1)

    @always_inline
    def token_rows[
        world_size: Int
    ](
        self,
        src_info: TileTensor[mut=True, .int32, ...],
        jt: Int,
        tok: Int,
        num_tokens: Int,
        my_rank: Int32,
    ) -> Tuple[Array[Int, Self.top_k + 1], Array[Int, Self.top_k + 1]]:
        """Returns token `tok`'s rows and their scale rows, the shared
        expert's last, and writes its routed rows' source info.

        `jt` is the token's index among this CTA's.
        """
        var tid = thread_idx.x
        if tid < Self.top_k:
            var dst = self.slot_row[unsafe_offset=jt * Self.top_k + tid]
            if dst >= 0:
                src_info[Int(dst), 0] = Int32(tok)
                src_info[Int(dst), 1] = Int32(tid)
        var rows = Array[Int, Self.top_k + 1](fill=-1)
        var scale_rows = Array[Int, Self.top_k + 1](fill=-1)
        comptime for k in range(Self.top_k):
            rows[k] = Int(self.slot_row[unsafe_offset=jt * Self.top_k + k])
            comptime if Self.sf_block > 0:
                scale_rows[k] = Int(
                    self.slot_sf_row[unsafe_offset=jt * Self.top_k + k]
                )
        comptime if Self.fused_shared_expert:
            var shared_begin, shared_count = ep_shared_expert_rows[world_size](
                UInt32(num_tokens), UInt32(my_rank)
            )
            if UInt32(tok) >= shared_begin and UInt32(tok) < (
                shared_begin + shared_count
            ):
                # The shared expert is group 0, whose blocks come first.
                rows[Self.top_k] = tok - Int(shared_begin)
                scale_rows[Self.top_k] = rows[Self.top_k]
        return (rows^, scale_rows^)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_EP_SELF_THREADS)
    )
)
@__llvm_arg_metadata(token_handler, `nvvm.grid_constant`)
@__name(
    t"ep_local_dispatch_{input_type}_{token_fmt_type.get_type_name()}_{n_local}_{fused_shared_expert}_{world_size}",
)
def _dispatch_kernel[
    input_type: DType,
    input_tokens_layout: TensorLayout,
    topk_ids_layout: TensorLayout,
    row_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    src_info_layout: TensorLayout,
    token_fmt_type: TokenFormat,
    n_local: Int,
    fused_shared_expert: Bool,
    world_size: Int,
    InputScalesFnType: InputScalesFn,
    has_input_scales: Bool,
    ExpertLoadT: ExpertLoadSink = NullExpertLoad,
    *,
    InputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
    TopkIdsEngine: TensorEngine = DefaultEngine[element_width=1],
    RowOffsetsEngine: TensorEngine = DefaultEngine[element_width=1],
    ExpertIdsEngine: TensorEngine = DefaultEngine[element_width=1],
    SrcInfoEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input_tokens: TileTensor[
        input_type,
        input_tokens_layout,
        ImmUntrackedOrigin,
        Engine=InputTokensEngine,
    ],
    topk_ids: TileTensor[
        .int32, topk_ids_layout, ImmUntrackedOrigin, Engine=TopkIdsEngine
    ],
    token_handler: token_fmt_type,
    row_offsets: TileTensor[
        .uint32, row_offsets_layout, MutUntrackedOrigin, Engine=RowOffsetsEngine
    ],
    expert_ids: TileTensor[
        .int32, expert_ids_layout, MutUntrackedOrigin, Engine=ExpertIdsEngine
    ],
    src_info: TileTensor[
        .int32, src_info_layout, MutUntrackedOrigin, Engine=SrcInfoEngine
    ],
    ep_counters: EPLocalSyncCounters[n_local],
    chunk_counts: MutPointer[Int32, MutUntrackedOrigin],
    tokens_per_cta: Int32,
    my_rank: Int32,
    expert_load: ExpertLoadT,
    input_scales_fn: InputScalesFnType,
):
    """Writes this device's routed and shared expert rows in protocol layout.

    CTA `b` takes the tokens `[b * tokens_per_cta, (b + 1) * tokens_per_cta)`.
    It marks every slot of its chunk in a per-expert bitmask and reads each
    expert's slots in the earlier chunks from `chunk_counts` (a step of one
    chunk needs none), so it knows every row it writes without waiting on
    another CTA. The token format writes each token to all of its rows, and
    CTA 0 also writes the grouped matmul's offsets and expert ids, and the
    row ranges that both all-reduce-mode combines read from the counter
    buffer. With a blocked scale layout, CTA 0 also writes each group's
    block offset, and the CTAs zero the groups' last-block tails between
    them.

    Parameters:
        input_type: The type of the input tokens.
        input_tokens_layout: The layout of the input tokens.
        topk_ids_layout: The layout of the top-k ids.
        row_offsets_layout: The layout of the row offsets.
        expert_ids_layout: The layout of the expert ids.
        src_info_layout: The layout of the source info.
        token_fmt_type: The token format, which writes the rows.
        n_local: Experts on this device.
        fused_shared_expert: Whether the shared expert's rows lead the
            output, as its group 0.
        world_size: Devices of the all-reduce, which split the shared
            expert's rows.
        InputScalesFnType: The type of `input_scales_fn`.
        has_input_scales: Whether `input_scales_fn` supplies the input scale.
        ExpertLoadT: Type of `expert_load`.
        InputTokensEngine: Storage engine backing `input_tokens`.
        TopkIdsEngine: Storage engine backing `topk_ids`.
        RowOffsetsEngine: Storage engine backing `row_offsets`.
        ExpertIdsEngine: Storage engine backing `expert_ids`.
        SrcInfoEngine: Storage engine backing `src_info`.

    Args:
        input_tokens: This device's tokens, `[num_tokens, hid_dim]`.
        topk_ids: Every token's expert ids.
        token_handler: Writes a token to its rows.
        row_offsets: The grouped matmul's row offsets.
        expert_ids: The grouped matmul's expert ids.
        src_info: Each routed row's `(token, top-k slot)`.
        ep_counters: The dispatch's counter buffer.
        chunk_counts: `_count_kernel`'s counts, read only when the step has
            more than one chunk.
        tokens_per_cta: Tokens per CTA; divides `_EP_SELF_CHUNK`.
        my_rank: The rank of this device.
        expert_load: Receives each local expert's routed-token count.
        input_scales_fn: The input scale, read when `has_input_scales`.
    """
    comptime hid_dim = token_fmt_type.hid_dim
    comptime top_k = token_fmt_type.top_k
    comptime Tables = LocalDispatchTables[
        n_local, top_k, fused_shared_expert, token_fmt_type.scale_block_rows
    ]
    comptime width = simd_width_of[input_type]()
    comptime n_vecs = hid_dim // width
    comptime per_thread = ceildiv(n_vecs, _EP_SELF_THREADS)
    comptime assert (
        Tables.n_groups < _EP_SELF_THREADS
    ), "one thread per expert group"
    comptime assert input_tokens.flat_rank == 2
    comptime assert row_offsets.flat_rank == 1
    comptime assert expert_ids.flat_rank == 1
    comptime assert src_info.flat_rank == 2

    var tables = Tables(
        unsafe_stack_allocation[
            Tables.layout_words, UInt32, address_space=.SHARED
        ](),
        unsafe_stack_allocation[
            Tables.slot_words, UInt32, address_space=.SHARED
        ](),
    )
    tables.clear()

    var num_tokens = Int(input_tokens.dim(0))
    var first_tok = block_idx.x * Int(tokens_per_cta)
    var n_mine = max(min(Int(tokens_per_cta), num_tokens - first_tok), 0)
    var shared_count = ep_shared_expert_rows[world_size](
        UInt32(num_tokens), UInt32(my_rank)
    )[1]

    with PDL():
        # The first token's row and the chunk's expert ids load together
        # while the layout is built, and each token's processing has the next
        # one's row in flight.
        var ids = Tables.chunk_ids(topk_ids, first_tok)
        var row = Array[SIMD[input_type, width], per_thread](fill=0)
        if n_mine > 0:
            row = _load_row[input_type, width, n_vecs, per_thread](
                input_tokens.ptr_at_offset((first_tok, Idx[0]))
            )

        barrier()
        tables.mark(
            ids,
            chunk_counts,
            num_tokens,
            first_tok,
            n_mine,
            my_rank * Int32(n_local),
        )
        barrier()
        tables.total(num_tokens)
        barrier()
        tables.starts[token_fmt_type.expert_m_padding](shared_count)
        barrier()

        if block_idx.x == 0:
            tables.publish(row_offsets, expert_ids, ep_counters, expert_load)
            comptime if Tables.sf_block > 0:
                token_handler.pad_expert_offsets[Tables.n_groups](
                    tables.group_start
                )

        # Each group's last scale block past its rows reads as zero, as the
        # protocol dispatch's 128-row scale boxes leave it.
        comptime if Tables.sf_block > 0:
            comptime atoms = token_fmt_type.scale_atoms
            comptime warps = _EP_SELF_THREADS // WARP_SIZE
            for item in range(
                block_idx.x * warps + warp_id(),
                tables.tail_items(atoms),
                grid_dim.x * warps,
            ):
                var begin, end, atom = tables.tail_item(item, atoms)
                token_handler.zero_scale_atom(begin, end, atom)

        tables.place(first_tok, n_mine)
        barrier()

        var input_scale = Float32(1.0)
        comptime if has_input_scales:
            input_scale = input_scales_fn[.float32](0)

        for jt in range(n_mine):
            var tok = first_tok + jt
            var cur = row.copy()
            if jt + 1 < n_mine:
                row = _load_row[input_type, width, n_vecs, per_thread](
                    input_tokens.ptr_at_offset((tok + 1, Idx[0]))
                )
            var rows, scale_rows = tables.token_rows[world_size](
                src_info, jt, tok, num_tokens, my_rank
            )
            token_handler.copy_token_to_output_rows[_EP_SELF_THREADS](
                cur, rows, scale_rows, input_scale
            )


@always_inline
def ep_local_dispatch[
    token_fmt_type: TokenFormat,
    RowOffsetsEngine: TensorEngine,
    ExpertIdsEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    AtomicCountersEngine: TensorEngine,
    InputTokensEngine: TensorEngine,
    TopkIdsEngine: TensorEngine,
    SendPtrsEngine: TensorEngine,
    input_type: DType,
    InputScalesFnType: InputScalesFn,
    //,
    n_local: Int,
    fused_shared_expert: Bool,
    world_size: Int,
    has_input_scales: Bool,
    ExpertLoadT: ExpertLoadSink = NullExpertLoad,
](
    token_handler: token_fmt_type,
    row_offsets: TileTensor[.uint32, Engine=RowOffsetsEngine, ...],
    expert_ids: TileTensor[.int32, Engine=ExpertIdsEngine, ...],
    src_info: TileTensor[.int32, Engine=SrcInfoEngine, ...],
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[
        mut=False, input_type, Engine=InputTokensEngine, ...
    ],
    topk_ids: TileTensor[mut=False, .int32, Engine=TopkIdsEngine, ...],
    send_ptrs: TileTensor[.uint64, Engine=SendPtrsEngine, ...],
    context: DeviceContext,
    input_scales_fn: InputScalesFnType,
    expert_load: ExpertLoadT = NullExpertLoad(),
) raises:
    """Runs the all-reduce-mode dispatch of this device's tokens on it alone.

    Writes what `ep_fused_dispatch_kernel_api` writes under `skip_a2a`: the
    rows, row offsets, expert ids, source info and the counter buffer's row
    ranges. One launch for a step of up to 256 tokens, two for a larger one,
    whose per-chunk counts this device's send buffer holds between them.

    Parameters:
        token_fmt_type: The token format, which writes the rows (inferred).
        RowOffsetsEngine: Storage engine backing `row_offsets` (inferred).
        ExpertIdsEngine: Storage engine backing `expert_ids` (inferred).
        SrcInfoEngine: Storage engine backing `src_info` (inferred).
        AtomicCountersEngine: Storage engine backing `atomic_counters`
            (inferred).
        InputTokensEngine: Storage engine backing `input_tokens` (inferred).
        TopkIdsEngine: Storage engine backing `topk_ids` (inferred).
        SendPtrsEngine: Storage engine backing `send_ptrs` (inferred).
        input_type: The type of the input tokens (inferred).
        InputScalesFnType: The type of `input_scales_fn` (inferred).
        n_local: Experts on this device.
        fused_shared_expert: Whether the shared expert's rows lead the
            output.
        world_size: Devices of the all-reduce.
        has_input_scales: Whether `input_scales_fn` supplies the input scale.
        ExpertLoadT: Type of `expert_load`.

    Args:
        token_handler: The token format's handle on the output rows and
            their scales.
        row_offsets: The grouped matmul's row offsets.
        expert_ids: The grouped matmul's expert ids.
        src_info: Each routed row's `(token, top-k slot)`.
        atomic_counters: The dispatch's counter buffer.
        input_tokens: This device's tokens.
        topk_ids: Every token's expert ids.
        send_ptrs: The dispatch send buffers.
        context: The device context.
        input_scales_fn: The input scale, read on the device when
            `has_input_scales`.
        expert_load: Receives each local expert's routed-token count.

    Raises:
        Error: If a kernel fails to compile or launch.

    Constraints:
        The target must be a GPU and the token format must write local rows
        (`TokenFormat.local_rows`), under its own constraints on the hidden
        size. The shared expert and the local experts take fewer than 256
        groups, one thread each. `input_tokens` and `topk_ids` must have the
        format's hidden size and top-k as static widths.
    """
    comptime assert context.target.is_gpu(), "EP is only supported on GPU."
    comptime assert (
        token_fmt_type.local_rows
    ), "the token format writes no local rows"
    comptime assert input_tokens.static_shape[1] == token_fmt_type.hid_dim
    comptime assert topk_ids.static_shape[1] == token_fmt_type.top_k

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.device().id)
    var my_rank = Int32(gpu_id)
    comptime sm_count = gpu_ctx.default_device_info.sm_count

    var num_tokens = Int(input_tokens.dim(0))
    var n_chunks = ceildiv(num_tokens, _EP_SELF_CHUNK)
    # The all-reduce mode sends nothing, so the send buffer holds the
    # per-chunk expert counts.
    var chunk_counts = MutPointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=Int(send_ptrs[gpu_id])
    )

    if n_chunks > 1:
        comptime count = _count_kernel[
            topk_ids.LayoutType,
            token_fmt_type.top_k,
            n_local,
            TopkIdsEngine=TopkIdsEngine,
        ]
        gpu_ctx.compile_function[count]()._call_with_pack_checked(
            gpu_ctx,
            topk_ids,
            chunk_counts,
            my_rank,
            grid_dim=n_chunks,
            block_dim=_EP_SELF_THREADS,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

    var tokens_per_cta = _tokens_per_cta(num_tokens, sm_count)
    comptime dispatch = _dispatch_kernel[
        input_type,
        input_tokens.LayoutType,
        topk_ids.LayoutType,
        row_offsets.LayoutType,
        expert_ids.LayoutType,
        src_info.LayoutType,
        token_fmt_type,
        n_local,
        fused_shared_expert,
        world_size,
        InputScalesFnType,
        has_input_scales,
        ExpertLoadT,
        InputTokensEngine=InputTokensEngine,
        TopkIdsEngine=TopkIdsEngine,
        RowOffsetsEngine=RowOffsetsEngine,
        ExpertIdsEngine=ExpertIdsEngine,
        SrcInfoEngine=SrcInfoEngine,
    ]
    gpu_ctx.compile_function[dispatch]()._call_with_pack_checked(
        gpu_ctx,
        input_tokens,
        topk_ids,
        token_handler,
        row_offsets,
        expert_ids,
        src_info,
        EPLocalSyncCounters[n_local](
            MutPointer[Int32, MutUntrackedOrigin](
                unsafe_from_address=Int(atomic_counters.ptr)
            )
        ),
        chunk_counts,
        Int32(tokens_per_cta),
        my_rank,
        expert_load,
        host_args=(input_scales_fn,),
        grid_dim=local_dispatch_grid[token_fmt_type.scale_block_rows](
            num_tokens, tokens_per_cta
        ),
        block_dim=_EP_SELF_THREADS,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


# ===-----------------------------------------------------------------------===#
# Combine
# ===-----------------------------------------------------------------------===#


@always_inline
def _row_ranges[
    n_local: Int
](
    ep_counters: EPLocalSyncCounters[n_local],
    row_start: _SharedPtr[.int32],
    row_prefix: _SharedPtr[.int32],
):
    """Reads each local expert's first row and running row count.

    The dispatch's Region A holds them; `row_start[e]` gets the first row of
    expert `e`, and `row_prefix[e]` the rows of the experts before it. Every
    thread of the CTA must call it.
    """
    comptime assert n_local <= _EP_SELF_THREADS, "one thread per local expert"
    var tid = thread_idx.x
    var count = Int32(0)
    if tid < n_local:
        var end_count = ep_counters.get_combine_async_ptr().unsafe_load[
            width=2, alignment=align_of[SIMD[.int32, 2]]()
        ](2 * tid)
        count = end_count[1]
        row_start[unsafe_offset=tid] = end_count[0] - EP_DATA_READY_FLAG - count
    var prefix = block_prefix_sum[n_local](count)
    if tid < n_local:
        row_prefix[unsafe_offset=tid + 1] = prefix
    if tid == 0:
        row_prefix[unsafe_offset=0] = 0
    barrier()


@always_inline
def _row_of[
    n_local: Int
](j: Int, row_start: _SharedPtr[.int32], row_prefix: _SharedPtr[.int32]) -> Int:
    """Returns the row of the `j`-th routed row across the local experts."""
    var lo = 0
    var hi = n_local
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if Int(row_prefix[unsafe_offset=mid]) <= j:
            lo = mid
        else:
            hi = mid
    return (
        Int(row_start[unsafe_offset=lo]) + j - Int(row_prefix[unsafe_offset=lo])
    )


@always_inline
def _index_slot_rows[
    SrcInfoEngine: TensorEngine,
    StoreSlotFn: def(Int, Int, Int) -> None,
    //,
    n_local: Int,
    top_k: Int,
    batch: Int,
](
    src_info: TileTensor[mut=False, .int32, Engine=SrcInfoEngine, ...],
    row_start: _SharedPtr[.int32],
    row_prefix: _SharedPtr[.int32],
    first: Int,
    stride: Int,
    tok_begin: Int,
    tok_end: Int,
    store_slot: StoreSlotFn,
):
    """Hands each routed row of the tokens in a range to `store_slot`.

    The calling thread takes the routed rows `first, first + stride, ...`,
    `batch` of them in flight at once, and calls `store_slot(token, k, row)`
    for each one whose `(token, top-k slot)` has a token in
    `[tok_begin, tok_end)`.
    """
    var n_rows = Int(row_prefix[unsafe_offset=n_local])
    for base in range(first, n_rows, stride * batch):
        var rows = Array[Int, batch](fill=0)
        var info = Array[SIMD[DType.int32, 2], batch](fill=-1)
        comptime for i in range(batch):
            var j = base + i * stride
            if j < n_rows:
                rows[i] = _row_of[n_local](j, row_start, row_prefix)
                info[i] = src_info.load[width=2]((rows[i], Idx[0]))
        comptime for i in range(batch):
            var tok = Int(info[i][0])
            var k = Int(info[i][1])
            if tok >= tok_begin and tok < tok_end and k >= 0 and k < top_k:
                store_slot(tok, k, rows[i])


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_EP_SELF_THREADS)
    )
)
@__name(t"ep_local_combine_index_{top_k}_{n_local}")
def _combine_index_kernel[
    src_info_layout: TensorLayout,
    top_k: Int,
    n_local: Int,
    *,
    SrcInfoEngine: TensorEngine = DefaultEngine[element_width=1],
](
    src_info: TileTensor[
        .int32, src_info_layout, ImmUntrackedOrigin, Engine=SrcInfoEngine
    ],
    ep_counters: EPLocalSyncCounters[n_local],
    slot_rows: MutPointer[Int32, MutUntrackedOrigin],
    num_tokens: Int32,
):
    """Stores each routed row's index at its `(token, top-k slot)`.

    `slot_rows[token * top_k + k]` gets the row of every slot routed here;
    the other entries keep whatever they held.

    Parameters:
        src_info_layout: The layout of the source info.
        top_k: Experts each token is routed to.
        n_local: Experts on this device.
        SrcInfoEngine: Storage engine backing `src_info`.

    Args:
        src_info: Each routed row's `(token, top-k slot)`.
        ep_counters: The counter buffer the dispatch wrote the rows to.
        slot_rows: The `[num_tokens, top_k]` row index this writes.
        num_tokens: Tokens on this device.
    """
    comptime assert src_info.flat_rank == 2
    var row_start = unsafe_stack_allocation[
        n_local, Int32, address_space=.SHARED
    ]()
    var row_prefix = unsafe_stack_allocation[
        n_local + 1, Int32, address_space=.SHARED
    ]()

    @always_inline
    def store_slot(tok: Int, k: Int, row: Int) {imm} -> None:
        slot_rows.unsafe_store(tok * top_k + k, Int32(row))

    with PDL():
        _row_ranges[n_local](ep_counters, row_start, row_prefix)
        _index_slot_rows[n_local, top_k, 1](
            src_info,
            row_start,
            row_prefix,
            block_idx.x * _EP_SELF_THREADS + thread_idx.x,
            grid_dim.x * _EP_SELF_THREADS,
            0,
            Int(num_tokens),
            store_slot,
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_EP_SELF_THREADS)
    )
)
@__name(
    t"ep_local_combine_{dtype}_{top_k}_{n_local}_{fused_shared_expert}_{world_size}_{indexed}",
)
def _combine_kernel[
    dtype: DType,
    input_tokens_layout: TensorLayout,
    src_info_layout: TensorLayout,
    top_k: Int,
    n_local: Int,
    fused_shared_expert: Bool,
    world_size: Int,
    indexed: Bool,
    StoreFnType: EPCombineStoreFn,
    RouterWeightsFnType: RouterWeightsFn,
    *,
    InputTokensEngine: TensorEngine = DefaultEngine[element_width=1],
    SrcInfoEngine: TensorEngine = DefaultEngine[element_width=1],
](
    input_tokens: TileTensor[
        dtype,
        input_tokens_layout,
        ImmUntrackedOrigin,
        Engine=InputTokensEngine,
    ],
    src_info: TileTensor[
        .int32, src_info_layout, ImmUntrackedOrigin, Engine=SrcInfoEngine
    ],
    ep_counters: EPLocalSyncCounters[n_local],
    topk_ids: ImmPointer[Int32, ImmUntrackedOrigin],
    slot_rows: MutPointer[Int32, MutUntrackedOrigin],
    num_tokens: Int32,
    tokens_per_cta: Int32,
    my_rank: Int32,
    store_fn: StoreFnType,
    router_weights_fn: RouterWeightsFnType,
):
    """Sums each token's routed rows by weight and adds its shared expert row.

    CTA `b` takes the tokens `[b * tokens_per_cta, (b + 1) * tokens_per_cta)`,
    one warp per 256-element chunk of a token. It finds the rows of its slots
    in the source info of every routed row, or reads them from `slot_rows`
    when `indexed`. The arithmetic is the protocol combine's under
    `skip_a2a`, routed rows in top-k order, so the outputs match it bit for
    bit.

    Parameters:
        dtype: The type of the expert outputs and the output.
        input_tokens_layout: The layout of the expert outputs.
        src_info_layout: The layout of the source info.
        top_k: Experts each token is routed to.
        n_local: Experts on this device.
        fused_shared_expert: Whether the shared expert's rows lead
            `input_tokens`.
        world_size: Devices of the all-reduce, which split the shared
            expert's rows.
        indexed: Whether `_combine_index_kernel` wrote `slot_rows`.
        StoreFnType: The type of `store_fn`.
        RouterWeightsFnType: The type of `router_weights_fn`.
        InputTokensEngine: Storage engine backing `input_tokens`.
        SrcInfoEngine: Storage engine backing `src_info`.

    Args:
        input_tokens: The expert outputs, one row per dispatched row.
        src_info: Each routed row's `(token, top-k slot)`.
        ep_counters: The counter buffer the dispatch wrote the rows to.
        topk_ids: Every token's expert ids.
        slot_rows: Each slot's row, when `indexed`.
        num_tokens: Tokens on this device.
        tokens_per_cta: Tokens per CTA, at most
            `_EP_SELF_MAX_TOKENS_PER_CTA`.
        my_rank: The rank of this device.
        store_fn: Stores one output vector at `(token, element)`.
        router_weights_fn: The router weights.
    """
    comptime hid_dim = input_tokens.static_shape[1]
    comptime width = simd_width_of[dtype]()
    comptime chunk_elems = WARP_SIZE * width
    comptime n_chunks = hid_dim // chunk_elems
    comptime n_warps = _EP_SELF_THREADS // WARP_SIZE
    # A small step takes two chunks a warp, so one wait covers both; a
    # large one takes one, which halves the registers and doubles the CTAs
    # an SM holds.
    comptime unit_chunks = 2 if not indexed and n_chunks % 2 == 0 else 1
    comptime units_per_token = n_chunks // unit_chunks
    comptime store_align = align_of[
        SIMD[.uint8, simd_width_of[DType.uint8]()]
    ]() // size_of[dtype]()
    comptime assert (
        hid_dim % chunk_elems == 0
    ), "hid_dim must be divisible by a warp's elements"
    comptime assert top_k <= WARP_SIZE
    comptime assert input_tokens.flat_rank == 2
    comptime assert src_info.flat_rank == 2

    var row_start = unsafe_stack_allocation[
        n_local, Int32, address_space=.SHARED
    ]()
    var row_prefix = unsafe_stack_allocation[
        n_local + 1, Int32, address_space=.SHARED
    ]()
    var mine = unsafe_stack_allocation[
        _EP_SELF_MAX_TOKENS_PER_CTA * top_k, Int32, address_space=.SHARED
    ]()

    var tid = thread_idx.x
    var lane = lane_id()
    var first_tok = block_idx.x * Int(tokens_per_cta)
    var end_tok = min(first_tok + Int(tokens_per_cta), Int(num_tokens))
    var n_rows_max = Int(input_tokens.dim(0))
    comptime if not indexed:
        for i in range(
            tid, _EP_SELF_MAX_TOKENS_PER_CTA * top_k, _EP_SELF_THREADS
        ):
            mine[unsafe_offset=i] = -1

    var shared_begin, shared_count = ep_shared_expert_rows[world_size](
        UInt32(num_tokens), UInt32(my_rank)
    )
    var n_units = (end_tok - first_tok) * units_per_token

    @always_inline
    def routing(tok: Int) {imm} -> Tuple[Int32, Float32]:
        # Lane k holds the token's k-th expert and weight.
        if lane < top_k:
            return (
                topk_ids.unsafe_load(tok * top_k + lane),
                router_weights_fn[1](tok, lane)[0],
            )
        return (Int32(-1), Float32(0))

    @always_inline
    def store_mine(tok: Int, k: Int, row: Int) {imm} -> None:
        mine[unsafe_offset=(tok - first_tok) * top_k + k] = Int32(row)

    with PDL():
        # The first token's routing loads with the row ranges.
        var routed_jt = warp_id() // units_per_token
        var expert = Int32(-1)
        var weight = Float32(0)
        if warp_id() < n_units:
            expert, weight = routing(first_tok + routed_jt)

        comptime if not indexed:
            _row_ranges[n_local](ep_counters, row_start, row_prefix)
            _index_slot_rows[n_local, top_k, 4](
                src_info,
                row_start,
                row_prefix,
                tid,
                _EP_SELF_THREADS,
                first_tok,
                end_tok,
                store_mine,
            )
            barrier()

        var row_jt = -1
        var row = Int32(-1)
        for unit in range(warp_id(), n_units, n_warps):
            var jt, group = divmod(unit, units_per_token)
            var tok = first_tok + jt
            if jt != row_jt:
                if jt != routed_jt:
                    routed_jt = jt
                    expert, weight = routing(tok)
                row_jt = jt
                row = -1
                if ufloordiv(Int(expert), n_local) == Int(my_rank):
                    comptime if indexed:
                        row = slot_rows.unsafe_load(tok * top_k + lane)
                    else:
                        row = mine[unsafe_offset=jt * top_k + lane]
                    if Int(row) >= n_rows_max:
                        row = -1

            var rows = Array[Int32, top_k](fill=-1)
            var weights = Array[Float32, top_k](fill=0)
            comptime for k in range(top_k):
                rows[k] = warp.shuffle_idx(row, UInt32(k))
                weights[k] = warp.shuffle_idx(weight, UInt32(k))

            # Every chunk's loads are in flight before any is summed.
            var outs = Array[SIMD[dtype, width], unit_chunks * top_k](fill=0)
            var shared_vals = Array[SIMD[dtype, width], unit_chunks](fill=0)
            comptime for c in range(unit_chunks):
                var elem = (
                    group * unit_chunks + c
                ) * chunk_elems + lane * width
                comptime for k in range(top_k):
                    if rows[k] >= 0:
                        outs[c * top_k + k] = input_tokens.load[width=width](
                            (Int(rows[k]), elem)
                        )
                comptime if fused_shared_expert:
                    if UInt32(tok) >= shared_begin and UInt32(tok) < (
                        shared_begin + shared_count
                    ):
                        shared_vals[c] = input_tokens.load[width=width](
                            (tok - Int(shared_begin), elem)
                        ).cast[dtype]()

            comptime for c in range(unit_chunks):
                var elem = (
                    group * unit_chunks + c
                ) * chunk_elems + lane * width
                # The protocol combine's sum over the local slots in top-k
                # order. Its `accum += w * x` compiles to one FMA a term, so
                # each term here is an explicit FMA rather than whatever
                # contraction this kernel's code would get.
                var accum = SIMD[.float32, width](0)
                comptime for k in range(top_k):
                    if rows[k] >= 0:
                        accum = SIMD[.float32, width](weights[k]).fma(
                            outs[c * top_k + k].cast[.float32](), accum
                        )
                comptime if fused_shared_expert:
                    store_fn[alignment=store_align](
                        Coord(tok, elem), accum.cast[dtype]() + shared_vals[c]
                    )
                else:
                    store_fn[alignment=store_align](
                        Coord(tok, elem), accum.cast[dtype]()
                    )


@always_inline
def ep_local_combine[
    InputTokensEngine: TensorEngine,
    SrcInfoEngine: TensorEngine,
    AtomicCountersEngine: TensorEngine,
    RecvPtrsEngine: TensorEngine,
    StoreFnType: EPCombineStoreFn,
    RouterWeightsFnType: RouterWeightsFn,
    dtype: DType,
    //,
    top_k: Int,
    n_local: Int,
    fused_shared_expert: Bool,
    world_size: Int,
](
    num_tokens: Int,
    atomic_counters: TileTensor[.int32, Engine=AtomicCountersEngine, ...],
    input_tokens: TileTensor[mut=False, dtype, Engine=InputTokensEngine, ...],
    src_info: TileTensor[mut=False, .int32, Engine=SrcInfoEngine, ...],
    recv_ptrs: TileTensor[.uint64, Engine=RecvPtrsEngine, ...],
    context: DeviceContext,
    store_fn: StoreFnType,
    router_weights_fn: RouterWeightsFnType,
    topk_ids: ImmPointer[Int32, ImmUntrackedOrigin],
) raises:
    """Runs the all-reduce-mode combine of this device's tokens on it alone.

    Stores what `ep_fused_combine_kernel_api` stores under `skip_a2a`, for the
    rows of either all-reduce-mode dispatch: it finds them through the source
    info and the counter buffer's row ranges. One launch for a step of up to
    256 tokens; a larger step first indexes every row into this device's
    combine receive buffer.

    Parameters:
        InputTokensEngine: Storage engine backing `input_tokens` (inferred).
        SrcInfoEngine: Storage engine backing `src_info` (inferred).
        AtomicCountersEngine: Storage engine backing `atomic_counters`
            (inferred).
        RecvPtrsEngine: Storage engine backing `recv_ptrs` (inferred).
        StoreFnType: The type of `store_fn` (inferred).
        RouterWeightsFnType: The type of `router_weights_fn` (inferred).
        dtype: The type of the expert outputs and the output (inferred).
        top_k: Experts each token is routed to.
        n_local: Experts on this device.
        fused_shared_expert: Whether the shared expert's rows lead
            `input_tokens`.
        world_size: Devices of the all-reduce.

    Args:
        num_tokens: Tokens on this device.
        atomic_counters: The counter buffer the dispatch wrote.
        input_tokens: The expert outputs, one row per dispatched row.
        src_info: Each routed row's `(token, top-k slot)`.
        recv_ptrs: The combine receive buffers.
        context: The device context.
        store_fn: Stores one output vector at `(token, element)`. It runs on
            the device, so it must copy what it captures.
        router_weights_fn: The router weights. It runs on the device, so it
            must copy what it captures.
        topk_ids: Every token's expert ids.

    Raises:
        Error: If a kernel fails to compile or launch.

    Constraints:
        The target must be a GPU. Each warp must hold whole vectors of a
        row: the hidden size is a multiple of 32 SIMD vectors of `dtype`.
        `top_k` is at most 32, one lane each, and `n_local` at most 256, one
        thread each.
    """
    comptime assert context.target.is_gpu(), "EP is only supported on GPU."
    if num_tokens == 0:
        return

    var gpu_ctx = context
    var gpu_id = Int(gpu_ctx.device().id)
    var my_rank = Int32(gpu_id)
    comptime sm_count = gpu_ctx.default_device_info.sm_count
    var ep_counters = EPLocalSyncCounters[n_local](
        MutPointer[Int32, MutUntrackedOrigin](
            unsafe_from_address=Int(atomic_counters.ptr)
        )
    )
    var slot_rows = MutPointer[Int32, MutUntrackedOrigin](
        unsafe_from_address=Int(recv_ptrs[gpu_id])
    )
    var tokens_per_cta = _tokens_per_cta(num_tokens, sm_count)
    var grid = ceildiv(num_tokens, tokens_per_cta)

    @always_inline
    def launch[indexed: Bool]() raises {imm}:
        comptime combine = _combine_kernel[
            dtype,
            input_tokens.LayoutType,
            src_info.LayoutType,
            top_k,
            n_local,
            fused_shared_expert,
            world_size,
            indexed,
            StoreFnType,
            RouterWeightsFnType,
            InputTokensEngine=InputTokensEngine,
            SrcInfoEngine=SrcInfoEngine,
        ]
        gpu_ctx.compile_function[combine]()._call_with_pack_checked(
            gpu_ctx,
            input_tokens,
            src_info,
            ep_counters,
            topk_ids,
            slot_rows,
            Int32(num_tokens),
            Int32(tokens_per_cta),
            my_rank,
            host_args=(store_fn, router_weights_fn),
            grid_dim=grid,
            block_dim=_EP_SELF_THREADS,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

    if num_tokens <= _EP_SELF_COMBINE_SCAN_MAX_TOKENS:
        launch[False]()
        return

    comptime index = _combine_index_kernel[
        src_info.LayoutType, top_k, n_local, SrcInfoEngine=SrcInfoEngine
    ]
    gpu_ctx.compile_function[index]()._call_with_pack_checked(
        gpu_ctx,
        src_info,
        ep_counters,
        slot_rows,
        Int32(num_tokens),
        grid_dim=ceildiv(num_tokens * top_k, _EP_SELF_THREADS),
        block_dim=_EP_SELF_THREADS,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
    launch[True]()
