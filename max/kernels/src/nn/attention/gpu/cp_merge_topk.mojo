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
"""Merges a decode context-parallel group's local indexer top-k selections.

Under decode context parallelism (DCP) each rank of a group selects the top-k
of its own key slots (`mla_indexer_ragged_float8_paged_cp`), and the group
all-gathers the selections' scores, rank-major: rank `r`'s `k` scores are
columns `[r * k, (r + 1) * k)` of a `[T, dcp * k]` matrix. The group's top-k
is the top-k of those `dcp * k` candidates, exactly up to ties, because a
token in the group's top-k is in its owner's local top-k. Every rank computes
the same global choice from the same matrix, then keeps the rows of its own
candidates that made it.

Per row, a two-pass radix select over the bf16 scores' order-preserving 16-bit
keys finds the k-th largest score `t` and how many candidates scoring exactly
`t` the top-k takes; those ties go to the lowest columns, so the choice is
deterministic and agreed on by every rank without communication.

Over an NVFP4 latent cache each selection carries a second row, its row in
the cache's scale pool, which the indexer writes `top_k` columns after the
value row (`PagedKVScaleRemap`). With `halves = 2` a row of `local_rows` and
`out_rows` is `2 * top_k` wide, and the merge moves each kept selection's
scale row with its value row, so both halves come out compacted the same way.
"""

from std.atomic import Atomic, Ordering
from std.memory import bitcast, unsafe_stack_allocation
from std.utils.static_tuple import StaticTuple

from layout import TensorLayout
from layout.tile_tensor import ImmTileTensor, MutTileTensor

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.sync import barrier
from shmem.ep_comm import BLOCK_SCOPE

from nn.attention.gpu.sparse_index_fp8_sm100_prefill import CPRankPtr


# One CTA per token row: a decode step's few rows each still get a full CTA's
# loads in flight, and an 8192-row prefill chunk fills the machine many times.
comptime _MERGE_THREADS = 512

# Radix digit width: two 8-bit passes cover a bf16 score's 16-bit key.
comptime _DIGIT_BITS = 8
comptime _BINS = 1 << _DIGIT_BITS

# The candidates' keys live in shared memory for all passes; this caps them
# at `dcp * k <= 16384` (a group of 8 at the indexer's k of 2048).
comptime _MAX_CANDIDATES = 16384

# bf16 `-inf`: a padding slot past a rank's local candidates.
comptime _NEG_INF_BITS = UInt16(0xFF80)


@inline(.always)
def _score_key(score: BFloat16) -> UInt16:
    """Maps a bf16 score to a 16-bit key with the same order; `0` is padding.

    Positive scores set the sign bit and negative ones flip every bit, so
    keys compare as unsigned integers in score order. `-inf` maps to `0`,
    below every real score's key (only a negative NaN would also reach it).
    """
    var bits = bitcast[DType.uint16](score)
    if bits == _NEG_INF_BITS:
        return 0
    if bits & 0x8000 != 0:
        return ~bits
    return bits | 0x8000


@inline(.always)
def _select_digit[
    origin: MutOrigin, //
](
    hist: MutPointer[UInt32, origin, address_space=.SHARED],
    need: UInt32,
    ctl: MutPointer[UInt32, origin, address_space=.SHARED],
):
    """Finds the digit holding the `need`-th largest key; warp 0 only.

    Writes the digit to `ctl[0]` and how many of the keys with that digit
    the selection still needs to `ctl[1]`. Requires `need >= 1` and at least
    `need` keys in `hist`.
    """
    comptime PER_LANE = _BINS // WARP_SIZE
    var lane = Int(lane_id())
    # Lane `l` owns the `PER_LANE` bins just below `_BINS - l * PER_LANE`, so
    # an inclusive prefix over lanes counts the keys at or above a lane's
    # lowest bin.
    var top = _BINS - 1 - lane * PER_LANE
    var lane_sum = UInt32(0)
    comptime for i in range(PER_LANE):
        lane_sum += hist[top - i]
    var incl = warp.prefix_sum(lane_sum)
    var above = incl - lane_sum
    if above < need and incl >= need:
        comptime for i in range(PER_LANE):
            var count = hist[top - i]
            if above < need and above + count >= need:
                ctl[0] = UInt32(top - i)
                ctl[1] = need - above
            above += count


@__name(t"cp_merge_topk_{dcp}x{top_k}x{halves}")
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_MERGE_THREADS))
)
def _cp_merge_topk_kernel[
    dcp: Int, top_k: Int, halves: Int
](
    local_rows: ImmPointer[Int32, ImmUntrackedOrigin],
    scores: ImmPointer[BFloat16, ImmUntrackedOrigin],
    cp_rank: CPRankPtr,
    out_rows: MutPointer[Int32, MutUntrackedOrigin],
    out_lengths: MutPointer[Int32, MutUntrackedOrigin],
):
    """Merges one token row; see `cp_merge_topk`."""
    comptime N = dcp * top_k
    comptime NT = _MERGE_THREADS
    comptime MASK = UInt16(_BINS - 1)
    # Row stride of `local_rows` and `out_rows`: the value rows, then (with
    # `halves == 2`) the matching scale rows.
    comptime W = halves * top_k
    var tid = thread_idx.x
    var row = block_idx.x

    var keys = unsafe_stack_allocation[N, UInt16, address_space=.SHARED]()
    var hist = unsafe_stack_allocation[_BINS, UInt32, address_space=.SHARED]()
    # [0] digit, [1] keys still needed at that digit.
    var ctl = unsafe_stack_allocation[2, UInt32, address_space=.SHARED]()

    with PDL():
        var rank = Int(cp_rank[0])
        # The rank is read from the device; out of range, `own_begin` below
        # would index past the `dcp * top_k` staged keys.
        debug_assert(
            rank >= 0 and rank < dcp,
            "cp_merge_topk: cp_rank outside [0, dcp)",
        )
        var row_scores = scores.unsafe_offset(row * N)

        # Pass 1: stage the keys, and histogram their high digit.
        for i in range(tid, _BINS, NT):
            hist[i] = 0
        barrier()
        var n_valid = UInt32(0)
        for i in range(tid, N, NT):
            var key = _score_key(row_scores[i])
            keys[i] = key
            if key != 0:
                n_valid += 1
                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](hist.unsafe_offset(Int(key >> _DIGIT_BITS)), UInt32(1))
        var total = block.sum[block_size=NT](n_valid)
        barrier()

        # With at most `top_k` real candidates the group's top-k is all of
        # them; otherwise the threshold key `t` and the count of ties at it
        # the selection takes, lowest columns first.
        var take_all = total <= UInt32(top_k)
        var threshold = UInt16(0)
        var ties_needed = UInt32(0)
        var ties_total = UInt32(0)
        if not take_all:
            if warp_id() == 0:
                _select_digit(hist, UInt32(top_k), ctl)
            barrier()
            var hi = UInt16(ctl[0])
            var need = ctl[1]
            barrier()
            # Pass 2: the low digit among keys with high digit `hi`.
            for i in range(tid, _BINS, NT):
                hist[i] = 0
            barrier()
            for i in range(tid, N, NT):
                var key = keys[i]
                if key != 0 and key >> _DIGIT_BITS == hi:
                    _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                        ordering=Ordering.RELAXED
                    ](hist.unsafe_offset(Int(key & MASK)), UInt32(1))
            barrier()
            if warp_id() == 0:
                _select_digit(hist, need, ctl)
            barrier()
            threshold = (hi << _DIGIT_BITS) | UInt16(ctl[0])
            ties_needed = ctl[1]
            ties_total = hist[Int(ctl[0])]
            barrier()

        # Ties in lower ranks' columns come first in the column order.
        var own_begin = rank * top_k
        var ties_before = UInt32(0)
        if ties_needed < ties_total:
            var mine = UInt32(0)
            for i in range(tid, own_begin, NT):
                if keys[i] == threshold:
                    mine += 1
            ties_before = block.sum[block_size=NT](mine)
            barrier()

        # Walk this rank's columns in order: rank each tie among all ties by
        # column, and compact the selected rows to the front.
        var n_sel = UInt32(0)
        for base in range(0, top_k, NT):
            var j = base + tid
            var live = j < top_k
            var key = keys[own_begin + j] if live else UInt16(0)
            var tie = live and key == threshold and key != 0
            var tie_rank = ties_before + block.prefix_sum[
                block_size=NT, exclusive=True
            ](UInt32(Int(tie)))
            barrier()
            var local_row = local_rows[row * W + j] if live else Int32(-1)
            var sel = (
                key != 0
                and local_row >= 0
                and (
                    take_all
                    or key > threshold
                    or (tie and tie_rank < ties_needed)
                )
            )
            var slot = n_sel + block.prefix_sum[block_size=NT, exclusive=True](
                UInt32(Int(sel))
            )
            barrier()
            if sel:
                out_rows[row * W + Int(slot)] = local_row
                comptime for h in range(1, halves):
                    out_rows[row * W + h * top_k + Int(slot)] = local_rows[
                        row * W + h * top_k + j
                    ]
            ties_before += block.sum[block_size=NT](UInt32(Int(tie)))
            barrier()
            n_sel += block.sum[block_size=NT](UInt32(Int(sel)))
            barrier()

        for i in range(Int(n_sel) + tid, top_k, NT):
            comptime for h in range(halves):
                out_rows[row * W + h * top_k + i] = -1
        if tid == 0:
            out_lengths[row] = Int32(n_sel)


def cp_merge_topk[
    rows_layout: TensorLayout,
    scores_layout: TensorLayout,
    out_layout: TensorLayout,
    lengths_layout: TensorLayout,
    //,
    dcp: Int,
    top_k: Int,
    halves: Int = 1,
](
    out_rows: MutTileTensor[.int32, out_layout, _],
    out_lengths: MutTileTensor[.int32, lengths_layout, _],
    local_rows: ImmTileTensor[.int32, rows_layout, _],
    scores: ImmTileTensor[.bfloat16, scores_layout, _],
    cp_rank: ImmPointer[Int32, _],
    ctx: DeviceContext,
) raises:
    """Keeps this rank's candidates that are in its group's top-k.

    Each of the `dcp` ranks of a decode context-parallel group selected the
    top-k of its own key slots. Given every rank's selection scores, this
    picks the `min(top_k, candidates)` highest of the `dcp * top_k` -- ties
    at the threshold go to the lowest columns, so every rank makes the same
    choice -- and writes this rank's rows among them, compacted to the front
    of each row. Padding is a `-inf` score, which never selects; a selected
    candidate whose row is `-1` is dropped.

    With `halves = 2` each row of `local_rows` is a selection's value rows
    then its scale-pool rows (an NVFP4 cache's indexer output), and each kept
    selection's scale row moves with its value row: `out_rows`' second half is
    compacted exactly as its first, and `out_lengths` counts either half.

    Parameters:
        rows_layout: Layout of `local_rows`.
        scores_layout: Layout of `scores`.
        out_layout: Layout of `out_rows`.
        lengths_layout: Layout of `out_lengths`.
        dcp: Ranks in the group.
        top_k: Selections per token on each rank and in the merged result.
        halves: Rows per selection, 1, or 2 for a value row and its scale
            row `top_k` columns later (defaults to 1).

    Args:
        out_rows: Dense `[T, halves * top_k]` this rank's selected rows, in
            column order, then `-1`; with `halves = 2` their scale rows
            follow at column `top_k`, compacted the same way.
        out_lengths: Dense `[T]` how many of `out_rows`' slots are selected.
        local_rows: Dense `[T, halves * top_k]` this rank's local
            selections, the rows matching
            `scores[:, rank * top_k :][:, :top_k]`, then (with `halves = 2`)
            their scale rows.
        scores: Dense `[T, dcp * top_k]` bf16 scores of every rank's selections,
            rank-major, `-inf` for padding.
        cp_rank: This rank in `[0, dcp)`, one `int32` on the device.
        ctx: Device context.
    """
    comptime assert dcp >= 1 and top_k >= 1
    comptime assert (
        halves == 1 or halves == 2
    ), "a selection is a value row, or a value row and its scale row"
    comptime assert dcp * top_k <= _MAX_CANDIDATES, (
        "the candidates' keys are staged in shared memory, which holds at most"
        " 16384"
    )
    comptime assert out_rows.flat_rank == 2 and local_rows.flat_rank == 2
    comptime assert scores.flat_rank == 2 and out_lengths.flat_rank == 1
    var rows = Int(local_rows.dim[0]())
    if (
        Int(local_rows.dim[1]()) != halves * top_k
        or Int(out_rows.dim[0]()) != rows
        or Int(out_rows.dim[1]()) != halves * top_k
        or Int(scores.dim[0]()) != rows
        or Int(scores.dim[1]()) != dcp * top_k
        or Int(out_lengths.dim[0]()) != rows
    ):
        raise Error(
            "cp_merge_topk: expected local_rows and out_rows"
            " [T, halves * top_k],"
            " scores [T, dcp * top_k] and out_lengths [T]"
        )
    if rows == 0:
        return
    comptime kernel = _cp_merge_topk_kernel[dcp, top_k, halves]
    ctx.enqueue_function[kernel](
        local_rows.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        scores.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
        cp_rank.unsafe_origin_cast[ImmUntrackedOrigin](),
        out_rows.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        out_lengths.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        grid_dim=rows,
        block_dim=_MERGE_THREADS,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
