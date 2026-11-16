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
"""The filtered top-k select for long indexer rows that fill the GPU.

`persistent_topk_block_split` selects such a row with the full histogram
select, which spends tens of instructions on every column. When `K` is a few
percent of the row, a strided sample bounds the K-th score from below, one
pass keeps the columns at or above the bound, and an exact select over those
runs in shared memory. A row whose bound does not hold is flagged, and a
second launch runs the full select on the flagged rows only. A filtered row
takes the lowest columns among those tied at the K-th score; a flagged row
keeps the full select's unspecified choice among them.

`FilteredTopk` is the `IndexerTopk` that takes this select. Hy4's indexer
runs it at every step, through the
`attention.mla.indexer.ragged.float8.paged.remap.filtered` op; the filtered
launches run only for score windows whose rows fill the GPU, which in
Hy4's serving config is a prefill chunk. Its Moka
component is `Kernels/test/gpu/Moka/kernels/select/filtered_select_plan.mojo`,
which runs `filtered_topk_block`, the same CTA body.
"""

from std.sys import align_of, is_nvidia_gpu, size_of

from std.bit import count_trailing_zeros, pop_count

from std.atomic import Atomic, Ordering
from shmem.ep_comm import BLOCK_SCOPE
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp
from max.gpu.host import (
    DeviceContext,
    FuncAttribute,
    default_accelerator,
)
from std.memory import bitcast, unsafe_stack_allocation
from std.math import sqrt
from std.utils.static_tuple import StaticTuple
from structured_kernels.trace_buf import NullTrace

from nn.attention.gpu.mla_index_fp8 import IndexerTopk
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBound,
    TopkSink,
    _HSEL_BINS,
    _HSEL_BITS,
    _HSEL_SCAN_ITEMS,
    _HSEL_SMEM_BYTES,
    _HSEL_TAIL_BITS,
    _PTOPK_BLOCK,
    _histsel_topk_impl,
    _hsel_half_rounds,
    _hsel_phi_dtype,
    _hsel_sig_bits,
    _key_column,
    _load_scan_group,
    _pack_key,
    _phi_of,
    _score_bits,
    persistent_topk_block_split,
)

comptime FILTER_SAMPLES: Int = 2048
"""Scores sampled per row to bound its K-th score."""
comptime FILTER_MIN_RATIO: Int = 4
"""The shortest row the filtered select takes, as a multiple of `K`."""
comptime FILTER_SMEM_BYTES: Int = _HSEL_SMEM_BYTES
"""Shared memory a CTA of the filtered select takes: its histogram, then its
keys. The full select takes the same, so a flagged row fits either kernel."""
comptime FILTER_POISONS: Int = 4
"""Planted faults, `poison` 1 through 4, each of which a test must catch:
1 writes only the keys strictly above the threshold key, 2 picks the wrong
column from the tie list, 3 takes one too many keys above the bf16 step bin,
4 shifts the plateau's tied columns one slot up."""

comptime _KEY_CAP: Int = (
    FILTER_SMEM_BYTES - _HSEL_BINS * size_of[UInt32]()
) // size_of[UInt64]()
comptime _TIE_CAP: Int = _PTOPK_BLOCK
comptime FILTER_BLOCKS_PER_SM: Int = 3
"""CTAs of the filtered select an SM holds. The select is latency-bound per
row (barriers and dependent loads, not issue slots), so rows in flight are its
throughput, and its registers are capped to fit three CTAs."""


@inline(.always)
def _filter_shift[sig_bits: Int, r: Int]() -> Int:
    """Lowest `phi` bit read by round `r` of the score rounds: 12 bits, then 10
    at a time, up to the score's significant width (two rounds at bf16, three
    at f32)."""
    return 32 - _HSEL_BITS - min(sig_bits - _HSEL_BITS, _HSEL_TAIL_BITS * r)


@inline(.always)
def _zero[
    o: MutOrigin
](hist: MutPointer[UInt32, o, address_space=.SHARED], n: Int, tid: Int):
    """Zeroes the first `n` bins of `hist`; the caller barriers after."""
    var z = tid
    while z < n:
        hist[z] = 0
        z += _PTOPK_BLOCK


@inline(.always)
def _split[
    ho: MutOrigin, co: MutOrigin
](
    hist: MutPointer[UInt32, ho, address_space=.SHARED],
    ctl: MutPointer[UInt32, co, address_space=.SHARED],
    n: Int,
    need: Int,
    tid: Int,
):
    """Writes the bin of `hist`'s `need`-th largest key to `ctl[0]`, the count
    of keys in higher bins to `ctl[1]` and in the bin itself to `ctl[2]`, then
    barriers.

    Thread `t` owns the descending run of bins ending at `n-1 - t*run`, so a
    prefix sum over threads is a suffix sum over bins, and the one thread whose
    run crosses `need` writes the answer.
    """
    var run = max(1, n // _PTOPK_BLOCK)
    var top = n - 1 - tid * run
    var local = UInt32(0)
    if top >= 0:
        for j in range(run):
            local += hist[top - j]
    var inclusive = block.prefix_sum[block_size=_PTOPK_BLOCK](local)
    var exclusive = inclusive - local
    var uneed = UInt32(need)
    if exclusive < uneed and uneed <= inclusive:
        var acc = exclusive
        var bin = top
        while True:
            acc += hist[bin]
            if acc >= uneed:
                break
            bin -= 1
        ctl[0] = UInt32(bin)
        ctl[1] = acc - hist[bin]
        ctl[2] = hist[bin]
    barrier()


@inline(.always)
def _round[
    shift: Int,
    hi: Int,
    ho: MutOrigin,
    co: MutOrigin,
    ko: MutOrigin,
](
    hist: MutPointer[UInt32, ho, address_space=.SHARED],
    ctl: MutPointer[UInt32, co, address_space=.SHARED],
    keys: MutPointer[UInt64, ko, address_space=.SHARED],
    n_keys: Int,
    need: Int,
    mask: UInt64,
    prefix: UInt64,
    tid: Int,
):
    """One digit round over the `n_keys` kept keys: histograms key bits
    `[shift, hi)` of the keys that agree with `prefix` on every bit of `mask`
    at or above `hi` (what earlier rounds resolved) and splits at `need`."""
    comptime nb = 1 << (hi - shift)
    var live_mask = UInt64(0)
    comptime if hi < 64:
        live_mask = mask & ~((UInt64(1) << UInt64(hi)) - 1)
    _zero(hist, nb, tid)
    barrier()
    var i = tid
    while i < n_keys:
        var key = keys[i]
        if key & live_mask == prefix & live_mask:
            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](
                hist.unsafe_offset(
                    Int((key >> UInt64(shift)) & UInt64(nb - 1))
                ),
                1,
            )
        i += _PTOPK_BLOCK
    barrier()
    _split(hist, ctl, nb, need, tid)


@inline(.always)
def _keep_pass[
    in_dtype: DType,
    ko: MutOrigin,
    co: MutOrigin,
    //,
    strict: Bool,
    by_phi: Bool,
](
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    keys: MutPointer[UInt64, ko, address_space=.SHARED],
    ctl: MutPointer[UInt32, co, address_space=.SHARED],
    row: Int,
    count: Int,
    bound: UInt32,
    bound_score: Scalar[in_dtype],
    tid: Int,
) -> Int:
    """Keeps a row's columns at or above `bound` (strictly above when
    `strict`) as packed keys, returning how many there are; past `_KEY_CAP` it
    only counts. `ctl[3]` must be zero on entry.

    One group ahead, so two loads are in flight per thread. A warp's kept
    columns go out behind one cursor claim, placed by a prefix sum over its
    lanes, and a lane walks only its kept bits: with a few percent kept, a
    branch per column would cost the warp every column's store path. Scores
    compare in their own dtype against `bound_score`, the lowest score at or
    above `bound`, unless `by_phi`, which a zero bound and a strict test need
    so -0 stays below +0. The loop runs on the warp's first lane's base so the
    shuffles stay converged; lanes past `count` load padding, below any bound.
    """
    comptime scan_items = _HSEL_SCAN_ITEMS
    comptime scan_step = _PTOPK_BLOCK * scan_items
    comptime assert by_phi or not strict, "a strict test compares `phi`"
    var lane = Int(lane_id())
    var base = tid * scan_items
    var warp_base = base - lane * scan_items
    var cur = _load_scan_group[scan_items](in_scores, row + base, base, count)
    while warp_base < count:
        var nb = base + scan_step
        var nxt = _load_scan_group[scan_items](in_scores, row + nb, nb, count)
        var keep: SIMD[DType.bool, scan_items]
        comptime if by_phi:
            var ph = _phi_of[in_dtype, .uint32, scan_items](cur)
            comptime if strict:
                keep = ph.gt(SIMD[.uint32, scan_items](bound))
            else:
                keep = ph.ge(SIMD[.uint32, scan_items](bound))
        else:
            keep = bitcast[in_dtype, scan_items](cur).ge(
                SIMD[in_dtype, scan_items](bound_score)
            )
        var kept = UInt32(0)
        comptime for j in range(scan_items):
            kept |= UInt32(Int(keep[j])) << UInt32(j)
        var n_keep = pop_count(kept)
        var end = warp.prefix_sum(n_keep)
        var warp_kept = warp.shuffle_idx(end, UInt32(WARP_SIZE - 1))
        if warp_kept != 0:
            var start = UInt32(0)
            if lane == 0:
                start = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl.unsafe_offset(3), warp_kept)
            var slot = Int(warp.shuffle_idx(start, 0) + end - n_keep)
            while kept != 0:
                var col = base + Int(count_trailing_zeros(kept))
                if slot < _KEY_CAP:
                    var ph = _phi_of[in_dtype, .uint32, 1](
                        bitcast[_score_bits[in_dtype](), 1](
                            SIMD[in_dtype, 1](in_scores[row + col])
                        )
                    )[0]
                    keys[slot] = _pack_key(ph, ~UInt32(col))
                slot += 1
                kept &= kept - 1
        cur = nxt
        base = nb
        warp_base += scan_step
    barrier()
    return Int(ctl[3])


@inline(.always)
def _plateau_ties[
    in_dtype: DType,
    S: TopkSink,
    co: MutOrigin,
    //,
    poison: Int,
](
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    sink: S,
    ctl: MutPointer[UInt32, co, address_space=.SHARED],
    row: Int,
    count: Int,
    bound: UInt32,
    out_base: Int,
    need: Int,
    tid: Int,
):
    """Writes the `need` lowest columns whose `phi` is `bound` to output slots
    `[out_base, out_base + need)`, scanning in column order and stopping once
    it has them. A step covers `_PTOPK_BLOCK` groups of contiguous columns in
    thread order, so a block prefix sum of the tie counts ranks each tie by
    column."""
    comptime scan_items = _HSEL_SCAN_ITEMS
    comptime scan_step = _PTOPK_BLOCK * scan_items
    var taken = 0
    var step = 0
    while step < count and taken < need:
        var base = step + tid * scan_items
        var ph = _phi_of[in_dtype, .uint32, scan_items](
            _load_scan_group[scan_items](in_scores, row + base, base, count)
        )
        var tied = UInt32(0)
        comptime for j in range(scan_items):
            if ph[j] == bound and base + j < count:
                tied |= UInt32(1) << UInt32(j)
        var n_tied = pop_count(tied)
        var inclusive = block.prefix_sum[block_size=_PTOPK_BLOCK](n_tied)
        var rank = taken + Int(inclusive - n_tied) + Int(poison == 4)
        while tied != 0 and rank < need:
            sink.store(
                out_base + rank, Int32(base + Int(count_trailing_zeros(tied)))
            )
            rank += 1
            tied &= tied - 1
        if tid == _PTOPK_BLOCK - 1:
            ctl[7] = inclusive
        barrier()
        taken += Int(ctl[7])
        # The next step's write must not race this step's reads.
        barrier()
        step += scan_step


@inline(.always)
def _filtered_select[
    in_dtype: DType, S: TopkSink, so: MutOrigin, //, sig_bits: Int, poison: Int
](
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    sink: S,
    smem: MutPointer[UInt32, so, address_space=.SHARED],
    N: Int,
    K: Int,
    count: Int,
    row_idx: Int,
    tid: Int,
) -> Bool:
    """Selects row `row_idx`'s top `K` among the columns at or above a sampled
    bound.

    Only the bound comes from the sample; the selection is exact over every
    column above it, so a poor sample costs speed, never correctness. The `K`
    columns go out in no particular order. The set is deterministic:
    descending score, then ascending column among the columns tied at the K-th
    score. A bound on a plateau (more columns at or above it than the keys
    hold) resolves the plateau as the threshold when fewer than `K` columns
    lie strictly above it.

    Returns:
        False, having written nothing, when the bound does not hold: fewer than
        `K` columns at or above it, `K` or more strictly above a plateau, or a
        row too short for a sample to bound tightly.
    """
    comptime nbins = _HSEL_BINS
    comptime n_samples = FILTER_SAMPLES
    comptime sig_pad = 32 - sig_bits
    if count < FILTER_MIN_RATIO * K:
        return False

    var row = row_idx * N
    # The histogram and the keys are two regions of one buffer.
    var hist = smem.unsafe_origin_cast[MutUntrackedOrigin]()
    var keys = hist.unsafe_offset(nbins).unsafe_bitcast[UInt64]()
    # [0] split bin, [1] keys strictly above it, [2] keys in it, [3] kept-key
    # cursor, [4] output cursor, [5] tie-list cursor, [6] threshold column,
    # [7] ties taken per step.
    var ctl = unsafe_stack_allocation[8, UInt32, address_space=.SHARED]()

    # The bound: the m-th largest score of a strided sample, m five standard
    # deviations above the sample's expected count above the K-th score. Its
    # 12-bit bin is the bound when the samples at or above that bin predict a
    # key count well inside the cap; a row whose scores sit close together
    # puts too many in one bin and takes the rest of the significant bits.
    comptime per_thread = n_samples // _PTOPK_BLOCK
    comptime fine_shift = _filter_shift[sig_bits, 1]()
    comptime fine_bins = 1 << (32 - _HSEL_BITS - fine_shift)
    _zero(hist, nbins, tid)
    if tid < 8:
        ctl[tid] = 0
    barrier()
    var sample = SIMD[.uint32, per_thread]()
    comptime for i in range(per_thread):
        var col = ((tid + i * _PTOPK_BLOCK) * count) // n_samples
        sample[i] = _phi_of[in_dtype, .uint32, 1](
            bitcast[_score_bits[in_dtype](), 1](
                SIMD[in_dtype, 1](in_scores[row + col])
            )
        )[0]
        _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
            ordering=Ordering.RELAXED
        ](hist.unsafe_offset(Int(sample[i] >> UInt32(32 - _HSEL_BITS))), 1)
    barrier()
    var mean = Float32(n_samples * K) / Float32(count)
    var m = Int(mean + 5 * sqrt(mean)) + 4
    if m >= n_samples:
        return False
    _split(hist, ctl, nbins, m, tid)
    var bound_bin = ctl[0]
    var m_fine = m - Int(ctl[1])
    var bound = bound_bin << UInt32(32 - _HSEL_BITS)
    if (Int(ctl[1]) + Int(ctl[2])) * count > n_samples * (_KEY_CAP // 2):
        _zero(hist, fine_bins, tid)
        barrier()
        comptime for i in range(per_thread):
            if sample[i] >> UInt32(32 - _HSEL_BITS) == bound_bin:
                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](
                    hist.unsafe_offset(
                        Int(
                            (sample[i] >> UInt32(fine_shift))
                            & UInt32(fine_bins - 1)
                        )
                    ),
                    1,
                )
        barrier()
        _split(hist, ctl, fine_bins, m_fine, tid)
        bound |= ctl[0] << UInt32(fine_shift)
    # The bound's score is the lowest one at or above it: its low bits are
    # zero above a positive and ones above a negative. A bf16 bound keeps the
    # high half by truncation; rounding would carry a negative bound's ones
    # into the next step down. A zero bound would admit -0 below a +0 bound,
    # so it compares `phi` instead.
    var bound_bits = bound ^ UInt32(0x80000000) if bound >> 31 != 0 else ~bound
    var bound_score: Scalar[in_dtype]
    comptime if size_of[Scalar[in_dtype]]() == 2:
        bound_score = bitcast[in_dtype, 1](UInt16(bound_bits >> 16))
    else:
        bound_score = bitcast[.float32, 1](bound_bits).cast[in_dtype]()
    var n_keys: Int
    if bound_score == 0:
        n_keys = _keep_pass[strict=False, by_phi=True](
            in_scores, keys, ctl, row, count, bound, bound_score, tid
        )
    else:
        n_keys = _keep_pass[strict=False, by_phi=False](
            in_scores, keys, ctl, row, count, bound, bound_score, tid
        )
    if n_keys < K:
        return False
    if n_keys > _KEY_CAP:
        # A plateau the sample could not see past: keep the columns strictly
        # above it. At least `K` of them hold the K; fewer, and the K-th score
        # is the plateau's, so the K are every column above it and the
        # plateau's lowest columns. Every thread must read the first count
        # before the reset.
        barrier()
        if tid == 0:
            ctl[3] = 0
        barrier()
        n_keys = _keep_pass[strict=True, by_phi=True](
            in_scores, keys, ctl, row, count, bound, bound_score, tid
        )
        if n_keys > _KEY_CAP:
            return False
    if n_keys < K:
        var i = tid
        while i < n_keys:
            sink.store(row_idx * K + i, _key_column(keys[i]))
            i += _PTOPK_BLOCK
        _plateau_ties[poison](
            in_scores,
            sink,
            ctl,
            row,
            count,
            bound,
            row_idx * K + n_keys,
            K - n_keys,
            tid,
        )
        return True

    # The K-th largest score among the kept keys, which are every column at or
    # above the bound. At bf16 one round resolves it: the K-th score sits
    # within a few hundred bf16 steps of the bound, so the step count above it
    # is the digit, the top bin catching the rest. Wider scores, or a split in
    # the top bin, take the digit rounds.
    var t_key = UInt64(0)
    var need = K
    var resolved = False
    comptime if sig_bits == 16:
        comptime step_bins = 1024
        var bound16 = bound >> UInt32(16)
        _zero(hist, step_bins, tid)
        barrier()
        var i = tid
        while i < n_keys:
            var steps = UInt32(keys[i] >> 48) - bound16
            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](hist.unsafe_offset(Int(min(steps, UInt32(step_bins - 1)))), 1)
            i += _PTOPK_BLOCK
        barrier()
        _split(hist, ctl, step_bins, K, tid)
        if Int(ctl[0]) < step_bins - 1:
            t_key = UInt64(bound16 + ctl[0]) << 48
            need = K - Int(ctl[1]) + Int(poison == 3)
            resolved = True
        barrier()
    if not resolved:
        comptime rounds = _hsel_half_rounds[_HSEL_TAIL_BITS, sig_bits]()
        comptime for r in range(rounds):
            comptime shift = 32 + _filter_shift[sig_bits, r]()
            comptime hi = 64 if r == 0 else 32 + _filter_shift[
                sig_bits, r - 1
            ]()
            _round[shift, hi](
                hist, ctl, keys, n_keys, need, ~UInt64(0), t_key, tid
            )
            t_key |= UInt64(ctl[0]) << UInt64(shift)
            need -= Int(ctl[1])
            barrier()
    var n_tied = Int(ctl[2])

    # Among the columns tied at the threshold score the `need` lowest go out,
    # which sets the threshold key's column half: a few tied columns are
    # ranked in a list, a plateau takes the column half's digit rounds. A key's
    # score half carries `sig_pad` low bits `phi` fills uniformly; no round
    # resolves them, so every comparison masks them out.
    comptime pad_mask = ~(
        (UInt64(1) << UInt64(32 + sig_pad)) - (UInt64(1) << UInt64(32))
    )
    if n_tied > need and n_tied <= _TIE_CAP:
        var t_score = t_key >> 32
        var i = tid
        while i < n_keys:
            var key = keys[i]
            if (key & pad_mask) >> 32 == t_score:
                var t = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl.unsafe_offset(5), 1)
                hist[Int(t)] = UInt32(key & 0xFFFFFFFF)
            i += _PTOPK_BLOCK
        barrier()
        # The tied column with `need - 1` tied columns below it (above it in
        # `~column`) is the threshold key's column half.
        if tid < n_tied:
            var rcol = hist[tid]
            var rank = 0
            for t in range(n_tied):
                rank += Int(hist[t] > rcol)
            if rank == need - 1 + Int(poison == 2):
                ctl[6] = rcol
        barrier()
        t_key |= UInt64(ctl[6])
    elif n_tied > need:
        comptime col_rounds = _hsel_half_rounds[_HSEL_TAIL_BITS, 32]()
        comptime for r in range(col_rounds):
            comptime shift = _filter_shift[32, r]()
            comptime hi = 32 if r == 0 else _filter_shift[32, r - 1]()
            _round[shift, hi](
                hist, ctl, keys, n_keys, need, pad_mask, t_key, tid
            )
            t_key |= UInt64(ctl[0]) << UInt64(shift)
            need -= Int(ctl[1])
            barrier()

    var out_row = row_idx * K
    var i = tid
    while i < n_keys:
        var key = keys[i]
        var take: Bool
        comptime if poison == 1:
            take = key & pad_mask > t_key
        else:
            take = key & pad_mask >= t_key
        if take:
            var slot = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](ctl.unsafe_offset(4), 1)
            sink.store(out_row + Int(slot), _key_column(key))
        i += _PTOPK_BLOCK
    return True


@inline(.always)
def filtered_topk_block[
    in_dtype: DType,
    RB: RowBound,
    so: MutOrigin,
    //,
    sig_bits: Int = _hsel_sig_bits[in_dtype](),
    poison: Int = 0,
](
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: MutPointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    row_bounds: RB,
    fallback: MutPointer[UInt8, MutAnyOrigin],
    smem: MutPointer[UInt32, so, address_space=.SHARED],
    row: Int,
    tid: Int,
):
    """Selects one row with the filtered select, or flags it for the full one.

    One CTA's work: `tid` is the thread in a block of `_PTOPK_BLOCK`, `smem`
    the CTA's `FILTER_SMEM_BYTES` of shared memory (8-byte aligned). Writes
    `fallback[row]`, 1 when the row's bound did not hold and nothing else was
    written for it.

    Parameters:
        in_dtype: Score dtype (inferred).
        RB: Row bound type (inferred).
        so: Origin of `smem` (inferred).
        sig_bits: Significant bits of a score's `phi`.
        poison: A planted fault (`FILTER_POISONS`), 0 for none.

    Args:
        in_scores: Row-major scores `[rows, N]`.
        out_idxs: Row-major output `[rows, K]`.
        N: Score columns per row, the row stride.
        K: Columns selected per row.
        row_bounds: Per-row live column counts and output sinks.
        fallback: One flag per row.
        smem: The CTA's shared memory.
        row: The row.
        tid: The thread in the block.
    """
    var bound, sink = row_bounds.row(row, out_idxs)
    var count = min(N, max(0, bound))
    var done = False
    comptime if is_nvidia_gpu():
        done = _filtered_select[sig_bits=sig_bits, poison=poison](
            in_scores, sink, smem, N, K, count, row, tid
        )
    if tid == 0:
        fallback[row] = UInt8(not done)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_PTOPK_BLOCK)),
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(FILTER_BLOCKS_PER_SM))
@__name(t"filtered_topk_bounded_{in_dtype}")
def _filtered_topk_kernel[
    in_dtype: DType, RB: RowBound, poison: Int
](
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: MutPointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    row_bounds: RB,
    fallback: MutPointer[UInt8, MutAnyOrigin],
):
    """`filtered_topk_block` for row `block_idx.x`. The full select stays out
    of this kernel so its shared memory does not limit the rows in flight."""
    with PDL():
        filtered_topk_block[poison=poison](
            in_scores,
            out_idxs,
            Int(N),
            Int(K),
            row_bounds,
            fallback,
            external_memory[
                UInt32, address_space=.SHARED, alignment=align_of[UInt64]()
            ](),
            Int(block_idx.x),
            Int(thread_idx.x),
        )


@__name(t"filtered_topk_fallback_{in_dtype}")
def _fallback_topk_kernel[
    in_dtype: DType, RB: RowBound
](
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: MutPointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    row_bounds: RB,
    fallback: ImmPointer[UInt8, ImmutAnyOrigin],
):
    """The full select over the rows the filtered launch flagged; every other
    row's block exits at once."""
    with PDL():
        if fallback[block_idx.x] == 0:
            return
        var bound, sink = row_bounds.row(Int(block_idx.x), out_idxs)
        _histsel_topk_impl[
            NullTrace,
            ordered=False,
            deterministic=False,
            sig_bits=_hsel_sig_bits[in_dtype](),
            phi_dtype=_hsel_phi_dtype[in_dtype](),
        ](in_scores, sink, N, K, min(Int(N), max(0, bound)), NullTrace())


def filtered_topk_launch[
    in_dtype: DType, RB: RowBound, //, poison: Int = 0
](
    ctx: DeviceContext,
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: MutPointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    rows: Int,
    row_bounds: RB,
    fallback: MutPointer[UInt8, MutAnyOrigin],
) raises:
    """Runs the filtered select's two launches, with the caller's flags.

    Every row takes the filtered launch; the rows it flags in `fallback`
    (`[rows]` bytes) take the full select in the second. Callers go through
    `filtered_topk_block_split`, which decides when this pays; tests call it
    directly to read the flags.

    Parameters:
        in_dtype: Score dtype (inferred).
        RB: Row bound type (inferred).
        poison: A planted fault (`FILTER_POISONS`), 0 for none.

    Args:
        ctx: Device context.
        in_scores: Row-major scores `[rows, N]`, 16-byte aligned rows.
        out_idxs: Row-major output `[rows, K]`.
        N: Score columns per row, the row stride.
        K: Columns selected per row, at most `PERSISTENT_TOPK_MAX_N`.
        rows: Score rows.
        row_bounds: Per-row live column counts and output sinks.
        fallback: One flag per row, written by the first launch.
    """
    ctx.enqueue_function[_filtered_topk_kernel[in_dtype, RB, poison]](
        in_scores,
        out_idxs,
        Int32(N),
        Int32(K),
        row_bounds,
        fallback,
        grid_dim=rows,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=FILTER_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(FILTER_SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
    ctx.enqueue_function[_fallback_topk_kernel[in_dtype, RB]](
        in_scores,
        out_idxs,
        Int32(N),
        Int32(K),
        row_bounds,
        fallback.as_imm(),
        grid_dim=rows,
        block_dim=_PTOPK_BLOCK,
        shared_mem_bytes=FILTER_SMEM_BYTES,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(FILTER_SMEM_BYTES)
        ),
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


def filtered_topk_block_split[
    in_dtype: DType, RB: RowBound, //
](
    ctx: DeviceContext,
    in_scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: MutPointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    rows: Int,
    row_bounds: RB,
) raises:
    """`persistent_topk_block_split` with `ordered=False, deterministic=False`,
    except that bounded rows which fill the GPU take the filtered select.

    That is the case where `persistent_topk_block_split` runs one full select
    per row (`_histsel_topk_bounded_kernel`): rows at least the SM count and
    wider than `PERSISTENT_TOPK_MAX_N`. The filtered select also needs rows of
    more than `FILTER_MIN_RATIO * K` columns; below that stride no row could
    take it. Every other shape, and every non-NVIDIA GPU, keeps
    `persistent_topk_block_split`. An empty launch does nothing.

    Parameters:
        in_dtype: Score dtype, bf16 or f32 (inferred).
        RB: Row bound type (inferred).

    Args:
        ctx: Device context.
        in_scores: Row-major scores `[rows, N]`, 16-byte aligned rows.
        out_idxs: Row-major output `[rows, K]`.
        N: Score columns per row, the row stride.
        K: Columns selected per row, at most `PERSISTENT_TOPK_MAX_N`.
        rows: Score rows.
        row_bounds: Per-row live column counts and output sinks.
    """
    if rows == 0:
        return
    comptime if default_accelerator().is_nvidia_gpu():
        if N > PERSISTENT_TOPK_MAX_N and N > FILTER_MIN_RATIO * K:
            var sm_count = ctx.device().multiprocessor_count()
            if rows >= sm_count:
                var fallback = ctx.enqueue_create_buffer[.uint8](rows)
                filtered_topk_launch(
                    ctx,
                    in_scores,
                    out_idxs,
                    N,
                    K,
                    rows,
                    row_bounds,
                    fallback.unsafe_ptr(),
                )
                _ = fallback
                return
    persistent_topk_block_split[RB=RB](
        ctx, in_scores, out_idxs, N, K, rows, Optional(row_bounds)
    )


struct FilteredTopk(IndexerTopk):
    """The indexer top-k with the filtered select (`filtered_topk_block_split`).
    """

    @staticmethod
    def select[
        in_dtype: DType, RB: RowBound
    ](
        ctx: DeviceContext,
        scores: ImmPointer[Scalar[in_dtype], ImmutAnyOrigin],
        out_idxs: MutPointer[Int32, MutAnyOrigin],
        n: Int,
        k: Int,
        rows: Int,
        row_bounds: RB,
    ) raises:
        """Writes the top-`k` columns of each row of `scores` to `out_idxs`.

        Parameters:
            in_dtype: Score element type.
            RB: Type of `row_bounds`.

        Args:
            ctx: Device context.
            scores: Row-major scores, `[rows, n]`.
            out_idxs: Row-major output, `[rows, k]`.
            n: Score columns per row, the row stride.
            k: Columns selected per row.
            rows: Score rows.
            row_bounds: Per-row live column counts and output sinks.
        """
        filtered_topk_block_split(ctx, scores, out_idxs, n, k, rows, row_bounds)
