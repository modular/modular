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
"""Deterministic cluster-cooperative top-k over bf16 scores.

The `ordered=False, deterministic=True` contract of
`persistent_topk_block_split`, for rows that under-fill the GPU. It writes the
same output as the ranked streaming select, slot for slot: the `K` largest
`(score, column)` keys, descending score and ascending column within a score,
then `-1` past a row's live columns. That select spends one SM on a row, and
its counting rank, which is what makes its output reproducible, degrades on
the bf16 score plateaus an indexer row is made of. This kernel spreads a row
over a cluster of `CS` CTAs and orders the selection with a comparison sort,
whose cost does not depend on the scores.

Per row, every rank owns a contiguous, 16 B-aligned slice of `[0, count)`:

1. Each rank histograms the top 12 bits of its scores' `phi16` (the
   order-preserving bf16 bijection `_phi_of` defines). Every rank reads the
   peers' histograms over DSMEM, first as per-warp group totals and then the
   split group's bins, and finds the same split bin.
2. A 4-bit round over the split bin's columns resolves the exact threshold
   key `t`, the `K`-th largest `phi16`. Each rank publishes its 16 fine bins
   and its count above the split bin, which is also what tells every rank how
   many columns each rank holds above and at `t`.
3. Every column above `t` is selected, and of the columns at `t` the
   lowest-numbered ones that complete `K`, which the ranks below take first.
   A block scan places each selected column at its position `g` in column
   order. Nothing is placed by arrival, so the set and `g` are functions of
   the input alone.
4. The key `(phi16 << 16) | (0xFFFF - g)` orders a selection exactly as the
   ranked select does, since ascending `g` is ascending column. Rank `r`
   receives the keys of positions `[r L, (r + 1) L)`, `L = 2048 / CS`, and
   sorts them; then every rank reads the other sorted chunks and places each
   of its keys at its own index plus the count of larger keys in every other
   chunk, and writes it there.

A row with at most `K` live columns selects all of them, already in column
order, so it starts at step 4.

Every count is a histogram sum or a scan, and every slot is written by
exactly one thread at a position those counts determine, so one input always
gives one output, whatever order the CTAs and warps run in.

Every DSMEM producer and consumer are separated by a full `cluster_sync()`,
whose arrive releases and whose wait acquires at cluster scope: the
semantics `_ctopk_cluster_sync_cta_release` in `topk_cluster.mojo` documents
from the PTX ISA. compute-sanitizer's racecheck still reports "potential RAW"
hazards on one rank's stores into a peer's `pairs` and the peer's reads after
that barrier; it reports the same class on the shipped non-deterministic
cluster kernel at 16 ranks (a 128k-key row), and nothing at one rank, so
these are read as the tool not modelling cluster barriers.
"""

from std.atomic import Atomic, Ordering
from std.bit import log2_floor
from std.collections.optional import OptionalReg
from std.math import ceildiv
from std.memory import bitcast, unsafe_stack_allocation
from std.utils.static_tuple import StaticTuple

from shmem.ep_comm import BLOCK_SCOPE

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, Dim
from max.gpu.sync import barrier, syncwarp
import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_arrive,
    cluster_sync,
    cluster_wait,
    load_cluster_smem,
    store_cluster_smem,
)

from nn.attention.gpu.nvidia.sm100.attention_utils import clusters_per_wave
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    RowBound,
    TopkSink,
    _assert_scores_16b_aligned,
    _phi_of,
    _warp_sum_u32,
)

comptime _DTOPK_BLOCK: Int = 512
comptime _DTOPK_KEYS: Int = PERSISTENT_TOPK_MAX_N
comptime _DTOPK_COARSE_BITS: Int = 12
comptime _DTOPK_BINS: Int = 1 << _DTOPK_COARSE_BITS
comptime _DTOPK_FINE_BITS: Int = 16 - _DTOPK_COARSE_BITS
comptime _DTOPK_FINE: Int = 1 << _DTOPK_FINE_BITS
# Round 0's bins a thread owns, and a group: the bins of one warp.
comptime _DTOPK_RUN: Int = _DTOPK_BINS // _DTOPK_BLOCK
comptime _DTOPK_GROUP: Int = _DTOPK_RUN * WARP_SIZE
comptime _DTOPK_GROUPS: Int = _DTOPK_BINS // _DTOPK_GROUP
# What a rank publishes after the fine round: its count above the split bin,
# then the split bin's 16 fine counts. The buffer has one word past them,
# which takes the counts of columns in neither, so counting needs no branch.
comptime _DTOPK_PUB: Int = 1 + _DTOPK_FINE
comptime _DTOPK_MAX_CS: Int = 16
# bf16 columns in a 16 B load, and how many such groups a thread keeps in
# registers across the passes.
comptime _DTOPK_VEC: Int = 8
comptime _DTOPK_CACHED: Int = 4
# The key's low half carries `0xFFFF - g`, so `g` must fit below it.
comptime _DTOPK_G_MASK: UInt32 = 0xFFFF

comptime _DTOPK_CLUSTER_SIZES: List[Int] = [16, 8, 4, 2]

comptime _Group = SIMD[.uint32, _DTOPK_VEC]


@inline(.always)
def _dtopk_sort_items[L: Int]() -> Int:
    """Keys a sorting thread holds for a chunk of `L`: one while the chunk
    fits the block, so the steps run over as many threads as there are keys.
    """
    return max(1, L // _DTOPK_BLOCK)


@inline(.always)
def _dtopk_load(
    row: ImmPointer[BFloat16, ImmUntrackedOrigin], col: Int
) -> _Group:
    """`phi16` of columns `[col, col + 8)`, one 16 B load.

    `col` is a multiple of 8 below the row's live bound, and the row stride
    is a multiple of 8 at least that bound, so the group lies inside the row.
    """
    return _phi_of[.bfloat16, .uint16, _DTOPK_VEC](
        bitcast[.uint16, _DTOPK_VEC](
            row.unsafe_load[width=_DTOPK_VEC, alignment=16](col)
        )
    ).cast[.uint32]()


@inline(.always)
def _dtopk_cas[T: Int, a: Int, b: Int](mut v: SIMD[.uint32, T], desc: Bool):
    """Orders lanes `a < b` of `v`: the larger first when `desc`."""
    var hi = max(v[a], v[b])
    var lo = min(v[a], v[b])
    v[a] = hi if desc else lo
    v[b] = lo if desc else hi


@inline(.always)
def _dtopk_sort_desc[
    L: Int, b_origin: MutOrigin
](
    keys_in: SIMD[.uint32, _dtopk_sort_items[L]()],
    buf: Pointer[UInt32, b_origin, address_space=.SHARED],
) -> SIMD[.uint32, _dtopk_sort_items[L]()]:
    """Bitonic sort of `L` keys, descending.

    Thread `t < L / T` holds positions `[T t, T t + T)` and gets them back
    sorted; the rest of the block only joins the barriers, which every thread
    must call. Partners inside a thread compare in registers, inside a warp
    through shuffles, and across warps through `buf`, which needs `2 * L`
    words: alternating halves lets one barrier per cross-warp step stand in
    for two.

    A block whose merge direction is ascending is sorted as the complement of
    its keys descending, so every step of a level spanning threads applies
    one rule: the lower position keeps the larger key.
    """
    comptime T = _dtopk_sort_items[L]()
    comptime assert L % T == 0 and L // T <= _DTOPK_BLOCK
    var tid = thread_idx.x
    var active = tid < L // T
    # Shuffles need whole warps: a chunk shorter than a warp still runs one.
    var wactive = tid < max(L // T, WARP_SIZE)
    var i0 = T * tid
    var v = keys_in
    var flip = 0
    comptime for ls in range(1, log2_floor(L) + 1):
        comptime s = 1 << ls
        comptime if s < T:
            # Directions are per position inside the thread, known at compile
            # time since `i0` is a multiple of `T`.
            if active:
                comptime for step in range(ls):
                    comptime d = 1 << (ls - 1 - step)
                    comptime for a in range(T):
                        comptime if (a & d) == 0:
                            _dtopk_cas[T, a, a + d](v, (a & s) == 0)
        else:
            var mask = SIMD[.uint32, T](UInt32.MAX if (i0 & s) != 0 else 0)
            v ^= mask
            comptime for step in range(ls):
                comptime d = 1 << (ls - 1 - step)
                comptime if d < T:
                    if active:
                        comptime for a in range(T):
                            comptime if (a & d) == 0:
                                _dtopk_cas[T, a, a + d](v, True)
                else:
                    var p = v
                    comptime if d < T * WARP_SIZE:
                        if wactive:
                            comptime for a in range(T):
                                p[a] = warp.shuffle_xor(v[a], UInt32(d // T))
                    else:
                        var half = buf.unsafe_offset(flip * L)
                        if active:
                            half.unsafe_store[width=T](i0, v)
                        barrier()
                        if active:
                            p = half.unsafe_load[width=T](i0 ^ d)
                        flip ^= 1
                    if active:
                        v = max(v, p) if (i0 & d) == 0 else min(v, p)
            v ^= mask
    return v


@inline(.always)
def _dtopk_place[
    S: TopkSink,
    //,
    CS: Int,
    x_origin: MutOrigin,
    c_origin: MutOrigin,
    a_origin: MutOrigin,
    p_origin: MutOrigin,
](
    keys: SIMD[.uint32, _dtopk_sort_items[_DTOPK_KEYS // CS]()],
    xbuf: Pointer[UInt32, x_origin, address_space=.SHARED],
    chunk: Pointer[UInt32, c_origin, address_space=.SHARED],
    allk: Pointer[UInt32, a_origin, address_space=.SHARED],
    pairs: Pointer[UInt32, p_origin, address_space=.SHARED],
    sink: S,
    out_base: Int,
    rank: Int,
    M: Int,
    K: Int,
    cols_are_g: Bool,
):
    """Sorts this rank's chunk of keys and writes each to its output slot.

    `keys` are thread `t`'s positions `[T t, T t + T)` of the chunk, `0`
    where a position holds no key: a real key's low half is at least
    `0xFFFF - 2047`, so `0` sorts below every one. A key's column is `g`
    itself when `cols_are_g`, else `pairs[2 (g - rank L) + 1]`. Every rank
    must call this, cluster-uniformly; it ends with this rank's part of the
    cluster's last barrier.
    """
    comptime L = _DTOPK_KEYS // CS
    comptime T = _dtopk_sort_items[L]()
    var tid = thread_idx.x
    var sorted = _dtopk_sort_desc[L](keys, xbuf)
    if tid < L // T:
        chunk.unsafe_store[width=T](T * tid, sorted)
    comptime if CS > 1:
        cluster_sync()
        # Every rank gathers every sorted chunk, then lets the cluster go:
        # nothing reads a peer's shared memory after this. The arrive has to
        # release those reads -- a peer's matching wait is what keeps its
        # shared memory alive until they land, so the relaxed form would let it
        # leave first.
        comptime per = _DTOPK_KEYS // _DTOPK_BLOCK
        var w = per * tid
        allk.unsafe_store[width=per, alignment=16](
            w,
            load_cluster_smem[.uint32, per](
                chunk.unsafe_offset(w % L), UInt32(w // L)
            ),
        )
        cluster_arrive()
        barrier()
    else:
        barrier()

    # A key's slot is its index in its own chunk plus the keys above it in
    # every other chunk, found by binary search. `tpk` threads share a key,
    # each searching every `tpk`-th chunk.
    comptime tpk = max(1, _DTOPK_BLOCK // L)
    comptime kpt = max(1, L // _DTOPK_BLOCK)
    var sub = tid % tpk
    comptime for kk in range(kpt):
        var li = tid // tpk + kk * (_DTOPK_BLOCK // tpk)
        var key = chunk[li]
        var above = UInt32(0)
        comptime if CS > 1:
            comptime for c0 in range(0, CS, tpk):
                var c = c0 + sub
                if c < CS and c != rank:
                    var arr = allk.unsafe_offset(c * L)
                    var pos = 0
                    comptime for st in range(log2_floor(L) + 1):
                        comptime step = L >> st
                        if pos + step <= L and arr[pos + step - 1] > key:
                            pos += step
                    above += UInt32(pos)
            comptime for o in range(log2_floor(tpk)):
                above += warp.shuffle_xor(above, UInt32(1 << o))
        if sub == 0 and key != 0:
            var g = Int(_DTOPK_G_MASK - (key & _DTOPK_G_MASK))
            # The mask is the identity on a real key, which is in this
            # chunk; it keeps the load in bounds where the compiler hoists
            # it above the test, as it does in the one-rank instantiation.
            var col = Int32(g) if cols_are_g else Int32(
                Int(pairs[2 * ((g - rank * L) & (L - 1)) + 1])
            )
            sink.store(out_base + li + Int(above), col)
    # Slots past the selection.
    var q = rank * L + tid
    while q < (rank + 1) * L:
        if q >= M and q < K:
            sink.store(out_base + q, Int32(-1))
        q += _DTOPK_BLOCK
    comptime if CS > 1:
        cluster_wait()


@inline(.always)
def _dtopk_live(lo: Int, hi: Int, gi: Int) -> Int:
    """Live columns of a thread's group `gi`, for its range `[lo, hi)`."""
    return min(_DTOPK_VEC, max(0, hi - lo - gi * _DTOPK_VEC))


@inline(.always)
def _dtopk_hist_group[
    h_origin: MutOrigin
](
    phi: _Group,
    live: Int,
    hist: Pointer[UInt32, h_origin, address_space=.SHARED],
):
    """Counts a group into round 0's bins; dead columns go to the spare bin."""
    comptime for j in range(_DTOPK_VEC):
        var b = Int(phi[j] >> UInt32(_DTOPK_FINE_BITS))
        _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
            ordering=Ordering.RELAXED
        ](
            hist.unsafe_offset(b if j < live else _DTOPK_BINS),
            UInt32(1),
        )


@inline(.always)
def _dtopk_fine_group[
    u_origin: MutOrigin
](
    phi: _Group,
    live: Int,
    split: UInt32,
    mut n_above: UInt32,
    pub: Pointer[UInt32, u_origin, address_space=.SHARED],
):
    """Counts a group's columns above the split bin, and its columns in the
    split bin into their fine bins; the rest go to the spare word."""
    comptime for j in range(_DTOPK_VEC):
        var cb = phi[j] >> UInt32(_DTOPK_FINE_BITS)
        var ok = j < live
        n_above += UInt32(1) if ok and cb > split else UInt32(0)
        var w = 1 + Int(phi[j] & UInt32(_DTOPK_FINE - 1))
        _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
            ordering=Ordering.RELAXED
        ](
            pub.unsafe_offset(w if ok and cb == split else _DTOPK_PUB),
            UInt32(1),
        )


@inline(.always)
def _dtopk_count_group(
    phi: _Group, live: Int, t: UInt32, mut n_gt: UInt32, mut n_eq: UInt32
):
    """Counts a group's columns above and at the threshold."""
    comptime for j in range(_DTOPK_VEC):
        var ok = j < live
        n_gt += UInt32(1) if ok and phi[j] > t else UInt32(0)
        n_eq += UInt32(1) if ok and phi[j] == t else UInt32(0)


@inline(.always)
def _dtopk_emit_group[
    CS: Int, p_origin: MutOrigin
](
    phi: _Group,
    live: Int,
    col0: Int,
    t: UInt32,
    take: UInt32,
    mut eq_i: UInt32,
    mut g: UInt32,
    pairs: Pointer[UInt32, p_origin, address_space=.SHARED],
):
    """Delivers a group's selected `(key, column)` pairs to the ranks that
    own their positions, in column order."""
    comptime L = _DTOPK_KEYS // CS
    comptime for j in range(_DTOPK_VEC):
        if j < live:
            var phi_j = phi[j]
            var sel = phi_j > t
            if phi_j == t:
                sel = eq_i < take
                eq_i += 1
            if sel:
                var pair = SIMD[.uint32, 2](
                    (phi_j << 16) | (_DTOPK_G_MASK - g), UInt32(col0 + j)
                )
                var slot = 2 * (Int(g) % L)
                comptime if CS > 1:
                    store_cluster_smem[.uint32, 2](
                        pairs.unsafe_offset(slot), UInt32(Int(g) // L), pair
                    )
                else:
                    pairs.unsafe_store[width=2, alignment=8](slot, pair)
                g += 1


@inline(.always)
def _topk_cluster_det_impl[
    S: TopkSink, //, CS: Int
](
    in_scores: ImmPointer[BFloat16, ImmUntrackedOrigin],
    sink: S,
    N: Int,
    K: Int,
    count: Int,
):
    """The kernel body: one row per cluster; see the module docstring.

    `count` is the row's live column count, already clamped to `[0, N]`.
    """
    comptime L = _DTOPK_KEYS // CS
    comptime T = _dtopk_sort_items[L]()
    var tid = thread_idx.x
    var row_idx = block_idx.x
    var rank: Int
    comptime if CS > 1:
        rank = Int(block_rank_in_cluster())
    else:
        rank = 0
    var row = in_scores.unsafe_offset(row_idx * N)
    var out_base = row_idx * K

    # One spare bin past the histogram takes the dead columns' counts.
    var hist = unsafe_stack_allocation[
        _DTOPK_BINS + 4, UInt32, alignment=16, address_space=.SHARED
    ]()
    # `(key, column)` of each position of this rank's chunk, written by the
    # rank that selected it.
    var pairs = unsafe_stack_allocation[
        2 * L, UInt32, alignment=16, address_space=.SHARED
    ]()
    var chunk = unsafe_stack_allocation[
        L, UInt32, alignment=16, address_space=.SHARED
    ]()
    # Once the threshold is known nothing reads the histogram again, so the
    # sort's exchange buffer and the gathered chunks take it over.
    comptime assert 2 * L + (_DTOPK_KEYS if CS > 1 else 0) <= _DTOPK_BINS
    var xbuf = hist
    var allk = hist.unsafe_offset(_DTOPK_BINS - _DTOPK_KEYS)
    var wsum = unsafe_stack_allocation[
        _DTOPK_GROUPS, UInt32, address_space=.SHARED
    ]()
    var pub = unsafe_stack_allocation[
        _DTOPK_PUB + 1, UInt32, address_space=.SHARED
    ]()
    var peers = unsafe_stack_allocation[
        _DTOPK_PUB * _DTOPK_MAX_CS, UInt32, address_space=.SHARED
    ]()
    # [0] split group, [1] count above it, [2] split bin, then fine bin, [3]
    # count above the split bin, [4] threshold `phi16`, [5] where this rank's
    # positions start, [6] how many of its threshold columns it takes, [7]
    # count above the threshold.
    var ctl = unsafe_stack_allocation[8, UInt32, address_space=.SHARED]()

    var a = SIMD[.uint32, T](0)
    var select_all = count <= K
    if select_all:
        # Every live column fits: the selection is the whole row in column
        # order, so `g` is the column and rank `r` loads its own chunk. The
        # launcher guarantees `N >= _DTOPK_KEYS`, so every position is inside
        # the row and the load need not wait for the bound; positions past
        # it are dropped.
        #
        # TODO(DISTINF-832): the keys are built without branches because
        # with the general path's equivalent load under a per-key branch,
        # the one-rank instantiation writes wrong slots (2045 of 2048 at a
        # 4000-key row, every run). Ruled out: both branch-free forms are
        # correct; a `barrier()` after the guarded load is correct too;
        # racecheck is clean at one rank; the PTX of both forms issues the
        # loads after the same `bar.sync`, predicated or `selp`'d, and so
        # does the SASS. What differs is the SASS's reconvergence: the
        # branchy form opens one `BSSY B0` in the prologue and closes it
        # only after the guarded loads, so the selection's barriers and
        # loads all run inside one open scope, which the branch-free form
        # does not have. That points at ptxas, not a missing fence; the
        # dispatch never runs one rank, and the tests cover every size.
        var active = tid < L // T
        var g0 = rank * L + T * (tid % (L // T))
        var phi = _phi_of[.bfloat16, .uint16, T](
            bitcast[.uint16, T](row.unsafe_load[width=T](g0))
        ).cast[.uint32]()
        comptime for j in range(T):
            var key = (phi[j] << 16) | (_DTOPK_G_MASK - UInt32(g0 + j))
            a[j] = key if active and g0 + j < count else UInt32(0)
    else:
        # This rank's slice, in 16 B units balanced over the ranks, and this
        # thread's contiguous share of it: column order is (thread, column).
        var units = ceildiv(count, _DTOPK_VEC)
        var base_units = units // CS
        var extra = units - base_units * CS
        var lo = (rank * base_units + min(rank, extra)) * _DTOPK_VEC
        var hi = min(
            lo + (base_units + (1 if rank < extra else 0)) * _DTOPK_VEC,
            count,
        )
        var per_thread = ceildiv(
            ceildiv(max(hi - lo, 0), _DTOPK_VEC), _DTOPK_BLOCK
        )
        var my_lo = min(lo + tid * per_thread * _DTOPK_VEC, hi)
        var my_hi = min(my_lo + per_thread * _DTOPK_VEC, hi)
        var ng = ceildiv(my_hi - my_lo, _DTOPK_VEC)
        # A thread's first groups stay in registers for every pass; only a
        # thread with more re-reads the rest, from L1 after the first pass.
        var cached = Array[_Group, _DTOPK_CACHED](fill=_Group(0))
        comptime for gi in range(_DTOPK_CACHED):
            if gi < ng:
                cached[gi] = _dtopk_load(row, my_lo + gi * _DTOPK_VEC)

        comptime run = _DTOPK_RUN
        hist.unsafe_store[width=run, alignment=16](
            run * tid, SIMD[.uint32, run](0)
        )
        # The spare words too: nothing reads them, but a count that is never
        # zeroed is one refactor away from being read.
        if tid == 0:
            hist[_DTOPK_BINS] = 0
        if tid <= _DTOPK_PUB:
            pub[tid] = 0
        barrier()

        comptime for gi in range(_DTOPK_CACHED):
            if gi < ng:
                _dtopk_hist_group(
                    cached[gi], _dtopk_live(my_lo, my_hi, gi), hist
                )
        for gi in range(_DTOPK_CACHED, ng):
            _dtopk_hist_group(
                _dtopk_load(row, my_lo + gi * _DTOPK_VEC),
                _dtopk_live(my_lo, my_hi, gi),
                hist,
            )
        barrier()

        # Warp `w`'s threads own the bins of group `w`.
        var mine = hist.unsafe_load[width=run, alignment=16](run * tid)
        var wtot = _warp_sum_u32(mine.reduce_add())
        if lane_id() == 0:
            wsum[Int(warp_id())] = wtot
        comptime if CS > 1:
            cluster_sync()
        else:
            barrier()

        var need = UInt32(K)
        # Lane `l` of warp 0 takes group `_DTOPK_GROUPS - 1 - l`, so a prefix
        # over lanes is a suffix over groups.
        comptime assert _DTOPK_GROUPS <= WARP_SIZE
        if warp_id() == 0:
            var grp = _DTOPK_GROUPS - 1 - Int(lane_id())
            var gv = UInt32(0)
            if grp >= 0:
                comptime if CS > 1:
                    comptime for p in range(CS):
                        gv += load_cluster_smem[.uint32, 1](
                            wsum.unsafe_offset(grp), UInt32(p)
                        )[0]
                else:
                    gv = wsum[grp]
            var gincl = warp.prefix_sum(gv)
            var gexcl = gincl - gv
            if grp >= 0 and gexcl < need and need <= gincl:
                ctl[0] = UInt32(grp)
                ctl[1] = gexcl
        barrier()
        var split_group = Int(ctl[0])
        var above_group = ctl[1]

        # The split group's bins, the top one first.
        var bv = UInt32(0)
        var bin = split_group * _DTOPK_GROUP + _DTOPK_GROUP - 1 - tid
        if tid < _DTOPK_GROUP:
            comptime if CS > 1:
                comptime for p in range(CS):
                    bv += load_cluster_smem[.uint32, 1](
                        hist.unsafe_offset(bin), UInt32(p)
                    )[0]
            else:
                bv = hist[bin]
        var bincl = block.prefix_sum[block_size=_DTOPK_BLOCK](bv)
        var bexcl = bincl - bv
        var need_g = need - above_group
        if tid < _DTOPK_GROUP and bexcl < need_g and need_g <= bincl:
            ctl[2] = UInt32(bin)
            ctl[3] = above_group + bexcl
        barrier()
        var split = ctl[2]
        var above_split = ctl[3]

        var n_above = UInt32(0)
        comptime for gi in range(_DTOPK_CACHED):
            if gi < ng:
                _dtopk_fine_group(
                    cached[gi],
                    _dtopk_live(my_lo, my_hi, gi),
                    split,
                    n_above,
                    pub,
                )
        for gi in range(_DTOPK_CACHED, ng):
            _dtopk_fine_group(
                _dtopk_load(row, my_lo + gi * _DTOPK_VEC),
                _dtopk_live(my_lo, my_hi, gi),
                split,
                n_above,
                pub,
            )
        n_above = _warp_sum_u32(n_above)
        if lane_id() == 0:
            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](pub, n_above)
        comptime if CS > 1:
            cluster_sync()
            if tid < _DTOPK_PUB * CS:
                peers[tid] = load_cluster_smem[.uint32, 1](
                    pub.unsafe_offset(tid % _DTOPK_PUB),
                    UInt32(tid // _DTOPK_PUB),
                )[0]
        else:
            barrier()
            if tid < _DTOPK_PUB:
                peers[tid] = pub[tid]
        barrier()

        # The threshold, and which ranks take how many columns at it. Every
        # rank reads the same published counts, so every rank agrees.
        comptime assert _DTOPK_FINE <= WARP_SIZE and CS <= WARP_SIZE
        if warp_id() == 0:
            var lane = Int(lane_id())
            # Lane `l` takes fine bin `15 - l`: a prefix over lanes is a
            # suffix over the split bin's values.
            var fb = _DTOPK_FINE - 1 - lane
            var h = UInt32(0)
            if fb >= 0:
                comptime for p in range(CS):
                    h += peers[p * _DTOPK_PUB + 1 + fb]
            var hincl = above_split + warp.prefix_sum(h)
            if fb >= 0 and hincl - h < need and need <= hincl:
                ctl[2] = UInt32(fb)
                ctl[7] = hincl - h
            syncwarp()
            var f = Int(ctl[2])
            var need_eq = need - ctl[7]
            # Lane `p` takes rank `p`: its columns above the threshold and
            # at it. The ranks below take their threshold columns first.
            var gt_p = UInt32(0)
            var eq_p = UInt32(0)
            if lane < CS:
                gt_p = peers[lane * _DTOPK_PUB]
                for f2 in range(f + 1, _DTOPK_FINE):
                    gt_p += peers[lane * _DTOPK_PUB + 1 + f2]
                eq_p = peers[lane * _DTOPK_PUB + 1 + f]
            var eq_before = warp.prefix_sum(eq_p) - eq_p
            var take_p = min(eq_p, need_eq - min(eq_before, need_eq))
            var sel_p = gt_p + take_p
            var g_before = warp.prefix_sum(sel_p) - sel_p
            if lane == rank:
                ctl[5] = g_before
                ctl[6] = take_p
            if lane == 0:
                ctl[4] = (split << UInt32(_DTOPK_FINE_BITS)) | UInt32(f)
        barrier()
        var t = ctl[4]
        var g_base = ctl[5]
        var take = ctl[6]

        # This thread's columns above and at the threshold, packed so one
        # scan places both.
        var n_gt = UInt32(0)
        var n_eq = UInt32(0)
        comptime for gi in range(_DTOPK_CACHED):
            if gi < ng:
                _dtopk_count_group(
                    cached[gi], _dtopk_live(my_lo, my_hi, gi), t, n_gt, n_eq
                )
        for gi in range(_DTOPK_CACHED, ng):
            _dtopk_count_group(
                _dtopk_load(row, my_lo + gi * _DTOPK_VEC),
                _dtopk_live(my_lo, my_hi, gi),
                t,
                n_gt,
                n_eq,
            )
        var packed = UInt64(n_gt) | (UInt64(n_eq) << 32)
        var pexcl = block.prefix_sum[block_size=_DTOPK_BLOCK, exclusive=True](
            packed
        )

        var eq_i = UInt32(pexcl >> 32)
        var g = g_base + UInt32(pexcl & 0xFFFFFFFF) + min(eq_i, take)
        comptime for gi in range(_DTOPK_CACHED):
            if gi < ng:
                _dtopk_emit_group[CS](
                    cached[gi],
                    _dtopk_live(my_lo, my_hi, gi),
                    my_lo + gi * _DTOPK_VEC,
                    t,
                    take,
                    eq_i,
                    g,
                    pairs,
                )
        for gi in range(_DTOPK_CACHED, ng):
            _dtopk_emit_group[CS](
                _dtopk_load(row, my_lo + gi * _DTOPK_VEC),
                _dtopk_live(my_lo, my_hi, gi),
                my_lo + (gi) * _DTOPK_VEC,
                t,
                take,
                eq_i,
                g,
                pairs,
            )

        # Every rank's pairs are in place once every rank has written them.
        comptime if CS > 1:
            cluster_sync()
        else:
            barrier()
        # Without branches; see the select-all load above.
        comptime for j in range(T):
            var pos = T * tid + j
            var kv = pairs[2 * (pos & (L - 1))]
            a[j] = kv if (pos < L and rank * L + pos < K) else UInt32(0)

    _dtopk_place[CS](
        a,
        xbuf,
        chunk,
        allk,
        pairs,
        sink,
        out_base,
        rank,
        count if select_all else K,
        K,
        select_all,
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_DTOPK_BLOCK)),
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__name(t"topk_cluster_det_bounded_cs{CS}")
def _topk_cluster_det_bounded_kernel[
    CS: Int, RB: RowBound
](
    in_scores: ImmPointer[BFloat16, ImmUntrackedOrigin],
    out_idxs: MutPointer[Int32, MutUntrackedOrigin],
    N: Int32,
    K: Int32,
    row_bounds: RB,
):
    """`_topk_cluster_det_impl` with a per-row live-column bound."""
    with PDL():
        var bound, sink = row_bounds.row(block_idx.x, out_idxs)
        _topk_cluster_det_impl[CS](
            in_scores,
            sink,
            Int(N),
            Int(K),
            min(Int(N), max(0, bound)),
        )


def topk_cluster_det_cluster_size[sm_count: Int](rows: Int) -> Int:
    """The widest cluster that keeps every row resident in one wave, or 1.

    A rank holds an SM, so `rows` clusters of `CS` must fit
    `clusters_per_wave`'s per-GPC packing.
    """
    comptime if sm_count == 148 or sm_count == 160:
        comptime for cs in _DTOPK_CLUSTER_SIZES:
            if rows <= clusters_per_wave[cs, sm_count]():
                return cs
    return 1


def topk_cluster_det[
    RB: RowBound
](
    ctx: DeviceContext,
    in_scores: ImmPointer[BFloat16, ImmUntrackedOrigin],
    out_idxs: MutPointer[Int32, MutUntrackedOrigin],
    N: Int,
    K: Int,
    total_seq_len: Int,
    row_bounds: RB,
    cluster_size: Int,
) raises:
    """Launches the deterministic cluster top-k, one row per `cluster_size`
    CTAs.

    Writes what `persistent_topk_block_split[ordered=False,
    deterministic=True]` writes for these rows, slot for slot.

    Args:
        ctx: Device context.
        in_scores: Flat bf16 score buffer `[total_seq_len x N]`, 16 B aligned
            with `N` a multiple of 8 and at least `PERSISTENT_TOPK_MAX_N`.
        out_idxs: Output buffer `[total_seq_len x K]` (int32).
        N: Score columns per row (the row stride).
        K: Top-k count per row, at most `PERSISTENT_TOPK_MAX_N`.
        total_seq_len: Number of rows.
        row_bounds: Per-row live-column counts and output sinks.
        cluster_size: CTAs per row: 1, 2, 4, 8 or 16.

    Raises:
        If `K` exceeds the selection width, the row stride is not a whole
        number of 16 B units of at least `PERSISTENT_TOPK_MAX_N` columns, or
        `cluster_size` is unsupported.
    """
    comptime assert (
        ctx.target._is_sm_9x_or_newer()
        and not ctx.target._is_sm_120x_or_newer()
    ), "topk_cluster_det needs thread-block clusters on NVIDIA SM90+"
    if total_seq_len <= 0 or K <= 0:
        return
    if K > PERSISTENT_TOPK_MAX_N:
        raise Error("topk_cluster_det selects at most ", PERSISTENT_TOPK_MAX_N)
    if N % 8 != 0 or N < PERSISTENT_TOPK_MAX_N:
        raise Error(
            (
                "topk_cluster_det needs a row stride of whole 16 B units and at"
                " least "
            ),
            PERSISTENT_TOPK_MAX_N,
            " columns",
        )
    # The alignment helper is shared with the top-k kernel host and takes its
    # scores as any-origin pointers; this entry point keeps the untracked form.
    _assert_scores_16b_aligned(
        rebind[ImmPointer[BFloat16, ImmutAnyOrigin]](in_scores),
        N,
        total_seq_len,
    )

    @inline(.always)
    def launch[CS: Int]() raises {imm}:
        var cluster = OptionalReg[Dim]()
        comptime if CS > 1:
            cluster = OptionalReg[Dim](Dim(1, CS, 1))
        ctx.enqueue_function[_topk_cluster_det_bounded_kernel[CS, RB]](
            in_scores,
            out_idxs,
            Int32(N),
            Int32(K),
            row_bounds,
            grid_dim=(total_seq_len, CS, 1),
            block_dim=_DTOPK_BLOCK,
            cluster_dim=cluster,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

    if cluster_size == 16:
        launch[16]()
    elif cluster_size == 8:
        launch[8]()
    elif cluster_size == 4:
        launch[4]()
    elif cluster_size == 2:
        launch[2]()
    elif cluster_size == 1:
        launch[1]()
    else:
        raise Error(
            "topk_cluster_det: cluster_size must be 1, 2, 4, 8 or 16, got ",
            cluster_size,
        )
