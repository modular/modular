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
"""Cluster-cooperative top-k for the sparse-attention indexer.

Port of vLLM's `cooperative_topk` structure (`csrc/cooperative_topk.cuh`): one
query row per thread-block cluster of `CS` CTAs, 1024 threads each, combined
through distributed shared memory. A decode batch whose rows under-fill the
GPU gets `CS` times the CTAs on the same rows, which is the whole point -- at
24 rows a B200 leaves 124 of 148 SMs idle while each row's scan runs serially.

Where this deviates from vLLM it is because MAX's contract is different, not
because the port is incomplete:

- No global tie workspace and no truncated tie list. Like vLLM, round 0
  bins a coarse monotone key (`_ctopk_key0`, 16 bins per octave as vLLM's
  fp16-rounded bin) and only the threshold bin's candidates are refined, on
  rank 0; but the candidates gather into rank 0's shared memory over DSMEM,
  and a bin too dense for that list falls back to refining over the whole
  slice rather than dropping candidates. Either way the score half of the key
  resolves fully, so the threshold is an exact score and the plateau at it
  may be cut first-come -- the `ordered=False, deterministic=False` contract
  the GLM indexer ships.
- Scores are read at `in_dtype`, f32 or bf16, and widened exactly on load
  (`_widen_scores`' contract); bf16 halves the score-buffer traffic and the
  staging.
- The streaming flavor can stage its slices with the 1D `cp.async.bulk`
  copy through a two-slot mbarrier pipeline (vLLM's `tma_stream_pass`), on
  SM100+ where that copy compiles and behind an off-by-default switch: the
  plain-load streaming measured faster on B200, so the pipeline is opt-in
  (`use_tma`, pinned by tests) rather than the route. A row
  stride the copy cannot serve (not a whole 16 B unit per row) keeps the
  plain loads, so the correct-at-any-alignment contract holds everywhere.
- Rows carry a live-column bound (`RowBound`) instead of a lengths tensor, and
  every output slot goes through a `TopkSink`.

As in vLLM, a row short enough for one CTA's registers (`_ctopk_solo_cap`)
is selected by rank 0 alone, vLLM's `histogram_4096_topk`, while the other
ranks exit before any cluster synchronization.

One kernel body, comptime on `CS` and the score dtype. The launch is
`grid = (rows, CS)`, `block = 1024`, `cluster = (1, CS, 1)`; `block_idx.x` is
the row and `block_rank_in_cluster()` the rank. A row stride rank 0 covers
puts every row on the mid-length path, so it launches `grid = (rows, 1)`
without a cluster, each CTA its own rank 0, and only the mid-length path's
shared memory.
"""

from std.bit import (
    count_leading_zeros,
    count_trailing_zeros,
    pop_count,
)
from std.collections.optional import OptionalReg
from std.math import align_down, align_up, ceildiv
from std.memory import bitcast
from std.sys import size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import _is_sm_100x_or_newer

from std.atomic import Atomic, Ordering
from shmem.ep_comm import BLOCK_SCOPE

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext, Dim, FuncAttribute
from max.gpu.host.info import B200, H100
from max.gpu.memory import (
    cp_async_bulk_shared_cluster_global,
    external_memory,
    fence_mbarrier_init,
)
from max.gpu.intrinsics import Scope, byte_permute
from max.gpu.sync import barrier, named_barrier, syncwarp
import max.gpu.primitives.warp as warp
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_arrive_relaxed,
    cluster_sync,
    cluster_wait,
    load_cluster_smem,
    store_cluster_smem,
)

from layout.tma_async import SharedMemBarrier

from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import clusters_per_wave
from nn.topk_bitonic import (
    PERSISTENT_TOPK_MAX_N,
    _assert_scores_16b_aligned,
    PlainSink,
    RowBound,
    RowBoundsPtr,
    TopkSink,
    _HSEL_SCAN_ITEMS,
    _V4_ALIGN,
    _elems_per_16b,
    _hsel_sig_bits,
    _phi_of,
    _score_bits,
    _score_pad,
    _warp_sum_u32,
)

# One CTA of this many threads per rank; also the coarse histogram's warp
# count, so the threshold scan below assumes exactly `WARP_SIZE` warps.
comptime _CTOPK_BLOCK: Int = 1024

# Coarse-bin width of the first round. vLLM's `kHistBits`; 1024 bins keeps the
# per-rank histogram at 4 KB and lets the first round fold the whole row.
comptime _CTOPK_HIST_BITS: Int = 10
comptime _CTOPK_BINS: Int = 1 << _CTOPK_HIST_BITS

# The per-block dynamic shared-memory opt-in: the SM's capacity less the 1 KiB
# the hardware reserves per CTA. Every target the kernel admits (SM90 through
# SM11x) shares Hopper's and Blackwell's 228 KiB, so the smaller of the two is
# the budget. A 1024-thread CTA is already one per SM, so filling it costs no
# occupancy.
comptime _CTOPK_SMEM_OPTIN: Int = (
    min(
        H100.shared_memory_per_multiprocessor,
        B200.shared_memory_per_multiprocessor,
    )
    - 1024
)

# Streaming-flavor pipeline slots the staging area is carved into. Two: the
# double buffer with an epilogue prefetch is vLLM's `kStreamingStagesCS4`.
# More stages would buy nothing at these slice lengths -- the scan's
# consumption of a stage is slower than its copy -- and would shrink the
# single-pass capacity the same area serves.
comptime _CTOPK_TMA_STAGES: Int = 2

# Round 0 bins a monotone 10-bit key of the score (`_ctopk_key0`), not the top
# bits of `phi`: 16 bins per octave of |score| over [2^-16, 2^16), one half of
# the bins per sign, saturating outside. `phi`'s top 10 bits are 2 bins per
# octave, so a Gaussian's threshold bin holds 1400-4200 columns of a 16K-164K
# row; this key holds 200-800 (vLLM bins its fp16-rounded score for the same
# reason). `_CTOPK_KEY0_LO` is the 12-bit `exponent:mantissa[22:19]` of 2^-16.
comptime _CTOPK_KEY0_LO: UInt32 = 111 * 16
comptime _CTOPK_KEY0_HALF: UInt32 = UInt32(_CTOPK_BINS // 2)

# Threshold-bin candidates a peer may stage as `(column, phi)` pairs, in
# `selbuf`'s upper half past its `[0, K)` strictly-above columns. Rank 0 writes
# its above columns straight to the output instead, so its whole `selbuf` is
# the cluster's list: `PERSISTENT_TOPK_MAX_N` pairs, its own from the bottom
# and the peers' pushed from the top. The candidate path therefore needs no
# shared memory beyond the scatter staging and leaves the slice staging free to
# grow. At the bench's Gaussian rows the split bin holds 200-800 columns (at
# most 420 on one rank at CS2), against 2048 and 1024 here.
comptime _CTOPK_CAND_CAP: Int = PERSISTENT_TOPK_MAX_N // 2


@inline(.always)
def _ctopk_stage_bytes() -> Int:
    """Bytes the slice staging gets: whatever the opt-in leaves past the fixed
    carve (histograms, scratch, counts, the scatter staging) and the wider
    dtype's mbarrier tail.

    Rounded down to whole scan groups per streaming slot at the wider score,
    so `_ctopk_tma_stage_elems`' slot split holds at either dtype. On B200
    this is 207488 B: 51872 f32 or 103744 bf16 columns per rank.
    """
    comptime granule = (
        _CTOPK_TMA_STAGES * _HSEL_SCAN_ITEMS * size_of[DType.float32]()
    )
    comptime tail = max(
        _ctopk_nbar[DType.float32](), _ctopk_nbar[DType.bfloat16]()
    ) * size_of[SharedMemBarrier]()
    return align_down(
        _CTOPK_SMEM_OPTIN - _ctopk_off_stage[_CTOPK_MAX_CS]() - tail, granule
    )


@inline(.always)
def _ctopk_res_elems[in_dtype: DType]() -> Int:
    """Slice elements the single-pass flavor can hold in staging.

    The staging holds the slice at the score's own width (the rescan widens),
    so the same byte budget buys twice the columns at bf16.
    """
    return _ctopk_stage_bytes() // size_of[in_dtype]()


@inline(.always)
def _ctopk_rounds[in_dtype: DType]() -> Int:
    """Most refinement rounds after round 0, at either refinement site.

    Ten bits per round from the highest bit a round-0 bin leaves open (every
    bin's `phi` range shares at least the top bit) down to the insignificant
    pad `_phi` fills uniformly: at most four at f32 and two at bf16. Below
    the pad a bracket holds exactly one representable key, which is what lets
    the scatter cut the threshold plateau first-come without invalidating the
    selection.
    """
    return ceildiv(_hsel_sig_bits[in_dtype]() - 1, _CTOPK_HIST_BITS)


@inline(.always)
def _ctopk_key0[
    bits_dtype: DType, //, in_dtype: DType, width: Int
](bits: SIMD[bits_dtype, width]) -> SIMD[.uint32, width]:
    """Round 0's bin of each score, from its raw bits.

    The magnitude's `exponent:mantissa[22:19]` clamped to the window above,
    mirrored for a negative score: `512 + m` above zero, `511 - m` below. That
    is non-decreasing in `phi` (so a column in a higher bin scores strictly
    higher), and a bin is a contiguous `phi` range that
    `_ctopk_key0_bounds` recovers, so the scatter classifies on `phi` alone.
    A bf16 reads the same field of its exact f32 widening.
    """
    comptime assert bits_dtype == _score_bits[in_dtype]()
    comptime sh = 16 if in_dtype == .bfloat16 else 0
    var b = bits.cast[.uint32]()
    var s = b >> UInt32(31 - sh)
    var mag = (b >> UInt32(19 - sh)) & UInt32(0xFFF)
    var m = min(
        max(mag, SIMD[.uint32, width](_CTOPK_KEY0_LO)) - _CTOPK_KEY0_LO,
        SIMD[.uint32, width](_CTOPK_KEY0_HALF - 1),
    )
    return m ^ (_CTOPK_KEY0_HALF - s)


@inline(.always)
def _ctopk_key0_bounds(b0: UInt32) -> Tuple[UInt32, UInt32]:
    """The inclusive `phi` range `[lo, hi]` of `_ctopk_key0` bin `b0`.

    Bins at either end of a sign's window run to that side's limit, so their
    range is wider than the 2^19-aligned block of an interior bin.
    """
    var pos = b0 >= _CTOPK_KEY0_HALF
    var m = b0 - _CTOPK_KEY0_HALF if pos else _CTOPK_KEY0_HALF - 1 - b0
    var mlo = UInt32(0) if m == 0 else (_CTOPK_KEY0_LO + m) << 19
    var mhi = (
        UInt32(0x7FFFFFFF) if m
        == _CTOPK_KEY0_HALF - 1 else ((_CTOPK_KEY0_LO + m + 1) << 19) - 1
    )
    if pos:
        return (mlo | 0x80000000, mhi | 0x80000000)
    return (UInt32(0x7FFFFFFF) - mhi, UInt32(0x7FFFFFFF) - mlo)


@inline(.always)
def _ctopk_tma_stage_elems[in_dtype: DType]() -> Int:
    """Elements per streaming pipeline slot: the staging area split evenly.

    A whole number of 16 B units and of scan groups, so every slot boundary
    both issues a whole `cp.async.bulk` and leaves no group straddling a slot.
    """
    comptime elems = _ctopk_res_elems[in_dtype]() // _CTOPK_TMA_STAGES
    comptime assert (
        elems % _HSEL_SCAN_ITEMS == 0
        and elems % _elems_per_16b[in_dtype]() == 0
    ), "a TMA stage must hold a whole number of 16 B units and of scan groups"
    return elems


@inline(.always)
def _ctopk_nbar[in_dtype: DType]() -> Int:
    """Streaming mbarriers: a fresh pair per pass, so no phase state crosses
    a pass boundary. Round 0, the candidate scatter, each multi-round
    refinement round and the multi-round scatter."""
    return (_ctopk_rounds[in_dtype]() + 3) * _CTOPK_TMA_STAGES


# Offsets into the one dynamic shared-memory block the kernel carves up. Order
# only matters for alignment: the scatter staging and the slice staging both
# load as 16 B vectors, so their offsets are 16 B aligned. Round 0 owns its
# histogram (`HIST0`): peers read it after the round's one cluster barrier,
# so nothing may write it again, and every later round bins into `HIST`.
# `WSUM` carries round 0's 32-bin group totals to the peers the same way
# before it is threshold-scan scratch again.
comptime _CTOPK_OFF_HIST = 0
comptime _CTOPK_OFF_HIST0 = 4 * _CTOPK_BINS
comptime _CTOPK_OFF_WSUM = 8 * _CTOPK_BINS
comptime _CTOPK_OFF_MTCH = _CTOPK_OFF_WSUM + 4 * WARP_SIZE
comptime _CTOPK_OFF_CTL = _CTOPK_OFF_MTCH + 4 * 4
comptime _CTOPK_OFF_COUNTS = _CTOPK_OFF_CTL + 4 * 8
# Widest cluster `topk_cluster_split` launches.
comptime _CTOPK_MAX_CS: Int = 16
# The candidate path's counts exchange: two words per rank, then its
# mbarriers, `cbar` for the counts and `pbar` (rank 0's) for the pairs.
comptime _CTOPK_OFF_CBAR = _CTOPK_OFF_COUNTS + 4 * 2 * _CTOPK_MAX_CS
comptime _CTOPK_OFF_PBAR = _CTOPK_OFF_CBAR + size_of[SharedMemBarrier]()


@inline(.always)
def _ctopk_off_sel[CS: Int]() -> Int:
    """Offset of the scatter staging: above-class at `[0, K)`, ties above it."""
    comptime assert CS <= _CTOPK_MAX_CS, "the counts area holds every rank"
    comptime assert _CTOPK_OFF_CBAR % 8 == 0, "an mbarrier is 8 B aligned"
    return align_up(_CTOPK_OFF_PBAR + size_of[SharedMemBarrier](), _V4_ALIGN)


@inline(.always)
def _ctopk_off_stage[CS: Int]() -> Int:
    return _ctopk_off_sel[CS]() + 4 * 2 * PERSISTENT_TOPK_MAX_N


@inline(.always)
def _ctopk_off_mbar[in_dtype: DType, CS: Int]() -> Int:
    """Offset of the streaming mbarriers, past the single-pass staging."""
    return (
        _ctopk_off_stage[CS]()
        + _ctopk_res_elems[in_dtype]() * size_of[in_dtype]()
    )


@inline(.always)
def _ctopk_smem_bytes[in_dtype: DType, CS: Int]() -> Int:
    """Dynamic shared-memory bytes one rank requests, for the launch.

    The mbarrier tail is requested at every target, not only where the TMA
    streaming compiles, so the kernel's carve math never runs past the
    allocation on a target that cannot use it.
    """
    comptime bytes = (
        _ctopk_off_mbar[in_dtype, CS]()
        + _ctopk_nbar[in_dtype]() * size_of[SharedMemBarrier]()
    )
    comptime assert (
        bytes <= _CTOPK_SMEM_OPTIN
    ), "the carve overflows the per-block shared-memory opt-in"
    return bytes


@inline(.always)
def _ctopk_hist_reduce[
    CS: Int, nbins: Int, h_origin: MutOrigin
](hist: Pointer[UInt32, h_origin, address_space=.SHARED]):
    """Sums the CS per-rank histograms into every rank's copy, over DSMEM.

    vLLM's `dsmem_hist_reduce`, generalized to any bin count and any `CS`:
    each CTA is the sole writer of one segment of the bins
    (`ceildiv(nbins, CS)` wide), one lane per bin reading that bin from all
    CS peers, summing, and writing the sum back to every peer. Sole-writer
    ownership is the point: a bin is read and written by exactly one lane of
    one CTA, so no CTA's store can land under another CTA's load of the same
    remote slot -- a scheme where every CTA folds every bin races exactly
    there and double-counts. One lane per bin, rather than vLLM's CS lanes
    folded by a sub-warp `shuffle_xor`, keeps CS independent loads in flight
    and needs no power-of-two `CS`.

    Every CTA ends up holding the complete global histogram, so the threshold
    scan after it is block- and cluster-uniform. The entries and the trailing
    exit are `cluster_sync`s, so callers reach this converged with their
    round's histogram written and leave it converged with the reduced bins
    visible. Only the multi-round fallback, which rebins one buffer every
    round, uses this; round 0 reads its peers' bins in place
    (`_ctopk_gather_threshold`, or `_ctopk_hist_gather` at 2 and 3 ranks).
    """
    cluster_sync()
    var tx = Int(thread_idx.x)
    comptime kLocal = ceildiv(nbins, CS)
    var b = kLocal * Int(block_rank_in_cluster()) + tx
    if tx < kLocal and b < nbins:
        var v = UInt32(0)
        comptime for p in range(CS):
            v += load_cluster_smem[.uint32, 1](
                hist.unsafe_offset(b), UInt32(p)
            )[0]
        comptime for p in range(CS):
            store_cluster_smem[.uint32, 1](
                hist.unsafe_offset(b), UInt32(p), SIMD[.uint32, 1](v)
            )
    cluster_sync()


@inline(.always)
def _ctopk_cluster_sync_cta_release():
    """A cluster barrier whose arrive releases only shared::cta accesses.

    `fence.release.sync_restrict::shared::cta.cluster` then a relaxed
    arrive: a release at cluster scope restricted to this CTA's own shared
    memory, which ptxas lowers to `MEMBAR.ALL.CTA` instead of the default
    arrive's `MEMBAR.ALL.GPU`. The wait keeps its default acquire, so peers'
    released writes are visible to the loads after it. Only for a barrier
    with no remote (`shared::cluster`) or global access before it that a
    peer depends on.

    PTX ISA 9.4: the fence's effect "only applies to operations performed on
    objects in .shared::cta state space" (section 9.7.15.4, `fence`), and a
    release fence before a relaxed `barrier.cluster.arrive`, then
    `barrier.cluster.wait`, is the ISA's own pairing in the
    `clusterlaunchcontrol.try_cancel` example (section 9.7.15.18). Tests
    cannot tell the fence from no fence; its sufficiency rests on that text.

    `.sync_restrict` needs PTX ISA 8.6, but the sm_90a target emits 8.5, so
    SM90 falls back to the default (release-arrive) `cluster_sync`.
    """
    comptime if _is_sm_100x_or_newer():
        inlined_assembly[
            "fence.release.sync_restrict::shared::cta.cluster;",
            NoneType,
            has_side_effect=True,
            constraints="",
        ]()
        cluster_arrive_relaxed()
        cluster_wait()
    else:
        cluster_sync()


@inline(.always)
def _ctopk_opaque(x: UInt32) -> UInt32:
    """Returns `x` through a copy LLVM cannot see through.

    LLVM otherwise rebuilds the slice length from its parts at every loop
    that tests it, keeping three of them live across the whole kernel. Past
    8 ranks, where the two-warp gather adds its own pressure, the f32
    bounded kernel spilled them under ptxas 13.1. At 8 ranks or fewer the
    copy is left out: it only reshuffles registers there, and it cost the
    bf16 single-CTA path up to 2% at 5 ranks.
    """
    return inlined_assembly["mov.u32 $0, $1;", UInt32, constraints="=r,r"](x)


@inline(.always)
def _ctopk_st_async(
    local_dst: Pointer[mut=True, UInt32, _, address_space=.SHARED],
    local_bar: Pointer[mut=True, SharedMemBarrier, _, address_space=.SHARED],
    peer: UInt32,
    val: SIMD[.uint32, 2],
):
    """Stores `val` to peer `peer`'s copy of `local_dst` with `st.async`,
    completing 8 bytes of transaction on the peer's copy of `local_bar`.

    The peer observes the store by waiting on that mbarrier (acquire at
    cluster scope); no cluster barrier, and so no device-scope fence, is
    involved.
    """
    inlined_assembly[
        """{
        .reg .b32 ra, rb;
        mapa.shared::cluster.u32 ra, $0, $2;
        mapa.shared::cluster.u32 rb, $1, $2;
        st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.u32 [ra], {$3, $4}, [rb];
        }""",
        NoneType,
        constraints="r,r,r,r,r",
        has_side_effect=True,
    ](
        UInt32(Int(local_dst)),
        UInt32(Int(local_bar)),
        peer,
        val[0],
        val[1],
    )


@inline(.always)
def _ctopk_hist_gather[
    CS: Int, peer_prefix: Bool, h_origin: MutOrigin
](hist: Pointer[UInt32, h_origin, address_space=.SHARED], rank: Int) -> Tuple[
    UInt32, UInt32
]:
    """Bin `thread_idx.x` of round 0's cluster-wide histogram, over DSMEM,
    and with `peer_prefix` its count over peers `1..rank-1`.

    Every CTA reads every peer's copy of its own thread's bin and sums them
    in a register, so nothing is written remotely and the round costs one
    cluster barrier, where `_ctopk_hist_reduce`'s write-back needs a second.
    The prefix is where this rank's candidates go in rank 0's list, should
    the bin be the split bin.

    The caller must have passed a cluster barrier after every rank finished
    binning `hist`, and no rank may write `hist` again: it is round 0's own.
    """
    var tx = Int(thread_idx.x)
    var v = UInt32(0)
    var pre = UInt32(0)
    comptime for p in range(CS):
        var x = load_cluster_smem[.uint32, 1](
            hist.unsafe_offset(tx), UInt32(p)
        )[0]
        v += x
        comptime if peer_prefix and p > 0:
            if p < rank:
                pre += x
    return (v, pre)


@inline(.always)
def _ctopk_publish_sums[
    h_origin: MutOrigin, s_origin: MutOrigin
](
    hist: Pointer[UInt32, h_origin, address_space=.SHARED],
    sums: Pointer[UInt32, s_origin, address_space=.SHARED],
):
    """Writes `sums[w]`, the total of round 0's bins `[32w, 32w + 32)`, the
    bins warp `w`'s lanes own, for `_ctopk_gather_threshold`'s first level.

    Opens with a block barrier so `hist` is complete; the caller's cluster
    barrier after it publishes `sums` to the peers."""
    barrier()
    var tx = UInt32(thread_idx.x)
    var t = _warp_sum_u32(hist[Int(tx)])
    if UInt32(lane_id()) == 0:
        sums[Int(tx // UInt32(WARP_SIZE))] = t


@inline(.always)
def _ctopk_gather_threshold[
    CS: Int,
    h_origin: MutOrigin,
    s_origin: MutOrigin,
    x_origin: MutOrigin,
    m_origin: MutOrigin,
](
    hist: Pointer[UInt32, h_origin, address_space=.SHARED],
    sums: Pointer[UInt32, s_origin, address_space=.SHARED],
    scratch: Pointer[UInt32, x_origin, address_space=.SHARED],
    mtch: Pointer[UInt32, m_origin, address_space=.SHARED],
    need: UInt32,
    rank: UInt32,
):
    """Round 0's split bin over the cluster's histograms `hist`, read over
    DSMEM, recorded to `mtch` as `_ctopk_find_threshold` does, with the
    bin's count over peers `1..rank-1` in `mtch[3]`: where this rank's
    candidates go in rank 0's list.

    One warp reads the peers in two levels. The first sums every rank's
    `sums` (`_ctopk_publish_sums`) and finds the 32-bin group holding the
    split bin; the second sums that group's bins, and the lane whose bin has
    `count(> b) < need <= count(>= b)` records `(b, above, equal, pre)`.
    That reads `64 * CS` words a rank, where reading every bin from every
    peer read `1024 * CS`, at the price of a second dependent DSMEM round
    trip and the block barrier in `_ctopk_publish_sums`.

    Past 8 peers two warps read half the peers each, so no lane holds more
    than 8 loads in flight at the register ceiling (one warp spilled the
    bounded f32 10-rank kernel under the driver JIT). The halves meet in
    `scratch[0, 192)`, which must be zero on entry and is zero again on
    exit: the caller passes the refinement histogram, zeroed before round 0
    and next read by rank 0's first refinement round.

    The caller must have passed a cluster barrier after every rank published
    its `sums`, and no rank may write `hist` or `sums` again until every peer
    is past this read (rank 0's refinement and the multi-round path reuse
    `sums` only after a barrier or mbarrier wait that orders them so). The
    trailing block barrier publishes `mtch`.
    """
    comptime NW = 1 if CS <= 8 else 2
    comptime PW = CS // NW
    comptime assert NW * PW == CS, "the peers split evenly over the warps"
    var tx = UInt32(thread_idx.x)
    if tx < UInt32(NW * WARP_SIZE):
        var lane = UInt32(lane_id())
        var w = tx // UInt32(WARP_SIZE)
        var t = UInt32(0)
        comptime for q in range(PW):
            t += load_cluster_smem[.uint32, 1](
                sums.unsafe_offset(Int(lane)), w * UInt32(PW) + UInt32(q)
            )[0]
        comptime if NW > 1:
            scratch[Int(tx)] = t
            named_barrier[Int32(NW * WARP_SIZE)](1)
            t = 0
            comptime for k in range(NW):
                t += scratch[k * WARP_SIZE + Int(lane)]
        var tsum = warp.prefix_sum(t)
        var above = warp.shuffle_idx(tsum, UInt32(WARP_SIZE - 1)) - tsum
        var g = count_trailing_zeros(
            warp.vote[DType.uint32](above < need and above + t >= need)
        )
        # `count(> b) = count(> group) + group total - inclusive prefix`.
        var gtop = warp.shuffle_idx(above + t, g)
        var b = g * UInt32(WARP_SIZE) + lane
        var v = UInt32(0)
        var pre = UInt32(0)
        comptime for q in range(PW):
            var p = w * UInt32(PW) + UInt32(q)
            var x = load_cluster_smem[.uint32, 1](
                hist.unsafe_offset(Int(b)), p
            )[0]
            v += x
            if p > 0 and p < rank:
                pre += x
        comptime if NW > 1:
            comptime o = NW * WARP_SIZE
            scratch[o + Int(tx)] = v
            scratch[2 * o + Int(tx)] = pre
            named_barrier[Int32(NW * WARP_SIZE)](1)
            if w == 0:
                v = 0
                pre = 0
                comptime for k in range(NW):
                    v += scratch[o + k * WARP_SIZE + Int(lane)]
                    pre += scratch[2 * o + k * WARP_SIZE + Int(lane)]
                comptime for k in range(3 * NW):
                    scratch[k * WARP_SIZE + Int(lane)] = 0
        if w == 0:
            var vsum = warp.prefix_sum(v)
            var babove = gtop - vsum
            if babove < need and babove + v >= need:
                mtch[0] = b
                mtch[1] = babove
                mtch[2] = v
                mtch[3] = pre
    barrier()


@inline(.always)
def _ctopk_find_threshold[
    nbins: Int,
    with_pre: Bool,
    m_origin: MutOrigin,
](
    value: UInt32,
    wsum: Pointer[UInt32, m_origin, address_space=.SHARED],
    mtch: Pointer[UInt32, m_origin, address_space=.SHARED],
    need: Int,
    pre: UInt32,
):
    """Finds the split bin, vLLM's `find_threshold`, on the reduced histogram.

    `value` is this thread's bin of the reduced histogram (0 past `nbins`).
    Exactly one bin `b` has `count(> b) < need <= count(>= b)`, because
    `count(>= b)` is non-increasing in `b` and `need` lies within the
    histogram's total; that one thread records `(b, above, equal)` to `mtch`.
    Every thread derives the same record, so the bracket update after the
    trailing barrier is block-uniform without a broadcast.

    `wsum` needs `WARP_SIZE` slots and `mtch` three, four `with_pre`, which
    records the split bin's `pre` (`_ctopk_hist_gather`'s peer prefix) in
    `mtch[3]`; both are scratch.
    """
    # 32-bit throughout: `Int` lane and warp indices cost 64-bit compares.
    var tx = UInt32(thread_idx.x)
    var lane = UInt32(lane_id())
    var wid = tx // UInt32(WARP_SIZE)
    var winc = warp.prefix_sum(value)
    if lane == UInt32(WARP_SIZE - 1):
        wsum[Int(wid)] = winc
    barrier()
    var tmp = wsum[Int(lane)]
    var total = _warp_sum_u32(tmp)
    var pfx = _warp_sum_u32(tmp if lane < wid else UInt32(0)) + winc
    var above = total - pfx
    var uneed = UInt32(need)
    if tx < UInt32(nbins) and above < uneed and above + value >= uneed:
        mtch[0] = tx
        mtch[1] = above
        mtch[2] = value
        comptime if with_pre:
            mtch[3] = pre
    barrier()


@inline(.always)
def _ctopk_phi[
    bits_dtype: DType, //, in_dtype: DType, width: Int
](bits: SIMD[bits_dtype, width]) -> SIMD[.uint32, width]:
    """`_phi` of each score's raw bits, as the refinement and scatter read it.

    At f32 this is `_phi_of`. At bf16 one `prmt` per 32-bit word replicates
    each column's sign byte across its half (selector nibbles 9 and B), so
    both columns convert in two instructions; the odd column's key is then
    the word as is, its low 16 bits the even column's key rather than
    `_phi_of`'s fill. Every consumer reads only the top 16 bits there: the
    refinement digits sit at or above bit 16 (`pad`), and every bracket's low
    16 bits are all zeros (`slo`) or all ones (`shi`), so each bracket test
    agrees with `_phi_of`'s.
    """
    comptime if size_of[in_dtype]() == 4:
        return _phi_of[in_dtype, .uint32, width](bits)
    else:
        comptime assert width % 2 == 0, "bf16 keys pair columns per word"
        comptime words = width // 2
        var w = bitcast[.uint32, words](bits)
        var k = SIMD[.uint32, words](0)
        comptime for i in range(words):
            var m = byte_permute(w[i], 0, 0xBB99)
            k[i] = w[i] ^ (m | UInt32(0x80008000))
        return rebind[SIMD[.uint32, width]]((k << 16).interleave(k))


@inline(.always)
def _ctopk_stage_key[
    bits_dtype: DType, //, in_dtype: DType, width: Int
](bits: SIMD[bits_dtype, width]) -> SIMD[bits_dtype, width]:
    """What the single-pass staging holds for each score.

    At f32 its `phi`, so the rescans read the key with no work. At bf16 the
    raw bits: staging the per-word key instead moves its two instructions per
    word into round 0, which measured 0.7% slower overall (up to 1.9% at the
    cluster cells), where at f32 the move measured 0.6% faster (up to 1.7%).
    `_ctopk_staged_phi` turns either back into `_ctopk_phi`'s value.
    """
    comptime if size_of[in_dtype]() == 4:
        return rebind[SIMD[bits_dtype, width]](_ctopk_phi[in_dtype](bits))
    else:
        return bits


@inline(.always)
def _ctopk_staged_phi[
    bits_dtype: DType, //, in_dtype: DType, width: Int
](key: SIMD[bits_dtype, width]) -> SIMD[.uint32, width]:
    """`_ctopk_phi` of the scores `_ctopk_stage_key` staged as `key`."""
    comptime if size_of[in_dtype]() == 4:
        return rebind[SIMD[.uint32, width]](key)
    else:
        return _ctopk_phi[in_dtype](key)


@inline(.always)
def _ctopk_load[
    in_dtype: DType, //, scan_items: Int
](
    src: Pointer[Scalar[in_dtype], ImmutAnyOrigin, address_space=.GLOBAL],
    base: UInt32,
    my_len: UInt32,
    aligned: Bool,
) -> SIMD[_score_bits[in_dtype](), scan_items]:
    """The group at slice position `base` of the slice starting at `src`.

    `_load_scan_group` with 32-bit slice positions and the alignment tested
    once by the caller: `aligned` is the slice start's, and every group sits
    a whole number of 16 B units past it. Positions past `my_len` take
    `_score_pad`'s all-ones bits.
    """
    comptime bits_dtype = _score_bits[in_dtype]()
    if base + UInt32(scan_items) <= my_len and aligned:
        return bitcast[bits_dtype, scan_items](
            src.unsafe_load[width=scan_items, alignment=_V4_ALIGN](Int(base))
        )
    var v = SIMD[bits_dtype, scan_items](Scalar[bits_dtype].MAX)
    comptime for j in range(scan_items):
        if base + UInt32(j) < my_len:
            v[j] = bitcast[bits_dtype, 1](
                SIMD[in_dtype, 1](src[Int(base) + j])
            )[0]
    return v


@inline(.always)
def _ctopk_stage_halves(base: UInt32) -> Tuple[Int, Int]:
    """Staging positions of a 32 B group's two 16 B halves, at slice
    position `base`.

    Groups sit 32 B apart, so a quarter-warp's 16 B accesses to the same half
    span 256 B and meet every bank twice. A thread whose group is an odd one
    of its 64 B pair swaps its halves, which spreads each quarter-warp's
    accesses over all 32 banks. Every staged access goes through here, and
    the swap stays inside the group, so the staging still holds exactly the
    slice's positions.
    """
    var lo = base ^ ((base >> 3) & 4)
    return (Int(lo), Int(lo ^ 4))


@inline(.always)
def _ctopk_stage_load[
    bits_dtype: DType, //, scan_items: Int, st_origin: MutOrigin
](
    stage: Pointer[Scalar[bits_dtype], st_origin, address_space=.SHARED],
    base: UInt32,
) -> SIMD[bits_dtype, scan_items]:
    """The staged group at slice position `base` (`_ctopk_stage_halves`)."""
    comptime if scan_items * size_of[bits_dtype]() == 32:
        var lo, hi = _ctopk_stage_halves(base)
        comptime half = scan_items // 2
        return rebind[SIMD[bits_dtype, scan_items]](
            stage.unsafe_load[width=half, alignment=_V4_ALIGN](lo).join(
                stage.unsafe_load[width=half, alignment=_V4_ALIGN](hi)
            )
        )
    else:
        return stage.unsafe_load[width=scan_items, alignment=_V4_ALIGN](
            Int(base)
        )


@inline(.always)
def _ctopk_stage_store[
    bits_dtype: DType, scan_items: Int, st_origin: MutOrigin
](
    stage: Pointer[Scalar[bits_dtype], st_origin, address_space=.SHARED],
    base: UInt32,
    v: SIMD[bits_dtype, scan_items],
):
    """Stages the group at slice position `base` (`_ctopk_stage_halves`)."""
    comptime if scan_items * size_of[bits_dtype]() == 32:
        var lo, hi = _ctopk_stage_halves(base)
        comptime half = scan_items // 2
        stage.unsafe_store[alignment=_V4_ALIGN](lo, v.slice[half, offset=0]())
        stage.unsafe_store[alignment=_V4_ALIGN](
            hi, v.slice[half, offset=half]()
        )
    else:
        stage.unsafe_store[width=scan_items, alignment=_V4_ALIGN](Int(base), v)


@inline(.always)
def _ctopk_hist_group[
    scan_items: Int,
    count_all: Bool,
    masked: Bool,
    h_origin: MutOrigin,
](
    hist: Pointer[UInt32, h_origin, address_space=.SHARED],
    ph: SIMD[.uint32, scan_items],
    base: UInt32,
    my_len: UInt32,
    slo: UInt32,
    shi: UInt32,
    sh: UInt32,
    dmask: UInt32,
):
    """Bins one loaded group into `hist`, at slice position `base`.

    `count_all` is round 0: `ph` is already the `_ctopk_key0` bin. Otherwise
    `ph` is `phi`, and a column in the bracket `[slo, shi]` bins its digit
    `(phi >> sh) & dmask`.

    Columns past `my_len` are never binned, except unmasked round 0. A
    streaming slot's copy rounds to whole 16 B units past a clamped bound,
    and a group can span that tail (at f32, where the group rounds to 8
    elements against the copy's 4, even past it into stale bytes) --
    counting it would inflate a bin exactly where a masked row's threshold
    can land. Unmasked round 0 reads `_ctopk_load`'s pads, which all bin 0,
    and `_ctopk_scan` takes their known count back out of it.
    """
    comptime for j in range(scan_items):
        var live = True
        comptime if masked or not count_all:
            live = base + UInt32(j) < my_len
        comptime if count_all:
            if live:
                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](hist.unsafe_offset(Int(ph[j])), UInt32(1))
        else:
            if live and ph[j] >= slo and ph[j] <= shi:
                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](hist.unsafe_offset(Int((ph[j] >> sh) & dmask)), UInt32(1))


@inline(.always)
def _ctopk_tma_stream[
    in_dtype: DType,
    //,
    scan_items: Int,
    count_all: Bool,
    hist_mode: Bool,
    cand: Bool,
    h_origin: MutOrigin,
    st_origin: MutOrigin,
    tb_origin: MutOrigin,
    ctl_origin: MutOrigin,
    sel_origin: MutOrigin,
    S: TopkSink,
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin],
    row_base: Int,
    my_len: UInt32,
    col0: Int32,
    hist: Pointer[UInt32, h_origin, address_space=.SHARED],
    stage: Pointer[
        Scalar[_score_bits[in_dtype]()], st_origin, address_space=.SHARED
    ],
    tbar: Pointer[SharedMemBarrier, tb_origin, address_space=.SHARED],
    t_phi: UInt32,
    ctl: Pointer[UInt32, ctl_origin, address_space=.SHARED],
    selbuf: Pointer[Int32, sel_origin, address_space=.SHARED],
    K: Int,
    slo: UInt32,
    shi: UInt32,
    sh: UInt32,
    dmask: UInt32,
    sink: S,
    out_base: Int,
    root: Bool,
):
    """One streaming pass over a rank's slice, the only copy of the protocol.

    vLLM's `tma_stream_pass`: the prologue fills both slots, each iteration
    consumes one while the next copy is in flight, and the epilogue refills
    the slot just consumed. Lane 0 of each warp holds the mbarrier wait --
    the default `.acquire` parity wait -- and publishes it to its lanes with
    `syncwarp`; the refill of a slot may only be issued once every thread
    has left it, which the trailing `barrier()` guarantees. The caller hands
    a fresh barrier pair per pass (`tbar`), so the wait's phase is
    `(it >> 1) & 1` with no state carried between passes.

    `hist_mode` picks the consumer; the other mode's arguments are passed
    anyway (the caller holds them all) and go unread -- every read is under
    the `comptime if`, so an unused pointer is not even an instantiation.
    `cand` is the candidate scatter (`_ctopk_scatter_group`), its bin floor
    in `slo`.
    """
    var tid = Int(thread_idx.x)
    comptime step = UInt32(_CTOPK_BLOCK * scan_items)
    comptime stage_elems = _ctopk_tma_stage_elems[in_dtype]()
    comptime align_u = _elems_per_16b[in_dtype]()
    var lane = Int(lane_id())
    # The last slot's copy rounds the slice up to a whole 16 B unit, so the
    # bulk copy stays 16 B on both ends; the columns the rounding adds sit
    # past `my_len` and past the live bound, so no consumer sees them. The
    # stride-alignment gate that selected this flavor keeps the read inside
    # the row.
    var la = align_up(Int(my_len), align_u)
    var ni = ceildiv(Int(my_len), stage_elems)
    var gsrc = in_scores.unsafe_address_space_cast[.GLOBAL]()
    if tid == 0:
        var i = 0
        while i < min(_CTOPK_TMA_STAGES, ni):
            var o = i * stage_elems
            var bytes = Int32(min(stage_elems, la - o) * size_of[in_dtype]())
            tbar[i].expect_bytes(bytes)
            cp_async_bulk_shared_cluster_global(
                stage.unsafe_offset(i * stage_elems),
                gsrc.unsafe_offset(row_base).unsafe_offset(o),
                bytes,
                tbar[i].unsafe_ptr(),
            )
            i += 1
    var it = 0
    while it < ni:
        var b = it & 1
        var o = UInt32(it * stage_elems)
        if lane == 0:
            tbar[b].wait(UInt32((it >> 1) & 1))
        syncwarp()
        var slot = stage.unsafe_offset(b * stage_elems)
        var base = UInt32(tid * scan_items)
        var end = min(UInt32(stage_elems), my_len - o)
        while base < end:
            var lv = slot.unsafe_load[width=scan_items, alignment=_V4_ALIGN](
                Int(base)
            )
            comptime if hist_mode:
                var ph: SIMD[.uint32, scan_items]
                comptime if count_all:
                    ph = _ctopk_key0[in_dtype](lv)
                else:
                    ph = _ctopk_phi[in_dtype](lv)
                _ctopk_hist_group[scan_items, count_all, masked=True](
                    hist, ph, o + base, my_len, slo, shi, sh, dmask
                )
            else:
                _ctopk_scatter_group[scan_items, cand](
                    _ctopk_phi[in_dtype](lv),
                    o + base,
                    my_len,
                    col0,
                    t_phi,
                    slo,
                    ctl,
                    selbuf,
                    K,
                    sink,
                    out_base,
                    root,
                )
            base += step
        barrier()
        if tid == 0 and it + _CTOPK_TMA_STAGES < ni:
            var no = (it + _CTOPK_TMA_STAGES) * stage_elems
            var bytes = Int32(min(stage_elems, la - no) * size_of[in_dtype]())
            tbar[b].expect_bytes(bytes)
            cp_async_bulk_shared_cluster_global(
                slot,
                gsrc.unsafe_offset(row_base).unsafe_offset(no),
                bytes,
                tbar[b].unsafe_ptr(),
            )
        it += 1


@inline(.always)
def _ctopk_scan[
    in_dtype: DType,
    //,
    scan_items: Int,
    count_all: Bool,
    from_stage: Bool,
    stage_slice: Bool,
    prefetch: Bool,
    h_origin: MutOrigin,
    st_origin: MutOrigin,
](
    src: Pointer[Scalar[in_dtype], ImmutAnyOrigin, address_space=.GLOBAL],
    my_len: UInt32,
    aligned: Bool,
    hist: Pointer[UInt32, h_origin, address_space=.SHARED],
    stage: Pointer[
        Scalar[_score_bits[in_dtype]()], st_origin, address_space=.SHARED
    ],
    slo: UInt32,
    shi: UInt32,
    sh: UInt32,
    dmask: UInt32,
):
    """One rank's histogram pass over its slice, from global or the staging.

    The slice (`src` is its first column) is read from global (prefetching
    one group ahead when `prefetch`) or rescanned from the staging
    (`from_stage`, the single-pass flavor); round 0 of the single-pass flavor
    also writes the staging (`stage_slice`) so every later pass serves one
    global read. `count_all` is round 0, which bins every column by
    `_ctopk_key0`; later rounds bin only the bracket `[slo, shi]`, by the
    `phi` digit `(phi >> sh) & dmask`. The streaming flavor's TMA pipeline
    lives in `_ctopk_tma_stream`, not here.

    Round 0 bins the slice's partial group whole, pads included (every pad
    bins 0), and thread 0 subtracts the pad count from bin 0, so the
    histogram counts exactly the live columns the scatter classifies: the
    peer prefix depends on the two agreeing.
    """
    var tid = UInt32(thread_idx.x)
    comptime step = UInt32(_CTOPK_BLOCK * scan_items)
    comptime items = UInt32(scan_items)
    comptime bits_dtype = _score_bits[in_dtype]()
    comptime if count_all:
        var pads = (items - my_len % items) % items
        if tid == 0 and pads != 0:
            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](hist, UInt32(0) - pads)
    var base = tid * items
    var cur = SIMD[bits_dtype, scan_items](0)
    comptime if prefetch:
        comptime assert (
            not from_stage
        ), "a staged rescan has nothing to prefetch"
        if base < my_len:
            cur = _ctopk_load[scan_items](src, base, my_len, aligned)
    while base < my_len:
        var nb = base + step
        var lv: SIMD[bits_dtype, scan_items]
        comptime if from_stage:
            lv = _ctopk_stage_load[scan_items](stage, base)
        elif prefetch:
            lv = cur
            if nb < my_len:
                cur = _ctopk_load[scan_items](src, nb, my_len, aligned)
        else:
            lv = _ctopk_load[scan_items](src, base, my_len, aligned)
        comptime if stage_slice:
            _ctopk_stage_store(stage, base, _ctopk_stage_key[in_dtype](lv))
        var ph: SIMD[.uint32, scan_items]
        comptime if count_all:
            ph = _ctopk_key0[in_dtype](lv)
        elif from_stage:
            ph = _ctopk_staged_phi[in_dtype](lv)
        else:
            ph = _ctopk_phi[in_dtype](lv)
        _ctopk_hist_group[scan_items, count_all, masked=False](
            hist, ph, base, my_len, slo, shi, sh, dmask
        )
        base = nb


@inline(.always)
def _ctopk_cand_list[
    o: MutOrigin
](selbuf: Pointer[Int32, o, address_space=.SHARED], root: Bool) -> Pointer[
    UInt32, o, address_space=.SHARED
]:
    """A rank's `(column, phi)` candidate pairs: rank 0's are the whole of
    `selbuf`, which also takes every peer's; a peer's sit past its
    strictly-above columns."""
    return (
        selbuf.unsafe_offset(0 if root else PERSISTENT_TOPK_MAX_N)
    ).unsafe_bitcast[UInt32]()


@inline(.always)
def _ctopk_cand_cap(root: Bool) -> Int:
    """Pairs `_ctopk_cand_list` holds."""
    return PERSISTENT_TOPK_MAX_N if root else _CTOPK_CAND_CAP


@inline(.always)
def _ctopk_scatter_group[
    scan_items: Int,
    cand: Bool,
    ctl_origin: MutOrigin,
    sel_origin: MutOrigin,
    S: TopkSink,
](
    ph: SIMD[.uint32, scan_items],
    base: UInt32,
    my_len: UInt32,
    col0: Int32,
    t_phi: UInt32,
    t_lo: UInt32,
    ctl: Pointer[UInt32, ctl_origin, address_space=.SHARED],
    selbuf: Pointer[Int32, sel_origin, address_space=.SHARED],
    K: Int,
    sink: S,
    out_base: Int,
    root: Bool,
):
    """Stages one loaded group's selections, at slice position `base` (row
    column `col0 + base`). Columns past `my_len` never stage, so neither a
    TMA slot's 16 B-rounded tail nor `_ctopk_load`'s pads reach `selbuf`.

    Columns above `t_phi` stage at `[0, K)`. The rest of `[t_lo, t_phi]` is
    the second class: with `cand` it is round 0's threshold bin, staged as
    `(column, phi)` pairs (`_ctopk_cand_list`); otherwise it is the final
    bracket -- the exact plateau, or a split bin taken whole -- staged as
    columns at `[K, 2K)`.

    With `cand`, rank 0 (`root`) writes its above columns straight to the
    sink: its slots start at 0 whatever the peers hold, and that frees its
    whole `selbuf` for the cluster's candidates. `sink`, `out_base` and
    `root` go unread otherwise."""
    # A whole group, every group but the slice's last, skips the per-column
    # position test.
    var n_live = min(my_len - base, UInt32(scan_items))
    var gcol = col0 + Int32(base)
    if n_live == UInt32(scan_items):
        _ctopk_scatter_cols[scan_items, cand, masked=False](
            ph, n_live, gcol, t_phi, t_lo, ctl, selbuf, K, sink, out_base, root
        )
    else:
        _ctopk_scatter_cols[scan_items, cand, masked=True](
            ph, n_live, gcol, t_phi, t_lo, ctl, selbuf, K, sink, out_base, root
        )


@inline(.always)
def _ctopk_scatter_cols[
    scan_items: Int,
    cand: Bool,
    masked: Bool,
    ctl_origin: MutOrigin,
    sel_origin: MutOrigin,
    S: TopkSink,
](
    ph: SIMD[.uint32, scan_items],
    n_live: UInt32,
    gcol: Int32,
    t_phi: UInt32,
    t_lo: UInt32,
    ctl: Pointer[UInt32, ctl_origin, address_space=.SHARED],
    selbuf: Pointer[Int32, sel_origin, address_space=.SHARED],
    K: Int,
    sink: S,
    out_base: Int,
    root: Bool,
):
    """`_ctopk_scatter_group`'s columns `[0, n_live)`, `masked` testing
    position; the group's first column is `gcol`."""
    var k32 = UInt32(K)
    comptime for j in range(scan_items):
        var live = True
        comptime if masked:
            live = UInt32(j) < n_live
        if live:
            var p = ph[j]
            var col = gcol + Int32(j)
            if p > t_phi:
                var pos = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl, UInt32(1))
                comptime if cand:
                    if root:
                        sink.store(out_base + Int(pos), col)
                    elif pos < k32:
                        selbuf[Int(pos)] = col
                else:
                    if pos < k32:
                        selbuf[Int(pos)] = col
            elif p >= t_lo:
                var pos = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl.unsafe_offset(1), UInt32(1))
                comptime if cand:
                    if pos < UInt32(_ctopk_cand_cap(root)):
                        _ctopk_cand_list(selbuf, root).unsafe_store[
                            width=2, alignment=8
                        ](Int(2 * pos), SIMD[.uint32, 2](UInt32(col), p))
                else:
                    if pos < k32:
                        selbuf[Int(k32 + pos)] = col


@inline(.always)
def _ctopk_scatter[
    in_dtype: DType,
    //,
    scan_items: Int,
    from_stage: Bool,
    prefetch: Bool,
    cand: Bool,
    st_origin: MutOrigin,
    ctl_origin: MutOrigin,
    sel_origin: MutOrigin,
    S: TopkSink,
](
    src: Pointer[Scalar[in_dtype], ImmutAnyOrigin, address_space=.GLOBAL],
    col0: Int32,
    my_len: UInt32,
    aligned: Bool,
    stage: Pointer[
        Scalar[_score_bits[in_dtype]()], st_origin, address_space=.SHARED
    ],
    t_phi: UInt32,
    t_lo: UInt32,
    ctl: Pointer[UInt32, ctl_origin, address_space=.SHARED],
    selbuf: Pointer[Int32, sel_origin, address_space=.SHARED],
    K: Int,
    sink: S,
    out_base: Int,
    root: Bool,
):
    """One rank's selection pass against the final bracket `[t_lo, t_phi]`,
    or, with `cand`, against round 0's threshold bin.

    Columns strictly above `t_phi` stage at `[0, K)` of `selbuf` -- there are
    fewer than `K` of them globally, so nothing is dropped. Without `cand`,
    columns in the bracket stage at `[K, 2K)`, capped at `K` per rank: which
    ties survive the cap is arrival order, the accepted first-come
    relaxation, and the cap is safe because `sum(min(eq_r, K)) >= K -
    count(>)` whenever the bracket holds enough to fill the row, which the
    threshold rounds guaranteed. With `cand` the bracket's columns are
    candidates (`_ctopk_scatter_group`).

    Placement is the caller's; this pass only stages and counts, in `ctl[0]`
    and `ctl[1]`. The streaming flavor's TMA pipeline lives in
    `_ctopk_tma_stream`, not here.
    """
    var tid = UInt32(thread_idx.x)
    comptime step = UInt32(_CTOPK_BLOCK * scan_items)
    comptime bits_dtype = _score_bits[in_dtype]()
    var base = tid * UInt32(scan_items)
    var cur = SIMD[bits_dtype, scan_items](0)
    comptime if prefetch:
        comptime assert (
            not from_stage
        ), "a staged rescan has nothing to prefetch"
        if base < my_len:
            cur = _ctopk_load[scan_items](src, base, my_len, aligned)
    while base < my_len:
        var nb = base + step
        var lv: SIMD[bits_dtype, scan_items]
        comptime if from_stage:
            lv = _ctopk_stage_load[scan_items](stage, base)
        elif prefetch:
            lv = cur
            if nb < my_len:
                cur = _ctopk_load[scan_items](src, nb, my_len, aligned)
        else:
            lv = _ctopk_load[scan_items](src, base, my_len, aligned)
        var ph: SIMD[.uint32, scan_items]
        comptime if from_stage:
            ph = _ctopk_staged_phi[in_dtype](lv)
        else:
            ph = _ctopk_phi[in_dtype](lv)
        _ctopk_scatter_group[scan_items, cand](
            ph,
            base,
            my_len,
            col0,
            t_phi,
            t_lo,
            ctl,
            selbuf,
            K,
            sink,
            out_base,
            root,
        )
        base = nb


# Geometry of the mid-length path: rank 0's 1024 threads hold up to `W` 16 B
# loads each (a group of `_ctopk_solo_group` columns per load), binned by
# vLLM's 12-bit coarse key. A threshold bin of up to `_CTOPK_SOLO_TIES`
# columns is ranked on its exact keys; a wider one falls back to radix rounds
# over them.
comptime _CTOPK_SOLO_BITS: Int = 12
comptime _CTOPK_SOLO_BINS: Int = 1 << _CTOPK_SOLO_BITS
comptime _CTOPK_SOLO_TIES: Int = 128
# Payload words a thread holds, two bins each; the class masks give every
# word one bit per half, so 16 words fill them.
comptime _CTOPK_SOLO_MAX_WORDS: Int = 16
# The path's own carve of the dynamic block, which it has to itself: the
# cluster's other ranks exit without touching any cluster state, and a launch
# whose stride rank 0 covers runs no cluster code at all. A column past the
# row bins one slot below the histogram, into a scratch word, so the
# histogram starts a 16 B unit in.
comptime _CTOPK_SOLO_OFF_HIST = 16
comptime _CTOPK_SOLO_OFF_CTL = _CTOPK_SOLO_OFF_HIST + 4 * _CTOPK_SOLO_BINS
comptime _CTOPK_SOLO_OFF_WSUM = _CTOPK_SOLO_OFF_CTL + 4 * 32
comptime _CTOPK_SOLO_OFF_MTCH = _CTOPK_SOLO_OFF_WSUM + 4 * WARP_SIZE
comptime _CTOPK_SOLO_OFF_TIES = _CTOPK_SOLO_OFF_MTCH + 4 * 4
comptime _CTOPK_SOLO_SMEM = _CTOPK_SOLO_OFF_TIES + 8 * _CTOPK_SOLO_TIES


@inline(.always)
def _ctopk_solo_group[in_dtype: DType]() -> Int:
    """Columns one 16 B load per thread covers: 4096 at f32, 8192 at bf16."""
    return _CTOPK_BLOCK * _elems_per_16b[in_dtype]()


@inline(.always)
def _ctopk_solo_loads[in_dtype: DType, CS: Int]() -> Int:
    """Loads per thread rank 0 holds: 32768 columns at either dtype (the 16
    payload words the class masks hold), 24576 at `CS` 5 to 8.

    Measured on B200 (`bench_topk_3way sweep mode=mid`, warm L2): one CTA
    beats or ties the cluster on rows of up to 24576 columns at every cluster
    size (0.85-0.91x at 16384, 0.97-1.02x at 24576). At 28672 and 32768
    columns a cluster of 5 or more ranks wins by 2-9% at either dtype, while
    against 4 ranks the single CTA ties and against 2 it wins (0.80-0.83x).
    The dispatch picks 10 and 16 only for strides past 65536 and 81920,
    where a row this short is never a launch's slowest, so they keep the
    full width.
    """
    comptime full = 8 if in_dtype == .float32 else 4
    comptime if CS >= 5 and CS <= 8:
        return full * 3 // 4
    return full


@inline(.always)
def _ctopk_solo_cap[in_dtype: DType, CS: Int]() -> Int:
    """The longest row rank 0 selects alone; longer rows take the cluster."""
    return _ctopk_solo_loads[in_dtype, CS]() * _ctopk_solo_group[in_dtype]()


@inline(.always)
def _ctopk_solo_fields[
    bits_dtype: DType, //, in_dtype: DType, width: Int
](bits: SIMD[bits_dtype, width]) -> SIMD[.uint32, width // 2]:
    """Coarse bins of `width` raw scores, two per word as `0x8000 + bin`.

    The bin is vLLM's `extract_coarse_bin_N`: the score rounded to fp16, its
    bits made monotone (sign flipped, a negative complemented), the top 12
    bits. fp16's 5 exponent bits give 64 bins per octave where scores sit.
    The conversion packs two columns per `cvt.rn.f16x2.f32`, and one `prmt`
    replicating each score's sign byte across its half orders both halves at
    once. The `0x8000` guard keeps each half non-negative under the class
    tests' subtraction, so they run on both halves of a word together.

    The sign comes from the score, not the conversion, which returns every
    NaN as the positive canonical NaN: setting the half's sign bit from the
    score before the flip, in the same 3-input logic op, bins a negative NaN
    0, lowest, as its `phi` is.
    """
    comptime assert width % 2 == 0, "bins pair columns per word"
    var h: SIMD[.uint32, width // 2]
    var sg = SIMD[.uint32, width // 2](0)
    comptime if in_dtype == .float32:
        h = bitcast[.uint32, width // 2](
            bitcast[.float32, width](bits).cast[.float16]()
        )
        var x = bitcast[.uint32, width](bits)
        comptime for i in range(width // 2):
            sg[i] = byte_permute(x[2 * i], x[2 * i + 1], 0xFFBB)
    else:
        # bf16 widens to f32 by placement alone: the high half as is, the
        # low half shifted up.
        var w = bitcast[.uint32, width // 2](bits)
        var x = SIMD[.uint32, width](0)
        comptime for i in range(width // 2):
            x[2 * i] = w[i] << 16
            x[2 * i + 1] = w[i] & UInt32(0xFFFF0000)
        h = bitcast[.uint32, width // 2](
            bitcast[.float32, width](x).cast[.float16]()
        )
        comptime for i in range(width // 2):
            sg[i] = byte_permute(x[2 * i], x[2 * i + 1], 0xFFBB)
    var f = SIMD[.uint32, width // 2](0)
    comptime for i in range(width // 2):
        comptime c = UInt32(0x80008000)
        var k = (h[i] | (sg[i] & c)) ^ (sg[i] | c)
        f[i] = ((k >> 4) & UInt32(0x0FFF0FFF)) | UInt32(0x80008000)
    return f


@inline(.always)
def _ctopk_solo_ge_bits(f: UInt32, b: UInt32) -> UInt32:
    """Bit 15 (31) set when the low (high) half's bin is at least `b`."""
    return (f - b * UInt32(0x10001)) & UInt32(0x80008000)


@inline(.always)
def _ctopk_solo_col[in_dtype: DType](tid: UInt32, b: UInt32) -> UInt32:
    """Column of class-mask bit `b`: word `b & 15`, its half `b >> 4`."""
    comptime vi = _elems_per_16b[in_dtype]()
    comptime wpl = vi // 2
    var i = b & 15
    return (
        (i // UInt32(wpl)) * UInt32(_ctopk_solo_group[in_dtype]())
        + tid * UInt32(vi)
        + 2 * (i % UInt32(wpl))
        + (b >> 4)
    )


@inline(.always)
def _ctopk_solo_phi[
    in_dtype: DType
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin, address_space=.GLOBAL],
    row_base: Int,
    c: UInt32,
) -> UInt32:
    """`phi` of one live column, re-read from global (L1/L2-resident by now)."""
    return _phi_of[in_dtype, .uint32, 1](
        bitcast[_score_bits[in_dtype](), 1](in_scores[row_base + Int(c)])
    )[0]


@inline(.always)
def _ctopk_solo_load[
    in_dtype: DType
](
    src: Pointer[Scalar[in_dtype], ImmutAnyOrigin, address_space=.GLOBAL],
    c0: UInt32,
    count: UInt32,
    aligned: Bool,
    whole: Bool,
) -> SIMD[_score_bits[in_dtype](), _elems_per_16b[in_dtype]()]:
    """The raw bits of one thread's 16 B of a group, at row column `c0`:
    one vector load, or scalar loads with `_score_pad` past `count` where the
    group straddles the row's end or the row base is not 16 B aligned.

    `whole` is the block-uniform verdict that the whole group is live, so
    every group but a row's last skips the per-thread position tests.
    """
    comptime bits_dtype = _score_bits[in_dtype]()
    comptime vi = _elems_per_16b[in_dtype]()
    if aligned and (whole or c0 + UInt32(vi) <= count):
        return bitcast[bits_dtype, vi](
            src.unsafe_load[width=vi, alignment=_V4_ALIGN](Int(c0))
        )
    var r = _score_pad[in_dtype, vi]()
    comptime for j in range(vi):
        if c0 + UInt32(j) < count:
            r[j] = bitcast[bits_dtype, 1](SIMD[in_dtype, 1](src[Int(c0) + j]))[
                0
            ]
    return r


@inline(.always)
def _ctopk_solo_select[
    in_dtype: DType, //, W: Int, S: TopkSink
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin, address_space=.GLOBAL],
    sink: S,
    row_base: Int,
    out_base: Int,
    count: UInt32,
    K: UInt32,
    smem: Pointer[UInt8, MutUntrackedOrigin, address_space=.SHARED],
):
    """One CTA selects a row of at most `W` groups of live columns, in
    registers.

    vLLM's `histogram_4096_topk`: the row is read once, `W` 16 B loads per
    thread, one 4096-bin histogram of the coarse bins finds the threshold bin,
    and one pass writes the columns above it straight to the sink. The bin's
    own columns are resolved last: all of them when they fit exactly, by a
    rank on the exact key when there are at most `_CTOPK_SOLO_TIES`, and
    otherwise by 8-bit radix rounds that cut the final plateau first-come --
    the `ordered=False, deterministic=False` contract of the cluster path.

    Registers hold only the bins, two to a word: the exact key is needed for
    the threshold bin's members alone, so those re-read their score.
    """
    comptime assert (
        _CTOPK_SOLO_SMEM <= _ctopk_stage_bytes()
    ), "the single-CTA carve must fit inside every rank's staging"
    comptime bits_dtype = _score_bits[in_dtype]()
    comptime vi = _elems_per_16b[in_dtype]()
    comptime wpl = vi // 2
    comptime nw = W * wpl
    comptime assert nw <= _CTOPK_SOLO_MAX_WORDS, "the class masks hold 16 words"
    comptime grp = UInt32(_ctopk_solo_group[in_dtype]())
    comptime nbins = _CTOPK_SOLO_BINS
    comptime per_thread = nbins // _CTOPK_BLOCK
    var tid = UInt32(thread_idx.x)
    var lane = UInt32(lane_id())
    var wid = UInt32(warp_id())
    var hist = (smem.unsafe_offset(_CTOPK_SOLO_OFF_HIST)).unsafe_bitcast[
        UInt32
    ]()
    var ctl = (smem.unsafe_offset(_CTOPK_SOLO_OFF_CTL)).unsafe_bitcast[UInt32]()
    var wsum = (smem.unsafe_offset(_CTOPK_SOLO_OFF_WSUM)).unsafe_bitcast[
        UInt32
    ]()
    var mtch = (smem.unsafe_offset(_CTOPK_SOLO_OFF_MTCH)).unsafe_bitcast[
        UInt32
    ]()
    var ties = (smem.unsafe_offset(_CTOPK_SOLO_OFF_TIES)).unsafe_bitcast[
        UInt64
    ]()
    var src = in_scores.unsafe_offset(row_base)
    # The vector load needs the row base 16 B aligned; every group then is.
    var aligned = (row_base & (vi - 1)) == 0

    # Up to four loads per thread are in flight before any binning, so the
    # row's L2 round trips overlap rather than chain group by group; past
    # four, each group binned frees its registers for the load four ahead.
    comptime depth = 4 if W >= 4 else (2 if W >= 2 else 1)
    var raw = SIMD[bits_dtype, depth * vi](0)
    comptime for v in range(depth):
        raw = raw.insert[offset=v * vi](
            _ctopk_solo_load(
                src,
                tid * UInt32(vi) + UInt32(v) * grp,
                count,
                aligned,
                UInt32(v + 1) * grp <= count,
            )
        )

    hist.unsafe_store[width=per_thread, alignment=_V4_ALIGN](
        Int(tid) * per_thread, SIMD[.uint32, per_thread](0)
    )
    if tid < 32:
        ctl[Int(tid)] = 0
    if tid < UInt32(_CTOPK_SOLO_TIES):
        ties[Int(tid)] = 0
    barrier()

    # Bins are counted from `0x8000` up, hence the offset base. A column past
    # the row takes `0x7FFF`, one below bin 0: it bins into the scratch word
    # below the histogram, and no class test can pass it.
    var hbase = hist.unsafe_offset(-0x8000)
    var pk = SIMD[.uint32, _CTOPK_SOLO_MAX_WORDS](0)
    comptime for v in range(W):
        var c0 = tid * UInt32(vi) + UInt32(v) * grp
        var whole = UInt32(v + 1) * grp <= count
        var f = _ctopk_solo_fields[in_dtype](
            raw.slice[vi, offset=(v % depth) * vi]()
        )
        comptime if v + depth < W:
            raw = raw.insert[offset=(v % depth) * vi](
                _ctopk_solo_load(
                    src,
                    c0 + UInt32(depth) * grp,
                    count,
                    aligned,
                    UInt32(v + depth + 1) * grp <= count,
                )
            )
        if not whole and c0 + UInt32(vi) > count:
            comptime for j in range(vi):
                if c0 + UInt32(j) >= count:
                    comptime if j % 2 == 0:
                        f[j // 2] = (f[j // 2] & 0xFFFF0000) | 0x7FFF
                    else:
                        f[j // 2] = (f[j // 2] & 0xFFFF) | 0x7FFF0000
        comptime for i in range(wpl):
            pk[v * wpl + i] = f[i]
        # A group wholly past the row is not binned, warp by warp.
        if whole or c0 < count:
            comptime for i in range(wpl):
                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](hbase.unsafe_offset(Int(f[i] & 0xFFFF)), UInt32(1))
                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](hbase.unsafe_offset(Int(f[i] >> 16)), UInt32(1))
    barrier()

    # Thread `t` scans bins `[4t, 4t + 4)`; the bins above a bin are every
    # live column minus the bins at or below it.
    var h = hist.unsafe_load[width=per_thread, alignment=_V4_ALIGN](
        Int(tid) * per_thread
    )
    var local = h.reduce_add()
    var winc = warp.prefix_sum(local)
    if lane == UInt32(WARP_SIZE - 1):
        wsum[Int(wid)] = winc
    barrier()
    var wtot = wsum[Int(lane)]
    var total = _warp_sum_u32(wtot)
    var pre = _warp_sum_u32(wtot if lane < wid else UInt32(0)) + winc - local
    comptime for i in range(per_thread):
        pre += h[i]
        var above = total - pre
        if above < K and above + h[i] >= K:
            ctl[16] = tid * UInt32(per_thread) + UInt32(i)
            ctl[17] = above
            ctl[18] = h[i]
    barrier()

    var thr = ctl[16]
    var above = ctl[17]
    var eq = ctl[18]
    var need = K - above
    var exact = eq == need
    var listed = eq <= UInt32(_CTOPK_SOLO_TIES)

    # Class masks: word `i`'s halves are bits `i` and `16 + i`.
    var gmask = UInt32(0)
    var emask = UInt32(0)
    comptime for i in range(nw):
        var g = _ctopk_solo_ge_bits(pk[i], thr + 1)
        var e = _ctopk_solo_ge_bits(pk[i], thr) & ~g
        gmask |= g >> UInt32(15 - i)
        emask |= e >> UInt32(15 - i)
    # Both class counts ride one cursor, above-class high: neither total can
    # reach 2^16 in a row of at most 32768 columns.
    var packed = (UInt32(pop_count(gmask)) << 16) | UInt32(pop_count(emask))
    var wpre = warp.prefix_sum[exclusive=True](packed)
    var wb = UInt32(0)
    if lane == UInt32(WARP_SIZE - 1):
        wb = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
            ordering=Ordering.RELAXED
        ](ctl, wpre + packed)
    var pos = warp.shuffle_idx(wb, UInt32(WARP_SIZE - 1)) + wpre
    var gp = pos >> 16
    var ep = pos & UInt32(0xFFFF)
    # Walked by set bit: an eighth of the slots are selected, so a test per
    # slot is mostly branches that skip.
    var m = gmask
    while m != 0:
        var c = _ctopk_solo_col[in_dtype](tid, count_trailing_zeros(m))
        m &= m - 1
        sink.store(out_base + Int(gp), c.cast[.int32]())
        gp += 1
    m = emask
    while m != 0:
        var c = _ctopk_solo_col[in_dtype](tid, count_trailing_zeros(m))
        m &= m - 1
        if exact:
            sink.store(out_base + Int(above + ep), c.cast[.int32]())
        elif listed:
            ties[Int(ep)] = (
                UInt64(_ctopk_solo_phi(in_scores, row_base, c)) << 32
            ) | UInt64(~c)
        ep += 1
    if exact:
        return

    if listed:
        # Exact key high, complemented column low: a strict total order, so
        # the bin's ranks are a permutation and exactly `need` of them land.
        # Each of the first `eq` threads ranks its own key against the whole
        # list, read as warp-uniform broadcasts, eight keys a step: the
        # prologue zeroed the list, and a zero key outranks nothing.
        barrier()
        if tid < eq:
            var mine = ties[Int(tid)]
            var rank = UInt32(0)
            var j = 0
            while j < Int(eq):
                comptime for q in range(4):
                    var kp = ties.unsafe_load[width=2, alignment=16](j + 2 * q)
                    rank += UInt32(kp[0] > mine) + UInt32(kp[1] > mine)
                j += 8
            if rank < need:
                sink.store(
                    out_base + Int(above + rank),
                    (~mine.cast[.uint32]()).cast[.int32](),
                )
        return

    # A bin too wide to rank: radix rounds over the exact key of its members,
    # 8 bits a round down to the score's last significant bit.
    comptime rounds = _hsel_sig_bits[in_dtype]() // 8
    var active = emask
    var base = above
    var remain = need
    comptime for r in range(rounds):
        comptime sh = 24 - 8 * r
        if tid < 256:
            hist[Int(tid)] = 0
        barrier()
        # Walked by set bit rather than unrolled: this path is rare, and the
        # unrolled re-reads in flight would set the kernel's register peak.
        m = active
        while m != 0:
            var c = _ctopk_solo_col[in_dtype](tid, count_trailing_zeros(m))
            m &= m - 1
            var p = _ctopk_solo_phi(in_scores, row_base, c)
            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](hist.unsafe_offset(Int((p >> UInt32(sh)) & 0xFF)), UInt32(1))
        barrier()
        _ctopk_find_threshold[256, with_pre=False](
            hist[Int(tid)] if tid < 256 else UInt32(0),
            wsum,
            mtch,
            Int(remain),
            UInt32(0),
        )
        var dthr = mtch[0]
        var na = mtch[1]
        var nv = mtch[2]
        # The last round's bracket is one exact key, as is any bracket that
        # fills the row exactly: either way its members are a plateau.
        var last = r == rounds - 1 or na + nv == remain
        m = active
        while m != 0:
            var bit = count_trailing_zeros(m)
            m &= m - 1
            var c = _ctopk_solo_col[in_dtype](tid, bit)
            var d = (
                _ctopk_solo_phi(in_scores, row_base, c) >> UInt32(sh)
            ) & 0xFF
            if d > dthr:
                var p = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl.unsafe_offset(1).unsafe_offset(2 * r), UInt32(1))
                sink.store(out_base + Int(base + p), c.cast[.int32]())
                active &= ~(UInt32(1) << bit)
            elif d < dthr:
                active &= ~(UInt32(1) << bit)
            elif last:
                var p = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl.unsafe_offset(2).unsafe_offset(2 * r), UInt32(1))
                if p < remain - na:
                    sink.store(out_base + Int(base + na + p), c.cast[.int32]())
        if last:
            return
        base += na
        remain -= na


@inline(.never)
def _ctopk_solo_entry[
    in_dtype: DType, //, loads: Int, S: TopkSink
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin],
    sink: S,
    row_base: Int,
    out_base: Int,
    count: Int,
    K: Int,
    smem: Pointer[UInt8, MutUntrackedOrigin, address_space=.SHARED],
):
    """The single-CTA path's dispatch on payload width, kept out of line.

    Inlined, it would share the kernel's register allocation and instruction
    stream with the cluster path, which measured 1-2% slower on rows the
    cluster takes. It runs once per CTA, so the call costs little. The
    argument arrives as a generic pointer, so it is cast back to the global
    space for the loads to stay `ld.global.nc`. `loads` is the caller's
    `_ctopk_solo_loads` rather than its cluster size, so the kernels of one
    width share one copy: a copy per size spilled the bounded f32 10-rank
    kernel under the driver JIT.
    """
    var gscores = in_scores.unsafe_address_space_cast[.GLOBAL]()
    comptime grp = _ctopk_solo_group[in_dtype]()
    # The payload width follows the row, since every pass tests all of a
    # thread's words.
    var ucount = UInt32(count)
    comptime for w in [1, 2, 3, 4, 6, 8]:
        comptime if w <= loads:
            if ucount <= UInt32(w * grp):
                _ctopk_solo_select[w](
                    gscores,
                    sink,
                    row_base,
                    out_base,
                    ucount,
                    UInt32(K),
                    smem,
                )
                return


def _topk_cluster_impl[
    in_dtype: DType, CS: Int, TMA: Bool, S: TopkSink
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin],
    sink: S,
    N: Int,
    K: Int,
    count: Int,
):
    """Top-K over one row per cluster, the body both entry points share.

    `count` is the row's live column count (the full stride `N` unbounded, the
    `RowBound` bound otherwise), clamped by the callers. `count <= K` is the
    trivial path: every live column is selected, rank 0 writes `[0, count)`
    and a `-1` tail. A row no longer than `_ctopk_solo_cap` is the mid-length
    path, vLLM's `histogram_4096_topk`: rank 0 selects it alone in registers
    (`_ctopk_solo_select`) and the other ranks exit. Otherwise:

    1. The row's `[0, count)` is partitioned into `CS` 16 B-aligned slices,
       one per rank, load-balanced (vLLM's split).
    2. Each rank histograms its slice into the 1024 bins of `_ctopk_key0`,
       and every rank finds the same split bin by reading its peers'
       histograms over DSMEM: from 4 ranks only their 32-bin group totals
       and then the split group's bins (`_ctopk_gather_threshold`). A slice that fits
       the staging is read from global once and rescanned from shared
       memory after (single-pass); a longer one streams, re-reading global
       per pass -- through the `cp.async.bulk` double buffer when `TMA`,
       plain prefetched loads otherwise.
    3. One scatter stages each rank's columns above the split bin and the
       bin's candidates as `(column, phi)` pairs, and each peer appends its
       pairs to rank 0's list over DSMEM. A prefix sum of the published
       above counts assigns each rank its slots in `[0, above)`; rank 0 runs
       the remaining radix rounds over the candidates alone, with no
       cluster barrier, and writes `[above, K)`.
    4. If the split bin holds more than rank 0's list, or one peer more than
       `_CTOPK_CAND_CAP` (a tie-dense row), every rank instead refines the
       split bin over its whole slice: 10-bit rounds on `phi`, each with its
       DSMEM reduce, then a scatter of the strictly-above columns and the
       exact plateau, placed by the same prefix sum. Both decisions read
       state every rank holds identically, so the cluster stays on one
       path.

    The selection is the `ordered=False, deterministic=False` contract: the
    strictly-above columns are all present, the exact-score plateau at the
    threshold is cut first-come, and nothing about the arrangement within a
    class is promised.
    """
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var rank = Int(block_rank_in_cluster())

    # One dynamic shared-memory block, carved by the offsets above. Everything
    # the kernel needs is here; static shared memory tops out at 48 KiB, and
    # the slice staging alone takes most of the opt-in.
    var smem = external_memory[
        UInt8, address_space=.SHARED, alignment=_V4_ALIGN
    ]()
    var hist = smem.unsafe_bitcast[UInt32]()
    var hist0 = (smem.unsafe_offset(_CTOPK_OFF_HIST0)).unsafe_bitcast[UInt32]()
    var wsum = (smem.unsafe_offset(_CTOPK_OFF_WSUM)).unsafe_bitcast[UInt32]()
    var mtch = (smem.unsafe_offset(_CTOPK_OFF_MTCH)).unsafe_bitcast[UInt32]()
    var ctl = (smem.unsafe_offset(_CTOPK_OFF_CTL)).unsafe_bitcast[UInt32]()
    var counts = (smem.unsafe_offset(_CTOPK_OFF_COUNTS)).unsafe_bitcast[
        UInt32
    ]()
    var cbar = (smem.unsafe_offset(_CTOPK_OFF_CBAR)).unsafe_bitcast[
        SharedMemBarrier
    ]()
    var pbar = (smem.unsafe_offset(_CTOPK_OFF_PBAR)).unsafe_bitcast[
        SharedMemBarrier
    ]()
    comptime off_sel = _ctopk_off_sel[CS]()
    comptime off_stage = _ctopk_off_stage[CS]()
    var selbuf = (smem.unsafe_offset(off_sel)).unsafe_bitcast[Int32]()
    var stage = (smem.unsafe_offset(off_stage)).unsafe_bitcast[
        Scalar[_score_bits[in_dtype]()]
    ]()
    var mbar = (
        smem.unsafe_offset(_ctopk_off_mbar[in_dtype, CS]())
    ).unsafe_bitcast[SharedMemBarrier]()

    var out_base = row * K

    # Trivial path: a short row selects every live column. Cluster-uniform
    # (`count` is the same row for every rank), so the early return cannot
    # strand a peer at a cluster barrier.
    if count <= K:
        if rank == 0:
            var q = tid
            while q < K:
                sink.store(out_base + q, Int32(q) if q < count else Int32(-1))
                q += _CTOPK_BLOCK
        return

    # Mid-length path: rank 0 selects the row alone and every other rank
    # exits. Cluster-uniform for the same reason as the trivial path, and no
    # cluster barrier or DSMEM access follows it, so no live rank can wait on
    # or address an exited one.
    if count <= _ctopk_solo_cap[in_dtype, CS]():
        if rank == 0:
            _ctopk_solo_entry[_ctopk_solo_loads[in_dtype, CS]()](
                in_scores, sink, row * N, out_base, count, K, smem
            )
        return

    # 16 B-aligned slice split, load-balanced over ranks: `mla_index_fp8`
    # pads the row stride but not the bound, so the last slice clamps to
    # `count` and a rank past it can end up empty.
    comptime align_u = _elems_per_16b[in_dtype]()
    comptime scan_items = _HSEL_SCAN_ITEMS
    # 32-bit: `count` is bounded by the `Int32` stride, and a 64-bit divide by
    # a non-power-of-two `CS` is a wide magic multiply.
    var units = UInt32(ceildiv(count, align_u))
    var base_units = units // UInt32(CS)
    var extra = units - base_units * UInt32(CS)
    var urank = UInt32(rank)
    var ustart = (urank * base_units + min(urank, extra)) * UInt32(align_u)
    var uend = min(
        ustart
        + (base_units + UInt32(1 if urank < extra else 0)) * UInt32(align_u),
        UInt32(count),
    )
    var my_start = Int(ustart)
    var len_u = uend - min(ustart, uend)
    comptime if CS > 8:
        len_u = _ctopk_opaque(len_u)
    var my_len = Int(len_u)
    var row_base = row * N + my_start
    var single_pass = my_len <= _ctopk_res_elems[in_dtype]()
    # Slice positions are 32-bit: a slice is at most one row, whose stride the
    # launch passes as `Int32`. The row column is `col0 + position`.
    var src = in_scores.unsafe_address_space_cast[.GLOBAL]().unsafe_offset(
        row_base
    )
    var len32 = UInt32(my_len)
    var col0 = Int32(my_start)
    var aligned = (row_base & (align_u - 1)) == 0

    # The streaming flavor's mbarriers, one fresh pair per pass. Published by
    # the histogram zeroing's trailing `barrier()` below, which every round
    # runs before its first bulk copy. `cbar` takes the candidate path's
    # counts exchange: thread 0's arrival registers the CS two-word messages
    # every rank (this one included) will `st.async` into `counts`; rank 0's
    # `pbar` takes the peers' candidate pairs, its bytes registered once the
    # counts say how many. The mbarrier-init fence followed by round 0's
    # cluster barrier is the PTX ISA's pattern for publishing an mbarrier to
    # the cluster before peers signal it.
    comptime if TMA:
        if tid < _ctopk_nbar[in_dtype]():
            mbar[tid].init()
    if tid == 0:
        cbar[].init()
        cbar[].expect_bytes(Int32(4 * 2 * CS))
        pbar[].init()
    fence_mbarrier_init()

    comptime sig_bits = _hsel_sig_bits[in_dtype]()
    comptime pad = 32 - sig_bits
    comptime assert (
        _CTOPK_BINS == _CTOPK_BLOCK
    ), "one histogram bin per thread: the zeroing below is one store each"
    var root = rank == 0

    # Round 0: every column binned by `_ctopk_key0`, into its own `hist0`.
    # The barrier also publishes the zeroed `hist` that rank 0's first
    # refinement round, or the multi-round path's first round, bins into.
    hist0[tid] = 0
    hist[tid] = 0
    if tid < 8:
        ctl[tid] = 0
    barrier()
    if single_pass:
        _ctopk_scan[
            scan_items,
            count_all=True,
            from_stage=False,
            stage_slice=True,
            prefetch=True,
        ](
            src,
            len32,
            aligned,
            hist0,
            stage,
            UInt32(0),
            UInt32.MAX,
            UInt32(0),
            UInt32(0),
        )
    else:
        comptime if TMA:
            _ctopk_tma_stream[
                scan_items,
                count_all=True,
                hist_mode=True,
                cand=False,
            ](
                in_scores,
                row_base,
                len32,
                col0,
                hist0,
                stage,
                mbar,
                UInt32(0),
                ctl,
                selbuf,
                K,
                UInt32(0),
                UInt32.MAX,
                UInt32(0),
                UInt32(0),
                sink,
                out_base,
                root,
            )
        else:
            _ctopk_scan[
                scan_items,
                count_all=True,
                from_stage=False,
                stage_slice=False,
                prefetch=True,
            ](
                src,
                len32,
                aligned,
                hist0,
                stage,
                UInt32(0),
                UInt32.MAX,
                UInt32(0),
                UInt32(0),
            )
    # One cluster barrier, then every rank reads the round's bins from its
    # peers. Before it a rank has written only its own shared memory
    # (`hist0`, its group totals, the zeroing, `ctl`, the staging) and read
    # global, so its arrive releases only shared::cta writes. No rank exits
    # before the counts exchange, which each sends after its read. From 4
    # ranks the two-level read wins (1-1.5% at 4, 6-7% at 8 and 10, on
    # B200); at 2 and 3 reading every bin is as fast or faster.
    comptime if CS > 3:
        _ctopk_publish_sums(hist0, wsum)
        _ctopk_cluster_sync_cta_release()
        _ctopk_gather_threshold[CS](
            hist0, wsum, hist, mtch, UInt32(K), UInt32(rank)
        )
    else:
        _ctopk_cluster_sync_cta_release()
        var v0, pre0 = _ctopk_hist_gather[CS, peer_prefix=(CS > 2)](hist0, rank)
        _ctopk_find_threshold[_CTOPK_BINS, with_pre=(CS > 2)](
            v0, wsum, mtch, K, pre0
        )

    # `[slo, shi]` brackets the K-th largest key's phi; `need` is how many of
    # the top K the bracket still owes after the columns above it.
    var slo, shi = _ctopk_key0_bounds(mtch[0])
    var need = K - Int(mtch[1])
    var above0 = Int(mtch[1])
    var done = Int(mtch[2]) == need
    # Every rank holds the same reduced bins, so this is cluster-uniform: a
    # split bin past rank 0's list skips straight to the multi-round path.
    var try_cand = Int(mtch[2]) <= PERSISTENT_TOPK_MAX_N

    if try_cand:
        # Candidate path: one scatter stages the columns above the split bin
        # and the bin's `(column, phi)` candidates; every peer appends its
        # candidates to rank 0's list, and rank 0 alone resolves the exact
        # threshold over them.
        if single_pass:
            _ctopk_scatter[
                scan_items, from_stage=True, prefetch=False, cand=True
            ](
                src,
                col0,
                len32,
                aligned,
                stage,
                shi,
                slo,
                ctl,
                selbuf,
                K,
                sink,
                out_base,
                root,
            )
        else:
            comptime if TMA:
                _ctopk_tma_stream[
                    scan_items,
                    count_all=False,
                    hist_mode=False,
                    cand=True,
                ](
                    in_scores,
                    row_base,
                    len32,
                    col0,
                    hist,
                    stage,
                    mbar.unsafe_offset(_CTOPK_TMA_STAGES),
                    shi,
                    ctl,
                    selbuf,
                    K,
                    slo,
                    shi,
                    UInt32(0),
                    UInt32(0),
                    sink,
                    out_base,
                    root,
                )
            else:
                _ctopk_scatter[
                    scan_items, from_stage=False, prefetch=True, cand=True
                ](
                    src,
                    col0,
                    len32,
                    aligned,
                    stage,
                    shi,
                    slo,
                    ctl,
                    selbuf,
                    K,
                    sink,
                    out_base,
                    root,
                )
        barrier()

        # A peer pushes its pairs into rank 0's list from the top, below the
        # lower-ranked peers' (round 0's peer prefix of the split bin, which
        # counts exactly the pairs those peers staged); rank 0's own pairs
        # fill it from the bottom. A count past its cap is published as
        # `cap + 1`, so every rank reads the same overflow verdict off the
        # same counts and takes the same branch.
        #
        # Pairs and counts both travel by `st.async`: each rank sends every
        # rank (itself included) `(counts, pairs pushed)` on that rank's
        # `cbar`, and each pair completes on rank 0's `pbar`, whose bytes
        # rank 0 registers once its counts say how many were pushed (an
        # mbarrier's tx-count may run negative before that arrival). No
        # cluster barrier closes the exchange. Exit safety: a CTA's shared
        # memory is touched remotely only by peers' round-0 gathers (loads
        # whose values a peer consumed before computing the counts it sends)
        # and by the exchange's stores into it. A rank leaves only after its
        # `cbar` wait, which completes once every rank's counts landed, and
        # rank 0 also after its `pbar` wait, once every pair landed; so no
        # peer reads or writes a CTA that has exited. Its own stores into a
        # peer may still be in flight at exit; that peer waits for them.
        var lc = Int(ctl[1])
        var pushed = UInt32(0)
        if not root and lc <= _CTOPK_CAND_CAP:
            # 32-bit: the room above the lower peers' pairs is below 2^11.
            var room = UInt32(PERSISTENT_TOPK_MAX_N)
            comptime if CS > 2:
                room -= min(mtch[3], room)
            pushed = min(UInt32(lc), room)
            var mine = _ctopk_cand_list(selbuf, False)
            var dst = _ctopk_cand_list(selbuf, True)
            var i = UInt32(tid)
            while i < pushed:
                _ctopk_st_async(
                    dst.unsafe_offset(Int(2 * (room - 1 - i))),
                    pbar,
                    UInt32(0),
                    mine.unsafe_load[width=2, alignment=8](Int(2 * i)),
                )
                i += UInt32(_CTOPK_BLOCK)
        if tid < CS:
            var lcw = UInt32(min(lc, _ctopk_cand_cap(root) + 1))
            _ctopk_st_async(
                counts.unsafe_offset(2 * rank),
                cbar,
                UInt32(tid),
                SIMD[.uint32, 2]((lcw << 16) | ctl[0], pushed),
            )
        if tid == 0:
            cbar[].wait_acquire[Scope.CLUSTER](0)
            if root:
                var n_pushed = UInt32(0)
                comptime for j in range(1, CS):
                    n_pushed += counts[2 * j + 1]
                pbar[].expect_bytes(Int32(8 * n_pushed))
                pbar[].wait_acquire[Scope.CLUSTER](0)
        barrier()

        var fits = Int(counts[0] >> 16) <= PERSISTENT_TOPK_MAX_N
        var my_above = 0
        var n_cand = 0
        comptime for j in range(CS):
            var cj = counts[2 * j]
            comptime if j > 0:
                fits = fits and Int(cj >> 16) <= _CTOPK_CAND_CAP
            if j < rank:
                my_above += Int(cj & UInt32(0xFFFF))
            n_cand += Int(cj >> 16)
        fits = fits and n_cand <= PERSISTENT_TOPK_MAX_N

        if fits:
            # No DSMEM access follows, so a peer may exit as soon as its own
            # columns are out.
            if not root:
                var la0 = Int(ctl[0])
                var i0 = tid
                while i0 < la0:
                    sink.store(out_base + my_above + i0, selbuf[i0])
                    i0 += _CTOPK_BLOCK
                return

            # Rank 0's list: its own pairs at `[0, lc)`, the peers' at the
            # top. Radix rounds over the candidates' `phi`, below the bits
            # the bin already fixed, until the bracket is one key, the split
            # bin is taken whole, or it holds no more than a warp (`few`).
            # Only f32 brackets outlive one round: a bf16 interior bin's
            # round reaches the pad, so bf16 compiles `few` out.
            comptime few_ok = in_dtype == DType.float32
            var cl = _ctopk_cand_list(selbuf, True)
            var gap = PERSISTENT_TOPK_MAX_N - n_cand
            var w = 32 - Int(count_leading_zeros(slo ^ shi))
            var blo = slo & ~((UInt32(1) << UInt32(w)) - 1)
            var first = True
            var few = False
            while w > pad and not done and not few:
                var w_out = max(pad, w - _CTOPK_HIST_BITS)
                var dmask = (UInt32(1) << UInt32(w - w_out)) - 1
                if not first:
                    hist[tid] = 0
                    barrier()
                first = False
                var g = tid
                while g < n_cand:
                    var k = cl[2 * (g if g < lc else g + gap) + 1]
                    if k >= slo and k <= shi:
                        _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                            ordering=Ordering.RELAXED
                        ](
                            hist.unsafe_offset(
                                Int((k >> UInt32(w_out)) & dmask)
                            ),
                            UInt32(1),
                        )
                    g += _CTOPK_BLOCK
                barrier()
                _ctopk_find_threshold[_CTOPK_BINS, with_pre=False](
                    hist[tid], wsum, mtch, need, UInt32(0)
                )
                blo = blo | (mtch[0] << UInt32(w_out))
                slo = max(slo, blo)
                shi = min(shi, blo | ((UInt32(1) << UInt32(w_out)) - 1))
                need -= Int(mtch[1])
                w = w_out
                done = Int(mtch[2]) == need
                comptime if few_ok:
                    few = w > pad and not done and Int(mtch[2]) <= WARP_SIZE

            # Candidates above the bracket all go; the bracket's own are the
            # exact plateau (or a bin taken whole), cut first-come. A `few`
            # bracket is still unresolved: its pairs park in the bins, dead
            # since the last threshold scan, for one warp to rank.
            var n_hi = K - above0 - need
            var parked = hist
            var g = tid
            while g < n_cand:
                var e = cl.unsafe_load[width=2, alignment=8](
                    2 * (g if g < lc else g + gap)
                )
                if e[1] > shi:
                    var p = Int(
                        Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                            ordering=Ordering.RELAXED
                        ](ctl.unsafe_offset(5), UInt32(1))
                    )
                    sink.store(out_base + above0 + p, Int32(e[0]))
                elif e[1] >= slo:
                    var p = Int(
                        Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                            ordering=Ordering.RELAXED
                        ](ctl.unsafe_offset(6), UInt32(1))
                    )
                    if few_ok and few:
                        parked.unsafe_store[width=2, alignment=8](2 * p, e)
                    elif p < need:
                        sink.store(out_base + above0 + n_hi + p, Int32(e[0]))
                g += _CTOPK_BLOCK
            comptime if few_ok:
                if few:
                    # Lane `i`'s rank is the count of greater keys plus equal
                    # keys parked before it: distinct in `[0, m)`, so the `need`
                    # lowest take the bracket's slots, the plateau first-come.
                    barrier()
                    if tid < WARP_SIZE:
                        var m = Int(mtch[2])
                        var e = SIMD[.uint32, 2](0)
                        if tid < m:
                            e = parked.unsafe_load[width=2, alignment=8](
                                2 * tid
                            )
                        var r = 0
                        for j in range(m):
                            var kj = warp.shuffle_idx(e[1], UInt32(j))
                            if kj > e[1] or (kj == e[1] and j < tid):
                                r += 1
                        if tid < m and r < need:
                            sink.store(
                                out_base + above0 + n_hi + r, Int32(e[0])
                            )
            return
        # Every rank reads its verdict off `counts` before the multi-round
        # path may rewrite them.
        cluster_sync()

    # Multi-round path, for a split bin too dense for the candidate list.
    # It refines round 0's bracket over the whole slice, one cluster reduce
    # per round, until the bracket is one key or the split bin is taken
    # whole; the single-pass flavor rescans the staging round 0 wrote.
    var w = 32 - Int(count_leading_zeros(slo ^ shi))
    var blo = slo & ~((UInt32(1) << UInt32(w)) - 1)
    comptime for r in range(1, _ctopk_rounds[in_dtype]() + 1):
        if w <= pad or done:
            break
        var w_out = max(pad, w - _CTOPK_HIST_BITS)
        var sh = UInt32(w_out)
        var dmask = (UInt32(1) << UInt32(w - w_out)) - 1
        comptime if r > 1:
            hist[tid] = 0
        barrier()
        if single_pass:
            _ctopk_scan[
                scan_items,
                count_all=False,
                from_stage=True,
                stage_slice=False,
                prefetch=False,
            ](src, len32, aligned, hist, stage, slo, shi, sh, dmask)
        else:
            comptime if TMA:
                # The scatter-side arguments are the driver's `hist_mode`
                # dummies: every consumer read sits under the `comptime if`
                # above, so they go unread this pass.
                _ctopk_tma_stream[
                    scan_items,
                    count_all=False,
                    hist_mode=True,
                    cand=False,
                ](
                    in_scores,
                    row_base,
                    len32,
                    col0,
                    hist,
                    stage,
                    mbar.unsafe_offset((r + 1) * _CTOPK_TMA_STAGES),
                    UInt32(0),
                    ctl,
                    selbuf,
                    K,
                    slo,
                    shi,
                    sh,
                    dmask,
                    sink,
                    out_base,
                    False,
                )
            else:
                _ctopk_scan[
                    scan_items,
                    count_all=False,
                    from_stage=False,
                    stage_slice=False,
                    prefetch=True,
                ](src, len32, aligned, hist, stage, slo, shi, sh, dmask)

        _ctopk_hist_reduce[CS, _CTOPK_BINS](hist)
        _ctopk_find_threshold[_CTOPK_BINS, with_pre=False](
            hist[tid], wsum, mtch, need, UInt32(0)
        )
        blo = blo | (mtch[0] << sh)
        slo = max(slo, blo)
        shi = min(shi, blo | ((UInt32(1) << sh) - 1))
        need -= Int(mtch[1])
        w = w_out
        done = Int(mtch[2]) == need

    # Scatter: stage this rank's selections, then place them by a cluster
    # prefix sum (vLLM's `s_local_counts` scheme, the tie count split into its
    # own word so a 16-bit field never has to hold a slice length). The tie
    # class is the bracket: one key once resolved, or a split bin taken whole.
    if tid == 0:
        ctl[0] = 0
        ctl[1] = 0
    barrier()

    if single_pass:
        _ctopk_scatter[scan_items, from_stage=True, prefetch=False, cand=False](
            src,
            col0,
            len32,
            aligned,
            stage,
            shi,
            slo,
            ctl,
            selbuf,
            K,
            sink,
            out_base,
            False,
        )
    else:
        comptime if TMA:
            # The histogram-side arguments are the driver's scatter dummies,
            # comptime-elided along with the reads they feed.
            _ctopk_tma_stream[
                scan_items,
                count_all=False,
                hist_mode=False,
                cand=False,
            ](
                in_scores,
                row_base,
                len32,
                col0,
                hist,
                stage,
                mbar.unsafe_offset(
                    (_ctopk_rounds[in_dtype]() + 2) * _CTOPK_TMA_STAGES
                ),
                shi,
                ctl,
                selbuf,
                K,
                slo,
                shi,
                UInt32(0),
                UInt32(0),
                sink,
                out_base,
                False,
            )
        else:
            _ctopk_scatter[
                scan_items, from_stage=False, prefetch=True, cand=False
            ](
                src,
                col0,
                len32,
                aligned,
                stage,
                shi,
                slo,
                ctl,
                selbuf,
                K,
                sink,
                out_base,
                False,
            )
    barrier()

    var la = Int(ctl[0])
    var le = min(Int(ctl[1]), K)
    if tid < CS:
        var packed = (UInt32(le) << 16) | UInt32(la)
        # Publish my counts into every peer's `counts[my_rank]`.
        store_cluster_smem[.uint32, 1](
            counts.unsafe_offset(rank), UInt32(tid), SIMD[.uint32, 1](packed)
        )
    cluster_sync()

    # Every CTA holds the same CS counts; thread 0's serial prefix gives each
    # rank its slot base and the row totals.
    if tid == 0:
        var prefix = UInt32(0)
        var ta = UInt32(0)
        var te = UInt32(0)
        comptime for i in range(CS):
            if i == rank:
                ctl[2] = prefix
            ta += counts[i] & UInt32(0xFFFF)
            te += counts[i] >> 16
            prefix += counts[i]
        ctl[3] = ta
        ctl[4] = te
    barrier()

    var prefix_above = Int(ctl[2] & UInt32(0xFFFF))
    var prefix_equal = Int(ctl[2] >> 16)
    var total_above = Int(ctl[3])
    var total_equal = Int(ctl[4])

    var i = tid
    while i < la:
        sink.store(out_base + prefix_above + i, selbuf[i])
        i += _CTOPK_BLOCK
    i = tid
    while i < le:
        var p = total_above + prefix_equal + i
        if p < K:
            sink.store(out_base + p, selbuf[K + i])
        i += _CTOPK_BLOCK

    # In contract every slot is already written: the threshold rounds
    # guarantee `total_above + total_equal >= K`. The fill only covers rows a
    # contract violation (NaN scores) leaves short, and writes slots no peer
    # wrote: the equal class never reaches past `total_above + total_equal`.
    if rank == 0:
        var filled = min(K, total_above + total_equal)
        var s = tid + filled
        while s < K:
            sink.store(out_base + s, Int32(-1))
            s += _CTOPK_BLOCK


# The shared-memory carve already holds a rank to one CTA per SM, so the
# register budget is the full 64 per thread. Without the hint ptxas targets two
# 1024-thread CTAs (32 registers) and spills the bf16 kernels to get there.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_CTOPK_BLOCK)),
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__name(t"topk_cluster_{in_dtype}_cs{CS}_tma{Int(TMA)}")
def _topk_cluster_kernel[
    in_dtype: DType, CS: Int, TMA: Bool
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: Pointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
):
    """`_topk_cluster_impl` over the full row width `N`."""
    _topk_cluster_impl[in_dtype, CS, TMA, PlainSink](
        in_scores, PlainSink(out_idxs), Int(N), Int(K), Int(N)
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_CTOPK_BLOCK)),
)
@__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
@__name(t"topk_cluster_bounded_{in_dtype}_cs{CS}_tma{Int(TMA)}")
def _topk_cluster_bounded_kernel[
    in_dtype: DType, CS: Int, TMA: Bool, RB: RowBound
](
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: Pointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    row_bounds: RB,
):
    """`_topk_cluster_impl` with a per-row live-column bound.

    Same contract as the other bounded kernels: row `r` treats only
    `[0, row_bounds(r))` of its `N`-wide stripe as live. Columns past the
    bound are never selected and pad the output with `-1`; the plain loads
    never read them, and the TMA streaming's 16 B copy rounding reads no
    further than the padded row's end.
    """
    var bound, sink = row_bounds.row(Int(block_idx.x), out_idxs)
    _topk_cluster_impl[in_dtype, CS, TMA, RB.Sink](
        in_scores, sink, Int(N), Int(K), min(Int(N), max(0, bound))
    )


# ===----------------------------------------------------------------------=== #
# Host launcher
# ===----------------------------------------------------------------------=== #


comptime _CTOPK_CLUSTER_SIZES: List[Int] = [2, 3, 4, 5, 6, 8, 10, 16]


def _ctopk_resident_clusters[sm_count: Int](cs: Int) -> Int:
    """Clusters of `cs` ranks that can be resident at once on `sm_count` SMs.

    A rank holds a whole SM (1024 threads, the full shared-memory opt-in) and
    a cluster must fit inside one GPC, so the count is `clusters_per_wave`'s
    per-GPC packing, which on B200 equals `cudaOccupancyMaxActiveClusters`
    for this kernel at every size and puts the measured second-wave cliff of
    each size exactly where it predicts. A part `clusters_per_wave` does not
    model (H100 among them) gets the power-of-two counts scaled from B200 by
    SM count, an estimate. A row past the count waits for a second wave,
    which costs about 1.5x.
    """
    comptime if sm_count == 148 or sm_count == 160:
        comptime for c in _CTOPK_CLUSTER_SIZES:
            if cs == c:
                return clusters_per_wave[c, sm_count]()
        return 0
    else:
        var at_148: Int
        if cs <= 2:
            at_148 = 74
        elif cs <= 4:
            at_148 = 33
        else:
            at_148 = 15
        return at_148 * sm_count // 148


def _ctopk_max_rows[sm_count: Int]() -> Int:
    """The most rows the launcher sends to the cluster kernel: as many as
    clusters of two keep resident in one wave. Past it a row waits for a
    second wave, and the selects that give each row one CTA, which then fill
    the GPU, are faster."""
    return _ctopk_resident_clusters[sm_count](2)


def _ctopk_passes(N: Int, cs: Int) -> Int:
    """Block-wide passes of `_HSEL_SCAN_ITEMS` columns per thread over a
    `cs`-way slice of an `N`-column row: what each round costs."""
    return ceildiv(ceildiv(N, cs), _CTOPK_BLOCK * _HSEL_SCAN_ITEMS)


def _choose_topk_cluster_size[sm_count: Int](rows: Int, N: Int) -> Int:
    """Cluster size for a launch of `rows` rows of stride `N` on `sm_count`
    SMs.

    The widest of 8, 6, 5, 4, 3 and 2 that keeps every row resident in one
    wave (8, 4 and 2 on a part `clusters_per_wave` does not model), so on
    B200 8 up to 15 rows, 6 up to 22, 5 up to 26, 4 up to 33, 3 up to 45 and
    2 beyond. Where 8 fits, 10 replaces it if it fits too and its slices
    take one pass fewer than 8's, and 16 replaces either if it fits and even
    10's slices take more than one pass: on B200, 16 up to 7 rows for
    strides past 81920 and 10 up to 11 for strides past 65536. Each rank's
    fixed cost (round 0's gather reads every peer, and the counts exchange)
    grows with the size, so a wider cluster pays only where the slices are
    long: at 81920, where 10's slice is one pass, 16 measured up to 5% slower
    than 10, and at 98304-163840 up to 14% faster than 8 or 10. On B200, over
    32768-163840 columns and 1-64 rows, this is within 3.4% of the best of
    every size at every point and 0.3% on the geomean, at both score dtypes;
    vLLM's row-count table is 9% off on the geomean and up to 1.45x. A
    stride that `_ctopk_solo_cap` covers launches without a cluster at any
    size.
    """
    comptime modeled = sm_count == 148 or sm_count == 160
    var cs = 8
    while cs > 2 and rows > _ctopk_resident_clusters[sm_count](cs):
        cs = (6 if cs == 8 else cs - 1) if modeled else cs // 2
    comptime if modeled:
        if cs == 8:
            if (
                rows <= _ctopk_resident_clusters[sm_count](16)
                and _ctopk_passes(N, 10) > 1
            ):
                return 16
            if rows <= _ctopk_resident_clusters[sm_count](10) and _ctopk_passes(
                N, 10
            ) < _ctopk_passes(N, 8):
                return 10
    return cs


def topk_cluster_split[
    in_dtype: DType, //, RB: RowBound = RowBoundsPtr
](
    ctx: DeviceContext,
    in_scores: Pointer[Scalar[in_dtype], ImmutAnyOrigin],
    out_idxs: Pointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    total_seq_len: Int,
    row_bounds: Optional[RB],
    cluster_size: Int,
    use_tma: Bool = False,
) raises:
    """Launch the cluster-cooperative top-k, one row per `cluster_size` CTAs.

    The `ordered=False, deterministic=False` contract of
    `persistent_topk_block_split`, and only that contract: the selection is
    exact on the strictly-above columns, the threshold plateau is cut
    first-come, and the arrangement is unspecified. The launcher
    `persistent_topk_block_split` routes here under its cluster gate; this
    entry point also exists so tests and benchmarks can pin `cluster_size`
    and reach the kernel at shapes the dispatch would not choose.

    The score-dtype and pointer-alignment contract is
    `persistent_topk_block`'s, unchanged: 16 B-aligned buffer, row stride a
    multiple of `_elems_per_16b[in_dtype]()` (correct but slower otherwise --
    an unpadded stride always takes the plain-load streaming kernel; a padded
    one can take the `cp.async.bulk` streaming kernel on SM100+, opt-in via
    `use_tma`).

    Parameters:
        in_dtype: Element type of the score buffer, f32 or bf16 (inferred).
        RB: Type of `row_bounds`; see `RowBound`.

    Args:
        ctx: Device context.
        in_scores: Flat score buffer `[total_seq_len x N]` row-major.
        out_idxs: Output buffer `[total_seq_len x K]` row-major (int32).
        N: Score columns per token (the row stride).
        K: Top-k count per token, at most `PERSISTENT_TOPK_MAX_N`.
        total_seq_len: Number of rows; one cluster of `cluster_size` CTAs each.
        row_bounds: Optional per-row live-column counts; see
            `persistent_topk_block`.
        cluster_size: CTAs per row: 2, 3, 4, 5, 6, 8, 10 or 16. A size past 8 is
            nonportable; the runtime allows it on every function it loads.
        use_tma: Whether the streaming flavor stages its slices with
            `cp.async.bulk` (SM100+, padded row stride only; clamped to the
            plain-load kernel otherwise). Defaults to off: the plain-load
            streaming measures faster on B200 (the plan doc's M3 table), so
            only tests pinning the flavor opt in.

    Raises:
        If `K` exceeds the selection width or `cluster_size` is not a
        supported size. A `K` of at most 0 selects nothing and launches
        nothing.
    """
    # The device side asserts again through the cluster primitives' own
    # `_is_sm_9x_or_newer` checks, which run in the kernel's offload
    # compilation.
    comptime assert (
        ctx.target._is_sm_9x_or_newer()
        and not ctx.target._is_sm_120x_or_newer()
    ), (
        "topk_cluster_split needs thread-block clusters on NVIDIA SM90+"
        " (SM12x excluded, as vLLM excludes capability family 120)"
    )
    if total_seq_len <= 0 or K <= 0:
        return
    if K > PERSISTENT_TOPK_MAX_N:
        raise Error(
            "topk_cluster_split selects at most ", PERSISTENT_TOPK_MAX_N
        )
    _assert_scores_16b_aligned(in_scores, N, total_seq_len)

    # The streaming flavor's `cp.async.bulk` serves a whole 16 B unit per
    # copy, so it wants the row stride padded the way `mla_index_fp8` pads
    # it -- a host-known property, so the variant is picked here and a
    # captured graph replays the same route. An unpadded stride keeps the
    # plain-load streaming and `persistent_topk_block`'s
    # correct-at-any-alignment contract; a target without the 1D bulk copy
    # does the same through the `comptime if` below. `use_tma` defaults to
    # off (the plain arm measures faster); tests pin it to cover the
    # pipeline.
    var want_tma = use_tma and N % _elems_per_16b[in_dtype]() == 0

    @inline(.always)
    def launch[CS: Int, TMA: Bool]() raises {imm}:
        # A stride rank 0 selects alone sends every row down the mid-length
        # path (`count <= N`), whose idle ranks only exit: launched without
        # the cluster, they cost neither the CTAs nor the co-scheduling, and
        # the block needs only that path's shared memory. Outside a cluster
        # launch every CTA is rank 0 of its own.
        var ranks = CS
        var cluster = OptionalReg[Dim](Dim(1, CS, 1))
        var smem_bytes = _ctopk_smem_bytes[in_dtype, CS]()
        if N <= _ctopk_solo_cap[in_dtype, CS]():
            ranks = 1
            cluster = None
            smem_bytes = _CTOPK_SOLO_SMEM
        if case .Some(rb) = row_bounds:
            ctx.enqueue_function[
                _topk_cluster_bounded_kernel[in_dtype, CS, TMA, RB]
            ](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                rb,
                grid_dim=(total_seq_len, ranks, 1),
                block_dim=_CTOPK_BLOCK,
                cluster_dim=cluster,
                shared_mem_bytes=smem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(_ctopk_smem_bytes[in_dtype, CS]())
                ),
            )
        else:
            ctx.enqueue_function[_topk_cluster_kernel[in_dtype, CS, TMA]](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                grid_dim=(total_seq_len, ranks, 1),
                block_dim=_CTOPK_BLOCK,
                cluster_dim=cluster,
                shared_mem_bytes=smem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(_ctopk_smem_bytes[in_dtype, CS]())
                ),
            )

    @inline(.always)
    def launch_cs[CS: Int]() raises {imm}:
        # The TMA variant elaborates only where the 1D bulk copy compiles,
        # which the copy asserts at comptime with `_is_sm_100x_or_newer`.
        comptime if ctx.target._is_sm_100x_or_newer():
            if want_tma:
                launch[CS, True]()
            else:
                launch[CS, False]()
        else:
            launch[CS, False]()

    if cluster_size == 2:
        launch_cs[2]()
    elif cluster_size == 3:
        launch_cs[3]()
    elif cluster_size == 4:
        launch_cs[4]()
    elif cluster_size == 5:
        launch_cs[5]()
    elif cluster_size == 6:
        launch_cs[6]()
    elif cluster_size == 8:
        launch_cs[8]()
    elif cluster_size == 10:
        launch_cs[10]()
    elif cluster_size == 16:
        launch_cs[16]()
    else:
        raise Error(
            "topk_cluster_split: cluster_size must be 2-6, 8, 10 or 16, got ",
            cluster_size,
        )
