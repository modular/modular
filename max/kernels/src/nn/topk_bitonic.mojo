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
"""Block-wide bitonic sort top-k for the MLA/MSA indexer.

Replaces `topk_gpu`, which is pathological when `k ≈ N` (e.g. the
`k = N = 2048` DeepSeek-V3 / MiniMax-M3 indexer config). For `N > 2048` a
streaming variant folds `TILE`-wide tiles into a running top-`TILE`
champion, selecting `K ≤ TILE` out of arbitrarily large `N`.
"""

from std.sys import align_of, size_of

from std.gpu import barrier, block_idx, thread_idx
import std.gpu.primitives.warp as warp
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import ceildiv
from std.utils.numerics import min_or_neg_inf
from layout import TileTensor, row_major

# ===----------------------------------------------------------------------=== #
# Compile-time constants
# ===----------------------------------------------------------------------=== #

# Locked together: _PTOPK_BLOCK * _PTOPK_ITEMS == _PTOPK_TOTAL,
# _PTOPK_LOG2 == log2(_PTOPK_TOTAL), and
# PERSISTENT_TOPK_MAX_N == _PTOPK_TOTAL. Changing the top-k capacity
# means changing all of them.
comptime PERSISTENT_TOPK_MAX_N: Int = 2048

comptime _PTOPK_BLOCK: Int = 512
comptime _PTOPK_ITEMS: Int = 4
comptime _PTOPK_TOTAL: Int = _PTOPK_BLOCK * _PTOPK_ITEMS
comptime _PTOPK_LOG2: Int = 11
comptime _TILE: Int = _PTOPK_TOTAL


@always_inline
def _load_score_and_index(
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    row: Int,
    e: Int,
    N: Int,
) -> Tuple[Scalar[DType.float32], Scalar[DType.int32]]:
    if e < N:
        return (in_scores[row + e], Int32(e))
    return (min_or_neg_inf[DType.float32](), Int32(-1))


@always_inline
def _select_lane_after_xor(
    v: Scalar[DType.float32],
    i: Scalar[DType.int32],
    pv: Scalar[DType.float32],
    pi: Scalar[DType.int32],
    want_d: Bool,
    is_lo: Bool,
) -> Tuple[Scalar[DType.float32], Scalar[DType.int32]]:
    var do_swap: Bool
    if is_lo:
        do_swap = (v < pv) == want_d
    else:
        do_swap = (pv < v) == want_d
    if do_swap:
        return (pv, pi)
    return (v, i)


@always_inline
def _swap_pair_if(
    v0: Scalar[DType.float32],
    i0: Scalar[DType.int32],
    v1: Scalar[DType.float32],
    i1: Scalar[DType.int32],
    want_d: Bool,
) -> Tuple[
    Scalar[DType.float32],
    Scalar[DType.int32],
    Scalar[DType.float32],
    Scalar[DType.int32],
]:
    if (v0 < v1) == want_d:
        return (v1, i1, v0, i0)
    return (v0, i0, v1, i1)


# ===----------------------------------------------------------------------=== #
# Bitonic sort core (register + warp-shuffle + swizzled-SMEM)
# ===----------------------------------------------------------------------=== #


@always_inline
def _bitonic_sort_desc[
    sv_origin: MutOrigin,
    si_origin: MutOrigin,
](
    mut v0: Scalar[DType.float32],
    mut v1: Scalar[DType.float32],
    mut v2: Scalar[DType.float32],
    mut v3: Scalar[DType.float32],
    mut i0: Scalar[DType.int32],
    mut i1: Scalar[DType.int32],
    mut i2: Scalar[DType.int32],
    mut i3: Scalar[DType.int32],
    smem_v: UnsafePointer[
        Scalar[DType.float32], sv_origin, address_space=AddressSpace.SHARED
    ],
    smem_i: UnsafePointer[
        Scalar[DType.int32], si_origin, address_space=AddressSpace.SHARED
    ],
    tid: Int,
):
    """Full descending bitonic sort of `_PTOPK_TOTAL` elements in place.

    Thread `t` owns canonical elements `e0=4t..e3=4t+3` in registers on
    entry and holds them sorted-descending on exit (position 0 = largest).
    `smem_v`/`smem_i` must be `_PTOPK_TOTAL`-wide; they are scratch
    (contents undefined on return).
    """
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    comptime for s in range(1, 8):  # stages 1..7
        comptime for h in reversed(range(1, s + 1)):  # h = s..1
            comptime stride = 1 << (h - 1)
            var want_d = ((e0 >> s) & 1) == 0

            comptime if stride == 1:
                var want_d0 = ((e0 >> s) & 1) == 0
                var want_d2 = ((e2 >> s) & 1) == 0
                v0, i0, v1, i1 = _swap_pair_if(v0, i0, v1, i1, want_d0)
                v2, i2, v3, i3 = _swap_pair_if(v2, i2, v3, i3, want_d2)

            elif stride == 2:
                v0, i0, v2, i2 = _swap_pair_if(v0, i0, v2, i2, want_d)
                v1, i1, v3, i3 = _swap_pair_if(v1, i1, v3, i3, want_d)

            else:
                comptime ts = stride >> 2
                var pv0 = warp.shuffle_xor(v0, UInt32(ts))
                var pi0 = warp.shuffle_xor(i0, UInt32(ts))
                var pv1 = warp.shuffle_xor(v1, UInt32(ts))
                var pi1 = warp.shuffle_xor(i1, UInt32(ts))
                var pv2 = warp.shuffle_xor(v2, UInt32(ts))
                var pi2 = warp.shuffle_xor(i2, UInt32(ts))
                var pv3 = warp.shuffle_xor(v3, UInt32(ts))
                var pi3 = warp.shuffle_xor(i3, UInt32(ts))
                var is_lo = (tid & ts) == 0
                v0, i0 = _select_lane_after_xor(v0, i0, pv0, pi0, want_d, is_lo)
                v1, i1 = _select_lane_after_xor(v1, i1, pv1, pi1, want_d, is_lo)
                v2, i2 = _select_lane_after_xor(v2, i2, pv2, pi2, want_d, is_lo)
                v3, i3 = _select_lane_after_xor(v3, i3, pv3, pi3, want_d, is_lo)

    # XOR bank-swizzle (bijection) to keep the SMEM stages bank-conflict-free.
    var sw0 = (e0 & ~31) | ((e0 ^ (e0 >> 5)) & 31)
    var sw1 = (e1 & ~31) | ((e1 ^ (e1 >> 5)) & 31)
    var sw2 = (e2 & ~31) | ((e2 ^ (e2 >> 5)) & 31)
    var sw3 = (e3 & ~31) | ((e3 ^ (e3 >> 5)) & 31)

    comptime for s in range(8, _PTOPK_LOG2 + 1):  # stages 8..11
        smem_v[sw0] = v0
        smem_i[sw0] = i0
        smem_v[sw1] = v1
        smem_i[sw1] = i1
        smem_v[sw2] = v2
        smem_i[sw2] = i2
        smem_v[sw3] = v3
        smem_i[sw3] = i3
        barrier()

        comptime for h in reversed(range(8, s + 1)):  # h = s..8
            comptime stride = 1 << (h - 1)  # S = 128, 256, 512, 1024
            var want_d = ((e0 >> s) & 1) == 0
            comptime for item in range(4):
                var ei = e0 + item
                var ej = ei ^ stride
                if (ei & stride) == 0:
                    var si = (ei & ~31) | ((ei ^ (ei >> 5)) & 31)
                    var sj = (ej & ~31) | ((ej ^ (ej >> 5)) & 31)
                    var vi = smem_v[si]
                    var vj = smem_v[sj]
                    if (vi < vj) == want_d:
                        smem_v[si] = vj
                        smem_v[sj] = vi
                        smem_i[si], smem_i[sj] = smem_i[sj], smem_i[si]
            barrier()

        v0 = smem_v[sw0]
        i0 = smem_i[sw0]
        v1 = smem_v[sw1]
        i1 = smem_i[sw1]
        v2 = smem_v[sw2]
        i2 = smem_i[sw2]
        v3 = smem_v[sw3]
        i3 = smem_i[sw3]

        comptime for h in reversed(range(3, 8)):  # h = 7..3
            comptime stride = 1 << (h - 1)
            comptime ts = stride >> 2  # thread-stride ∈ [1..16]
            var want_d = ((e0 >> s) & 1) == 0
            var pv0 = warp.shuffle_xor(v0, UInt32(ts))
            var pi0 = warp.shuffle_xor(i0, UInt32(ts))
            var pv1 = warp.shuffle_xor(v1, UInt32(ts))
            var pi1 = warp.shuffle_xor(i1, UInt32(ts))
            var pv2 = warp.shuffle_xor(v2, UInt32(ts))
            var pi2 = warp.shuffle_xor(i2, UInt32(ts))
            var pv3 = warp.shuffle_xor(v3, UInt32(ts))
            var pi3 = warp.shuffle_xor(i3, UInt32(ts))
            var is_lo = (tid & ts) == 0
            v0, i0 = _select_lane_after_xor(v0, i0, pv0, pi0, want_d, is_lo)
            v1, i1 = _select_lane_after_xor(v1, i1, pv1, pi1, want_d, is_lo)
            v2, i2 = _select_lane_after_xor(v2, i2, pv2, pi2, want_d, is_lo)
            v3, i3 = _select_lane_after_xor(v3, i3, pv3, pi3, want_d, is_lo)

        var want_d2 = ((e0 >> s) & 1) == 0
        if (v0 < v2) == want_d2:
            v0, v2 = v2, v0
            i0, i2 = i2, i0
        if (v1 < v3) == want_d2:
            v1, v3 = v3, v1
            i1, i3 = i3, i1

        var want_d1 = ((e0 >> s) & 1) == 0
        if (v0 < v1) == want_d1:
            v0, v1 = v1, v0
            i0, i1 = i1, i0
        if (v2 < v3) == want_d1:
            v2, v3 = v3, v2
            i2, i3 = i3, i2


# ===----------------------------------------------------------------------=== #
# GPU kernels
# ===----------------------------------------------------------------------=== #


@__name(t"persistent_topk_2048")
def _persistent_topk_2048_kernel(
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    out_idxs: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    N: Int,
    K: Int,
):
    """Block-wide bitonic top-k for `N <= _PTOPK_TOTAL` (one block per row)."""
    var tid = thread_idx.x
    var token = block_idx.x

    var smem_v = stack_allocation[
        _PTOPK_TOTAL, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var smem_i = stack_allocation[
        _PTOPK_TOTAL, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    var row = token * N
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var v0: Scalar[DType.float32]
    var v1: Scalar[DType.float32]
    var v2: Scalar[DType.float32]
    var v3: Scalar[DType.float32]
    var i0: Scalar[DType.int32]
    var i1: Scalar[DType.int32]
    var i2: Scalar[DType.int32]
    var i3: Scalar[DType.int32]

    v0, i0 = _load_score_and_index(in_scores, row, e0, N)
    v1, i1 = _load_score_and_index(in_scores, row, e1, N)
    v2, i2 = _load_score_and_index(in_scores, row, e2, N)
    v3, i3 = _load_score_and_index(in_scores, row, e3, N)

    _bitonic_sort_desc(v0, v1, v2, v3, i0, i1, i2, i3, smem_v, smem_i, tid)

    var base = token * K
    if e0 < K:
        out_idxs[base + e0] = i0
    if e1 < K:
        out_idxs[base + e1] = i1
    if e2 < K:
        out_idxs[base + e2] = i2
    if e3 < K:
        out_idxs[base + e3] = i3


@__name(t"streaming_topk")
def _streaming_topk_kernel(
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    out_idxs: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    N: Int,
    K: Int,
):
    """Streaming top-K for `N > _PTOPK_TOTAL`, `K <= _TILE` (one block per row).

    Folds each `_TILE`-wide tile of the row into a running sorted
    top-`_TILE` champion in SMEM (Batcher half-cleaner merge), then
    writes the top-`K` indices.
    """
    var tid = thread_idx.x
    var token = block_idx.x

    var champ_v = stack_allocation[
        _TILE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var champ_i = stack_allocation[
        _TILE, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()
    var scratch_v = stack_allocation[
        _TILE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var scratch_i = stack_allocation[
        _TILE, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    var row = token * N
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var neg_inf = min_or_neg_inf[DType.float32]()
    champ_v[e0] = neg_inf
    champ_i[e0] = Int32(-1)
    champ_v[e1] = neg_inf
    champ_i[e1] = Int32(-1)
    champ_v[e2] = neg_inf
    champ_i[e2] = Int32(-1)
    champ_v[e3] = neg_inf
    champ_i[e3] = Int32(-1)
    barrier()

    var v0: Scalar[DType.float32]
    var v1: Scalar[DType.float32]
    var v2: Scalar[DType.float32]
    var v3: Scalar[DType.float32]
    var i0: Scalar[DType.int32]
    var i1: Scalar[DType.int32]
    var i2: Scalar[DType.int32]
    var i3: Scalar[DType.int32]

    var num_tiles = ceildiv(N, _TILE)
    for t in range(num_tiles):
        var g = t * _TILE
        v0, i0 = _load_score_and_index(in_scores, row, g + e0, N)
        v1, i1 = _load_score_and_index(in_scores, row, g + e1, N)
        v2, i2 = _load_score_and_index(in_scores, row, g + e2, N)
        v3, i3 = _load_score_and_index(in_scores, row, g + e3, N)

        _bitonic_sort_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        # Barrier before reusing `scratch` as the stash: the sort's
        # final swizzled reads alias the canonical write indices (WAR hazard).
        barrier()

        scratch_v[e0] = v0
        scratch_i[e0] = i0
        scratch_v[e1] = v1
        scratch_i[e1] = i1
        scratch_v[e2] = v2
        scratch_i[e2] = i2
        scratch_v[e3] = v3
        scratch_i[e3] = i3
        barrier()

        # Each thread touches only its own champion slots e0..e3, so the
        # merge needs no barrier until the write-back below.
        comptime for _pair in range(4):
            var e = e0 + _pair
            var cv = champ_v[e]
            var ci = champ_i[e]
            var bv = scratch_v[_TILE - 1 - e]
            var bi = scratch_i[_TILE - 1 - e]
            var mv = cv
            var mi = ci
            if bv > cv:
                mv = bv
                mi = bi
            comptime if _pair == 0:
                v0 = mv
                i0 = mi
            elif _pair == 1:
                v1 = mv
                i1 = mi
            elif _pair == 2:
                v2 = mv
                i2 = mi
            else:
                v3 = mv
                i3 = mi
        # Barrier: finish all champion/scratch reads before the re-sort
        # below overwrites them.
        barrier()

        _bitonic_sort_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v[e0] = v0
        champ_i[e0] = i0
        champ_v[e1] = v1
        champ_i[e1] = i1
        champ_v[e2] = v2
        champ_i[e2] = i2
        champ_v[e3] = v3
        champ_i[e3] = i3
        barrier()

    var base = token * K
    if e0 < K:
        out_idxs[base + e0] = champ_i[e0]
    if e1 < K:
        out_idxs[base + e1] = champ_i[e1]
    if e2 < K:
        out_idxs[base + e2] = champ_i[e2]
    if e3 < K:
        out_idxs[base + e3] = champ_i[e3]


# ===----------------------------------------------------------------------=== #
# Host launcher
# ===----------------------------------------------------------------------=== #


def persistent_topk_block(
    ctx: DeviceContext,
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    out_idxs: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    N: Int,
    K: Int,
    total_seq_len: Int,
) raises:
    """Launch block-wide bitonic top-k for `total_seq_len` score rows.

    For `N ≤ PERSISTENT_TOPK_MAX_N` (= 2048) a single block sorts the whole row.
    For `N > PERSISTENT_TOPK_MAX_N` a streaming variant folds `_TILE`-wide tiles
    into a running top-`_TILE` champion; this requires `K ≤ PERSISTENT_TOPK_MAX_N`
    (the champion width).  Call sites needing `K > PERSISTENT_TOPK_MAX_N` must
    use `topk_gpu`.

    Each row of `N` float32 scores yields the `K` highest-scoring column indices
    (as int32) in descending score order in `out_idxs`.

    Args:
        ctx: Device context.
        in_scores: Flat score buffer `[total_seq_len × N]` row-major.
        out_idxs: Output buffer `[total_seq_len × K]` row-major (int32).
        N: Score columns per token.
        K: Top-k count per token (≤ N, and ≤ PERSISTENT_TOPK_MAX_N when N > 2048).
        total_seq_len: Number of rows (one block per row).
    """
    if N <= PERSISTENT_TOPK_MAX_N:
        ctx.enqueue_function[_persistent_topk_2048_kernel](
            in_scores,
            out_idxs,
            N,
            K,
            grid_dim=total_seq_len,
            block_dim=_PTOPK_BLOCK,
        )
    else:
        ctx.enqueue_function[_streaming_topk_kernel](
            in_scores,
            out_idxs,
            N,
            K,
            grid_dim=total_seq_len,
            block_dim=_PTOPK_BLOCK,
        )
