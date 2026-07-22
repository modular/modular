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
from std.gpu.host import DeviceContext, DeviceAttribute
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

# Each thread owns the `_PTOPK_ITEMS` (=4) contiguous canonical elements
# `e0..e3 = tid*4 + {0,1,2,3}`, so those byte offsets are a multiple of 16.
# `_V4_ALIGN` (= 16 B) is the alignment of a width-4 f32/i32 SIMD; the `alignment`
# argument on the width-4 loads/stores below is REQUIRED — the default (element
# alignment) makes LLVM legalize the vector op back into 4 scalar accesses.
comptime _V4_ALIGN: Int = align_of[SIMD[DType.float32, _PTOPK_ITEMS]]()


@always_inline
def _load4_scores(
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    base: Int,
    index_base: Int,
    local0: Int,
    count: Int,
) -> Tuple[SIMD[DType.float32, _PTOPK_ITEMS], SIMD[DType.int32, _PTOPK_ITEMS]]:
    """Load the 4 contiguous scores a thread owns, plus their column indices.

    Reads `in_scores[base + index_base + local0 + j]` for `j` in `0..3`, storing
    the row-global column `index_base + local0 + j` as the index so a merged
    partial still carries row-global indices. Positions past `count` pad with
    (-inf, -1).

    Emits a single 128-bit vector load on the fast path — taken when the 4
    elements are fully in-bounds and the address is 16B-aligned. `base`,
    `index_base` and the tile offset folded into `local0` are uniform across a
    block and `tid*4` is a multiple of 4, so the alignment test is uniform: it
    only varies with `base` (the row/slice origin). Odd `N` (e.g. decode-long
    `N = 32769`) makes odd rows' bases non-16B-aligned and the boundary tile
    partially out of bounds; both fall to the scalar path, which a width-4
    aligned load would fault on.
    """
    var col0 = index_base + local0
    var off = base + col0
    if local0 + _PTOPK_ITEMS <= count and (off & (_PTOPK_ITEMS - 1)) == 0:
        var v = in_scores.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](off)
        var idx = SIMD[DType.int32, _PTOPK_ITEMS](Int32(col0)) + SIMD[
            DType.int32, _PTOPK_ITEMS
        ](0, 1, 2, 3)
        return (v, idx)

    var vv = SIMD[DType.float32, _PTOPK_ITEMS](min_or_neg_inf[DType.float32]())
    var ii = SIMD[DType.int32, _PTOPK_ITEMS](Int32(-1))
    comptime for j in range(_PTOPK_ITEMS):
        if local0 + j < count:
            vv[j] = in_scores[off + j]
            ii[j] = Int32(col0 + j)
    return (vv, ii)


@always_inline
def _halfclean4[
    cv_origin: MutOrigin,
    sv_origin: MutOrigin,
](
    champ_v: UnsafePointer[
        Scalar[DType.float32], cv_origin, address_space=AddressSpace.SHARED
    ],
    champ_i: UnsafePointer[
        Scalar[DType.int32], cv_origin, address_space=AddressSpace.SHARED
    ],
    scratch_v: UnsafePointer[
        Scalar[DType.float32], sv_origin, address_space=AddressSpace.SHARED
    ],
    scratch_i: UnsafePointer[
        Scalar[DType.int32], sv_origin, address_space=AddressSpace.SHARED
    ],
    tid: Int,
) -> Tuple[SIMD[DType.float32, _PTOPK_ITEMS], SIMD[DType.int32, _PTOPK_ITEMS]]:
    """Batcher half-cleaner: element-wise max of the champion and reversed tile.

    Reads the 4 canonical champion slots `champ[e0..e3]` and the mirrored
    partner run `scratch[_TILE-1-e0 .. _TILE-1-e3]`. That partner run is the
    contiguous descending block `scratch[_TILE-4-e0 .. _TILE-1-e0]`, so it loads
    as one 16B vector and reverses in registers (lane `p` = `scratch[_TILE-1-e_p]`).
    Bases are 16B-aligned (`_V4_ALIGN`-aligned allocations, `tid*4` a multiple of
    4), so both champion and scratch reads are single `LDS.128`s.
    """
    var e0 = tid * _PTOPK_ITEMS
    var cv = champ_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    var ci = champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    var rbase = _TILE - _PTOPK_ITEMS - e0
    var bv = scratch_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        rbase
    ).reversed()
    var bi = scratch_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        rbase
    ).reversed()
    var take = bv.gt(cv)  # element-wise mask; `>` is scalar-only on SIMD
    return (take.select(bv, cv), take.select(bi, ci))


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


@always_inline
def _bitonic_merge_desc[
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
    """Descending bitonic *merge* of a `_PTOPK_TOTAL`-element bitonic sequence.

    Precondition: the block-wide sequence held in registers (thread `t` owns
    canonical elements `e0=4t..e3=4t+3`) is already bitonic. On exit it is
    sorted descending in place. This is the final merge pass of
    `_bitonic_sort_desc` (stage `s = _PTOPK_LOG2`, where the direction is
    uniformly descending) factored out: a caller that has produced a bitonic
    sequence — e.g. a half-cleaner of two sorted runs — finishes in this one
    pass (strides 1024..1, ~5 block barriers) instead of a full sort
    (~14 barriers). `smem_v`/`smem_i` must be `_PTOPK_TOTAL`-wide scratch
    (contents undefined on return).
    """
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var sw0 = (e0 & ~31) | ((e0 ^ (e0 >> 5)) & 31)
    var sw1 = (e1 & ~31) | ((e1 ^ (e1 >> 5)) & 31)
    var sw2 = (e2 & ~31) | ((e2 ^ (e2 >> 5)) & 31)
    var sw3 = (e3 & ~31) | ((e3 ^ (e3 >> 5)) & 31)

    smem_v[sw0] = v0
    smem_i[sw0] = i0
    smem_v[sw1] = v1
    smem_i[sw1] = i1
    smem_v[sw2] = v2
    smem_i[sw2] = i2
    smem_v[sw3] = v3
    smem_i[sw3] = i3
    barrier()

    # Cross-warp substages (stride >= 128) go through swizzled SMEM; direction
    # is uniformly descending so the largest value moves to the lower index.
    comptime for h in reversed(range(8, _PTOPK_LOG2 + 1)):  # strides 1024..128
        comptime stride = 1 << (h - 1)
        comptime for item in range(4):
            var ei = e0 + item
            var ej = ei ^ stride
            if (ei & stride) == 0:
                var si = (ei & ~31) | ((ei ^ (ei >> 5)) & 31)
                var sj = (ej & ~31) | ((ej ^ (ej >> 5)) & 31)
                var vi = smem_v[si]
                var vj = smem_v[sj]
                if vi < vj:
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

    # Sub-warp substages (stride 64..4) via register shuffles, no barrier.
    comptime for h in reversed(range(3, 8)):
        comptime ts = (1 << (h - 1)) >> 2
        var pv0 = warp.shuffle_xor(v0, UInt32(ts))
        var pi0 = warp.shuffle_xor(i0, UInt32(ts))
        var pv1 = warp.shuffle_xor(v1, UInt32(ts))
        var pi1 = warp.shuffle_xor(i1, UInt32(ts))
        var pv2 = warp.shuffle_xor(v2, UInt32(ts))
        var pi2 = warp.shuffle_xor(i2, UInt32(ts))
        var pv3 = warp.shuffle_xor(v3, UInt32(ts))
        var pi3 = warp.shuffle_xor(i3, UInt32(ts))
        var is_lo = (tid & ts) == 0
        v0, i0 = _select_lane_after_xor(v0, i0, pv0, pi0, True, is_lo)
        v1, i1 = _select_lane_after_xor(v1, i1, pv1, pi1, True, is_lo)
        v2, i2 = _select_lane_after_xor(v2, i2, pv2, pi2, True, is_lo)
        v3, i3 = _select_lane_after_xor(v3, i3, pv3, pi3, True, is_lo)

    # Register substages (stride 2 then 1).
    if v0 < v2:
        v0, v2 = v2, v0
        i0, i2 = i2, i0
    if v1 < v3:
        v1, v3 = v3, v1
        i1, i3 = i3, i1
    if v0 < v1:
        v0, v1 = v1, v0
        i0, i1 = i1, i0
    if v2 < v3:
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

    var lv, li = _load4_scores(in_scores, row, 0, e0, N)
    v0 = lv[0]
    v1 = lv[1]
    v2 = lv[2]
    v3 = lv[3]
    i0 = li[0]
    i1 = li[1]
    i2 = li[2]
    i3 = li[3]

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

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var champ_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()

    var row = token * N
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var neg_inf = min_or_neg_inf[DType.float32]()
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[DType.float32, _PTOPK_ITEMS](neg_inf)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[DType.int32, _PTOPK_ITEMS](Int32(-1))
    )
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
        var lv, li = _load4_scores(in_scores, row, 0, g + e0, N)
        v0 = lv[0]
        v1 = lv[1]
        v2 = lv[2]
        v3 = lv[3]
        i0 = li[0]
        i1 = li[1]
        i2 = li[2]
        i3 = li[3]

        _bitonic_sort_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        # Barrier before reusing `scratch` as the stash: the sort's
        # final swizzled reads alias the canonical write indices (WAR hazard).
        barrier()

        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

        # Each thread touches only its own champion slots e0..e3, so the
        # merge needs no barrier until the write-back below.
        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        # Barrier: finish all champion/scratch reads before the re-sort
        # below overwrites them.
        barrier()

        # The half-cleaner above already made `v` a bitonic sequence (max of a
        # descending champion and a reversed-descending tile), so a bitonic
        # merge finishes it — no need to pay for a full sort.
        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var base = token * K
    var out_i = champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    if e0 < K:
        out_idxs[base + e0] = out_i[0]
    if e1 < K:
        out_idxs[base + e1] = out_i[1]
    if e2 < K:
        out_idxs[base + e2] = out_i[2]
    if e3 < K:
        out_idxs[base + e3] = out_i[3]


@__name(t"split_partial_topk")
def _split_partial_kernel(
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    part_v: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    part_i: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    N: Int,
    slice_len: Int,
    S: Int,
):
    """Phase 1 of the split streaming top-k.

    Grid is `rows * S` blocks. Block `b` folds slice `s = b % S` of row
    `r = b // S` (columns `[s*slice_len, min((s+1)*slice_len, N))`) into a
    sorted top-`_TILE` partial written to `part_v`/`part_i` at
    `(r*S + s)*_TILE`. The fold is identical to `_streaming_topk_kernel` but
    restricted to the slice and emitting the full champion (values + indices)
    so phase 2 can merge partials by value.
    """
    var tid = thread_idx.x
    var block = block_idx.x
    var row = block // S
    var slce = block % S

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var champ_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()

    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var neg_inf = min_or_neg_inf[DType.float32]()
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[DType.float32, _PTOPK_ITEMS](neg_inf)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[DType.int32, _PTOPK_ITEMS](Int32(-1))
    )
    barrier()

    var row_base = row * N
    var slice_base = slce * slice_len
    var slice_count = 0
    if slice_base < N:
        slice_count = min(slice_len, N - slice_base)

    var v0: Scalar[DType.float32]
    var v1: Scalar[DType.float32]
    var v2: Scalar[DType.float32]
    var v3: Scalar[DType.float32]
    var i0: Scalar[DType.int32]
    var i1: Scalar[DType.int32]
    var i2: Scalar[DType.int32]
    var i3: Scalar[DType.int32]

    var num_tiles = ceildiv(slice_count, _TILE)
    for t in range(num_tiles):
        var g = t * _TILE
        var lv, li = _load4_scores(
            in_scores, row_base, slice_base, g + e0, slice_count
        )
        v0 = lv[0]
        v1 = lv[1]
        v2 = lv[2]
        v3 = lv[3]
        i0 = li[0]
        i1 = li[1]
        i2 = li[2]
        i3 = li[3]

        _bitonic_sort_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        # Barrier before reusing `scratch` as the stash: the sort's final
        # swizzled reads alias the canonical write indices (WAR hazard).
        barrier()

        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        barrier()

        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var out_base = (row * S + slce) * _TILE
    part_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )
    part_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )


@__name(t"reduce_partials_topk")
def _reduce_partials_kernel(
    in_v: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    in_i: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    out_v: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    out_i: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    count_in: Int,
    count_out: Int,
    g: Int,
):
    """Tree phase-2 round: merge groups of `g` partials in parallel.

    Grid is `rows * count_out` blocks. Block `b` merges the input partials
    `[grp*g, min(grp*g + g, count_in))` of row `r` -- where `r = b // count_out`,
    `grp = b % count_out` -- into a single sorted top-`_TILE` partial written to
    `out_v`/`out_i` at `(r*count_out + grp)*_TILE`. Each input partial is
    already sorted descending, so folding one in is a half-cleaner plus a single
    bitonic merge. Fanning the `S`-way reduction across `rows * count_out` blocks
    (vs one block per row) is what unblocks the low-row decode regime.
    """
    var tid = thread_idx.x
    var block = block_idx.x
    var row = block // count_out
    var grp = block % count_out

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var champ_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()

    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var first = grp * g
    var n_parts = min(g, count_in - first)

    # Seed the champion with the group's first partial.
    var base0 = (row * count_in + first) * _TILE
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, in_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, in_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    barrier()

    # Seeded to keep definite-assignment happy; the half-cleaner overwrites all
    # four before the merge reads them.
    var neg_inf = min_or_neg_inf[DType.float32]()
    var v0 = neg_inf
    var v1 = neg_inf
    var v2 = neg_inf
    var v3 = neg_inf
    var i0 = Int32(-1)
    var i1 = Int32(-1)
    var i2 = Int32(-1)
    var i3 = Int32(-1)

    for s in range(1, n_parts):
        var bs = (row * count_in + first + s) * _TILE
        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, in_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, in_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        barrier()

        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        # Finish champion/scratch reads before the merge overwrites scratch.
        barrier()

        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var out_base = (row * count_out + grp) * _TILE
    out_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )
    out_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )


@__name(t"merge_partials_topk")
def _merge_partials_kernel(
    part_v: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    part_i: UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin],
    out_idxs: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    count: Int,
    K: Int,
):
    """Final round of the split streaming top-k (one block per row).

    Merges the `count` sorted top-`_TILE` partials of row `block_idx.x` into the
    final top-`K`. Each partial is already sorted descending, so folding one in
    is a half-cleaner (element-wise max against the reversed champion) followed
    by a single bitonic merge — no per-partial full sort. `count` is either the
    phase-1 split factor `S` (no tree reduction) or the residual partial count
    after `_reduce_partials_kernel` rounds.
    """
    var tid = thread_idx.x
    var row = block_idx.x

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var champ_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_v = stack_allocation[
        _TILE,
        Scalar[DType.float32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()
    var scratch_i = stack_allocation[
        _TILE,
        Scalar[DType.int32],
        alignment=_V4_ALIGN,
        address_space=AddressSpace.SHARED,
    ]()

    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    # Seed the champion with partial 0.
    var base0 = (row * count) * _TILE
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, part_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, part_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    barrier()

    # Seeded to keep definite-assignment happy; the half-cleaner below
    # overwrites all four before the merge reads them.
    var neg_inf = min_or_neg_inf[DType.float32]()
    var v0 = neg_inf
    var v1 = neg_inf
    var v2 = neg_inf
    var v3 = neg_inf
    var i0 = Int32(-1)
    var i1 = Int32(-1)
    var i2 = Int32(-1)
    var i3 = Int32(-1)

    for s in range(1, count):
        var bs = (row * count + s) * _TILE
        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, part_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, part_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        barrier()

        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        # Finish champion/scratch reads before the merge overwrites scratch.
        barrier()

        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[DType.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var base = row * K
    var out_i = champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    if e0 < K:
        out_idxs[base + e0] = out_i[0]
    if e1 < K:
        out_idxs[base + e1] = out_i[1]
    if e2 < K:
        out_idxs[base + e2] = out_i[2]
    if e3 < K:
        out_idxs[base + e3] = out_i[3]


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


@always_inline
def _choose_split_factor(rows: Int, num_tiles: Int, sm_count: Int) -> Int:
    """Pick the N-split factor `S` for the streaming top-k.

    Returns 1 (no split) when the rows already fill the GPU (splitting would
    only add merge overhead without new parallelism) or there is a single tile
    to fold. Otherwise splits as finely as the block budget allows: each of the
    `rows*S` phase-1 blocks then folds ~one tile (the cheapest phase 1), and the
    tree phase-2 reduces the resulting `S` partials in parallel so a large `S`
    is affordable. `S` is capped so `rows*S` stays within ~a couple of waves
    (`2*sm_count`); beyond that phase-1 blocks just serialize on the SMs.
    """
    if num_tiles <= 1 or rows >= sm_count:
        return 1
    var S = min(num_tiles, max(2, ceildiv(2 * sm_count, rows)))
    if S < 2:
        return 1
    return S


def persistent_topk_block_split(
    ctx: DeviceContext,
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    out_idxs: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    N: Int,
    K: Int,
    total_seq_len: Int,
) raises:
    """Launch bitonic top-k, splitting the N dimension when rows under-fill GPU.

    Same contract and output as `persistent_topk_block`. When the row count is
    small relative to the SM count and `N` spans many tiles (the long-context
    decode regime — a handful of blocks would otherwise each fold the whole row
    serially), the streaming fold is split across `rows * S` blocks (phase 1),
    each producing a sorted top-`_TILE` partial, then merged per row (phase 2).
    All other shapes fall back to `persistent_topk_block` unchanged.

    Args:
        ctx: Device context.
        in_scores: Flat score buffer `[total_seq_len × N]` row-major.
        out_idxs: Output buffer `[total_seq_len × K]` row-major (int32).
        N: Score columns per token.
        K: Top-k count per token (≤ N, and ≤ PERSISTENT_TOPK_MAX_N when N > 2048).
        total_seq_len: Number of rows.
    """
    if N <= PERSISTENT_TOPK_MAX_N:
        persistent_topk_block(ctx, in_scores, out_idxs, N, K, total_seq_len)
        return

    var num_tiles = ceildiv(N, _TILE)
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var S = _choose_split_factor(total_seq_len, num_tiles, sm_count)
    if S <= 1:
        persistent_topk_block(ctx, in_scores, out_idxs, N, K, total_seq_len)
        return

    var slice_len = ceildiv(N, S)
    var part_count = total_seq_len * S * _TILE

    # Phase 1: fan the streaming fold across `rows * S` blocks into buffer A.
    var buf_a_v = ctx.enqueue_create_buffer[DType.float32](part_count)
    var buf_a_i = ctx.enqueue_create_buffer[DType.int32](part_count)
    var a_v = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](
        buf_a_v.unsafe_ptr()
    )
    var a_i = rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
        buf_a_i.unsafe_ptr()
    )

    ctx.enqueue_function[_split_partial_kernel](
        in_scores,
        a_v,
        a_i,
        N,
        slice_len,
        S,
        grid_dim=total_seq_len * S,
        block_dim=_PTOPK_BLOCK,
    )

    # Phase 2: reduce the `S` partials per row toward `_MERGE_FANIN` via parallel
    # group merges (each round fanned across `rows * count_out` blocks),
    # ping-ponging A<->B, then a single per-row final merge to top-K. A serial
    # per-row merge would leave only `rows` blocks busy (8 SMs of 148 in the
    # long-context decode case), so the reduction is what fills the GPU.
    # Fan-in 3 is a deliberate latency trade: long-context decode -18.7% for
    # +1.5% on the MTP-decode shape (whose S = 5 then takes one reduce round),
    # accepted by the GLM serving owner; fan-in 5 keeps S = 5 reduce-free if
    # that trade is ever reversed.
    comptime _MERGE_FANIN = 3
    var count = S
    var src_v = a_v
    var src_i = a_i
    var do_reduce = S > _MERGE_FANIN
    var buf_b_v = ctx.enqueue_create_buffer[DType.float32](
        part_count if do_reduce else 1
    )
    var buf_b_i = ctx.enqueue_create_buffer[DType.int32](
        part_count if do_reduce else 1
    )
    var dst_v = rebind[UnsafePointer[Scalar[DType.float32], MutAnyOrigin]](
        buf_b_v.unsafe_ptr()
    )
    var dst_i = rebind[UnsafePointer[Scalar[DType.int32], MutAnyOrigin]](
        buf_b_i.unsafe_ptr()
    )

    while count > _MERGE_FANIN:
        var count_out = ceildiv(count, _MERGE_FANIN)
        ctx.enqueue_function[_reduce_partials_kernel](
            rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](src_v),
            rebind[UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin]](src_i),
            dst_v,
            dst_i,
            count,
            count_out,
            _MERGE_FANIN,
            grid_dim=total_seq_len * count_out,
            block_dim=_PTOPK_BLOCK,
        )
        var tv = src_v
        src_v = dst_v
        dst_v = tv
        var ti = src_i
        src_i = dst_i
        dst_i = ti
        count = count_out

    ctx.enqueue_function[_merge_partials_kernel](
        rebind[UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]](src_v),
        rebind[UnsafePointer[Scalar[DType.int32], ImmutAnyOrigin]](src_i),
        out_idxs,
        count,
        K,
        grid_dim=total_seq_len,
        block_dim=_PTOPK_BLOCK,
    )

    _ = buf_a_v
    _ = buf_a_i
    _ = buf_b_v
    _ = buf_b_i
