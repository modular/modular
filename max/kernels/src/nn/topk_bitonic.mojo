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

This is to replace the two-stage sequential-extraction `topk_gpu` is pathological when k ≈ N
(near-full sort).  A use case of `k = N = 2048` is the DeepSeek-V3 / MiniMax-M3 indexer
config.
"""

from std.sys import align_of, size_of

from std.gpu import barrier, block_idx, thread_idx
import std.gpu.primitives.warp as warp
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.numerics import min_or_neg_inf
from layout import TileTensor, row_major

# ===----------------------------------------------------------------------=== #
# Compile-time constants
# ===----------------------------------------------------------------------=== #

# Maximum N supported by the bitonic sort path.
# Increasing this requires raising _PTOPK_BLOCK or _PTOPK_ITEMS.
comptime PERSISTENT_TOPK_MAX_N: Int = 2048

comptime _PTOPK_BLOCK: Int = 512  # threads per block  (512 × 4 = 2048)
comptime _PTOPK_ITEMS: Int = 4  # elements per thread
comptime _PTOPK_TOTAL: Int = _PTOPK_BLOCK * _PTOPK_ITEMS  # = 2048
comptime _PTOPK_LOG2: Int = 11  # log2(2048)


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
# GPU kernel
# ===----------------------------------------------------------------------=== #


@__name(t"persistent_topk_2048")
def _persistent_topk_2048_kernel(
    in_scores: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    out_idxs: UnsafePointer[Scalar[DType.int32], MutAnyOrigin],
    N: Int,
    K: Int,
):
    """Block-wide bitonic sort for N ≤ 2048 (BLOCK=512, ITEMS=4).

    Consecutive element layout: thread `t` owns canonical elements
    `e0=4t`, `e1=4t+1`, `e2=4t+2`, `e3=4t+3`.

    With 4 items per thread:
    - S=1,2  (h=1,2): within-thread register compare-swaps (no barrier).
    - S=4..64 (h=3..7): warp.shuffle_xor, thread-stride S/4 in [1..16] (no barrier).
    - S≥128  (h≥8): SMEM + barrier (only in stages 8-11).

    This matches vLLM’s `persistent_topk_kernel<2048,4>` layout.

    Barrier count: 14 (vs. 20 for ITEMS=2, vs. 66 SMEM-only).
      Stage 8: 1 write + 1 SMEM sub-step = 2 barriers
      Stage 9: 1 write + 2 SMEM sub-steps = 3 barriers
      Stage 10: 1 + 3 = 4 barriers
      Stage 11: 1 + 4 = 5 barriers  → total 14.
    """
    var tid = thread_idx.x
    var token = block_idx.x

    # SMEM: used only during stages 8-11 (S ≥ 128, cross-warp).
    # Consecutive layout: smem_v[4t..4t+3] = thread-t’s four values.
    # smem_v[0..2047] then smem_i[0..2047] = 16 KB total.
    var smem_v = stack_allocation[
        _PTOPK_TOTAL, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    var smem_i = stack_allocation[
        _PTOPK_TOTAL, Scalar[DType.int32], address_space=AddressSpace.SHARED
    ]()

    # ── Phase 1: Load 4 elements per thread into registers ────────────────
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

    # ── Phase 2a: Stages 1-7 — registers + warp shuffles, ZERO barriers ──
    #
    # S=1 (h=1): e0↔0e1 and e2↔0e3 — within-thread register swap.
    # S=2 (h=2): e0↔0e2 and e1↔0e3 — within-thread register swap.
    # S=4..64 (h=3..7): warp.shuffle_xor, thread-stride ts=S/4 ∈ [1..16].
    #   Lower thread (tid & ts==0): swap if (mine < partner) == want_d.
    #   Upper thread: swap if (partner < mine) == want_d.
    #
    comptime for s in range(1, 8):  # stages 1..7
        comptime for h in reversed(range(1, s + 1)):  # h = s..1
            comptime stride = 1 << (h - 1)
            var want_d = ((e0 >> s) & 1) == 0

            comptime if stride == 1:
                # S=1: (e0,e1) and (e2,e3) within thread.
                # Compute direction separately: (e2>>s) ≠ (e0>>s) for s=1
                # because (4t+2)>>1 = 2t+1 (odd) vs (4t)>>1 = 2t (even).
                # For s≥2 they coincide, but using e2's own direction is always
                # correct.
                var want_d0 = ((e0 >> s) & 1) == 0
                var want_d2 = ((e2 >> s) & 1) == 0
                v0, i0, v1, i1 = _swap_pair_if(v0, i0, v1, i1, want_d0)
                v2, i2, v3, i3 = _swap_pair_if(v2, i2, v3, i3, want_d2)

            elif stride == 2:
                # S=2: (e0,e2) and (e1,e3) within thread.
                v0, i0, v2, i2 = _swap_pair_if(v0, i0, v2, i2, want_d)
                v1, i1, v3, i3 = _swap_pair_if(v1, i1, v3, i3, want_d)

            else:
                # S=4..64: warp shuffle, thread-stride ts = S/4 ∈ [1..16].
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

    # ── Phase 2b: Stages 8-11 — XOR-swizzled SMEM for S≥128 ─────────────
    #
    # XOR swizzle: swz(e) = (e & ~31) | ((e ^ (e>>5)) & 31)
    # Remaps each set of 32 consecutive canonical elements to 32 distinct SMEM
    # banks → zero bank conflicts for all SMEM reads and writes.
    # The swizzle is a bijection on [0,2047], transparent to the sort logic.
    #
    # Precompute swizzled addresses for this thread's 4 fixed elements.
    var sw0 = (e0 & ~31) | ((e0 ^ (e0 >> 5)) & 31)
    var sw1 = (e1 & ~31) | ((e1 ^ (e1 >> 5)) & 31)
    var sw2 = (e2 & ~31) | ((e2 ^ (e2 >> 5)) & 31)
    var sw3 = (e3 & ~31) | ((e3 ^ (e3 >> 5)) & 31)

    # Each stage s (8..11):
    #   1. Write registers → SMEM (swizzled) + barrier  (1 barrier).
    #   2. SMEM compare-swaps h=s..8 (swizzled)         (s-7 barriers).
    #   3. Read SMEM (swizzled) → registers.
    #   4. Shuffle h=7..3  (S=64..4, no barrier).
    #   5. Register h=2    (S=2, no barrier).
    #   6. Register h=1    (S=1, no barrier).
    #
    comptime for s in range(8, _PTOPK_LOG2 + 1):  # stages 8..11
        # Step 1: write to SMEM (swizzled) and sync.
        smem_v[sw0] = v0
        smem_i[sw0] = i0
        smem_v[sw1] = v1
        smem_i[sw1] = i1
        smem_v[sw2] = v2
        smem_i[sw2] = i2
        smem_v[sw3] = v3
        smem_i[sw3] = i3
        barrier()

        # Step 2: SMEM compare-swaps for h=s..8 (stride S ≥ 128).
        # Compute swizzled indices inline per pair.
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

        # Step 3: read back from SMEM (swizzled).
        v0 = smem_v[sw0]
        i0 = smem_i[sw0]
        v1 = smem_v[sw1]
        i1 = smem_i[sw1]
        v2 = smem_v[sw2]
        i2 = smem_i[sw2]
        v3 = smem_v[sw3]
        i3 = smem_i[sw3]

        # Step 4: shuffle h=7..3 (S=64..4, thread-stride 16..1).
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

        # Step 5: register h=2 (S=2): (e0,e2) and (e1,e3).
        var want_d2 = ((e0 >> s) & 1) == 0
        if (v0 < v2) == want_d2:
            v0, v2 = v2, v0
            i0, i2 = i2, i0
        if (v1 < v3) == want_d2:
            v1, v3 = v3, v1
            i1, i3 = i3, i1

        # Step 6: register h=1 (S=1): (e0,e1) and (e2,e3).
        var want_d1 = ((e0 >> s) & 1) == 0
        if (v0 < v1) == want_d1:
            v0, v1 = v1, v0
            i0, i1 = i1, i0
        if (v2 < v3) == want_d1:
            v2, v3 = v3, v2
            i2, i3 = i3, i2

    # ── Phase 3: Write top-K from registers to output ─────────────────────
    # After all stages v0/i0..v3/i3 hold the values at canonical positions
    # e0..e3. Position 0 (thread 0, item 0) is the highest-scoring element.
    var base = token * K
    if e0 < K:
        out_idxs[base + e0] = i0
    if e1 < K:
        out_idxs[base + e1] = i1
    if e2 < K:
        out_idxs[base + e2] = i2
    if e3 < K:
        out_idxs[base + e3] = i3


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
    """Launch block-wide bitonic sort top-k for `total_seq_len` score rows.

    Handles N ≤ PERSISTENT_TOPK_MAX_N (= 2048).  Call site must check this
    before calling; for larger N fall back to `topk_gpu`.

    Each row of `N` float32 scores is sorted descending in a single block,
    writing the `K` highest-scoring indices (as int32) to `out_idxs`.

    The SMEM budget is `2 * PERSISTENT_TOPK_MAX_N * 4 = 16 KB` per block.

    Args:
        ctx: Device context.
        in_scores: Flat score buffer `[total_seq_len × N]` row-major.
        out_idxs: Output buffer `[total_seq_len × K]` row-major (int32).
        N: Score columns per token (≤ PERSISTENT_TOPK_MAX_N).
        K: Top-k count per token (≤ N).
        total_seq_len: Number of rows (one block per row).
    """
    ctx.enqueue_function[_persistent_topk_2048_kernel](
        in_scores,
        out_idxs,
        N,
        K,
        grid_dim=total_seq_len,
        block_dim=_PTOPK_BLOCK,
    )
