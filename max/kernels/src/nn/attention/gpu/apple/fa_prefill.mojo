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
"""Apple (Metal) M5 MMA-based flash-attention (FA2) PREFILL kernel.

Apple silicon GPU (Metal 4, `compute_capability == 5`), prefill (many query
rows per sequence), paged KV cache via the `MHAOperand` contract, BF16/FP16
storage / FP32 accumulation.

Unlike decode (one query row -> Q.K^T is a GEMV), prefill has many query rows,
so Q.K^T and P.V are real matrix*matrix products. The kernel is two
`MmaOpApple` 16x16 simdgroup GEMMs sandwiching an online softmax -- it reuses
the Apple matmul MMA foundation directly. **16 simdgroups (256 query rows) share
one threadgroup** (`num_simdgroups=16`), each independently owning an
`Sq x depth` query-row tile for one `(batch, head)` and streaming the KV range
online; there is **no threadgroup memory and no `barrier()`** -- the simdgroups
are co-resident only so they share KV reads through the L2 (the wide threadgroup
beat both a single-simdgroup launch and an SMEM-staged variant; see KB
`apple-m5-gpu-performance-considerations`).

Per query-row tile, iterating `Sk`-wide KV tiles online:

```text
init: m = -inf, l = 0, O = 0
for each KV tile:
    QK = ScoreMma.mma(Q, K)   # Sq x Sk fp32 score tile (transpose_b in ScoreMma)
    apply MHAMask(QK) at absolute (b, head, q_abs, k)
    softmax.update(O, QK)                      # m,l update + rescale O
    P  = QK_softmaxed.cast[in_type]
    O += P @ V                                  # P fed register-resident; V via load_fragment
epilogue: O /= l; cast to out dtype; masked store
```

The crux -- a per-ROW max/sum over the score tile whose rows are scattered
across the MMA fragment -- is the XOR-butterfly reduction validated in
`test/gpu/linalg/test_apple_fa_frag_reduce.mojo` (PASS on M5), generalized here
to `frag_num_rows = 2` (the M5 lane owns two rows of each 16x16 subtile) and to
`num_n_mmas` fragments per row (a register combine across the `ni` fragments,
then the butterfly once). `air.simd_sum` (the decode reduction) is wrong here:
it reduces all 32 lanes, mixing rows.

Structured-Mojo data movement (the idiomatic shape, see KB
`exceptions/apple-mma-fragment-is-not-distribute-expressible`):
  * Q, K, V, and the output reach the kernel as `TileTensor`s; the ragged/BSHD
    offset is baked into each tile base, and `.tile[16, 16]()` descends to the
    16x16 sub-tiles the MMA consumes -- no raw pointer arithmetic.
  * K/V are resolved through `MHAOperand.block_paged_tile` per 16-row sub-tile
    (the paged-KV contract; see KB `apple-paged-kv-prefill-per-sub-tile`). The
    TileTensor layout carries the in-page row stride, so the old
    pointer-byte-difference stride hack is gone.
  * QK runs through `ScoreMma.mma(scores, q_tile, k_tile)` (both operands in
    memory). The MMA fragment load/store stays SIMD-level inside `MmaOpApple`
    (NOT a TileTensor `distribute` -- that is the wrong layer for the Apple
    bit-scatter fragment).
  * P.V keeps P register-resident: the QK score fragment, cast to `q_type`, is
    fed DIRECTLY as the P.V A operand via `_mma_apple_transposable` (the
    C-output and A-input fragment layouts are the identical
    `_apple_frag_layout` bit-scatter, so P never round-trips memory -- see KB
    `apple-fa-register-resident-p-fragment`). Only V loads memory->register, via
    `MmaOpApple.load_fragment` over a `block_paged_tile` sub-tile.

The host launcher `fa_prefill_apple` mirrors `mha_gpu_naive`'s `MHAOperand`
signature; `flash_attention_dispatch` selects it for Apple prefill
(`not is_token_generation`) by default (set
`MODULAR_ENABLE_APPLE_FA_PREFILL=0` to opt out, falling back to
`mha_gpu_naive`).
"""

from std.collections import OptionalReg
from std.gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.compute.arch.mma_apple import (
    _apple_frag_layout,
    _mma_apple_transposable,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_xor
from std.math import ceildiv, exp
from std.sys.info import size_of
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from layout import UNKNOWN_VALUE, Idx, Layout, LayoutTensor, TileTensor
from layout.coord import Coord
from layout.tile_layout import (
    Layout as TileLayout,
    RowMajorLayout,
    TensorLayout,
    row_major,
)

from linalg.arch.apple.mma import MmaOpApple

from nn.attention.mha_mask import CausalMask, MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand

comptime NEG_INF = Float32(-3.0e38)

# Dispatcher gate: head dims that are multiples of MMA_K=16 up to this cap take
# the prefill kernel; anything else falls back to mha_gpu_naive.
comptime FA_PREFILL_APPLE_MAX_HEAD_DIM = 256

# Default tile shape per simdgroup: Sq = NUM_M_MMAS*16 query rows, Sk =
# NUM_N_MMAS*16 KV cols. Tuned 1x2 (16x32) for the WIDE threadgroup
# (num_simdgroups=16). The output + score accumulators are
# NUM_M_MMAS*((depth/16) + NUM_N_MMAS) fp32 fragments PER simdgroup, and 16
# simdgroups co-reside, so the per-simdgroup register footprint is the binding
# constraint: a narrow tile (NUM_M_MMAS=1, NUM_N_MMAS=2) keeps 16 simdgroups
# resident, and a wide score tile collapses occupancy. Measured d=128 b1 null
# (full-work TFLOP/s): NUM_N_MMAS=2 (37.5) > 4 (17.9) > 8 (16.6) at sg16, and
# raising NUM_M_MMAS or num_simdgroups also regresses. (A single-simdgroup
# kernel would want the opposite, a wide 1x8 tile.)
comptime DEFAULT_NUM_M_MMAS = 1
comptime DEFAULT_NUM_N_MMAS = 2
comptime MMA_DIM = 16


# ===-------------------------------------------------------------------=== #
# AppleSoftmax — register-resident online softmax over the MmaOpApple accum.
# ===-------------------------------------------------------------------=== #
# The AMD `Softmax` (amd_structured/softmax.mojo) ports in *algebra* only: this
# struct drops the LOCAL-memory TileTensor scratch, the `lane_group_max`
# (Apple's row-sharing lanes are an XOR group, not a strided lane group), the
# cross-warp SMEM scratch (one simdgroup owns its row's whole KV span), and the
# `exp2`/log2e machinery (Apple uses plain `exp` + NEG_INF, as decode does). The
# load-bearing change is `frag_num_rows = 2` -- AMD's 16x16 MFMA fragment is a
# row-vector (frag_num_rows == 1); the M5 16x16 simdgroup fragment owns TWO rows
# per lane (`rb` and `rb+8`), so every per-row quantity is a `SIMD[fp32, 2]`.


struct AppleSoftmax[num_m_mmas: Int, num_n_mmas: Int]:
    """Online-softmax running state for an Apple M5 `Sq x Sk` score tile.

    One simdgroup owns the tile; `num_m_mmas` row-blocks of 16 query rows each.
    The lane owns rows `{rb, rb+8}` of every 16x16 subtile, so each row-block
    carries two running rows -- hence `m`/`l` are `SIMD[fp32, 2]` per row-block,
    stored in length-`num_m_mmas` `InlineArray`s. All state is register-resident
    (no shared memory, no barriers -- the Apple idiom).

    Parameters:
        num_m_mmas: Number of 16-row query blocks in the score tile (`Sq / 16`).
        num_n_mmas: Number of 16-col KV blocks in the score tile (`Sk / 16`).
    """

    comptime FRAG_ROWS = 2  # M5 lane owns rows {rb, rb+8} of each 16x16 subtile
    comptime RowState = InlineArray[
        SIMD[DType.float32, Self.FRAG_ROWS], Self.num_m_mmas
    ]

    # Running max (m) and running denominator (l), per row-block, per owned row.
    var m: Self.RowState
    var l: Self.RowState

    @always_inline
    def __init__(out self):
        self.m = Self.RowState(
            fill=SIMD[DType.float32, Self.FRAG_ROWS](NEG_INF)
        )
        self.l = Self.RowState(fill=SIMD[DType.float32, Self.FRAG_ROWS](0))

    @always_inline
    def seed_sink(mut self, sink_weight: Float32):
        """Pre-seed `(m, l)` with a sink token's contribution (init-state trick).

        Equivalent to a virtual "tile -1" with one element of score
        `sink_weight`: `exp(sink_weight - sink_weight) = 1` enters the running
        sum. After this, the hot loop is sink-agnostic. Apple uses plain `exp`
        (not `exp2`), so `sink_weight` is seeded directly with no `log2e` factor.
        The naive reference (`nn/softmax.mojo`) compares the RAW (unscaled) sink
        weight against the post-scale row max, so the caller passes the sink
        weight WITHOUT the `scale` multiply. Mirrors AMD's
        `amd-attention-sink-as-init-state`, retuned for base-`e` + unscaled sink.
        """
        comptime for mi in range(Self.num_m_mmas):
            self.m[mi] = SIMD[DType.float32, Self.FRAG_ROWS](sink_weight)
            self.l[mi] = SIMD[DType.float32, Self.FRAG_ROWS](1)

    @staticmethod
    @always_inline
    def _row_max(
        scores: MmaOpApple[
            DType.float32, DType.float32, Self.num_m_mmas, Self.num_n_mmas
        ].AccumType,
        mi: Int,
    ) -> SIMD[DType.float32, 2]:
        """Full-row max over the score row-block `mi` (across all `ni` cols).

        Combines the per-fragment max across the `num_n_mmas` column fragments
        (a register combine -- the same lane holds the matching rows of every
        `ni` fragment) BEFORE the butterfly, then reduces the 4 row-sharing
        lanes once. This is the `num_n_mmas > 1` row reduction from DESIGN.md.
        """
        # Per-lane, per-fragment column max, combined across ni.
        var r = SIMD[DType.float32, 2](NEG_INF)
        comptime for ni in range(Self.num_n_mmas):
            var frag = scores[mi * Self.num_n_mmas + ni]
            var r0 = max(max(frag[0], frag[1]), max(frag[2], frag[3]))
            var r1 = max(max(frag[4], frag[5]), max(frag[6], frag[7]))
            r = max(r, SIMD[DType.float32, 2](r0, r1))
        # One butterfly over the 4 row-sharing lanes {1, 8}.
        r[0] = max(r[0], shuffle_xor(r[0], UInt32(1)))
        r[0] = max(r[0], shuffle_xor(r[0], UInt32(8)))
        r[1] = max(r[1], shuffle_xor(r[1], UInt32(1)))
        r[1] = max(r[1], shuffle_xor(r[1], UInt32(8)))
        return r

    @staticmethod
    @always_inline
    def _row_sum(
        scores: MmaOpApple[
            DType.float32, DType.float32, Self.num_m_mmas, Self.num_n_mmas
        ].AccumType,
        mi: Int,
    ) -> SIMD[DType.float32, 2]:
        """Full-row sum over the score row-block `mi` (across all `ni` cols).

        Combine across `ni` (register add), then one butterfly. See `_row_max`.
        """
        var r = SIMD[DType.float32, 2](0)
        comptime for ni in range(Self.num_n_mmas):
            var frag = scores[mi * Self.num_n_mmas + ni]
            var r0 = frag[0] + frag[1] + frag[2] + frag[3]
            var r1 = frag[4] + frag[5] + frag[6] + frag[7]
            r = r + SIMD[DType.float32, 2](r0, r1)
        r[0] = r[0] + shuffle_xor(r[0], UInt32(1))
        r[0] = r[0] + shuffle_xor(r[0], UInt32(8))
        r[1] = r[1] + shuffle_xor(r[1], UInt32(1))
        r[1] = r[1] + shuffle_xor(r[1], UInt32(8))
        return r

    @always_inline
    def update[
        out_num_n_mmas: Int
    ](
        mut self,
        mut scores: MmaOpApple[
            DType.float32, DType.float32, Self.num_m_mmas, Self.num_n_mmas
        ].AccumType,
        mut output: MmaOpApple[
            DType.float32, DType.float32, Self.num_m_mmas, out_num_n_mmas
        ].AccumType,
    ):
        """One online-softmax step over the masked score tile `scores`.

        In place: `scores` becomes the (unnormalized) probabilities `P`,
        `output` is rescaled by the running-max correction, and `m`/`l` advance.
        The output's column count (`out_num_n_mmas = depth / 16`) differs from
        the score's (`num_n_mmas = Sk / 16`); both share the SAME `num_m_mmas`
        row-blocks and the same per-lane `(rb, rb+8)` row ownership, so the
        correction indexes by row-block `mi` and applies to all of the output's
        `ni` fragments.

        Sequence (ports the AMD `Softmax.full` algebra; plain `exp`, no log2e):
          1. row max `m_tile`  -> `m_new = max(m, m_tile)`
          2. correction `alpha = exp(m - m_new)`
          3. `P = exp(scores - m_new)` (the exp-scaled exact-zero-at-max form;
             OOB/masked entries are NEG_INF -> exp == 0)
          4. row sum of P, `l = l*alpha + rowsum`, `m = m_new`
          5. `output *= alpha`
        """
        comptime for mi in range(Self.num_m_mmas):
            var m_tile = Self._row_max(scores, mi)
            var m_new = max(self.m[mi], m_tile)
            # A row still fully masked here keeps its running max at the finite
            # NEG_INF floor (NEG_INF is finite, so the subtraction never NaNs --
            # as in nn/softmax.mojo); it resolves correctly once its first real
            # key arrives in a later tile. Covered by the small-window
            # multi-tile case in the prefill test.
            # alpha[0] -> row rb, alpha[1] -> row rb+8; broadcast across each
            # half-fragment in the rescale below.
            var alpha = exp(self.m[mi] - m_new)

            # P = exp(scores - m_new), per owned row. regs[0:4] -> row rb (m_new[0]),
            # regs[4:8] -> row rb+8 (m_new[1]). Masked entries are NEG_INF.
            comptime for ni in range(Self.num_n_mmas):
                var idx = mi * Self.num_n_mmas + ni
                var p = scores[idx]
                var p_lo = exp(
                    p.slice[4, offset=0]() - SIMD[DType.float32, 4](m_new[0])
                )
                var p_hi = exp(
                    p.slice[4, offset=4]() - SIMD[DType.float32, 4](m_new[1])
                )
                scores[idx] = p_lo.join(p_hi)

            var l_tile = Self._row_sum(scores, mi)
            self.l[mi] = self.l[mi] * alpha + l_tile
            self.m[mi] = m_new

            # Rescale the running output: every output fragment in row-block mi.
            comptime for ni in range(out_num_n_mmas):
                var oidx = mi * out_num_n_mmas + ni
                var o = output[oidx]
                var o_lo = o.slice[4, offset=0]() * SIMD[DType.float32, 4](
                    alpha[0]
                )
                var o_hi = o.slice[4, offset=4]() * SIMD[DType.float32, 4](
                    alpha[1]
                )
                output[oidx] = o_lo.join(o_hi)

    @always_inline
    def normalize[
        out_num_n_mmas: Int
    ](
        self,
        mut output: MmaOpApple[
            DType.float32, DType.float32, Self.num_m_mmas, out_num_n_mmas
        ].AccumType,
    ):
        """Final epilogue: divide each output row by its running denominator `l`.

        `l[mi][0]` normalizes the `rb` row, `l[mi][1]` the `rb+8` row, across all
        output column fragments in row-block `mi`.

        A fully-masked row has `l == 0` (no visible key contributed any
        probability and no sink seed). Dividing by it would give `0/0 = NaN`
        (the row's output `O` is also 0), so we follow the standard
        FlashAttention convention and emit 0 for such a row: clamp the inverse
        denominator to 0 when `l == 0`, making `O * inv == 0`. This is reachable
        with `SlidingWindowCausalMask` (or any non-causal mask) when a query row
        falls entirely outside its window AND the key range -- e.g. cross-
        attention with `num_keys < seq_len` and a small window, where a high
        query attends no key. Causal alone cannot hit it (every query attends
        its own position), and the sink seed makes `l >= 1`, so this guard is a
        no-op in those cases.
        """
        comptime for mi in range(Self.num_m_mmas):
            # `l == 0` -> inv 0 (output 0); else 1/l. Per owned row, branchless.
            var l_mi = self.l[mi]
            var inv = l_mi.gt(SIMD[DType.float32, 2](0)).select(
                SIMD[DType.float32, 2](1) / l_mi, SIMD[DType.float32, 2](0)
            )
            comptime for ni in range(out_num_n_mmas):
                var oidx = mi * out_num_n_mmas + ni
                var o = output[oidx]
                var o_lo = o.slice[4, offset=0]() * SIMD[DType.float32, 4](
                    inv[0]
                )
                var o_hi = o.slice[4, offset=4]() * SIMD[DType.float32, 4](
                    inv[1]
                )
                output[oidx] = o_lo.join(o_hi)


# ===-------------------------------------------------------------------=== #
# Prefill kernel: one simdgroup per (q_tile, head, batch).
# Grid (num_q_tiles, num_heads, batch); block = WARP_SIZE.
# ===-------------------------------------------------------------------=== #
def fa_prefill_apple_core[
    q_type: DType,
    output_type: DType,
    p_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    q_layout: TensorLayout,
    output_layout: TensorLayout,
    valid_length_layout: TensorLayout,
    sink_layout: TensorLayout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    Depth: Int,
    NumMMmas: Int,
    NumNMmas: Int,
    NumSimdgroups: Int = 1,
](
    output: TileTensor[output_type, output_layout, MutAnyOrigin],
    q: TileTensor[q_type, q_layout, ImmutAnyOrigin],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        DType.uint32,
        valid_length_layout,
        ImmutAnyOrigin,
    ],
    sink_weights: OptionalReg[TileTensor[q_type, sink_layout, ImmutAnyOrigin]],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
):
    """Per-simdgroup core of the prefill kernel (the module docstring covers the
    wide-threadgroup launch geometry and the no-SMEM design). Each simdgroup owns
    an `Sq x Depth` query-row tile for one `(batch, head)` and streams the KV
    range online.

    Q.K^T runs through `ScoreMma.mma` (both operands in memory); P.V runs on the
    M5 16x16 simdgroup MMA via `_mma_apple_transposable`. The QK score fragment
    is fed DIRECTLY (cast to `q_type`) as the PV A-operand -- the C-output and
    A-input fragment layouts are the identical `_apple_frag_layout` bit-scatter,
    so P never round-trips through memory (the register-resident FlashAttention
    trick; KB `apple-fa-register-resident-p-fragment`). Only K and V load
    memory->register, through `block_paged_tile` sub-tiles.

    Constraints:
        `Depth % 16 == 0` (MMA_K) and either `k.page_size == 0` (contiguous /
        ragged KV) or `k.page_size % 16 == 0` (paged KV). The paged gate is
        `% 16` because each MMA sub-tile is exactly 16 keys and we resolve the
        page per 16-row sub-tile (see the KV-tile load below); a page boundary
        that is a multiple of 16 therefore never bisects a sub-tile, so the
        in-page row stride stays valid. An odd page size (`page_size % 16 != 0`)
        would let a 16-row sub-tile straddle a boundary and is gated to
        `mha_gpu_naive` by the dispatcher.
    """
    comptime assert Depth % MMA_DIM == 0, "Depth must be a multiple of 16"
    comptime assert k_t.page_size == 0 or k_t.page_size % MMA_DIM == 0, (
        "fa_prefill_apple_core requires contiguous KV (page_size == 0) or a"
        " page size that is a multiple of MMA_DIM=16 (so a 16-row sub-tile"
        " never crosses a page boundary)"
    )
    comptime SQ = NumMMmas * MMA_DIM
    comptime SK = NumNMmas * MMA_DIM
    comptime DEPTH_MMAS = Depth // MMA_DIM
    debug_assert(depth == Depth, "runtime depth must match comptime Depth")

    var q_tile_id = Int(block_idx.x)
    var head_id = Int(block_idx.y)
    var batch_id = Int(block_idx.z)
    var kv_head = head_id // group
    var sg = Int(warp_id())  # this simdgroup's slot in the threadgroup
    var lane = Int(lane_id())
    var rb_cb = _apple_frag_layout(lane)
    var rb = rb_cb[0]
    var cb = rb_cb[1]

    # Offset math — mirror `_bmm0_bs` / `_bmm1_bs` (mha.mojo).
    var seq_start: Int
    var cur_query_len: Int
    var q_offset: Int
    var out_offset: Int
    var cur_cache_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        q_offset = depth * (seq_start * num_heads + head_id)
        out_offset = (seq_start * num_heads + head_id) * depth
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    elif _use_valid_length:
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        q_offset = depth * (head_id + num_heads * max_prompt_len * batch_id)
        out_offset = q_offset
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    else:
        seq_start = batch_id
        cur_query_len = max_prompt_len
        q_offset = depth * (head_id + num_heads * max_prompt_len * batch_id)
        out_offset = q_offset
        cur_cache_len = max_cache_size

    # This simdgroup's query-row block: each of the `NumSimdgroups` simdgroups
    # owns its own `SQ` contiguous rows within the threadgroup's `NumSimdgroups*
    # SQ`-row span. A simdgroup whose rows are entirely past the sequence just
    # returns -- safe because the simdgroups are independent (no barrier).
    var q_row0 = q_tile_id * (NumSimdgroups * SQ) + sg * SQ
    if q_row0 >= cur_query_len:
        return

    # Q/output row stride is num_heads*depth (the BSHD / ragged head stride);
    # depth is contiguous. Build this simdgroup's SQ-row TileTensor view by
    # baking the ragged/BSHD offset + the q_row0 row offset into the tile base,
    # then descending to 16x16 sub-tiles with `.tile[16, 16]()`. The score MMA
    # reads Q from this view; no raw pointer arithmetic in the QK loop.
    # Q tile shape (SQ, Depth) -- extents comptime, the row (token) stride a
    # runtime `Int` (num_heads*depth) and the inner depth stride a comptime 1.
    # Build the layout value, then derive the TileTensor type via `type_of`
    # (the matmul idiom for a runtime-stride layout).
    var q_row_stride = num_heads * depth
    var q_layout_val = TileLayout(
        Coord(Idx[SQ], Idx[Depth]), Coord(q_row_stride, Idx[1])
    )
    var q_tile = TileTensor[q_type, type_of(q_layout_val), ImmutAnyOrigin](
        ptr=q.ptr + q_offset + q_row0 * q_row_stride,
        layout=q_layout_val,
    )

    # KV token stride (within a page): difference of two consecutive key
    # pointers. Robust to the ragged-3D vs dense-4D head layout, and for paged
    # KV it is the IN-PAGE token stride: tokens 0 and 1 always share a page
    # (page_size >= 16 > 1), and the paged blocks layout
    # [total_blocks, page_size, num_heads, head_size] makes consecutive in-page
    # tokens contiguous with a constant stride (num_heads*head_size). We resolve
    # the page per 16-row sub-tile below, so this in-page stride is the only
    # stride a sub-tile load ever needs. The stride feeds the sub-tile
    # TileTensor's layout (`kv_layout_val`), so `MmaOpApple` reads it via
    # `_row_stride` -- no manual `ptr + row*stride` arithmetic in the load.
    var k_base0 = k.block_paged_ptr[1](
        UInt32(batch_id), UInt32(0), UInt32(kv_head), 0
    )
    var k_base1 = k.block_paged_ptr[1](
        UInt32(batch_id), UInt32(1), UInt32(kv_head), 0
    )
    # `Int(ptr)` is a BYTE address; the diff is in bytes. Convert to an ELEMENT
    # stride (the layout/`.tile`/`load_fragment` all index in elements).
    comptime kv_elt_size = size_of[Scalar[k_t.dtype]]()
    var kv_row_stride = (Int(k_base1) - Int(k_base0)) // kv_elt_size

    # KV sub-tile layout (16 keys x Depth): comptime extents + unit inner
    # stride, runtime token (row) stride. One value, reused for every K and V
    # sub-tile (loop-invariant). `block_paged_tile` infers the layout type from
    # this value, so each sub-tile is a TileTensor that carries the in-page
    # stride.
    var kv_layout_val = TileLayout(
        Coord(Idx[MMA_DIM], Idx[Depth]), Coord(kv_row_stride, Idx[1])
    )

    # ScoreMma loads K in its native dtype (`b_type = k_t.dtype`); the QK MMA
    # accepts mixed fp16/bf16 a/b operands. In attention q and k share the model
    # dtype, so `k_t.dtype == q_type` in practice.
    comptime ScoreMma = MmaOpApple[
        DType.float32,
        q_type,
        NumMMmas,
        1,
        b_type=k_t.dtype,
        transpose_b=True,
    ]
    # Type-only: holds the full Sk-wide score tile (NumNMmas col-blocks). Only
    # its `AccumType`/`zero_accum()` are used as score storage; the MMA itself
    # runs one col-block at a time via `ScoreMma` (num_n_mmas=1).
    comptime ScoreAccum = MmaOpApple[
        DType.float32, q_type, NumMMmas, NumNMmas, transpose_b=True
    ]
    comptime OutMma = MmaOpApple[DType.float32, q_type, NumMMmas, DEPTH_MMAS]

    var score_mma = ScoreMma()
    var out_mma = OutMma()
    var output_accum = OutMma.zero_accum()
    var softmax = AppleSoftmax[NumMMmas, NumNMmas]()

    comptime if sink:
        # Pre-seed (m, l) from the per-head sink weight. The naive softmax
        # reference (nn/softmax.mojo) compares the RAW sink weight
        # against the post-scale `row_max` and adds `exp(sink - row_max)` to the
        # denominator -- the sink weight is NOT multiplied by `scale`. Apple uses
        # plain `exp` (no log2e factor). Mirrors amd-attention-sink-as-init-state,
        # retuned for base-e + unscaled sink. `sink_weights` is the per-head 1D
        # tensor; index by `head_id`. The deref is comptime-gated on `sink`, so
        # the None case is never reached (KB `unsafepointer-is-non-nullable`:
        # nullable state is an OptionalReg, not a dangling pointer).
        var sw = rebind[Scalar[q_type]](sink_weights.value()[head_id])
        softmax.seed_sink(sw.cast[DType.float32]())

    # Absolute query positions for masking: row q_row0 + local -> score_row.
    var score_row_base = q_row0 + cur_cache_len - cur_query_len

    # The number of valid query rows in this tile (for output store bounds).
    var valid_q = min(SQ, cur_query_len - q_row0)

    # Causal (and any future monotonic-in-k mask): once a KV tile is fully
    # masked, every higher `kv0` is too, so we can `break` instead of
    # spinning the loop counter. Non-monotonic masks (ChunkedMask,
    # sliding-window) full-mask BOTH ends of the KV range, so they must
    # `continue` (skip the body) on each FULL_MASK tile, NOT break. Gated
    # comptime on the mask type so NullMask compiles the check away entirely
    # (it never full-masks) and only CausalMask takes the early-exit.
    # Mirrors the AMD prefill split: interior-FULL_MASK skip
    # (mha_prefill.mojo:266-275, `continue`) vs causal monotonic bounds
    # (mha_prefill.mojo:236-242, `has_interior_full_mask` gate). For Apple we
    # express the causal case as a monotonic early-exit on the same generic
    # `status` query rather than precomputing `last_masked_set_end`, so a
    # single code path stays correct for every `MHAMask`.
    comptime causal_monotonic = mask_t == CausalMask

    # Stream the KV range in SK-wide tiles.
    var kv0 = 0
    while kv0 < cur_cache_len:
        # --- Mask-derived tile skip. A FULL_MASK tile is a no-op on the
        # online-softmax state (all scores -> NEG_INF: row-max unchanged,
        # exp(NEG_INF - m) = 0 so `l` unchanged and P = 0 so PV adds nothing,
        # rescale exp(m_old - m_new) = 1 so O unchanged), hence skipping it is
        # exactly equivalent to processing it. The tile is at ABSOLUTE coords
        # so the mask sees the same geometry as the per-element `mask()` calls
        # below. Equivalence holds with the sink-seeded (m, l) init (a skipped
        # tile leaves the seed untouched) and with the per-element OOB masking
        # (the skipped elements would all have been NEG_INF anyway). For causal
        # every query attends at least its own position, so the diagonal tile
        # is never skipped and `softmax.normalize` never divides by l == 0. ---
        var tile_status = mask_functor.status[element_type=DType.uint32](
            UInt32(batch_id),
            Index[dtype=DType.uint32](score_row_base, kv0),
            Index[dtype=DType.uint32](SQ, SK),
        )
        if tile_status == TileMaskStatus.FULL_MASK:
            comptime if causal_monotonic:
                break
            else:
                kv0 += SK
                continue

        # --- QK: scores[Sq, Sk] = Q @ K^T (transpose_b). ---
        # Per 16-row KV sub-tile, resolved through the MHAOperand paging
        # contract. For contiguous/ragged KV `block_paged_tile` is a pointer
        # offset; for paged KV it resolves the page holding keys
        # [kv0+ni*16, kv0+ni*16+16). A 16-row sub-tile never crosses a page
        # boundary (page_size % 16 == 0, comptime-asserted above), so the
        # in-page `kv_row_stride` is valid within each sub-tile. The score MMA
        # runs once per `ni` with `num_n_mmas=1` (each column-block is its own
        # page-resolved sub-tile), writing into the full score accumulator.
        var scores = ScoreAccum.zero_accum()
        comptime for ni in range(NumNMmas):
            var k_tile = k.block_paged_tile[1](
                UInt32(batch_id),
                UInt32(kv0 + ni * MMA_DIM),
                UInt32(kv_head),
                kv_layout_val,
            )
            # One QK MMA for this key sub-tile: (SQ x Depth) @ (16 x Depth)^T ->
            # (SQ x 16). `ScoreMma.mma` loads BOTH Q and K from memory (the
            # fragment load stays SIMD-level inside MmaOpApple) and iterates the
            # Depth K-axis internally. `num_n_mmas=1` => one column-block (this
            # key sub-tile); copy each row-block into the full `NumNMmas`-wide
            # score accumulator.
            var score_col = ScoreMma.zero_accum()
            score_mma.mma(score_col, q_tile, k_tile)
            comptime for mi in range(NumMMmas):
                scores[mi * NumNMmas + ni] = rebind[
                    type_of(scores[mi * NumNMmas + ni])
                ](score_col[mi])

        # --- Mask + scale. Apply the mask functor at absolute coords; OOB keys
        # (kv0+col >= cur_cache_len) and OOB rows are set to NEG_INF. This is
        # why the QK MMA above can read K rows past num_keys unguarded: every
        # score from an OOB key is overwritten with NEG_INF here, so the garbage
        # QK product never reaches the softmax. PV cannot rely on the same trick
        # -- an OOB V row poisons via 0 * V_oob, so it bounds its load (below).
        comptime for mi in range(NumMMmas):
            comptime for ni in range(NumNMmas):
                var idx = mi * NumNMmas + ni
                var frag = scores[idx]
                comptime for el in range(8):
                    var lrow = mi * 16 + rb + (8 if el > 3 else 0)
                    var lcol = ni * 16 + cb + (el & 3)
                    var key = kv0 + lcol
                    var qrow = q_row0 + lrow
                    var s = frag[el] * scale
                    if qrow < cur_query_len and key < cur_cache_len:
                        frag[el] = mask_functor.mask(
                            Index(
                                batch_id,
                                head_id,
                                score_row_base + lrow,
                                key,
                            ),
                            s,
                        )
                    else:
                        frag[el] = NEG_INF
                scores[idx] = frag

        # --- Online softmax: scores -> P (in place), rescale output. ---
        softmax.update[DEPTH_MMAS](scores, output_accum)

        # --- PV: output += P @ V. P (the QK C-fragment) is fed directly as the
        # MMA A-operand -- same layout, no memory round-trip; V loads via
        # load_fragment. Only the last KV stage can be partial, and there an OOB
        # V row would poison the accumulator (0 * V_oob == NaN), so it bounds the
        # V load to zero-fill OOB rows. The full-vs-partial split is a single
        # per-kv0 branch, keeping the common full-tile path branchless. ---
        var sk_tile_full = kv0 + SK <= cur_cache_len
        if sk_tile_full:
            comptime for ki in range(NumNMmas):
                # P fragment (mi, ki) cast to q_type — the score accum IS the A
                # operand (FlashAttention register-resident trick).
                var p_frags = InlineArray[SIMD[q_type, 8], NumMMmas](
                    uninitialized=True
                )
                comptime for mi in range(NumMMmas):
                    p_frags[mi] = scores[mi * NumNMmas + ki].cast[q_type]()
                # V sub-tile for the 16 keys [kv0+ki*16, kv0+ki*16+16),
                # page-resolved once (invariant across the depth `ni` blocks).
                # Same page-per-16-row-sub-tile contract as the K load above.
                var v_tile = v.block_paged_tile[1](
                    UInt32(batch_id),
                    UInt32(kv0 + ki * MMA_DIM),
                    UInt32(kv_head),
                    kv_layout_val,
                )
                comptime for ni in range(DEPTH_MMAS):
                    # V fragment (ki, ni): row ki*16 (key), col-block ni*16
                    # (depth). depth col-block ni*16 is contiguous within a
                    # token, so it never crosses a page boundary.
                    # `.tile[16, 16](0, ni)` descends to the depth sub-tile;
                    # load_fragment formats the SIMD fragment (the bit-scatter
                    # stays inside MmaOpApple).
                    var v_sub = v_tile.tile[MMA_DIM, MMA_DIM](0, ni)
                    var v_frag = out_mma.load_fragment[v_t.dtype](v_sub).cast[
                        q_type
                    ]()
                    comptime for mi in range(NumMMmas):
                        var oidx = mi * DEPTH_MMAS + ni
                        _mma_apple_transposable(
                            output_accum[oidx],
                            p_frags[mi],
                            v_frag,
                            output_accum[oidx],
                            False,
                            False,
                        )
        else:
            # Partial last stage (`cur_cache_len % 16 != 0`): bound the V load
            # per sub-tile so OOB rows zero-fill.
            comptime for ki in range(NumNMmas):
                var p_frags = InlineArray[SIMD[q_type, 8], NumMMmas](
                    uninitialized=True
                )
                comptime for mi in range(NumMMmas):
                    p_frags[mi] = scores[mi * NumNMmas + ki].cast[q_type]()
                var v_tile = v.block_paged_tile[1](
                    UInt32(batch_id),
                    UInt32(kv0 + ki * MMA_DIM),
                    UInt32(kv_head),
                    kv_layout_val,
                )
                # Valid V rows in this sub-tile: keys [kv0+ki*16, cur_cache_len).
                var v_valid_rows = cur_cache_len - (kv0 + ki * MMA_DIM)
                comptime for ni in range(DEPTH_MMAS):
                    var v_sub = v_tile.tile[MMA_DIM, MMA_DIM](0, ni)
                    var v_frag: SIMD[v_t.dtype, 8]
                    if v_valid_rows >= MMA_DIM:
                        v_frag = out_mma.load_fragment[v_t.dtype](v_sub)
                    else:
                        v_frag = out_mma.load_fragment[v_t.dtype, bounded=True](
                            v_sub, v_valid_rows
                        )
                    var v_frag_q = v_frag.cast[q_type]()
                    comptime for mi in range(NumMMmas):
                        var oidx = mi * DEPTH_MMAS + ni
                        _mma_apple_transposable(
                            output_accum[oidx],
                            p_frags[mi],
                            v_frag_q,
                            output_accum[oidx],
                            False,
                            False,
                        )

        kv0 += SK

    # --- Epilogue: O /= l, cast to output dtype, masked store. ---
    softmax.normalize[DEPTH_MMAS](output_accum)

    # Output simdgroup view: SQ rows x Depth cols at this tile's base. The fp32
    # accumulator must be CAST to `output_type` (fp16/bf16) on store, so we
    # can't take the fp32-only `MmaOpApple.store_bounded` fast path; instead we
    # descend to each 16x16 sub-tile with `.tile[16, 16]()` (structured
    # addressing, no pointer math) and scatter the fragment with `out_mma`'s
    # `rb`/`cb` lane map, casting per element and bounds-checking the valid
    # query rows. This mirrors the matmul kernel's `_apply_epilogue`
    # (non-fp32-out path). Columns past `depth` are skipped (`depth == Depth`
    # here, so only the OOB query rows are clamped).
    var out_layout_val = TileLayout(
        Coord(Idx[SQ], Idx[Depth]), Coord(q_row_stride, Idx[1])
    )
    var out_tile = TileTensor[
        output_type, type_of(out_layout_val), MutAnyOrigin
    ](
        ptr=output.ptr + out_offset + q_row0 * q_row_stride,
        layout=out_layout_val,
    )
    comptime for mi in range(NumMMmas):
        comptime for ni in range(DEPTH_MMAS):
            var o_sub = out_tile.tile[MMA_DIM, MMA_DIM](mi, ni)
            comptime assert o_sub.flat_rank == 2, "output sub-tile must be 2D"
            var frag = output_accum[mi * DEPTH_MMAS + ni]
            comptime for el in range(8):
                var lrow = rb + (8 if el > 3 else 0)
                var lcol = cb + (el & 3)
                # `lrow`/`lcol` are coords WITHIN this 16x16 sub-tile; the
                # absolute query row is `mi*16 + lrow`, the absolute depth col
                # `ni*16 + lcol`. Bound on the valid query rows; depth is full.
                if (
                    mi * MMA_DIM + lrow < valid_q
                    and ni * MMA_DIM + lcol < depth
                ):
                    o_sub[lrow, lcol] = rebind[o_sub.ElementType](
                        SIMD[output_type, 1](frag[el].cast[output_type]())
                    )


# ===-------------------------------------------------------------------=== #
# Host launcher. Mirrors `mha_gpu_naive` / `naive_fa_decode_apple` (MHAOperand
# overload) signature; enqueues the prefill kernel. Dispatches the runtime
# `depth` to a compile-time `Depth` specialization over multiples of 16.
# ===-------------------------------------------------------------------=== #
def fa_prefill_apple[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    num_simdgroups: Int = 16,
](
    q: LayoutTensor[mut=False, address_space=AddressSpace.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: LayoutTensor[
        mut=True, output_type, address_space=AddressSpace.GENERIC, ...
    ],
    valid_length: LayoutTensor[
        mut=False, DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, q.dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
) raises:
    """Host launcher for the Apple M5 MMA-based flash-attention prefill kernel.

    Mirrors `mha_gpu_naive`'s `MHAOperand` overload so `flash_attention_dispatch`
    can route to it identically to the fallback. Specializes one kernel per
    supported `depth` (a multiple of 16 up to `FA_PREFILL_APPLE_MAX_HEAD_DIM`).
    The dispatcher guarantees `depth` matches a branch and that KV is either
    contiguous (`page_size == 0`) or paged with `page_size % 16 == 0` (the
    KV-tile load resolves the page per 16-row sub-tile); an odd page size keeps
    falling through to `mha_gpu_naive`.

    The external `q`/`output`/`valid_length` are `LayoutTensor`s (the
    `flash_attention_dispatch` ABI); they are converted to `TileTensor` at the
    enqueue boundary so the kernel-internal code is TileTensor-only.
    """
    # No `is_apple_gpu()` assert here — this launcher compiles for the host
    # target, where that target-query is always False. The Apple gate is the
    # caller's (`has_apple_gpu_accelerator()` in dispatch).
    comptime q_type = q.dtype
    comptime p_type = get_accum_type[q_type]()

    var num_keys = max_cache_size
    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    debug_assert(
        depth % MMA_DIM == 0 and depth <= FA_PREFILL_APPLE_MAX_HEAD_DIM,
        (
            "fa_prefill_apple requires depth %% 16 == 0 and depth <="
            " FA_PREFILL_APPLE_MAX_HEAD_DIM; the dispatcher must gate"
            " unsupported head dims to mha_gpu_naive"
        ),
    )

    comptime NumMMmas = DEFAULT_NUM_M_MMAS
    comptime NumNMmas = DEFAULT_NUM_N_MMAS
    comptime SQ = NumMMmas * MMA_DIM

    # Each threadgroup covers num_simdgroups*SQ contiguous query rows.
    var num_q_tiles = ceildiv(max_prompt_len, num_simdgroups * SQ)

    # Convert the external LayoutTensor ABI to flat 1D TileTensors over the
    # whole q/output buffers. The kernel bakes the ragged/BSHD + q_row0 offset
    # into each per-simdgroup tile base; the flat view just carries the device
    # pointer with TileTensor typing (no LayoutTensor inside the kernel).
    var q_flat = TileTensor(
        q.ptr.as_immutable().as_unsafe_any_origin(),
        row_major(Coord(Int(q.size()))),
    )
    var output_flat = TileTensor(
        output.ptr.as_unsafe_any_origin(),
        row_major(Coord(Int(output.size()))),
    )
    var valid_length_flat = TileTensor(
        valid_length.ptr.as_immutable().as_unsafe_any_origin(),
        row_major(Coord(Int(valid_length.size()))),
    )

    # Sink weights: a nullable `OptionalReg[TileTensor]` passed by value (NOT a
    # dangling `UnsafePointer` -- KB `unsafepointer-is-non-nullable`). When
    # sink=False this is None and never read (the seed is comptime-gated on
    # `sink` in the kernel). The per-head [num_heads] tensor is converted to a
    # TileTensor here so the kernel stays TileTensor-only; the kernel indexes
    # by `head_id`.
    var sink_layout_val = row_major(Coord(num_heads))
    comptime SinkTile = TileTensor[
        q_type, type_of(sink_layout_val), ImmutAnyOrigin
    ]
    var sink_tile: OptionalReg[SinkTile]
    comptime if sink:
        var sw = sink_weights.value()
        sink_tile = OptionalReg[SinkTile](
            SinkTile(
                sw.ptr.as_immutable().as_unsafe_any_origin(),
                sink_layout_val,
            )
        )
    else:
        sink_tile = None

    comptime MAX_D_STEPS = FA_PREFILL_APPLE_MAX_HEAD_DIM // MMA_DIM
    comptime for di in range(1, MAX_D_STEPS + 1):
        comptime D = di * MMA_DIM
        if depth == D:
            comptime core_kernel = fa_prefill_apple_core[
                q_type,
                output_type,
                p_type,
                k_t,
                v_t,
                mask_t,
                type_of(q_flat).LayoutType,
                type_of(output_flat).LayoutType,
                type_of(valid_length_flat).LayoutType,
                type_of(sink_layout_val),
                ragged=ragged,
                sink=sink,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
                Depth=D,
                NumMMmas=NumMMmas,
                NumNMmas=NumNMmas,
                NumSimdgroups=num_simdgroups,
            ]
            ctx.enqueue_function[core_kernel](
                output_flat,
                q_flat,
                k,
                v,
                mask_functor,
                valid_length_flat,
                sink_tile,
                scale,
                batch_size,
                max_prompt_len,
                max_cache_size,
                num_heads,
                depth,
                group,
                grid_dim=(num_q_tiles, num_heads, batch_size),
                block_dim=num_simdgroups * WARP_SIZE,
            )
