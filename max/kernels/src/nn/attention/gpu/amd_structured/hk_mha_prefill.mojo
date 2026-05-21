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
"""HKMhaPrefill — long-context BF16 MHA prefill for AMD MI355X (gfx950).

`run` interleaves QK MFMA, PV MFMA, softmax + rescale, and
K/V DMA across an explicit cluster schedule so each work class
overlaps the others' latency. 8 warps per block; each warp owns a
`Q_BLOCK_SIZE`-row stripe of Q.

## Cluster schedule

The main loop runs 8 clusters per iteration and advances `j` by 2,
so each iter processes two K/V tiles. Each cluster ends in a bare
`s_barrier`.

  | Cluster | Work                                                |
  |---------|-----------------------------------------------------|
  | C0      | QK[j-2] + tail softmax of tile (j-3)                |
  | C1      | DMA K[j] + LDS→register V[j-3]                      |
  | C2      | PV[j-3] strip-interleaved with partial softmax(j-2) |
  | C3      | DMA V[j-1] + LDS→register K[j-1]                    |
  | C4      | QK[j-1] + tail softmax of tile (j-2)                |
  | C5      | DMA K[j+1] + LDS→register V[j-2] + causal mask      |
  | C6      | PV[j-2] strip-interleaved with partial softmax(j-1) |
  | C7      | DMA V[j] + LDS→register K[j]                        |

Whole-tile K is pre-loaded one iteration ahead into the persistent
`k_reg`, so the QK clusters (C0/C4) contain MFMAs + VALU only — no
in-cluster `ds_read`. The prologue primes the pipeline and runs
QK[0] + partial softmax; the 13-cluster epilogue drains the final
four tiles `N-4..N-1`, with whole-V PV (no strip split) and an
unconditional normalizer rescale before the `o / norm_vec` divide.

## Key design choices

- **Whole-tile K pre-load with consumer-side waitcnt drains.** Each
  MFMA-consumer helper opens with `s_waitcnt[lgkmcnt=0]()` so
  SIInsertWaitcnts treats the cluster as a bracket reset and the per-
  consumer `lgkmcnt` staircase collapses (see KB
  `patterns/amd-explicit-lgkmcnt-drain-consumer-cluster`).

- **Kernel-scope BF16 P-cache.** Each softmax bulk-casts FP32 att to
  one persistent `att_block_bf16` register tile reused by the
  subsequent PV (see KB
  `known-limitations/llvm-amdgpu-cast-rematerialization`).

- **Lazy rescale (`RESCALE_THRESHOLD=8`).** In C2/C6, when the
  running max grows by more than 8 log2 units, `o_reg *= scale_vec`
  fires between PV strip 0 and strips 1-3 — strips 1-3 then
  contribute at the old scale into an already-rescaled accumulator.
  The 8 log2 cap bounds the inconsistency. When `rv_all_below`
  reports no lane exceeded the threshold, the rescale is skipped
  entirely. The epilogue rescales unconditionally with
  `scale_vec` initialized to ones, so the multiply is identity when
  no rescale ever fired.

- **Causal mask placement.** Tiles `0` (prologue), `(j - 1)` for
  each main-loop iter (C5), and `N - 3, N - 2, N - 1` (epilogue).
  Tiles `1, 3, …` in the main-loop range are unmasked by design;
  the `max_num_tiles` cap guarantees those positions are naturally
  fully unmasked.

- **Output transpose.** `col_l → row_l` is a zero-cost re-tag of the
  same register storage — no cross-lane permute, no data motion.

- **GQA-aware head remap.** `head_idx` is `(block_x % GROUP) *
  NUM_KV_HEADS + (block_x / GROUP)` — the transpose over the
  `(NUM_KV_HEADS, GROUP)` rectangle — so adjacent blocks visit
  different KV heads across CUs/XCDs. Bijective for any
  `NUM_HEADS == GROUP * NUM_KV_HEADS`; reduces to identity at MHA
  (`GROUP=1`) and MQA (`NUM_KV_HEADS=1`).

The cluster decomposition and overlap pattern are inspired by the
HipKittens project's attention kernel
(<https://github.com/HazyResearch/HipKittens>).
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    warp_id,
)
from std.gpu.host import DeviceContext
from std.gpu.host.compile import CompilationTarget
from std.math import ceildiv
from std.gpu.sync import (
    AMDScheduleBarrierMask,
    s_waitcnt,
    schedule_group_barrier,
)
from std.math import exp2 as math_exp2
from std.memory import AddressSpace
from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import (
    likely,
    llvm_intrinsic,
    readfirstlane,
    unlikely,
)
from std.utils import IndexList, StaticTuple

from layout import TensorLayout, TileTensor
from layout.coord import Coord
from layout.swizzle import Swizzle
from layout.tile_layout import (
    ComptimeInt,
    Idx,
    Layout as TileLayout,
    col_major,
    row_major,
)
from structured_kernels.amd_tile_io import (
    RegTile,
    RegTileEpilogue,
    RegTileLoader,
    SMemTile,
    SubTileLoaderLDS,
    SubTileLoaderLDS_HK_st_8x32,
    reg_alloc,
    smem_alloc,
)
from nn.attention.mha_mask import (
    CausalMask,
    MHAMask,
    NullMask,
    TileMaskStatus,
)
from nn.attention.mha_operand import MHAOperand
from std.sys.intrinsics import _type_is_eq

from .hk_mha_mask import _apply_causal_mask_fast, apply_mask_to_att_block
from .hk_mha_softmax import (
    col_max,
    col_max_acc,
    col_sum_acc,
    div_col,
    mul_col_inplace,
    rv_all_below,
    sub_col_inplace,
)

from .hk_mha_mma_op import HKMhaConfig, MhaMmaOp

from .iglp import (
    _iglp_opt,
    sched_barrier_exp_pairs,
    sched_barrier_pairs,
    sched_dsread_valu_pairs,
)
from std.sys import get_defined_bool, size_of


# AMDGPU instruction-priority + IGroupLP scheduling helpers.


@always_inline
def _s_setprio[priority: Int16]():
    """Sets MFMA wave instruction priority (0 = normal, 1 = high)."""
    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)


@always_inline
def _sched_barrier_zero():
    """`sched_barrier(0)` — hard reordering barrier that pins
    surrounding instructions to their source order."""
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))


@always_inline
def _asm_label[asm_str: StaticString]():
    """Emits an AMDGPU asm comment at the call site so disassembly diff
    against a reference kernel can be done by grep. Gated on
    `EMIT_ASM_LABELS`; when False this is a no-op. The
    `has_side_effect=True` + `~{memory}` form is a hard reordering
    barrier, so labels MUST be off when benchmarking — turn on for
    asm-level inspection only."""

    comptime emit_asm_label = get_defined_bool["EMIT_ASM_LABELS", False]()

    comptime if emit_asm_label:
        inlined_assembly[
            asm_str,
            NoneType,
            constraints="~{memory}",
            has_side_effect=True,
        ]()


@always_inline
def _s_barrier_raw():
    """Bare `s_barrier`, with NO release/acquire fences. Mojo's stdlib
    `barrier()` would also inject `s_waitcnt vmcnt(0) lgkmcnt(0)`, which
    would defeat the partial-drain `s_waitcnt vmcnt(N) lgkmcnt(0)` we
    emit before each barrier to let DMAs cross."""
    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()


@always_inline
def _cluster_barrier():
    """Cluster boundary: `sched_barrier(0)` + bare `s_barrier` +
    `sched_barrier(0)`. The `sched_barrier(0)` fences pin the
    `s_barrier` to its source position so IGroupLP cannot move
    cluster-N work past the wave sync into cluster-(N+1)."""
    _sched_barrier_zero()
    _s_barrier_raw()
    _sched_barrier_zero()


struct HKMhaPrefill[config: HKMhaConfig]:
    """8-warp MHA forward kernel parameterized by `HKMhaConfig`.

    Each block runs `config.num_warps` wave64 warps that share K/V
    SMEM via cooperative DMA. Warp `w` owns Q rows `[w * q_block_size,
    (w + 1) * q_block_size)` of the block's stripe and carries its own
    register-resident attention state.

    Parameters:
        config: Shape configuration (`HKMhaConfig`).
    """

    # Config aliases — expose `config` fields as struct comptime
    # constants so body code can keep referencing `Self.Q_BLOCK_SIZE`,
    # `Self.DEPTH`, etc. unchanged.
    comptime Q_BLOCK_SIZE = Self.config.q_block_size
    comptime KV_BLOCK = Self.config.kv_block
    comptime DEPTH = Self.config.depth
    comptime NUM_HEADS = Self.config.num_heads
    comptime NUM_KV_HEADS = Self.config.num_kv_heads
    comptime NUM_WARPS = Self.config.num_warps
    comptime RESCALE_THRESHOLD = Self.config.rescale_threshold

    comptime NUM_THREADS = Self.NUM_WARPS * 64
    comptime BM = Self.NUM_WARPS * Self.Q_BLOCK_SIZE
    comptime D_FRAG_PER_LANE = (Self.DEPTH * Self.Q_BLOCK_SIZE) // 64

    # MHA MMA operator (single source of truth for the MFMA shape, SMEM
    # sub-block geometry, register-tile layouts, and SMEM→register
    # loaders; specialized for `v_mfma_f32_32x32x16_bf16`).
    comptime _MmaOp = MhaMmaOp[DType.bfloat16, Self.config]

    # Layout types per operand role — captured from `_MmaOp` so the
    # `row_major[...]` expression isn't repeated at each `RegTile`
    # signature site.
    comptime _Q_LAYOUT_T = type_of(Self._MmaOp.Q_LAYOUT)
    comptime _K_LAYOUT_T = type_of(Self._MmaOp.K_LAYOUT)
    comptime _V_LAYOUT_T = type_of(Self._MmaOp.V_LAYOUT)
    comptime _ATT_LAYOUT_T = type_of(Self._MmaOp.ATT_LAYOUT)
    comptime _ATT_BF16_SUB_LAYOUT_T = type_of(Self._MmaOp.ATT_BF16_SUB_LAYOUT)
    comptime _ATT_BF16_FULL_LAYOUT_T = type_of(Self._MmaOp.ATT_BF16_FULL_LAYOUT)
    comptime _O_LAYOUT_T = type_of(Self._MmaOp.O_LAYOUT)
    comptime _O_T_LAYOUT_T = type_of(Self._MmaOp.O_T_LAYOUT)

    comptime _RV_LAYOUT_T = type_of(row_major[1, 1]())
    """Row-vector layout for softmax row state (`max_vec`, `norm_vec`,
    `scale_vec`). One element per lane, held redundantly across the
    paired half-warps that share each column of the col_l output."""

    # Derived block-level shape constants used across the kernel body.
    comptime _Q_ROW_STRIDE = Self.NUM_HEADS * Self.DEPTH
    comptime _KV_ROW_STRIDE = Self.NUM_KV_HEADS * Self.DEPTH
    comptime _ATT_PER_LANE = (Self.KV_BLOCK * Self.Q_BLOCK_SIZE) // 64
    """Per-lane element count for the FP32 att tile (col_l rt_32x32)."""
    comptime _ATT_HALF = Self._ATT_PER_LANE // 2
    """First-half / second-half split index for `exp2_inplace_range`."""
    comptime _NUM_PV_SUBTILES = Self.KV_BLOCK // 16
    """Number of 16-row PV strips in one K/V tile (`KV_BLOCK / 16`)."""

    # K SMEM swizzle: a `Swizzle(1, 1, 4)` + `Swizzle(1, 0, 6)` pair
    # realizes a byte-level `bit5 ^= bit9; bit4 ^= bit10` remap at vec
    # (8 BF16 = 16 B) worker scope, matching HK `st_32x32_s`. V uses
    # HK's identity-swizzle `st_8x32_s`.
    comptime k_swizzle = Optional(Swizzle(1, 1, 4))
    comptime k_swizzle2 = Optional(Swizzle(1, 0, 6))
    comptime v_swizzle = Optional[Swizzle](None)

    comptime KTileLoader = SubTileLoaderLDS[
        DType.bfloat16, Self.k_swizzle, Self.k_swizzle2
    ]

    comptime VTileLoader = SubTileLoaderLDS_HK_st_8x32[
        DType.bfloat16, Self.KV_BLOCK, Self.DEPTH, 32, Self.NUM_THREADS
    ]

    # Per-(batch, head) 2D layout TYPES used as `.reshape` targets after
    # slicing the input 4D / 3D tensors down to a per-(batch, head)
    # plane. Layout VALUES are constructed at the call site because the
    # `seq_len` / `num_keys` dim is runtime.
    comptime _QPerHeadLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.DEPTH]].element_types,
        Coord[ComptimeInt[Self._Q_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]
    comptime _KVPerHeadLayoutT = TileLayout[
        Coord[Int32, ComptimeInt[Self.DEPTH]].element_types,
        Coord[ComptimeInt[Self._KV_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]
    # Per-tile K/V layout (KV_BLOCK x DEPTH) used by the paged path. The
    # stride matches `_KV_ROW_STRIDE` (NUM_KV_HEADS * DEPTH) so the
    # cooperative loaders see the same row stride as the contiguous
    # path's `.tile[KV_BLOCK, DEPTH](t, 0)` sub-slice. Static shape +
    # static stride = constant folds everywhere.
    comptime _KvPerTileLayoutT = TileLayout[
        Coord[
            ComptimeInt[Self.KV_BLOCK], ComptimeInt[Self.DEPTH]
        ].element_types,
        Coord[ComptimeInt[Self._KV_ROW_STRIDE], ComptimeInt[1]].element_types,
    ]
    # IGLP `sched_group` IDs — one per scheduling-distinct cluster. The
    # numbers must be unique within the kernel; names document which
    # cluster each ID belongs to.
    comptime _SCHED_MAIN_C0_QK_TAIL = 1
    comptime _SCHED_MAIN_C2_PV_PARTIAL = 2
    comptime _SCHED_MAIN_C4_QK_TAIL = 3
    comptime _SCHED_MAIN_C6_PV_PARTIAL = 4
    comptime _SCHED_EPI_C0_TAIL = 5
    comptime _SCHED_EPI_C2_PV_PARTIAL = 6
    comptime _SCHED_EPI_C4_TAIL = 7
    comptime _SCHED_EPI_C6_PV_PARTIAL = 8
    comptime _SCHED_EPI_C8_TAIL = 9
    comptime _SCHED_EPI_C10_FULL = 10
    comptime _SCHED_MAIN_C5_DSREAD = 11
    comptime _SCHED_EPI_C5_DSREAD = 12
    comptime _SCHED_EPI_C9_DSREAD = 13

    @staticmethod
    @always_inline
    def load_q[
        layout: TensorLayout
    ](
        q_warp_2d: TileTensor[DType.bfloat16, layout, ...],
    ) -> RegTile[
        DType.bfloat16, Self._Q_LAYOUT_T, MutExternalOrigin
    ]:
        """Loads the warp's Q sub-tile from gmem into a row_l register
        tile via `RegTileLoader`.

        For d=128 / Q_BLOCK_SIZE=32: 8 `buffer_load_bf16x8` per lane
        (8 base tiles of MMA_K=16 cols each, distributed col_major[32, 2]
        with 8 BF16 per lane per base tile)."""
        comptime _FRAG = Self._MmaOp.FRAG_ELTS
        comptime _BK = Self._MmaOp.MMA_K
        comptime _num_k_tiles = Self.DEPTH // _BK
        comptime _q_thread_layout = col_major[
            Self.Q_BLOCK_SIZE, WARP_SIZE // Self.Q_BLOCK_SIZE
        ]()

        var q_reg = reg_alloc[DType.bfloat16](Self._MmaOp.Q_LAYOUT)
        var q_loader = RegTileLoader[
            DType.bfloat16, _q_thread_layout, warp_scope=True
        ](q_warp_2d)

        comptime for j in range(_num_k_tiles):
            var src = q_warp_2d.tile[Self.Q_BLOCK_SIZE, _BK](0, j)
            var dst = q_reg.tile[1, 1, _FRAG](0, j, 0).reshape(
                row_major[1, _FRAG]()
            )
            q_loader.load(dst, src.vectorize[1, _FRAG]())

        return q_reg

    @staticmethod
    @always_inline
    def _load_q_and_scale[
        layout: TensorLayout
    ](
        q_warp_2d: TileTensor[DType.bfloat16, layout, ...],
        scale_log2e: Float32,
    ) -> RegTile[DType.bfloat16, Self._Q_LAYOUT_T, MutExternalOrigin]:
        """Loads Q from gmem and prescales it by `scale * log2e`.

        The multiply is done per-fragment in FP32 then cast back to BF16,
        so only one FP32 fragment is alive at a time. The downstream QK
        MFMA consumes `q_reg` as B in pre-transpose form via
        `mma[swap_b=True]` — no explicit transpose tile needed."""
        var q_reg = Self.load_q(q_warp_2d)

        comptime _H = Self._Q_LAYOUT_T.static_shape[0]
        comptime _W = Self._Q_LAYOUT_T.static_shape[1]
        comptime _F = Self._Q_LAYOUT_T.static_shape[2]
        var q_v = q_reg.vectorize[1, 1, _F]()
        comptime assert q_v.flat_rank == 3
        comptime for h in range(_H):
            comptime for w in range(_W):
                q_v[h, w, 0] = (
                    q_v[h, w, 0].cast[DType.float32]() * scale_log2e
                ).cast[DType.bfloat16]()

        return q_reg

    # K/V DMA helpers: issue the gmem→LDS load only, with no barrier or
    # waitcnt so the caller can overlap MFMAs with in-flight DMAs. The
    # per-tile gmem_tile is supplied by the caller (built via
    # `_make_k_tile` / `_make_v_tile` from the MHAOperand), so the helpers
    # are agnostic to paged-vs-contiguous KV.

    @staticmethod
    @always_inline
    def _make_k_tile[
        k_t: MHAOperand,
        //,
    ](
        k_op: k_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
    ) -> TileTensor[
        DType.bfloat16, Self._KvPerTileLayoutT, ImmutAnyOrigin
    ]:
        """Builds the per-tile K gmem TileTensor at `(batch, t*KV_BLOCK,
        kv_head, 0)` via `MHAOperand.block_paged_tile`. For
        LayoutTensorMHAOperand this resolves to a pointer-arithmetic
        offset into a contiguous buffer; for KVCacheMHAOperand it
        resolves through the page table."""
        comptime assert (
            k_t.dtype == DType.bfloat16
        ), "HKMhaPrefill: K must be BF16"
        # rebind: comptime k_t.dtype == BF16 (asserted above), so the
        # returned TileTensor's dtype parameter is statically BF16.
        return rebind[
            TileTensor[DType.bfloat16, Self._KvPerTileLayoutT, ImmutAnyOrigin]
        ](
            k_op.block_paged_tile[Self.KV_BLOCK](
                batch_idx,
                UInt32(t * Self.KV_BLOCK),
                kv_head_idx,
                Self._KvPerTileLayoutT(
                    Coord(Idx[Self.KV_BLOCK], Idx[Self.DEPTH]),
                    Coord(Idx[Self._KV_ROW_STRIDE], Idx[1]),
                ),
            )
        )

    @staticmethod
    @always_inline
    def _make_v_tile[
        v_t: MHAOperand,
        //,
    ](
        v_op: v_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
    ) -> TileTensor[
        DType.bfloat16, Self._KvPerTileLayoutT, ImmutAnyOrigin
    ]:
        """Builds the per-tile V gmem TileTensor (see `_make_k_tile`)."""
        comptime assert (
            v_t.dtype == DType.bfloat16
        ), "HKMhaPrefill: V must be BF16"
        return rebind[
            TileTensor[DType.bfloat16, Self._KvPerTileLayoutT, ImmutAnyOrigin]
        ](
            v_op.block_paged_tile[Self.KV_BLOCK](
                batch_idx,
                UInt32(t * Self.KV_BLOCK),
                kv_head_idx,
                Self._KvPerTileLayoutT(
                    Coord(Idx[Self.KV_BLOCK], Idx[Self.DEPTH]),
                    Coord(Idx[Self._KV_ROW_STRIDE], Idx[1]),
                ),
            )
        )

    @staticmethod
    @always_inline
    def _dma_k[
        k_t: MHAOperand,
        //,
    ](
        k_smem_slot: SMemTile[DType.bfloat16, ...],
        k_op: k_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
        w_id: Int,
        l_id: Int,
    ):
        """Issues the K[t] DMA into `k_smem_slot` from the MHAOperand.

        Partition: at d=128 the K tile holds 8 `K_SUB_ROWS x K_SUB_COLS`
        sub-blocks (2 row-tiles × 4 col-tiles), matching `NUM_WARPS=8`
        → 1 warp / 1 sub-block; each warp loads a full 32×32 sub-block
        via two 16-row internal strips. At d=64 the K tile holds only
        4 sub-blocks (2 × 2) → 2 warps cooperate per sub-block, each
        loading a 16-row half-strip. The per-warp `SubTileLoaderLDS`
        bakes HK's two-XOR `st_32x32_s` swizzle into the DRAM-source
        lane mapping; `_MmaOp.load_K` unswizzles on read.

        Loader construction is per-tile: the buffer resource is bound
        to this tile's pointer + bounds, so the same helper composes
        with paged KV (where adjacent tiles live in different DRAM
        pages) without a separate code path. Cost is ~4 SGPR ops per
        call to materialize the SRD; negligible vs the DMA latency."""
        comptime _K_SUB_ROWS = Self._MmaOp.K_SUB_ROWS
        comptime _K_SUB_COLS = Self._MmaOp.K_SUB_COLS
        comptime _num_kv_subblocks = (
            (Self.KV_BLOCK // _K_SUB_ROWS) * (Self.DEPTH // _K_SUB_COLS)
        )
        comptime _warps_per_subblock = Self.NUM_WARPS // _num_kv_subblocks
        comptime _rows_per_warp = _K_SUB_ROWS // _warps_per_subblock
        comptime _num_block_cols_k = Self.DEPTH // _K_SUB_COLS
        comptime assert (
            Self.NUM_WARPS == _num_kv_subblocks * _warps_per_subblock
        ), (
            "HKMhaPrefill K DMA: NUM_WARPS must divide evenly into the"
            " K sub-block grid"
        )

        var subblock_id, row_strip = divmod(w_id, _warps_per_subblock)
        var sub_row, sub_col = divmod(subblock_id, _num_block_cols_k)

        var k_gmem_tile = Self._make_k_tile(k_op, batch_idx, kv_head_idx, t)
        var k_loader = Self.KTileLoader(k_gmem_tile)
        k_loader.load(
            k_smem_slot.tile[_K_SUB_ROWS, _K_SUB_COLS](subblock_id, 0).tile[
                _rows_per_warp, _K_SUB_COLS
            ](row_strip, 0),
            k_gmem_tile.tile[_K_SUB_ROWS, _K_SUB_COLS](sub_row, sub_col).tile[
                _rows_per_warp, _K_SUB_COLS
            ](row_strip, 0),
        )

    @staticmethod
    @always_inline
    def _dma_v[
        v_t: MHAOperand,
        //,
    ](
        v_smem_slot: SMemTile[DType.bfloat16, _, MutExternalOrigin, ...],
        v_op: v_t,
        batch_idx: UInt32,
        kv_head_idx: UInt32,
        t: Int,
        w_id: Int,
        l_id: Int,
    ):
        """Issues the V[t] DMA into `v_smem_slot` from the MHAOperand.
        V uses identity swizzle and the cooperative loader handles
        thread-id mapping over the whole `KV_BLOCK x DEPTH` tile.

        Loader construction is per-tile (see `_dma_k` docstring)."""
        var v_gmem_tile = Self._make_v_tile(v_op, batch_idx, kv_head_idx, t)
        var v_loader = Self.VTileLoader(v_gmem_tile)
        v_loader.load(v_smem_slot, v_gmem_tile, w_id, l_id)

    # Whole-tile K pre-load + consumer-side waitcnt drain. `_load_k_reg`
    # runs in a dedicated cluster; `_qk_with_kreg` consumes the result
    # without any in-cluster `ds_read`.

    @staticmethod
    @always_inline
    def _load_k_reg(
        mut k_reg: RegTile[DType.bfloat16, Self._K_LAYOUT_T, MutExternalOrigin],
        k_smem_slot: SMemTile[DType.bfloat16, _, MutExternalOrigin, ...],
    ):
        Self._MmaOp.load_K(k_reg, k_smem_slot)

    @staticmethod
    @always_inline
    def _qk_with_kreg(
        mut att_block: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
        mut k_reg: RegTile[DType.bfloat16, Self._K_LAYOUT_T, MutExternalOrigin],
        mut q_reg: RegTile[DType.bfloat16, Self._Q_LAYOUT_T, MutExternalOrigin],
    ):
        """QK MFMA over pre-loaded `k_reg`. The opening
        `s_waitcnt[lgkmcnt=0]()` collapses the per-VGPR staircase that
        SIInsertWaitcnts would otherwise emit at each consumer (KB
        `patterns/amd-explicit-lgkmcnt-drain-consumer-cluster`)."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        _ = att_block.fill(0)
        Self._MmaOp.mma_QK(att_block, k_reg, q_reg)

    @staticmethod
    @always_inline
    def _load_v_reg(
        mut v_reg: RegTile[DType.bfloat16, Self._V_LAYOUT_T, MutExternalOrigin],
        v_smem_slot: SMemTile[DType.bfloat16, _, MutExternalOrigin, ...],
    ):
        """LDS→register V load. No waitcnt: the consumer PV cluster
        drains LDS via its own `s_waitcnt[lgkmcnt=0]()`."""
        Self._MmaOp.load_V(v_reg, v_smem_slot)

    @staticmethod
    @always_inline
    def _att_bf16_subtile_jit[
        subtile_idx: Int,
    ](
        att_block: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
    ) -> RegTile[
        DType.bfloat16, Self._ATT_BF16_SUB_LAYOUT_T, MutExternalOrigin
    ]:
        """JIT-cast one PV-A subtile (16-row strip) from `att_block` FP32.
        The narrow lifetime lets RA fold the cast registers into the
        surrounding PV MFMA."""
        comptime _strip, _half = divmod(subtile_idx, 2)
        var result = reg_alloc[DType.bfloat16](Self._MmaOp.ATT_BF16_SUB_LAYOUT)
        var src_v = att_block.vectorize[1, 1, 16]()
        var dst_v = result.vectorize[1, 1, 8]()
        var bf16 = src_v[_strip, 0, 0].cast[DType.bfloat16]()
        dst_v[0, 0, 0] = bf16.slice[8, offset=_half * 8]()
        return result

    @staticmethod
    @always_inline
    def _att_bf16_full(
        mut dst: RegTile[
            DType.bfloat16, Self._ATT_BF16_FULL_LAYOUT_T, MutExternalOrigin
        ],
        att_block: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
    ):
        """Bulk-casts the FP32 `att_block` to BF16, writing into the
        caller-provided persistent destination `dst`.

        Must be called in the QK+softmax cluster, NOT inside the consumer
        PV cluster: placing the cast before the `s_barrier` lets the
        `v_cvt`s overlap with the barrier and the next cluster's DMA so
        PV's MFMAs run back-to-back. The persistent destination
        prevents LLVM from rematerializing the cast at each PV use site
        (KB `known-limitations/llvm-amdgpu-cast-rematerialization`)."""
        var src_v = att_block.vectorize[1, 1, 16]()
        var dst_v = dst.vectorize[1, 1, 8]()
        comptime for sub in range(Self._NUM_PV_SUBTILES):
            comptime _strip, _half = divmod(sub, 2)
            var bf16 = src_v[_strip, 0, 0].cast[DType.bfloat16]()
            dst_v[sub, 0, 0] = bf16.slice[8, offset=_half * 8]()

    @staticmethod
    @always_inline
    def _pv_whole(
        v_reg: RegTile[DType.bfloat16, Self._V_LAYOUT_T, MutExternalOrigin],
        att_bf16_full: RegTile[
            DType.bfloat16, Self._ATT_BF16_FULL_LAYOUT_T, MutExternalOrigin
        ],
        mut o_reg: RegTile[DType.float32, Self._O_LAYOUT_T, MutExternalOrigin],
    ):
        """Whole-V PV MFMA over a pre-cast `att_bf16_full`. No fused
        softmax — used by the epilogue PV clusters."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        comptime for i in range(Self._NUM_PV_SUBTILES):
            var v_sub = v_reg.tile[1, Self.DEPTH // 32, 8](i, 0, 0)
            var att_sub = att_bf16_full.tile[1, 1, 8](i, 0, 0)
            Self._MmaOp.mma_PV(o_reg, v_sub, att_sub)

    @staticmethod
    @always_inline
    def _pv_strip_with_partial_softmax[
        sched_group: Int,
    ](
        v_reg: RegTile[DType.bfloat16, Self._V_LAYOUT_T, MutExternalOrigin],
        att_bf16_full: RegTile[
            DType.bfloat16, Self._ATT_BF16_FULL_LAYOUT_T, MutExternalOrigin
        ],
        mut o_reg: RegTile[DType.float32, Self._O_LAYOUT_T, MutExternalOrigin],
        mut max_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut max_vec_prev: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut scale_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut att_block_qk: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
    ) -> Bool:
        """Main-loop C2/C6 body: strip-interleaved PV (using pre-loaded
        `v_reg`) + partial softmax of the next tile. Returns whether a
        rescale of `norm_vec` is pending for the next QK tail.

        APPROXIMATION: when the lazy rescale fires, `o_reg` is rescaled
        BETWEEN PV strip 0 and strips 1..3; strips 1..3 then add to
        already-rescaled `o_reg` at the OLD scale. Bounded by
        `RESCALE_THRESHOLD` and skipped on the `rv_all_below` path."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        _s_setprio[Int16(1)]()

        # PV strip 0.
        var v_sub_0 = v_reg.tile[1, Self.DEPTH // 32, 8](0, 0, 0)
        var att_sub_0 = att_bf16_full.tile[1, 1, 8](0, 0, 0)
        Self._MmaOp.mma_PV(o_reg, v_sub_0, att_sub_0)

        # col_max + lazy rescale decision.
        col_max_acc(max_vec, att_block_qk, max_vec_prev)
        # IGLP: 4×(1 MFMA + 5 VALU) interleaves the next 4 PV MFMAs with
        # the col_max + rescale VALU work.
        sched_barrier_pairs[4, valu_cnt=5, group=sched_group]()
        var pending_scale = False
        var all_ok = rv_all_below(max_vec_prev, max_vec, Self.RESCALE_THRESHOLD)
        if unlikely(not all_ok):
            scale_vec[0, 0] = math_exp2(max_vec_prev[0, 0] - max_vec[0, 0])
            mul_col_inplace(o_reg, scale_vec)
            max_vec_prev.copy_from(max_vec)
            pending_scale = True
        else:
            max_vec.copy_from(max_vec_prev)

        # PV strips 1..3.
        comptime for i in range(1, Self._NUM_PV_SUBTILES):
            var v_sub = v_reg.tile[1, Self.DEPTH // 32, 8](i, 0, 0)
            var att_sub = att_bf16_full.tile[1, 1, 8](i, 0, 0)
            Self._MmaOp.mma_PV(o_reg, v_sub, att_sub)

        # `att - max_vec` + first-half exp2 in preparation for the QK
        # tail softmax cluster that consumes `att_block_qk` next.
        sub_col_inplace(att_block_qk, max_vec)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block_qk)

        # IGLP: trailing PV MFMAs interleaved with `sub_col` (VALU) and
        # first-half `exp2` (TRANS).
        sched_barrier_pairs[
            Self._NUM_PV_SUBTILES + 2, valu_cnt=5, group=sched_group
        ]()
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        _s_setprio[Int16(0)]()
        return pending_scale

    @staticmethod
    @always_inline
    def _pv_whole_with_partial_softmax[
        sched_group: Int,
    ](
        v_reg: RegTile[DType.bfloat16, Self._V_LAYOUT_T, MutExternalOrigin],
        att_bf16_full: RegTile[
            DType.bfloat16, Self._ATT_BF16_FULL_LAYOUT_T, MutExternalOrigin
        ],
        mut o_reg: RegTile[DType.float32, Self._O_LAYOUT_T, MutExternalOrigin],
        mut max_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut max_vec_prev: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut scale_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut att_block_qk: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
    ):
        """Epilogue C2/C6 body: whole-V PV then UNCONDITIONAL rescale +
        partial softmax. Unlike `_pv_strip_with_partial_softmax`, the
        rescale fires AFTER all PV MFMAs so there is no strip-vs-rescale
        inconsistency — all of V's contribution lands at the old scale
        before `o_reg` is rescaled to the new one."""
        s_waitcnt[lgkmcnt=UInt32(0)]()
        _s_setprio[Int16(1)]()
        comptime for i in range(Self._NUM_PV_SUBTILES):
            var v_sub = v_reg.tile[1, Self.DEPTH // 32, 8](i, 0, 0)
            var att_sub = att_bf16_full.tile[1, 1, 8](i, 0, 0)
            Self._MmaOp.mma_PV(o_reg, v_sub, att_sub)
        col_max_acc(max_vec, att_block_qk, max_vec_prev)
        scale_vec[0, 0] = math_exp2(max_vec_prev[0, 0] - max_vec[0, 0])
        max_vec_prev.copy_from(max_vec)
        sub_col_inplace(att_block_qk, max_vec)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block_qk)
        # IGLP: interleave PV MFMAs with sub_col + mul_col (VALU) and
        # first-half exp2 (TRANS).
        sched_barrier_pairs[10, valu_cnt=5, group=sched_group]()
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        mul_col_inplace(o_reg, scale_vec)
        _s_setprio[Int16(0)]()

    @staticmethod
    @always_inline
    def _maybe_apply_mask[
        mask_t: MHAMask,
        layout: TensorLayout,
        //,
    ](
        mask_functor: mask_t,
        mut att_block: RegTile[DType.float32, layout, MutExternalOrigin],
        q_tile_idx: Int,
        k_tile_idx: Int,
        start_pos: Int,
        head_idx: UInt32,
        batch_idx: UInt32,
        l_id: Int,
    ):
        """Comptime dispatch wrapper around the mask application.

        - `CausalMask`: fast SIMD path. `q_start_pos < kv_end_pos`
          runtime check, then `_apply_causal_mask_fast`. The
          `start_pos` shift is folded into the q position.
        - `NullMask`: comptime no-op (the status would always be
          `NO_MASK` and the branch is unconditionally elided).
        - Any other `MHAMask`: generic path via
          `mask_functor.status(...)` + `apply_mask_to_att_block`. The
          runtime `status()` call + enum branching adds a few SGPR
          ops per tile; acceptable for the less-common masks where
          there is no comptime-specialized fast path.
        """
        comptime if _type_is_eq[mask_t, NullMask]():
            return
        elif _type_is_eq[mask_t, CausalMask]():
            var q_start_pos = q_tile_idx * Self.Q_BLOCK_SIZE + start_pos
            var kv_end_pos = (k_tile_idx + 1) * Self.KV_BLOCK
            if unlikely(q_start_pos < kv_end_pos):
                _apply_causal_mask_fast[
                    Q_BLOCK_SIZE=Self.Q_BLOCK_SIZE,
                    KV_BLOCK_SIZE=Self.KV_BLOCK,
                ](
                    att_block,
                    Int32(q_tile_idx),
                    Int32(k_tile_idx),
                    Int32(start_pos),
                    Int32(l_id),
                )
        else:
            var status = mask_functor.status(
                IndexList[2, element_type=DType.uint32](
                    Int(q_tile_idx * Self.Q_BLOCK_SIZE + start_pos),
                    Int(k_tile_idx * Self.KV_BLOCK),
                ),
                IndexList[2, element_type=DType.uint32](
                    Int(Self.Q_BLOCK_SIZE), Int(Self.KV_BLOCK)
                ),
            )
            if status != TileMaskStatus.NO_MASK:
                apply_mask_to_att_block[
                    Q_BLOCK_SIZE=Self.Q_BLOCK_SIZE,
                    KV_BLOCK_SIZE=Self.KV_BLOCK,
                ](
                    mask_functor,
                    att_block,
                    Int32(q_tile_idx),
                    Int32(k_tile_idx),
                    Int32(start_pos),
                    head_idx,
                    batch_idx,
                    Int32(l_id),
                    status,
                )

    @staticmethod
    @always_inline
    def _store_o_to_gmem[
        output_dtype: DType,
        epilogue_chunk_width: Int = 1,
    ](
        o_reg_t: RegTile[DType.float32, Self._O_T_LAYOUT_T, MutExternalOrigin],
        epilogue_writer: RegTileEpilogue[output_dtype, epilogue_chunk_width],
        l_id: Int,
    ):
        """Writes the FP32 row_l rt_32x32 accumulator to gmem via
        `RegTileEpilogue`, casting per-lane to `output_dtype`.

        The per-lane → (q_in_tile, output_col) mapping follows the
        rt_32x32 row_l fragment topology: lanes `[0, 32)` and `[32, 64)`
        own the two halves of the depth, offset by 4 to interleave.
        Since the col_l → row_l transpose is a zero-cost re-tag, the
        FP32-path stored values are bit-identical to the col_l
        accumulator; the BF16-path cast is a per-lane `v_cvt_pkrtz`
        (or scalar `v_cvt`) emitted just before the gmem store."""
        var q_in_tile = l_id & 31
        var d_extra = 4 if l_id >= 32 else 0

        comptime _D_FRAG = (Self.DEPTH * Self.Q_BLOCK_SIZE) // 64
        comptime for k_local in range(_D_FRAG):
            comptime i = k_local // 16
            comptime k_in_base = k_local % 16
            comptime d_within_4 = (k_in_base // 4) * 8 + (k_in_base % 4)
            var output_col = i * 32 + d_within_4 + d_extra
            var v_fp32 = SIMD[DType.float32, 1](o_reg_t.ptr[k_local])
            comptime if output_dtype == DType.float32:
                epilogue_writer.store(
                    rebind[SIMD[output_dtype, 1]](v_fp32),
                    m=q_in_tile,
                    n=output_col,
                )
            else:
                epilogue_writer.store(
                    v_fp32.cast[output_dtype](),
                    m=q_in_tile,
                    n=output_col,
                )

    @staticmethod
    @always_inline
    def _tail_softmax_unconditional[
        sched_group: Int,
    ](
        mut att_block: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
        mut norm_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut scale_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
    ):
        """Epilogue tail softmax: second-half `exp2` + UNCONDITIONAL
        `norm_vec *= scale_vec` + `col_sum_acc`. No BF16 cast — the
        consumer PV JIT-casts `att_block` per subtile inline."""
        Self._MmaOp.exp2_inplace_range[Self._ATT_HALF, Self._ATT_PER_LANE](
            att_block
        )
        norm_vec[0, 0] = norm_vec[0, 0] * scale_vec[0, 0]
        var norm_pre2 = reg_alloc[DType.float32](row_major[1, 1]())
        norm_pre2.copy_from(norm_vec)
        col_sum_acc(norm_vec, att_block, norm_pre2)
        # IGLP: interleave QK MFMAs with second-half exp2 (TRANS) and
        # col_sum_acc (VALU).
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        sched_barrier_pairs[10, valu_cnt=5, group=sched_group]()

    @staticmethod
    @always_inline
    def _qk_tail_softmax_cluster[
        sched_group: Int,
    ](
        mut att_block_qk: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
        mut att_block_softmax: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
        mut att_block_bf16: RegTile[
            DType.bfloat16, Self._ATT_BF16_FULL_LAYOUT_T, MutExternalOrigin
        ],
        mut k_reg: RegTile[DType.bfloat16, Self._K_LAYOUT_T, MutExternalOrigin],
        mut q_reg: RegTile[DType.bfloat16, Self._Q_LAYOUT_T, MutExternalOrigin],
        mut norm_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut scale_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        pending_scale: Bool,
    ):
        """Main-loop C0/C4 body: QK MFMA (consuming pre-loaded `k_reg`)
        + tail softmax of the previous tile + bulk BF16 cast into the
        persistent `att_block_bf16` for the next PV cluster.

        The `exp2` → optional rescale → `col_sum_acc` → cast order and
        the `norm_pre` register stash are tuned so the cast `v_cvt`s
        sit at the very end of the cluster and overlap the next
        barrier + DMA."""
        Self._qk_with_kreg(att_block_qk, k_reg, q_reg)
        Self._MmaOp.exp2_inplace_range[Self._ATT_HALF, Self._ATT_PER_LANE](
            att_block_softmax
        )
        if pending_scale:
            var norm_pre = reg_alloc[DType.float32](row_major[1, 1]())
            norm_pre.copy_from(norm_vec)
            norm_vec[0, 0] = norm_pre[0, 0] * scale_vec[0, 0]
        var norm_pre2 = reg_alloc[DType.float32](row_major[1, 1]())
        norm_pre2.copy_from(norm_vec)
        col_sum_acc(norm_vec, att_block_softmax, norm_pre2)
        Self._att_bf16_full(att_block_bf16, att_block_softmax)
        # IGLP: interleave QK MFMAs with second-half exp2 (TRANS) and
        # col_sum_acc + bulk BF16 cast `v_cvt`s (VALU).
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        sched_barrier_pairs[10, valu_cnt=5, group=sched_group]()

    @staticmethod
    @always_inline
    def _full_softmax_unconditional[
        sched_group: Int,
    ](
        mut att_block: RegTile[
            DType.float32, Self._ATT_LAYOUT_T, MutExternalOrigin
        ],
        mut max_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut max_vec_prev: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut scale_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
        mut norm_vec: RegTile[
            DType.float32, Self._RV_LAYOUT_T, MutExternalOrigin
        ],
    ):
        """Epilogue full softmax: both halves of `exp2` + UNCONDITIONAL
        norm rescale + `col_sum`. No cast — the consumer `_pv_whole`
        reuses an already-staged `att_block_bf16`."""
        col_max_acc(max_vec, att_block, max_vec_prev)
        scale_vec[0, 0] = math_exp2(max_vec_prev[0, 0] - max_vec[0, 0])
        max_vec_prev.copy_from(max_vec)
        sub_col_inplace(att_block, max_vec)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block)
        # IGLP: interleave PV-whole MFMAs with sub_col + mul_col (VALU)
        # and first-half exp2 (TRANS).
        sched_barrier_pairs[10, valu_cnt=5, group=sched_group]()
        sched_barrier_exp_pairs[6, exp_cnt=3, group=sched_group]()
        Self._MmaOp.exp2_inplace_range[Self._ATT_HALF, Self._ATT_PER_LANE](
            att_block
        )
        norm_vec[0, 0] = norm_vec[0, 0] * scale_vec[0, 0]
        var norm_pre2 = reg_alloc[DType.float32](row_major[1, 1]())
        norm_pre2.copy_from(norm_vec)
        col_sum_acc(norm_vec, att_block, norm_pre2)

    # `amdgpu-waves-per-eu` accepts a `"MIN,MAX"` string. The
    # `llvm.<name>` namespace in `@__llvm_metadata` routes through
    # `LowerKGENToLLVM.cpp`'s LLVM-passthrough path (drop `llvm.`
    # prefix, build an `(attr_name, attr_value)` ArrayAttr entry on
    # `llvm.func`'s `passthrough` slot). Lands on the LLVM function
    # without any MLIR ROCDL-dialect or LLVM-tarball patches.
    @__llvm_metadata(`llvm.amdgpu-waves-per-eu`=__mlir_attr.`"2,2"`)
    # The Modular-local LLVM patch at
    # `bazel/third-party/llvm-agpr-alloc-respect-author.patch` exposes a
    # `rocdl.agpr_alloc_min_required` decorator opting a function into
    # AGPR-form MFMA codegen. We intentionally omit it here: gfx950 has
    # no AGPR-side ALU, so online softmax's `o_reg *= exp2(...)` round-
    # trips every accumulator value through VGPRs (~25x regression
    # at MQA seq=8192 with MIN==MAX==64).
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.NUM_THREADS)
        )
    )
    @staticmethod
    def run[
        k_t: MHAOperand,
        v_t: MHAOperand,
        mask_t: MHAMask,
        q_dtype: DType,
        output_dtype: DType,
        q_layout: TensorLayout,
        o_layout: TensorLayout,
    ](
        q: TileTensor[q_dtype, q_layout, ImmutAnyOrigin],
        k: k_t,
        v: v_t,
        o: TileTensor[output_dtype, o_layout, MutAnyOrigin],
        mask_functor: mask_t,
        scale: Float32,
        num_keys: Int,
        start_pos: Int,
    ):
        """Multi-block 8-warp MHA forward (inference-only).

        Grid: `(NUM_HEADS, ceildiv(seq_len, BM), batch)`. Each block
        owns one `(batch, head, BM-tile)` slice; the 8 warps within
        split the BM-tile's Q rows.

        Expected layouts / shapes:

        - `q`, `o`: `(batch, seq_len, NUM_HEADS, DEPTH)` row-major
          TileTensor. `o`'s dtype matches `config.output_dtype` — BF16
          for the production dispatcher (which holds a BF16 output
          buffer) or FP32 if the caller wants the unnormalized
          accumulator.
        - `k`, `v`: any `MHAOperand` whose `block_paged_tile[KV_BLOCK]`
          returns `(KV_BLOCK, DEPTH)` tiles per `(batch, t*KV_BLOCK,
          kv_head, 0)`. `LayoutTensorMHAOperand` for contiguous test /
          bench buffers; `KVCacheMHAOperand` for paged production
          caches (`page_size >= KV_BLOCK = 64`).

        `batch` and `seq_len` / `num_keys` may be dynamic;
        `NUM_HEADS`, `NUM_KV_HEADS`, `DEPTH` must be static.
        `NUM_HEADS` must be a multiple of `NUM_KV_HEADS`
        (GROUP = `NUM_HEADS // NUM_KV_HEADS`).

        Args:
            q: Q tile tensor.
            k: K operand (MHAOperand).
            v: V operand (MHAOperand).
            o: Output tile tensor (`config.output_dtype`, same shape
                as `q`).
            mask_functor: Per-tile mask predicate (causal, sliding-window,
                etc.). Evaluated inside the QK→softmax cluster; identity
                for unmasked attention.
            scale: Softmax scale (typically `1 / sqrt(DEPTH)`).
            num_keys: Runtime length of the K/V sequence.
            start_pos: Position of the first Q row in the global sequence
                — non-zero for prefill chunks of a longer generation.
                Used by the mask functor to compute the causal cutoff.
        """
        # `q_dtype` and `output_dtype` are independent comptime type
        # parameters so callers can pass TileTensors with literal dtypes
        # (e.g. `DType.float32` from `enqueue_create_buffer[DType.float32]`)
        # without Mojo failing to unify a literal with a generic
        # parameter (`Self.config.output_dtype`, `DType.bfloat16`) even
        # when they evaluate equal. The asserts pin the contract: q is
        # BF16, o matches `config.output_dtype`. The downstream rebind
        # is a no-op identity given the assert.
        comptime assert (
            q_dtype == DType.bfloat16
        ), "HKMhaPrefill.run: `q.dtype` must be `DType.bfloat16`"
        comptime assert (
            output_dtype == Self.config.output_dtype
        ), "HKMhaPrefill.run: `o.dtype` must equal `config.output_dtype`"
        var q_bf16 = rebind[
            TileTensor[DType.bfloat16, q_layout, ImmutAnyOrigin]
        ](q)
        var seq_len = Int(q.dim[1]())
        var num_tiles = (num_keys + Self.KV_BLOCK - 1) // Self.KV_BLOCK
        comptime assert Self.NUM_HEADS % Self.NUM_KV_HEADS == 0, (
            "HKMhaPrefill: NUM_HEADS must be a multiple of NUM_KV_HEADS"
            " (GROUP = NUM_HEADS // NUM_KV_HEADS)"
        )
        # K/V SMEM byte layout: each slot is a 2D `(KV_BLOCK *
        # NUM_BLOCK_COLS, SUB_COLS)` row-major tile holding `SUB_ROWS x
        # SUB_COLS` sub-blocks linearized as
        # `subtile_id = block_row * (DEPTH / SUB_COLS) + block_col`.
        # `_dma_k` and `MhaMmaOp.load_V` both rely on this layout to
        # collapse subtile slicing to `tile[SUB_ROWS, SUB_COLS](id, 0)`.
        comptime _K_SUB_ROWS = Self._MmaOp.K_SUB_ROWS
        comptime _K_SUB_COLS = Self._MmaOp.K_SUB_COLS
        comptime _V_SUB_ROWS = Self._MmaOp.V_SUB_ROWS
        comptime _V_SUB_COLS = Self._MmaOp.V_SUB_COLS
        comptime _NUM_BLOCK_COLS_K = Self.DEPTH // _K_SUB_COLS
        comptime _NUM_BLOCK_COLS_V = Self.DEPTH // _V_SUB_COLS
        comptime _K_SLOT_ROWS = Self.KV_BLOCK * _NUM_BLOCK_COLS_K
        comptime _V_SLOT_ROWS = Self.KV_BLOCK * _NUM_BLOCK_COLS_V
        comptime smem_layout_k = row_major[_K_SLOT_ROWS, _K_SUB_COLS]()
        comptime smem_layout_v = row_major[_V_SLOT_ROWS, _V_SUB_COLS]()

        # Four independent `smem_alloc` calls (one per double-buffer
        # slot). Letting the LDS allocator place each slot separately
        # yields wave-uniform base pointers that `readfirstlane` to
        # SGPR; the comptime `k_smem[0]` / `k_smem[1]` subscripts then
        # resolve to a fixed SGPR operand per call site. Consolidating
        # into one allocation + runtime `.tile[](stage, 0)` indexing
        # defeats that scalarization and regresses ~5-10%.
        #
        # Per-slot alignment = one sub-block. For K (BF16 32x32 = 2 KiB)
        # this keeps the two-XOR swizzle `bit5 ^= bit9; bit4 ^= bit10`
        # invariant across sub-blocks within a slot; V needs only the
        # 16-B `ds_read_b128` alignment.
        comptime _K_ALIGN = (
            _K_SUB_ROWS * _K_SUB_COLS * size_of[DType.bfloat16]()
        )
        comptime _V_ALIGN = (
            _V_SUB_ROWS * _V_SUB_COLS * size_of[DType.bfloat16]()
        )
        var k_smem_0_tt = smem_alloc[DType.bfloat16, alignment=_K_ALIGN](
            smem_layout_k
        )
        var k_smem_1_tt = smem_alloc[DType.bfloat16, alignment=_K_ALIGN](
            smem_layout_k
        )
        var v_smem_0_tt = smem_alloc[DType.bfloat16, alignment=_V_ALIGN](
            smem_layout_v
        )
        var v_smem_1_tt = smem_alloc[DType.bfloat16, alignment=_V_ALIGN](
            smem_layout_v
        )

        var k_smem = Tuple(k_smem_0_tt, k_smem_1_tt)
        var v_smem = Tuple(v_smem_0_tt, v_smem_1_tt)

        var w_id = Int(readfirstlane(warp_id()))
        var l_id = Int(lane_id())

        # GQA-aware head index: permutes block_idx.x ∈ [0, NUM_HEADS)
        # into head_idx ∈ [0, NUM_HEADS) so blocks visiting the same
        # KV head are spaced GROUP apart in launch order, spreading KV
        # bandwidth across CUs/XCDs. View block_x as a row-major index
        # into a (NUM_KV_HEADS, GROUP) rectangle (`bx_div ∈ [0,
        # NUM_KV_HEADS)` indexes the KV head, `bx_mod ∈ [0, GROUP)`
        # indexes the Q within that group); `head_idx = bx_mod *
        # NUM_KV_HEADS + bx_div` transposes to column-major over the
        # same rectangle. The map is a bijection iff NUM_HEADS ==
        # GROUP * NUM_KV_HEADS, which the divisibility assert above
        # enforces. Reduces to identity at MHA (GROUP=1) and at MQA
        # (NUM_KV_HEADS=1), keeping `divmod(_, 1)` constant-foldable
        # for byte-identical asm on those paths.
        var block_x = Int(readfirstlane(Int32(block_idx.x)))
        comptime _GROUP = Self.NUM_HEADS // Self.NUM_KV_HEADS
        var bx_div, bx_mod = divmod(block_x, _GROUP)
        var head_idx = bx_mod * Self.NUM_KV_HEADS + bx_div
        var block_tile_idx = Int(readfirstlane(Int32(block_idx.y)))
        var batch_idx = Int(readfirstlane(Int32(block_idx.z)))
        var kv_head_idx = head_idx // _GROUP
        var tile_idx = block_tile_idx * Self.NUM_WARPS + w_id

        # Lower-half warps `[0..NUM_WARPS/2)` vs upper-half — controls
        # the prologue/conclusion stagger barrier.
        var stagger = w_id >= (Self.NUM_WARPS // 2)

        # Causal cap on `max_num_tiles`: skip tiles whose K range is
        # entirely past the block's last Q row (every entry would be
        # masked anyway).
        var max_tile_idx_local = (
            block_tile_idx * Self.NUM_WARPS + Self.NUM_WARPS - 1
        )
        var max_q_end_pos = (
            max_tile_idx_local + 1
        ) * Self.Q_BLOCK_SIZE + start_pos
        var max_num_tiles_calc = (
            max_q_end_pos + Self.KV_BLOCK - 1
        ) // Self.KV_BLOCK
        var max_num_tiles_local: Int
        # FULL_MASK skip at the loop boundary for `CausalMask` (the
        # common case). For an arbitrary `MHAMask` we can't statically
        # bound the K iteration count, so we iterate to `num_tiles` and
        # rely on the per-tile `mask_functor.status(...)` check at each
        # mask site to detect a fully-masked tile and zero `att_block`.
        # `NullMask` keeps the full iteration count and `status` always
        # returns `NO_MASK`, so per-tile mask work is comptime-elided.
        comptime if _type_is_eq[mask_t, CausalMask]():
            max_num_tiles_local = (
                max_num_tiles_calc if max_num_tiles_calc
                < num_tiles else num_tiles
            )
        else:
            max_num_tiles_local = num_tiles

        # Per-(batch, head) 2D views via `.tile(Coord shape, Coord
        # indices).reshape(2D layout)`. The 4D `.tile` selects the
        # singleton `(1, seq_len, 1, DEPTH)` sub-tensor at
        # `(batch_idx, 0, head_idx, 0)`; `.reshape` retags it as a 2D
        # MixedLayout that the per-warp `.tile[Q_BLOCK_SIZE, DEPTH]()`
        # downstream and the DMA loaders expect.
        var q_2d = q_bf16.tile(
            Coord(
                Idx[1],
                Int32(seq_len),
                Idx[1],
                Idx[Self.DEPTH],
            ),
            Coord(batch_idx, Idx[0], head_idx, Idx[0]),
        ).reshape(
            Self._QPerHeadLayoutT(
                Coord(
                    Int32(seq_len),
                    Idx[Self.DEPTH],
                ),
                Coord(Idx[Self._Q_ROW_STRIDE], Idx[1]),
            )
        )
        var o_2d = o.tile(
            Coord(
                Idx[1],
                Int32(seq_len),
                Idx[1],
                Idx[Self.DEPTH],
            ),
            Coord(batch_idx, Idx[0], head_idx, Idx[0]),
        ).reshape(
            Self._QPerHeadLayoutT(
                Coord(
                    Int32(seq_len),
                    Idx[Self.DEPTH],
                ),
                Coord(Idx[Self._Q_ROW_STRIDE], Idx[1]),
            )
        )
        var q_warp_block_idx = block_tile_idx * Self.NUM_WARPS + w_id
        var q_warp_2d = q_2d.tile[Self.Q_BLOCK_SIZE, Self.DEPTH](
            q_warp_block_idx, 0
        )
        var o_warp_2d = o_2d.tile[Self.Q_BLOCK_SIZE, Self.DEPTH](
            q_warp_block_idx, 0
        )

        # Per-(batch, kv_head) is captured via the MHAOperand interface.
        # `_dma_k` / `_dma_v` call `k.block_paged_tile[KV_BLOCK](...)` per
        # tile, returning a TileTensor anchored at `(batch_idx,
        # t*KV_BLOCK, kv_head_idx, 0)`. For LayoutTensorMHAOperand this
        # resolves to a pointer-arithmetic offset (contiguous K/V);
        # for KVCacheMHAOperand it resolves through the page table.
        # Cast indices to UInt32 once so the trait calls don't repeat
        # the conversion.
        var _batch_idx_u32 = UInt32(batch_idx)
        var _kv_head_idx_u32 = UInt32(kv_head_idx)

        # Q load + prescale by `scale * log2(e)`; multiply in FP32
        # per fragment to preserve ~1 ULP, cast back to BF16.
        var scale_log2e = scale * 1.4426950408889634
        var q_reg = Self._load_q_and_scale(q_warp_2d, scale_log2e)

        # Persistent kernel-scope state. `scale_vec` initialized to ones
        # so the epilogue's unconditional `norm_vec *= scale_vec` is a
        # safe no-op when no rescale ever fired.
        var o_reg = reg_alloc[DType.float32](Self._MmaOp.O_LAYOUT)
        var max_vec = reg_alloc[DType.float32](row_major[1, 1]())
        var max_vec_prev = reg_alloc[DType.float32](row_major[1, 1]())
        var norm_vec = reg_alloc[DType.float32](row_major[1, 1]())
        var scale_vec = reg_alloc[DType.float32](row_major[1, 1]())
        _ = o_reg.fill(0)
        _ = norm_vec.fill(0)
        _ = scale_vec.fill(1)
        _ = max_vec.fill(0)
        _ = max_vec_prev.fill(0)

        # Two FP32 `att_block` slots; the loop ping-pongs which one is
        # the QK destination and which the softmax source.
        var att_block_0 = reg_alloc[DType.float32](Self._MmaOp.ATT_LAYOUT)
        var att_block_1 = reg_alloc[DType.float32](Self._MmaOp.ATT_LAYOUT)

        # Persistent BF16 P-cache shared across all six `_att_bf16_full`
        # producers and their PV consumers. The persistent destination
        # keeps the cast from rematerializing at each PV use site (KB
        # `known-limitations/llvm-amdgpu-cast-rematerialization`).
        var att_block_bf16 = reg_alloc[DType.bfloat16](
            Self._MmaOp.ATT_BF16_FULL_LAYOUT
        )

        # Persistent `k_reg`: K is loaded once per K-tile in a dedicated
        # cluster (C3/C7) and consumed in the next QK cluster (C0/C4)
        # without any in-cluster `ds_read`s.
        var k_reg = reg_alloc[DType.bfloat16](Self._MmaOp.K_LAYOUT)

        # === Prologue ===
        # K[0] DMA, full drain, barrier.
        Self._dma_k(
            k_smem[0],
            k,
            _batch_idx_u32,
            _kv_head_idx_u32,
            0,
            w_id,
            l_id,
        )
        s_waitcnt[vmcnt=UInt32(0), lgkmcnt=UInt32(0)]()
        _sched_barrier_zero()
        _s_barrier_raw()

        # K[1] and V[0] DMAs.
        if num_tiles > 1:
            Self._dma_k(
                k_smem[1],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                1,
                w_id,
                l_id,
            )
        Self._dma_v(
            v_smem[0],
            v,
            _batch_idx_u32,
            _kv_head_idx_u32,
            0,
            w_id,
            l_id,
        )
        _sched_barrier_zero()
        _s_barrier_raw()

        Self._load_k_reg(k_reg, k_smem[0])

        # This `lgkmcnt(0)` is semantically redundant after the barrier
        # above, but removing it costs ~1.1% — likely acting as a
        # scheduler-region anchor that helps RA. Keep until understood.
        s_waitcnt[lgkmcnt=UInt32(0)]()

        # QK[0] over pre-loaded `k_reg`.
        Self._qk_with_kreg(att_block_0, k_reg, q_reg)
        _sched_barrier_zero()

        # Prologue tile 0 mask.
        Self._maybe_apply_mask(
            mask_functor,
            att_block_0,
            tile_idx,
            0,
            start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
        )

        # Tile-0 partial softmax (no rescale: first tile).
        col_max(max_vec, att_block_0)
        max_vec_prev.copy_from(max_vec)
        sub_col_inplace(att_block_0, max_vec)
        Self._MmaOp.exp2_inplace_range[0, Self._ATT_HALF](att_block_0)
        _sched_barrier_zero()

        # Stagger barrier — upper-half warps wait.
        if stagger:
            _sched_barrier_zero()
            _s_barrier_raw()

        # Pre-load K[1] ahead of main loop's first C0.
        if num_tiles > 1:
            Self._load_k_reg(k_reg, k_smem[1])

        # K[2], V[1] DMAs.
        if num_tiles > 2:
            Self._dma_k(
                k_smem[0],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                2,
                w_id,
                l_id,
            )
        if num_tiles > 1:
            Self._dma_v(
                v_smem[1],
                v,
                _batch_idx_u32,
                _kv_head_idx_u32,
                1,
                w_id,
                l_id,
            )
        _sched_barrier_zero()
        _s_barrier_raw()

        var pending_scale: Bool = False

        # === Main loop ===
        # 8 clusters per iteration, advancing `j` by 2 each pass.
        var j: Int = 3
        while j < max_num_tiles_local - 1:
            # C0: QK[j-2] + tail softmax of tile (j-3); bulk BF16 cast
            # into `att_block_bf16` happens inside the helper, pre-cast
            # for C2's PV.
            _asm_label["; HKMHA_MAIN_C0_BEGIN"]()
            Self._qk_tail_softmax_cluster[
                sched_group=Self._SCHED_MAIN_C0_QK_TAIL
            ](
                att_block_1,
                att_block_0,
                att_block_bf16,
                k_reg,
                q_reg,
                norm_vec,
                scale_vec,
                pending_scale,
            )
            _cluster_barrier()

            # C1: DMA K[j], load v_reg = V[j-3].
            _asm_label["; HKMHA_MAIN_C1_BEGIN"]()
            Self._dma_k(
                k_smem[1],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j,
                w_id,
                l_id,
            )
            var v_reg_c1 = reg_alloc[DType.bfloat16](Self._MmaOp.V_LAYOUT)
            Self._load_v_reg(v_reg_c1, v_smem[0])
            _cluster_barrier()

            # C2: PV[j-3] strip-interleaved with partial softmax of
            # tile (j-2), consuming `att_block_bf16` pre-cast in C0.
            _asm_label["; HKMHA_MAIN_C2_BEGIN"]()
            pending_scale = Self._pv_strip_with_partial_softmax[
                sched_group=Self._SCHED_MAIN_C2_PV_PARTIAL
            ](
                v_reg_c1,
                att_block_bf16,
                o_reg,
                max_vec,
                max_vec_prev,
                scale_vec,
                att_block_1,
            )
            _cluster_barrier()

            # C3: DMA V[j-1], pre-load k_reg = K[j-1] for the next C4.
            _asm_label["; HKMHA_MAIN_C3_BEGIN"]()
            Self._dma_v(
                v_smem[0],
                v,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j - 1,
                w_id,
                l_id,
            )
            Self._load_k_reg(k_reg, k_smem[0])
            _cluster_barrier()

            # C4: QK[j-1] + tail softmax of tile (j-2); bulk cast into
            # `att_block_bf16` pre-stages C6's PV-A.
            _asm_label["; HKMHA_MAIN_C4_BEGIN"]()
            Self._qk_tail_softmax_cluster[
                sched_group=Self._SCHED_MAIN_C4_QK_TAIL
            ](
                att_block_0,
                att_block_1,
                att_block_bf16,
                k_reg,
                q_reg,
                norm_vec,
                scale_vec,
                pending_scale,
            )
            _cluster_barrier()

            # C5: DMA K[j+1], load v_reg = V[j-2], causal-mask
            # att_block_0. No MFMAs — IGLP interleaves the 32 V `ds_read`s
            # with the causal-mask `v_cmp` / `v_cndmask` VALU pairs.
            _asm_label["; HKMHA_MAIN_C5_BEGIN"]()
            Self._dma_k(
                k_smem[0],
                k,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j + 1,
                w_id,
                l_id,
            )
            var v_reg_c5 = reg_alloc[DType.bfloat16](Self._MmaOp.V_LAYOUT)
            Self._load_v_reg(v_reg_c5, v_smem[1])
            Self._maybe_apply_mask(
                mask_functor,
                att_block_0,
                tile_idx,
                j - 1,
                start_pos,
                UInt32(head_idx),
                _batch_idx_u32,
                l_id,
            )
            sched_dsread_valu_pairs[
                32, valu_cnt=1, group=Self._SCHED_MAIN_C5_DSREAD
            ]()
            _cluster_barrier()

            # C6: PV[j-2] strip-interleaved with partial softmax of
            # tile (j-1), consuming `att_block_bf16` pre-cast in C4.
            _asm_label["; HKMHA_MAIN_C6_BEGIN"]()
            pending_scale = Self._pv_strip_with_partial_softmax[
                sched_group=Self._SCHED_MAIN_C6_PV_PARTIAL
            ](
                v_reg_c5,
                att_block_bf16,
                o_reg,
                max_vec,
                max_vec_prev,
                scale_vec,
                att_block_0,
            )
            _cluster_barrier()

            # C7: DMA V[j], pre-load k_reg = K[j] for the next iter's C0.
            _asm_label["; HKMHA_MAIN_C7_BEGIN"]()
            Self._dma_v(
                v_smem[1],
                v,
                _batch_idx_u32,
                _kv_head_idx_u32,
                j,
                w_id,
                l_id,
            )
            Self._load_k_reg(k_reg, k_smem[1])
            _cluster_barrier()

            j += 2

        # === Epilogue ===
        # 13 clusters draining the final 4 tiles `N-4..N-1`; assumes
        # `num_tiles >= 4`.
        var N = max_num_tiles_local

        # Epi-C0: QK[N-3] + tail softmax of tile (N-4). Tail rescale
        # is UNCONDITIONAL; `scale_vec` initialized to ones makes it
        # an identity when no rescale ever fired.
        _asm_label["; HKMHA_EPI_C0_BEGIN"]()
        Self._qk_with_kreg(att_block_1, k_reg, q_reg)
        Self._tail_softmax_unconditional[sched_group=Self._SCHED_EPI_C0_TAIL](
            att_block_0,
            norm_vec,
            scale_vec,
        )
        # Pre-cast for Epi-C2's PV.
        Self._att_bf16_full(att_block_bf16, att_block_0)
        _cluster_barrier()

        # Epi-C1: DMA K[N-1], load v_reg = V[N-4], causal-mask
        # att_block_1 for tile (N-3).
        _asm_label["; HKMHA_EPI_C1_BEGIN"]()
        Self._dma_k(
            k_smem[1],
            k,
            _batch_idx_u32,
            _kv_head_idx_u32,
            N - 1,
            w_id,
            l_id,
        )
        var v_reg_e1 = reg_alloc[DType.bfloat16](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e1, v_smem[0])
        Self._maybe_apply_mask(
            mask_functor,
            att_block_1,
            tile_idx,
            N - 3,
            start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
        )
        _cluster_barrier()

        # Epi-C2: PV[N-4] (whole) + partial softmax of tile (N-3),
        # consuming `att_block_bf16` pre-cast in Epi-C0.
        _asm_label["; HKMHA_EPI_C2_BEGIN"]()
        Self._pv_whole_with_partial_softmax[
            sched_group=Self._SCHED_EPI_C2_PV_PARTIAL
        ](
            v_reg_e1,
            att_block_bf16,
            o_reg,
            max_vec,
            max_vec_prev,
            scale_vec,
            att_block_1,
        )
        _cluster_barrier()

        # Epi-C3: DMA V[N-2], pre-load k_reg = K[N-2].
        _asm_label["; HKMHA_EPI_C3_BEGIN"]()
        Self._dma_v(
            v_smem[0],
            v,
            _batch_idx_u32,
            _kv_head_idx_u32,
            N - 2,
            w_id,
            l_id,
        )
        Self._load_k_reg(k_reg, k_smem[0])
        _cluster_barrier()

        # Epi-C4: QK[N-2] + tail softmax of tile (N-3); pre-cast for
        # Epi-C6's PV.
        _asm_label["; HKMHA_EPI_C4_BEGIN"]()
        Self._qk_with_kreg(att_block_0, k_reg, q_reg)
        Self._tail_softmax_unconditional[sched_group=Self._SCHED_EPI_C4_TAIL](
            att_block_1,
            norm_vec,
            scale_vec,
        )
        Self._att_bf16_full(att_block_bf16, att_block_1)
        _cluster_barrier()

        # Epi-C5: load v_reg = V[N-3], causal-mask att_block_0.
        _asm_label["; HKMHA_EPI_C5_BEGIN"]()
        var v_reg_e5 = reg_alloc[DType.bfloat16](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e5, v_smem[1])
        Self._maybe_apply_mask(
            mask_functor,
            att_block_0,
            tile_idx,
            N - 2,
            start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
        )
        sched_dsread_valu_pairs[
            32, valu_cnt=1, group=Self._SCHED_EPI_C5_DSREAD
        ]()
        _cluster_barrier()

        # Epi-C6: PV[N-3] (whole) + partial softmax of tile (N-2),
        # consuming `att_block_bf16` pre-cast in Epi-C4.
        _asm_label["; HKMHA_EPI_C6_BEGIN"]()
        Self._pv_whole_with_partial_softmax[
            sched_group=Self._SCHED_EPI_C6_PV_PARTIAL
        ](
            v_reg_e5,
            att_block_bf16,
            o_reg,
            max_vec,
            max_vec_prev,
            scale_vec,
            att_block_0,
        )
        _cluster_barrier()

        # Epi-C7: DMA V[N-1], pre-load k_reg = K[N-1].
        _asm_label["; HKMHA_EPI_C7_BEGIN"]()
        Self._dma_v(
            v_smem[1],
            v,
            _batch_idx_u32,
            _kv_head_idx_u32,
            N - 1,
            w_id,
            l_id,
        )
        Self._load_k_reg(k_reg, k_smem[1])
        _cluster_barrier()

        # Epi-C8: QK[N-1] + tail softmax of tile (N-2); pre-cast for
        # Epi-C10's PV.
        _asm_label["; HKMHA_EPI_C8_BEGIN"]()
        Self._qk_with_kreg(att_block_1, k_reg, q_reg)
        Self._tail_softmax_unconditional[sched_group=Self._SCHED_EPI_C8_TAIL](
            att_block_0,
            norm_vec,
            scale_vec,
        )
        Self._att_bf16_full(att_block_bf16, att_block_0)
        _cluster_barrier()

        # Epi-C9: load v_reg = V[N-2], causal-mask att_block_1.
        _asm_label["; HKMHA_EPI_C9_BEGIN"]()
        var v_reg_e9 = reg_alloc[DType.bfloat16](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e9, v_smem[0])
        Self._maybe_apply_mask(
            mask_functor,
            att_block_1,
            tile_idx,
            N - 1,
            start_pos,
            UInt32(head_idx),
            _batch_idx_u32,
            l_id,
        )
        sched_dsread_valu_pairs[
            32, valu_cnt=1, group=Self._SCHED_EPI_C9_DSREAD
        ]()
        _cluster_barrier()

        # Epi-C10: PV[N-2] (whole) + FULL softmax of tile (N-1) +
        # final `o_reg *= scale_vec`. The PV-A read happens before the
        # next `_att_bf16_full` overwrites the cache. Pre-casts for
        # Epi-C12's PV.
        _asm_label["; HKMHA_EPI_C10_BEGIN"]()
        Self._pv_whole(v_reg_e9, att_block_bf16, o_reg)
        Self._full_softmax_unconditional[sched_group=Self._SCHED_EPI_C10_FULL](
            att_block_1,
            max_vec,
            max_vec_prev,
            scale_vec,
            norm_vec,
        )
        Self._att_bf16_full(att_block_bf16, att_block_1)
        _sched_barrier_zero()
        mul_col_inplace(o_reg, scale_vec)
        _s_barrier_raw()
        _sched_barrier_zero()

        # Epi-C11: load v_reg = V[N-1].
        _asm_label["; HKMHA_EPI_C11_BEGIN"]()
        var v_reg_e11 = reg_alloc[DType.bfloat16](Self._MmaOp.V_LAYOUT)
        Self._load_v_reg(v_reg_e11, v_smem[1])
        _cluster_barrier()

        # Epi-C12: PV[N-1] (whole) over `att_block_bf16` pre-cast in
        # Epi-C10, then final `o_normalized = o_reg / norm_vec`.
        _asm_label["; HKMHA_EPI_C12_BEGIN"]()
        Self._pv_whole(v_reg_e11, att_block_bf16, o_reg)
        var o_normalized = reg_alloc[DType.float32](Self._MmaOp.O_LAYOUT)
        div_col(o_normalized, o_reg, norm_vec)
        _cluster_barrier()

        # Conclusion barrier — lower-half warps wait.
        if not stagger:
            _s_barrier_raw()

        # Output store. col_l → row_l is a zero-cost re-tag of the same
        # per-lane storage, so we construct a TileTensor view over
        # `o_normalized.ptr` directly under `O_T_LAYOUT` instead of
        # invoking a transpose primitive.
        comptime _o_view_layout = Self._MmaOp.O_T_LAYOUT
        var o_normalized_view = TileTensor[
            DType.float32,
            type_of(_o_view_layout),
            MutExternalOrigin,
            address_space=AddressSpace.LOCAL,
        ](o_normalized.ptr, _o_view_layout)
        var epilogue_writer = RegTileEpilogue[output_dtype, 1](o_warp_2d)
        Self._store_o_to_gmem[output_dtype](
            o_normalized_view, epilogue_writer, l_id
        )


@always_inline
def hk_mha_prefill[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: HKMhaConfig,
    compile_options: StaticString = CompilationTarget[
        DeviceContext.default_device_info.target()
    ].default_compile_options(),
](
    q: TileTensor[mut=False, ...],
    k: k_t,
    v: v_t,
    o: TileTensor[mut=True, ...],
    mask_functor: mask_t,
    scale: Float32,
    num_keys: Int,
    start_pos: Int,
    ctx: DeviceContext,
) raises:
    """Host launcher for `HKMhaPrefill`.

    Derives grid dimensions from the Q layout and `num_keys`, compiles
    `HKMhaPrefill[config].run` (with optional caller-supplied LLVM
    `compile_options` such as `amdgpu-igrouplp-exact-solver=true` for
    benchmarks), and enqueues it.

    - `q`, `o`: `(batch, seq_len, num_heads, depth)` TileTensor.
      `o`'s dtype matches `config.output_dtype` — BF16 for production
      inference (which the dispatcher uses) or FP32 if the caller wants
      the unnormalized accumulator.
    - `k`, `v`: any `MHAOperand` (`LayoutTensorMHAOperand` for tests/
      bench, `KVCacheMHAOperand` for paged production caches at
      `page_size >= KV_BLOCK = 64`).

    `batch` and `seq_len` / `num_keys` may be dynamic; the head and
    depth dims must be static.
    """
    # Operand dtypes are taken dtype-generic at the launcher boundary
    # because Mojo doesn't unify a caller-site literal (e.g.
    # `DType.float32` from `enqueue_create_buffer[DType.float32]`) with
    # a generic comptime parameter (`config.output_dtype`,
    # `DType.bfloat16`) even when they evaluate equal. Asserting
    # explicitly here gives the same safety without forcing
    # `rebind[TileTensor[DType.bfloat16, ...]](q)` at every call site
    # (tests, bench, and the production dispatcher).
    comptime assert (
        q.dtype == DType.bfloat16
    ), "hk_mha_prefill: `q.dtype` must be `DType.bfloat16`"
    comptime assert (
        o.dtype == config.output_dtype
    ), "hk_mha_prefill: `o.dtype` must equal `config.output_dtype`"

    var batch = Int(q.dim[0]())
    var seq_len = Int(q.dim[1]())

    comptime kernel = HKMhaPrefill[config]
    comptime kernel_run = kernel.run[
        k_t,
        v_t,
        mask_t,
        q.dtype,
        o.dtype,
        q.LayoutType,
        o.LayoutType,
    ]

    var compiled = ctx.compile_function[
        kernel_run, compile_options=compile_options
    ]()
    ctx.enqueue_function(
        compiled,
        q,
        k,
        v,
        o,
        mask_functor,
        scale,
        num_keys,
        start_pos,
        grid_dim=(
            config.num_heads,
            ceildiv(seq_len, kernel.BM),
            batch,
        ),
        block_dim=kernel.NUM_THREADS,
    )
