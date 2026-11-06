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

"""Sparse MLA decode kernel for SM100 (B200) -- unified gather.

The MMAs run in the Q dtype (`mma_type`): FP8 Q runs
`tcgen05.mma.kind::f8f6f4` and BF16 Q runs `kind::f16`. The K/V front end
is chosen by the config, and the compute core behind it is shared:

- Direct gather (`config.direct_gather`; every BF16 form, the FP8 Layout E
  form and the routed Layout G forms): WG3 TMA-gathers K/V straight into
  the 64-column SW128 MMA ring. Under Layout E a 2-byte operand's Q and two
  K/V stages leave no room for a P ring, so P lives in the rope block of
  its K/V stage (`MLA_SM100_Decode_Config.p_in_kv`); FP8 keeps its own P
  ring.
- Linear ring plus re-swizzle (packed NVFP4 at 64-key tiles, and FP8 Layout
  E under `MLA_SPARSE_DIRECT_GATHER_E=0`): warp 8 gathers whole rows with
  one INT64/SWIZZLE_NONE gather4 per four rows into a linear ring, and WG3
  writes each tile into the SW64 layout the FP8 operand descriptor expects.
  A gather4 under SW64 has a 64-byte box, so gathering straight into SW64
  would take nine times as many gather4 instructions for the same bytes.
  For packed NVFP4 the load warp also copies each row's E4M3 scales into
  the stage, and WG3 dequantizes `e2m1 * sf` to FP8 on the way.
- In-place convert (`config.kv_convert`; BF16 Q over an FP8 cache and
  packed NVFP4 at 128-key tiles): each tile lands at the top of its own MMA
  stage and WG3 converts it in place, so the gather ring and the MMA ring
  are one ring.

At 128 heads the FP8 Layout E form runs as a CTA pair (`cta_group == 2`):
one `cta_group::2` QK MMA over both CTAs' 64-head Q tiles and 64-key K
halves (the Z ring), and a PV MMA split on N so each CTA supplies the V
depth columns it owns (the V ring).

Roles, 4 warpgroups:
  WG0 (warps  0-3):   Softmax.
  WG1 (warps  4-7):   Correction.
  WG2 (warps  8-11):  Load (warp 8: Q, plus K/V under the linear front end)
                      / MMA QK (warp 9) / MMA PV (warp 10) / index producer
                      and store (warp 11).
  WG3 (warps 12-15):  K/V front end: the direct gather, the re-swizzle, or
                      the in-place convert.

Under `config.half_rows` with at most 16 softmax rows, two threads share each
row: thread t of a warp loads TMEM lane t & 15 and 8 of its warp's 16 key
columns through `tcgen05.ld.16x32bx2`, the pair combines its row max with one
shuffle, and P rows 16-31 are zeroed once instead of written per tile.

The gather ring and the MMA ring do not share a depth: `PV_REL(i)` frees the
MMA slot that WG3 refills for tile i + M (M = `num_kv_mma_stages`), and that
WAR edge is the back-edge of the loop that sets the steady-state rate, so
leftover SMEM buys depth on the MMA side. Barriers: the fixed pipeline
barriers, the output barriers, the index ring (`idx_bars`) and
`cvt_blk_bars`: the in-place convert's per-block barriers, plus the
`q_in_tmem` hold on the last MMA stage.
An NVFP4 index stage holds the tile's value rows, then its scale rows.
"""

from std.collections import OptionalReg
from std.memory import Pointer
from std.math import ceildiv, clamp
from std.math.constants import log2e
from std.memory import bitcast
from std.sys import get_defined_int, size_of, _RegisterPackType
from std.sys._assembly import inlined_assembly
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    thread_idx,
    warp_id,
    lane_id,
)
from max.gpu.sync import async_copy_arrive, barrier, named_barrier, syncwarp
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_arrive_relaxed,
    cluster_wait,
)
from max.gpu.intrinsics import (
    byte_permute,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from max.gpu.memory import (
    CacheEviction,
    async_copy,
    cp_async_bulk_prefetch,
    external_memory,
    fence_async_view_proxy,
    fence_mbarrier_init,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    mma_arrive_multicast,
    MMASmemDescriptorPair,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_cp,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.swizzle import make_swizzle
from nn.attention.gpu.nvidia.sm100.mla_stamp import (
    CTA_ALLOC,
    CTA_INDEX_START,
    CTA_INITS,
    CTA_INITS_UNFENCED,
    CTA_LATE_DEALLOC,
    CTA_LATE_INBOUND,
    CTA_LATE_RELEASED,
    CTA_OFFSETS,
    CTA_ROLES_DONE,
    CTA_SETUP,
    CTA_SYNC_IN,
    CTA_ZEROED,
    stamp_begin,
    stamp_cta_entry,
    stamp_cta_event,
    stamp_cta_event_here,
    stamp_cta_event_warp,
    stamp_cta_exit,
    stamp_cta_late,
    stamp_cta_keys,
    stamp_finish,
    stamp_park,
    stamp_tile,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _gather4_box_width,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    ComptimeInt,
    DefaultEngine,
    RowMajorLayout,
    TensorEngine,
    TileTensor,
    coord,
)
from layout.layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import NullPointer, OptionalPointer
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.nvfp4_kv_dequant import (
    NVFP4_MLA_ROPE_DIM,
    e4m3x2_to_f16x2,
    mla_kv_is_nvfp4,
    nvfp4_dequant_group,
    nvfp4_dequant_group_f16,
    nvfp4_mla_row_bytes,
    nvfp4_mla_scale_row_bytes,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    bulk_mma_ws,
    bulk_mma_ws_ts,
    elect,
    elect_mma_arrive,
    expect_bytes_pred,
    opaque_i64,
    opaque_nonneg_i32,
    opaque_split_bound,
    opaque_u32,
    llvm_opaque_tid,
    SharedMemPointer,
    MBarType,
    ConsumerPipeline,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    decode_pdl_wait,
    _MLA_PDL_ALWAYS,
    _MLA_RAGGED_EXIT_BREAK,
    _MLA_CG2_SLICE_BREAK,
    _MLA_SPARSE_CG2,
    _SPARSE_PAIR_IDX_STAGES,
    _MLA_SPARSE_DIRECT_GATHER_E,
    _MLA_NVFP4_DEQUANT,
    _MLA_NVFP4_BATCH,
    _MLA_Q_TMEM,
    SparseRowLayout,
    decode_head_tile,
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    sparse_kv_row,
    ORaggedTMATile,
    MLA_Decode_Pack,
    OffsetPosition,
    KVPipelineGeneric,
    DecodeSM100MiscMBars,
    DecodeSProducerN,
    DecodePConsumerN,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeKVProducer,
    DecodeKVConsumer,
    DecodeSM100QKTSS_FP8,
    DecodeSM100PVSS_FP8,
    ld_shared_v4_u32,
    st_shared_v4_b32_at_bf16_elem_off,
    cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4,
    e8m0_to_bf16_broadcast,
    hmul2_bf16x8_by_scalar,
    hr_row,
    SW64_COLS,
    _MLA_P6_BREAK,
)


@inline(.always)
def _role_block_x() -> Int:
    return Int(
        inlined_assembly[
            "mov.u32 $0, %ctaid.x;",
            UInt32,
            constraints="=r",
            has_side_effect=True,
        ]()
    )


@inline(.always)
def _role_block_z() -> Int:
    return Int(
        inlined_assembly[
            "mov.u32 $0, %ctaid.z;",
            UInt32,
            constraints="=r",
            has_side_effect=True,
        ]()
    )


@inline(.always)
def _wait_parity_word(word: UInt32):
    """Waits the mbarrier phase a `pair_tail_lane_wait` word names; 0 waits
    nothing."""
    if word != 0:
        inlined_assembly[
            """{
            .reg .pred P1;
            LAB_WAIT:
            mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;
            @P1 bra DONE;
            bra LAB_WAIT;
            DONE:
            }""",
            NoneType,
            constraints="r,r",
        ](word >> 1, word & 1)


@inline(.always)
def _tmem_base(
    ptr_tmem_addr: Pointer[mut=True, UInt32, _, address_space=.SHARED],
) -> UInt32:
    """Returns the TMEM base the roles address."""
    return ptr_tmem_addr[0]


comptime _MLA_NVFP4_BREAK = get_defined_int["MLA_NVFP4_BREAK", 0]()
comptime _MLA_NVFP4_WIDE_EARLY_STORE = get_defined_int[
    "MLA_NVFP4_WIDE_EARLY_STORE", 1
]()


struct MLA_SM100_Decode_Sparse_QKV_FP8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    AttnSinkPtrType: OptionalPointer = NullPointer[DType.float32],
    has_extra_kv: Bool = False,
    TopkLengthsPtrType: OptionalPointer = NullPointer[DType.int32],
    fold_shared_index: Bool = False,
    q_len_fold: Int = 1,
    physical_indices: Bool = True,
    sparse_row: SparseRowLayout = SparseRowLayout(),
    skip_masked_tiles: Bool = False,
    prepared_bounds: Bool = False,
    Engine: TensorEngine = DefaultEngine[element_width=1],
](TrivialRegisterPassable):
    """Sparse MLA decode for SM100 (B200), unified gather, FP8 or BF16 MMAs.

    The K/V front end is the config's: the direct SW128 gather
    (`config.direct_gather`), the linear ring plus re-swizzle into SW64, or
    the in-place convert (`config.kv_convert`). See the module docstring.
    """

    # Presence of the optional operands is carried by their pointer types.
    comptime has_attn_sink = not Self.AttnSinkPtrType.is_null
    comptime has_variable_topk = not Self.TopkLengthsPtrType.is_null

    comptime kv_type = Self.KVLUTType.dtype
    # Packed NVFP4 cache (sm_100): the gather lands half-width e2m1 rows plus
    # their E4M3 scales in the linear stage, and the re-swizzle warpgroup
    # expands them into the FP8 tile the MMA reads.
    comptime is_nvfp4_kv = mla_kv_is_nvfp4[Self.KVLUTType]()
    # Bytes of one packed row in the linear stage (the packed latent, then the
    # FP8 rope dims), and the E4M3 scales of the latent (one per 16 elements).
    # Each stage holds the packed rows first, then their scales, inside the
    # FP8-sized stage, so the SMEM budget is unchanged.
    comptime nvfp4_row_bytes = nvfp4_mla_row_bytes[Self.config.input_q_depth]()
    comptime nvfp4_sf_row_bytes = nvfp4_mla_scale_row_bytes[
        Self.config.input_q_depth
    ]()
    comptime nvfp4_sf_offset = Self.config.BN_QK * Self.nvfp4_row_bytes
    # An NVFP4 cache pages its scales apart from its values, so the indexer
    # writes each sparse index row as `[value rows | scale rows]`, `topk`
    # slots each. A stage of `idx_smem` mirrors it: the tile's value rows,
    # then its scale rows.
    comptime sparse_row_parts = 2 if Self.is_nvfp4_kv else 1
    comptime idx_stage_slots = Self.config.BN_QK * Self.sparse_row_parts
    # Q, K, V and P in SMEM, and the MMA kind: Q arrives in the dtype the
    # MMAs read.
    comptime ScalarArgsType = (
        StaticTuple[Int64, 0] if Self.prepared_bounds else TileTensor[
            DType.int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
            Engine=Self.Engine,
        ]
    )
    comptime mma_type = Self.q_type
    comptime AccumType = get_accum_type[Self.q_type]()
    comptime mma_bytes = size_of[Self.mma_type]()
    comptime role_local_lane = (
        Self.config.cta_group == 2
        and Self.mma_type == DType.float8_e4m3fn
        and Self.kv_type == DType.float8_e4m3fn
    )
    # A split NVFP4 decode reads its token and key span out of an SMEM record
    # instead of the entry-hoisted copies, which spill through every role.
    comptime role_remat = (
        Self.config.cta_group == 1
        and Self.is_nvfp4_kv
        and Self.skip_masked_tiles
        and Self.config.decoding_warp_split_k
        and Self.config.num_q_heads == 128
    )
    # The same re-derivation for the split bounds, which the h=64 masked arm
    # keeps in a stack slot that each role's loop reloads (`role_remat`'s
    # record is h=128-only, its dispatch stores the token row). Every value
    # is a sparse slot count or a slot offset, so it fits Int32.
    comptime role_split_remat = Self.is_nvfp4_kv and not Self.role_remat
    comptime role_metadata_bytes = 64 if Self.role_remat else 0
    # One C release and two O releases advance both pipeline phases together,
    # so Correction can read its phase from O instead of carrying its own.
    comptime corr_phase_from_o = (
        Self.role_remat and Self.config.depth // Self.config.MMA_PV_N == 2
    )
    # The clamped metadata bound keeps Correction's trip count inside an Int32
    # countdown, so neither 64-bit iteration word is live across its FP body.
    comptime corr_countdown = (
        Self.role_remat
        and Self.config.BN_QK == 64
        and Self.sparse_row_parts == 2
        and not Self.has_extra_kv
        and Self.MaskType.get_type_name() == "CausalMask"
    )
    # Q and P are stored as column blocks of 64 columns: SW64 for FP8, SW128
    # for a 2-byte operand.
    comptime operand_swizzle = Self.config.q_swizzle()
    comptime NumQKBlocks = Self.config.padded_q_depth // SW64_COLS
    comptime NumVOBlocks = Self.config.padded_depth // SW64_COLS
    # Q block: BM rows x 64 cols.
    comptime BlockElems = Self.config.BM * SW64_COLS
    # KV block: BN_QK rows x 64 cols (128 under `kv_sw128`), independent of
    # BM. Equal to BlockElems under Layout E only. A pair tile holds twice
    # the keys, but each CTA's K slot (Z) carries only its half, so the
    # block counts the PER-CTA rows.
    comptime kv_sw128 = Self.config.kv_sw128()
    comptime KVBlockCols = Self.config.kv_block_cols()
    comptime is_cg2 = Self.config.cta_group == 2
    # Keys each CTA gathers per tile: all of them, or half of the pair's.
    comptime CtaKeys = Self.config.BN_QK // Self.config.cta_group
    comptime KVBlockElems = Self.CtaKeys * Self.KVBlockCols
    comptime z_row_elems = Self.config.kv_row_elems()
    comptime NumKVBlocks = Self.z_row_elems // Self.KVBlockCols
    comptime NumVBlocks = Self.config.padded_depth // Self.KVBlockCols
    comptime kv_swizzle = (
        TensorMapSwizzle.SWIZZLE_128B if Self.kv_sw128 else TensorMapSwizzle.SWIZZLE_64B
    )
    comptime KVStageElems = Self.NumKVBlocks * Self.KVBlockElems
    # A pair's P stage spans the whole pair tile per CTA row: this CTA's
    # BM rows over all BN_QK pair keys (two 64-key SW64 blocks).
    comptime PStageElems = (Self.config.BM * Self.config.BN_QK)
    # The pair's V ring stage: both key halves' V depth columns this CTA
    # owns for the PV MMA's N split -- half of each 256-wide PV atom, two
    # SW128 blocks per half. Pair only.
    comptime EStageElems = Self.CtaKeys * Self.config.padded_depth
    # SW128 column blocks per (key half, PV atom) of the V-gather's V ring:
    # the 128 depth columns a CTA owns of each atom are one block at 1 byte,
    # two at 2 bytes.
    comptime v_boxes = (Self.config.MMA_PV_N // 2) // Self.KVBlockCols
    comptime NumQSmemBlocks = (0 if Self.config.q_in_tmem else Self.NumQKBlocks)
    # The pair's Z ring depth, equal to `num_stages` (asserted in the carve).
    comptime z_stages = Self.config.num_kv_mma_stages
    comptime e_ring_elems = Self.EStageElems * Self.config.num_kv_stages
    # P in the rope block of its K/V stage (`MLA_SM100_Decode_Config.p_in_kv`),
    # so its ring has no SMEM of its own and steps by whole K/V stages.
    comptime p_in_kv = Self.config.p_in_kv()
    comptime p_ring_elems = 0 if Self.p_in_kv else (
        Self.PStageElems * Self.config.num_kv_stages
    )
    comptime p_stage_stride = (
        Self.KVStageElems if Self.p_in_kv else Self.PStageElems
    )
    # Two softmax threads per row (`MLA_SM100_Decode_Config.half_rows`).
    comptime half_row_softmax = Self.config.half_row_softmax(
        Self.config.num_q_heads
        * (Self.q_len_fold if Self.fold_shared_index else 1)
    )
    comptime output_tile_width = (SW64_COLS // 2) * (
        4 // size_of[Self.output_type]()
    )

    # TMA gathers K/V straight into the SW128 ring (`config.direct_gather`).
    comptime direct_gather = Self.config.direct_gather
    # BF16 Q over an FP8 cache (`config.kv_convert`): each FP8 row lands in
    # the upper half of its own BF16 K/V stage and WG3 converts it in place,
    # so the gather ring and the MMA ring are one ring.
    comptime kv_convert = Self.config.kv_convert
    comptime blockwise = Self.config.scale_block_size > 0
    # Packed NVFP4 under `kv_convert`: the packed rows and their scales land
    # at the top of their own SW128 FP8 stage and WG3 dequantizes in place,
    # so the gather ring and the MMA ring are one ring.
    comptime nvfp4_ring = Self.is_nvfp4_kv and Self.kv_convert
    # Where a converting stage's gathered source starts: the upper half of
    # a BF16 stage, or the last packed-row-and-scale bytes of an FP8 one.
    comptime cvt_src_offset = (
        Self.KVStageElems
        - Self.config.BN_QK
        * (
            Self.nvfp4_row_bytes + Self.nvfp4_sf_row_bytes
        ) if Self.nvfp4_ring else Self.KVStageElems
        // 2
    )
    # The linear front end's gather ring: one 64-key tile per slot, released
    # by every re-swizzle thread. Under `kv_convert` a slot is the upper half
    # of a K/V stage and steps by whole stages. A packed NVFP4 tile's rows
    # and scales are smaller than a full stage, so the NVFP4 form sizes its
    # slot through the config (`MLA_NVFP4_PACKED_LINEAR`); the staging
    # assert below pins the packed tile to the slot exactly.
    comptime nvfp4_packed_linear = (
        Self.is_nvfp4_kv
        and not Self.kv_convert
        and not Self.direct_gather
        and Self.config.gather_stage_bytes > 0
    )
    comptime LinStageElems = (
        Self.KVStageElems if Self.kv_convert else (
            Self.config.gather_stage_bytes if Self.nvfp4_packed_linear else Self.config.BN_QK
            * Self.config.padded_q_depth
        )
    )
    # The converter's slot is also released by both MMAs (its BF16 output
    # overwrites the slot's FP8 rows), and blockwise scales arrive from the
    # 32 lanes of warp 11. An NVFP4 tile's scales arrive from the 32 lanes of
    # the load warp, through their `cp.async` copies.
    comptime gather_producers = 1 + (
        32 if (Self.kv_convert and Self.blockwise) or Self.is_nvfp4_kv else 0
    )
    comptime gather_consumers = WARPGROUP_SIZE + (2 if Self.kv_convert else 0)
    comptime GatherPipe = KVPipelineGeneric[
        num_kv_stages=Self.num_stages,
        num_qk_stages=1,
        num_producer=Self.gather_producers,
        num_consumer=Self.gather_consumers,
    ]
    comptime scale_ring_bytes = (
        Self.config.scale_smem_per_stage
        * Self.num_stages if Self.kv_convert else 0
    )
    # The SW64 ring's full barrier: the re-swizzle warpgroup's 128 threads,
    # or one elected gather lane per WG3 warp.
    comptime mma_producers = 4 if Self.direct_gather else WARPGROUP_SIZE
    comptime MmaPipe = KVPipelineGeneric[
        num_kv_stages=Self.config.num_kv_mma_stages,
        num_qk_stages=1,
        num_producer=Self.mma_producers,
        num_consumer=2,
    ]
    # Index ring consumers: the load warp, or the four WG3 gather lanes.
    comptime idx_consumers = 4 if Self.direct_gather else 32
    # The index ring's depth: the K/V ring's, or the pair's own
    # (`_SPARSE_PAIR_IDX_STAGES`).
    comptime idx_stages = (
        _SPARSE_PAIR_IDX_STAGES if Self.is_cg2 else Self.config.num_kv_stages
    )
    comptime IdxBars = DecodeSM100MiscMBars[
        num_stages=Self.idx_stages,
        num_producer=32,
        num_consumer=Self.idx_consumers,
    ]

    # Gather4 rows per descriptor box. Type-only for gather4, whose box is
    # one row; the per-call row count is a parameter of the issue loop.
    comptime GatherTileRows = 64
    # Linear front end: one INT64, SWIZZLE_NONE box covering the 576-byte
    # row (16 gather4 instructions per 64 keys instead of the SW64-fragmented
    # 144). Direct gather: the cache dtype, one SW128 column block per box.
    comptime k_tma_dtype = (
        Self.kv_type if Self.direct_gather
        and not Self.is_nvfp4_kv else DType.int64
    )
    # The row as stored, in INT64 words.
    comptime kv_row_words = (
        Self.nvfp4_row_bytes if Self.is_nvfp4_kv else Self.config.input_q_depth
        * size_of[Self.kv_type]()
    ) // 8
    comptime kv_gather4_tile_width = (
        Self.config.input_q_depth if Self.direct_gather
        and not Self.is_nvfp4_kv else Self.kv_row_words
    )
    comptime kv_gather4_swizzle = (
        Self.kv_swizzle if Self.direct_gather
        and not Self.is_nvfp4_kv else TensorMapSwizzle.SWIZZLE_NONE
    )
    comptime kv_gather4_box_w = _gather4_box_width[
        Self.k_tma_dtype,
        Self.kv_gather4_tile_width,
        Self.kv_gather4_swizzle,
    ]()
    comptime KTMA = TMATensorTile[
        Self.k_tma_dtype,
        tile_shape=coord[Self.GatherTileRows, Self.kv_gather4_box_w],
        desc_shape=coord[1, Self.kv_gather4_box_w],
    ]
    comptime PackedVTMA = TMATensorTile[
        DType.int64, tile_shape=coord[64, 8], desc_shape=coord[1, 8]
    ]
    # >0: the elected lane loads the row indices of this many 4-row chunks
    # with back-to-back ld.shared.v4 before issuing their gather4s, so the
    # SMEM latency is paid once per batch instead of once per gather4.
    comptime gather4_idx_batch = get_defined_int[
        "MLA_SPARSE_GATHER4_IDX_BATCH", 8
    ]()
    # Worst-case padding that 16-byte aligns idx_smem for ld.shared.v4; the
    # dispatch adds this to the SMEM it reserves.
    comptime idx_smem_align_pad = 12
    # The split's bulk L2 index prefetch and the next tile's register
    # prefetch. They pay off on the Layout G conversion's long walks, not
    # at Layout E.
    comptime idx_prefetch = (Self.direct_gather and not Self.is_nvfp4_kv) or (
        Self.kv_convert and Self.config.decode_layout_g and not Self.is_nvfp4_kv
    )

    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type=Self.mma_type,
        accum_type=Self.AccumType,
        config=Self.config,
        k_swizzle=Self.kv_swizzle,
        k_row_elems=Self.z_row_elems if Self.kv_sw128 else 0,
        q_swizzle=Self.operand_swizzle,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.mma_type,
        accum_type=Self.AccumType,
        config=Self.config,
        p_swizzle=Self.operand_swizzle,
        v_swizzle=Self.kv_swizzle,
    ]

    comptime num_stages = Self.config.num_kv_stages
    # Depth of the SW64 ring only. `PV_REL(i)` frees the slot that
    # `CVT_SW64_ACQ(i + num_mma_stages)` reclaims, and that WAR edge is the
    # back-edge of the loop that sets this kernel's steady-state rate, so the
    # bound is (cycle weight) / this depth. The linear gather ring is not on
    # that loop and stays at `num_stages`.
    comptime num_mma_stages = Self.config.num_kv_mma_stages
    # Layout E needs replicated Q in the columns occupied by TMEM correction
    # scalars, so this operand layout keeps those scalars in SMEM instead.
    comptime h64_q192_tmem = (
        Self.config.cta_group == 1
        and Self.config.num_q_heads == 64
        and Self.q_type == DType.bfloat16
        and Self.kv_type == DType.bfloat16
        and Self.config.BM == 64
        and Self.config.BN_QK == 64
        and Self.config.BK_QK == 576
        and Self.config.input_q_depth == 576
        and Self.config.padded_q_depth == 576
        and Self.num_stages == 2
        and Self.num_mma_stages == 2
        and Self.KVBlockCols == 64
        and Self.config.sm100_tmem_cols >= 512
        and Self.operand_swizzle == TensorMapSwizzle.SWIZZLE_128B
        and Self.direct_gather
        and not Self.kv_convert
        and not Self.config.decode_layout_g
        and not Self.fold_shared_index
        and Self.q_len_fold == 1
    )
    comptime corr_smem = Self.h64_q192_tmem
    comptime ml_elems = (4 if Self.corr_smem else 3) * WARPGROUP_SIZE
    # Under `kv_convert`, one barrier per K/V block per stage: WG3 publishes
    # each converted block so the QK MMA starts before the stage finishes.
    comptime cvt_blk_only_mbars = (
        Self.NumKVBlocks * Self.num_stages if Self.kv_convert else 0
    )
    # Under `config.q_in_tmem`, Q lands in the last SW64 MMA stage and the
    # QK warp copies it to TMEM; this barrier holds WG3's first fill of that
    # stage until the copy has read it.
    comptime q_in_tmem = Self.config.q_in_tmem
    comptime q_cp_mbar_idx = Self.cvt_blk_only_mbars
    comptime cvt_blk_mbars = Self.cvt_blk_only_mbars + (
        1 if Self.q_in_tmem else 0
    )

    # Barriers the carve in `kernel` lays out: Q, the gather and MMA K/V
    # rings, S, P, O, C, corr_done, the output pipeline, and the index ring.
    comptime num_carve_mbars = (
        Self.num_carve_mbars_pair() if Self.is_cg2 else (
            1
            + Int(Self.GatherPipe.num_mbars())
            + Int(Self.MmaPipe.num_mbars())
            + 2 * Self.num_stages
            + 2 * Self.num_stages
            + 2 * 2
            + 2 * 1
            + 2 * 2
            + Int(
                OutPipeline[
                    num_out_stages=DecodeOutProducer[
                        Self.output_type, Self.config
                    ].num_out_stages,
                    num_producer=WARPGROUP_SIZE,
                    num_consumer=1,
                ].num_mbars()
            )
            + 2 * Self.num_stages
            + Self.cvt_blk_mbars
        )
    )
    # The separate linear gather ring (the re-swizzle front end only).
    comptime kv_linear_elems = 0 if (
        Self.direct_gather or Self.kv_convert
    ) else Self.LinStageElems * Self.num_stages
    # The end of the carve in `kernel`, in bytes from the dynamic SMEM base,
    # with idx_smem's worst-case alignment pad.
    comptime carve_bytes = (
        (
            Self.BlockElems * Self.NumQSmemBlocks
            + Self.kv_linear_elems
            + (Self.e_ring_elems if Self.is_cg2 else 0)
            + Self.KVStageElems * Self.num_mma_stages
            + Self.p_ring_elems
        )
        * Self.mma_bytes
        + Self.ml_elems * size_of[Self.AccumType]()
        + Self.scale_ring_bytes
        + Self.num_carve_mbars * Self.config.mbar_size
        + size_of[UInt32]()
        + Self.idx_smem_align_pad
        + Self.idx_stages * Self.idx_stage_slots * size_of[Int32]()
        + Self.role_metadata_bytes
    )

    # The ground-truth SW64 destination layout the MMA operand descriptor
    # already reads today -- the re-swizzle WG's job is to reproduce this
    # exact byte permutation from the contiguously-gathered linear source.
    comptime sw_fp8 = make_swizzle[
        Self.mma_type, TensorMapSwizzle.SWIZZLE_64B
    ]()

    comptime Common_MLA_Op = MLA_SM100_Decode_Common[
        Self.q_type,
        Self.KVLUTType,
        Self.output_type,
        Self.SplitAccumType,
        Self.MaskType,
        Self.config,
        Self.ValidLengthType,
        Self._is_cache_length_accurate,
        Self.ragged,
    ]

    @staticmethod
    @inline(.always)
    def num_carve_mbars_pair() -> Int:
        """Returns the paired carve's barrier count (comptime).

        Per ring stage: z_tx, z_empty, z_full, the V ring's v_tx, v_full
        and v_free, S's full/free pair, P's full/free pair and the index
        ring's pair, 12 in all. Fixed: mbar_q, the O pipeline (2 stages),
        the C pair, the corr_done pair and the output pipeline.
        """
        return (
            1
            + 12 * Self.num_stages
            + 2 * (Self.idx_stages - Self.num_stages)
            + 4
            + 2
            + 4
            + Int(
                OutPipeline[
                    num_out_stages=DecodeOutProducer[
                        Self.output_type, Self.config
                    ].num_out_stages,
                    num_producer=WARPGROUP_SIZE,
                    num_consumer=1,
                ].num_mbars()
            )
        )

    @staticmethod
    @inline(.always)
    def v_block_offset[h: Int, a: Int, c: Int]() -> Int:
        """Returns where V-ring column block `c` of key half `h`, PV atom
        `a` starts in a V stage, in elements (comptime).

        At 1 byte a half's atom is one block of its `CtaKeys` keys, blocks
        half-major. At 2 bytes a block holds all `BN_QK` pair keys of one
        64-column box, half `h` at row `h * CtaKeys`, so each atom's two
        blocks are one MN-major operand of the whole pair tile, the layout
        of the unpaired kernel's V.
        """
        comptime if Self.mma_bytes == 1:
            return (2 * h + a) * Self.KVBlockElems
        return (
            a * Self.v_boxes + c
        ) * Self.config.BN_QK * Self.KVBlockCols + h * Self.KVBlockElems

    @staticmethod
    @inline(.always)
    def zero_input_tail(
        q_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
    ):
        """Zero the SMEM columns past the input row, once per CTA.

        Once per CTA is enough because the kernel is not persistent: `p_smem`
        is its own region, and the only later write to these columns is the
        epilogue's output staging (`out_smem` aliases `kv_smem`), after the
        last QK. A kernel that runs more than one work item per CTA must
        re-zero them before the next item's first QK.
        """
        comptime tail_block0 = Self.config.input_q_depth // SW64_COLS
        comptime tail_blocks = Self.NumQKBlocks - tail_block0
        comptime q_tail_elems = tail_blocks * Self.BlockElems
        comptime q_tail_off = tail_block0 * Self.BlockElems
        # SW128 K/V: TMA zero-fills a partly loaded block, so only the
        # blocks the gather skips need zeroing.
        comptime kv_block0 = ceildiv(
            Self.config.input_q_depth, Self.KVBlockCols
        ) if Self.kv_sw128 else tail_block0
        comptime kv_tail_elems = (
            Self.NumKVBlocks - kv_block0
        ) * Self.KVBlockElems
        comptime kv_tail_off = kv_block0 * Self.KVBlockElems
        comptime zero = Scalar[Self.mma_type](0)

        var tid = thread_idx.x
        for i in range(tid, q_tail_elems, Self.config.num_threads):
            q_smem[q_tail_off + i] = zero
        comptime for stage in range(Self.num_mma_stages):
            var base = kv_smem.unsafe_offset(
                stage * Self.KVStageElems + kv_tail_off
            )
            for i in range(tid, kv_tail_elems, Self.config.num_threads):
                base[i] = zero

    @staticmethod
    @inline(.always)
    def pdl_early_exit_all_q(
        split_idx: Int,
        batch_idx: Int,
        max_seq_len: Int,
        batch_size: Int,
        lse_accum_split_ptr: Self.SplitAccumType,
    ):
        """Writes -inf LSE for every query this CTA owns and releases PDL
        dependents, for a CTA that exits before its walk."""
        comptime if Self.fold_shared_index:
            comptime for q_local in range(Self.q_len_fold):
                Self.Common_MLA_Op.pdl_early_exit[fold_q=True](
                    split_idx,
                    batch_idx,
                    max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                    seq_idx_fold=UInt32(q_local),
                )
        else:
            Self.Common_MLA_Op.pdl_early_exit(
                split_idx,
                batch_idx,
                max_seq_len,
                batch_size,
                lse_accum_split_ptr,
            )

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(packed_v_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(extra_k_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](
            Int32(Self.config.cta_group), 1, 1
        )
    )
    def kernel_pair(
        q_tma: QOTMATile[
            dtype=Self.mma_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.operand_swizzle,
        ],
        # The K/V gather4 TMA: one INT64 SWIZZLE_NONE box per row under the
        # linear front end, SW128 column blocks under the direct gather.
        k_tma: Self.KTMA,
        packed_v_tma: Self.PackedVTMA,
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType=Self.ValidLengthType,
            MaskType=Self.MaskType,
            SplitAccumType=Self.SplitAccumType,
            prepared_bounds=Self.prepared_bounds,
        ],
        d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        indices_stride_dev: Int32,
        # Logical sparse indices (same layout as `d_indices`, -1 padding) for
        # position-based causal masking. See mla_decode_utils.mojo.
        logical_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        topk_lengths: Self.TopkLengthsPtrType,
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
        attn_sink_ptr: Self.AttnSinkPtrType,
        extra_k_tma: Self.KTMA,
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        extra_topk_lengths: Self.TopkLengthsPtrType,
        extra_indices_stride_dev: Int32,
        extra_scales_ptr: OptionalReg[Pointer[Float32, MutAnyOrigin]],
        scalar_args: Self.ScalarArgsType,
        # Positions between the token each sparse row was computed for and
        # the token that reads it.
        selection_lag_dev: Int32,
        stamp_addr: UInt64,
    ):
        """The pair's kernel entry: the same body as `kernel`, with the
        (2, 1, 1) cluster the cta_group::2 MMAs and TMAs require.

        A separate entry point because the cluster dimension is function
        metadata, and the unpaired forms launch without a cluster.
        """
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_Sparse_QKV_FP8 supports NullMask and CausalMask"
            " only. Sliding-window is not yet wired for sparse."
        )
        comptime assert (
            not Self.fold_shared_index
            or Self.config.num_q_heads * Self.q_len_fold <= Self.config.BM
        ), "fold_shared_index requires num_q_heads * q_len_fold <= BM."
        comptime assert not Self.config.decode_layout_g or (
            Self.config.MMA_M == 32 and Self.config.BN_QK == 64
        ), "Sparse Layout G requires MMA_M == 32 and BN_QK == 64."
        comptime assert not Self.direct_gather or (
            Self.config.decode_layout_g
            or Self.config.cta_group == 2
            or Self.mma_bytes == 2
            or (
                (_MLA_SPARSE_DIRECT_GATHER_E != 0 or Self.config.BN_QK == 128)
                and not Self.p_in_kv
            )
        ), (
            "The FP8 direct gather needs Layout G, the pair, or a separate P"
            " ring."
        )
        comptime assert (
            not Self.is_cg2 or _MLA_SPARSE_CG2 != 0
        ), "the CTA pair is behind MLA_SPARSE_CG2"
        comptime assert (
            _MLA_CG2_SLICE_BREAK >= 0 and _MLA_CG2_SLICE_BREAK <= 2
        ), "MLA_CG2_SLICE_BREAK is 0, 1 or 2"
        comptime assert not Self.is_cg2 or (
            Self.config.cta_group == 2
            and not Self.config.decode_layout_g
            and not Self.kv_convert
            and not Self.p_in_kv
            and not Self.fold_shared_index
            and Self.config.num_q_heads == 2 * Self.config.BM
        ), "the pair needs the Layout E direct-gather form at 128 heads"
        comptime assert (
            not Self.is_cg2 or Self.mma_bytes == 1 or Self.config.BN_QK == 64
        ), "the 2-byte pair runs a 64-key pair tile"
        comptime assert not Self.p_in_kv or (
            Self.mma_bytes == 2
        ), "P in the K/V stage needs the rope block of a 2-byte operand"
        comptime assert (
            Self.config.operand_size == Self.mma_bytes
        ), "the config's operand size is not Q's"
        comptime assert not Self.kv_convert or (
            (Self.mma_bytes == 2 or Self.nvfp4_ring)
            and size_of[Self.kv_type]() == 1
            and not Self.direct_gather
            and Self.num_stages == Self.num_mma_stages
        ), "kv_convert converts an FP8 or NVFP4 ring in place"
        comptime assert not Self.p_in_kv or (
            Self.num_stages == Self.num_mma_stages
        ), "P in the K/V stage needs the S/P and K/V rings in lockstep"
        comptime assert not (
            Self.fold_shared_index and Self.has_extra_kv
        ), "fold_shared_index does not support extra always-attend KV cache."
        comptime assert not Self.is_nvfp4_kv or (
            Self.physical_indices
            and not Self.has_extra_kv
            and Self.config.BN_QK == 128
            and Self.config.input_q_depth == 576
            and Self.num_stages >= 2
        ), "the NVFP4 pair needs physical indices and 128-key rotary tiles"

        # The converter holds a row's overlay blocks across its barrier, so
        # it takes the correction warpgroup's share. Each warpgroup grows
        # or shrinks from the 128-register launch share, so which of
        # `setmaxnreg.inc` and `.dec` it issues follows its budget.
        comptime num_reg_softmax = 192
        # The linear NVFP4 dequant holds a row's packed chunks across their
        # loads; below 80 registers WG3 spills, and up to 152 it still gains
        # (measured on B200), while correction needs 72.
        comptime nvfp4_linear = Self.is_nvfp4_kv and not Self.kv_convert
        comptime num_reg_correction = (
            72 if Self.kv_convert else (104 if nvfp4_linear else 184)
        )
        comptime num_reg_other = 64
        comptime num_reg_reswizzle = (
            184 if Self.kv_convert else (152 if nvfp4_linear else 72)
        )
        comptime assert (
            num_reg_softmax
            + num_reg_correction
            + num_reg_other
            + num_reg_reswizzle
            <= 512
        ), "the warpgroup register budgets overrun the register file"

        stamp_cta_entry()
        comptime _pdl_on = Self.config.decoding_warp_split_k or _MLA_PDL_ALWAYS
        var indices_stride = Int(indices_stride_dev)
        var topk_stride = indices_stride // Self.sparse_row_parts
        var extra_indices_stride = Int(extra_indices_stride_dev)
        var batch_size: Int
        var q_max_seq_len: Int
        comptime if Self.prepared_bounds:
            batch_size = mla_decode_pack.batch_size()
            q_max_seq_len = mla_decode_pack.q_max_seq_len()
        else:
            var dynamic_args = rebind[
                TileTensor[
                    DType.int64,
                    RowMajorLayout[ComptimeInt[3]],
                    MutAnyOrigin,
                    Engine=Self.Engine,
                ]
            ](scalar_args)
            batch_size = Int(dynamic_args[0])
            q_max_seq_len = Int(dynamic_args[1])
        var selection_lag = Int(selection_lag_dev)
        var num_partitions = Int(mla_decode_pack.num_partitions)
        var mask = mla_decode_pack.mask
        var valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr

        decode_pdl_wait[_pdl_on]()
        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ](
            kv_lut,
            rebind[
                Pointer[
                    Scalar[Self.ValidLengthType.dtype],
                    ImmutAnyOrigin,
                    address_space=.GENERIC,
                ]
            ](valid_length.value()),
            q_max_seq_len,
            num_partitions,
            batch_size,
            sparse_indices_stride=topk_stride,
            sparse_topk_lengths=topk_lengths,
            sparse_extra_indices_stride=extra_indices_stride,
            sparse_extra_topk_lengths=extra_topk_lengths,
            topk_lengths_per_token=Self.skip_masked_tiles,
        )

        comptime if Self.skip_masked_tiles:
            comptime assert (
                not Self.fold_shared_index
            ), "skip_masked_tiles bounds one token per CTA"
            offset_position.skip_masked_tiles[
                causal=Self.MaskType.get_type_name() == "CausalMask"
            ](num_partitions, topk_stride, topk_lengths)

        var extra_topk: Int = 0
        comptime if Self.has_extra_kv:
            comptime if Self.has_variable_topk:
                extra_topk = Int(
                    extra_topk_lengths.value()[Int(offset_position.batch_idx)]
                )
            else:
                extra_topk = extra_indices_stride
        # Main-segment slot count: the clamped topk, tile-padded when extra KV
        # follows it (see `OffsetPosition.sparse_topk`).
        var topk = offset_position.num_keys - extra_topk

        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)

        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                Self.pdl_early_exit_all_q(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                )
                return

        comptime if Self.ragged and (
            Self.config.decoding_warp_split_k or _MLA_RAGGED_EXIT_BREAK != 2
        ):
            if offset_position.seq_idx >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    Self.pdl_early_exit_all_q(
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                    )
                return

        comptime if Self.skip_masked_tiles and not Self.config.decoding_warp_split_k:
            if offset_position.num_keys == 0:
                Self.Common_MLA_Op.store_empty(
                    o_tma, offset_position.out_row_offset
                )
                # With an LSE output (decode context parallelism) an empty
                # shard also writes `-inf`, so a cross-rank merge weights it 0.
                comptime if not Self.SplitAccumType.is_null:
                    Self.Common_MLA_Op.store_empty_lse(
                        offset_position.q_token_idx, lse_accum_split_ptr
                    )
                return

        stamp_cta_event(CTA_OFFSETS)
        var q_smem = external_memory[
            Scalar[Self.mma_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        # Linear (SWIZZLE_NONE) gather destination -- the re-swizzle WG's
        # source, a slot the size of an SW64 stage. Empty under the direct
        # gather.
        var kv_smem_linear = q_smem.unsafe_offset(
            Self.BlockElems * Self.NumQSmemBlocks
        )
        # SW64 destination the native FP8 MMA reads -- byte-identical role
        # to the single-buffer kernel's `kv_smem`, one ring deeper.
        comptime kv_linear_elems = Self.kv_linear_elems
        var kv_smem = kv_smem_linear.unsafe_offset(kv_linear_elems)
        comptime if Self.kv_convert:
            kv_smem_linear = kv_smem.unsafe_offset(Self.cvt_src_offset)
        # The pair's V ring sits between the Z ring and the P ring;
        # `out_smem` still reuses the Z region at the epilogue.
        var e_smem = kv_smem.unsafe_offset(
            Self.KVStageElems * Self.num_mma_stages
        )
        var p_smem = e_smem.unsafe_offset(
            Self.e_ring_elems
        ) if Self.is_cg2 else kv_smem.unsafe_offset(
            Self.KVStageElems * Self.num_mma_stages
        )
        comptime if Self.p_in_kv:
            p_smem = kv_smem.unsafe_offset(Self.NumVBlocks * Self.KVBlockElems)
        comptime if Self.kv_sw128:
            # An SW128 operand's swizzle phase comes from its address (the
            # TMA writes and the MMA descriptors assume base offset 0). The
            # offsets are carve constants; the dynamic SMEM base is declared
            # only 128 B aligned, so its check stays a debug assert.
            comptime kv_off_bytes = (
                Self.BlockElems * Self.NumQSmemBlocks + kv_linear_elems
            ) * Self.mma_bytes
            comptime assert (
                kv_off_bytes % 1024 == 0
                and (Self.KVStageElems * Self.mma_bytes) % 1024 == 0
            ), "SW128 K/V stage offset is not 1 KiB aligned"
            comptime p_off_bytes = (
                kv_off_bytes
                + Self.KVStageElems * Self.num_mma_stages * Self.mma_bytes
                + (Self.e_ring_elems * Self.mma_bytes if Self.is_cg2 else 0)
            )
            comptime assert (
                Self.config.BK_PV != 128 or p_off_bytes % 1024 == 0
            ), "SW128 P stage offset is not 1 KiB aligned"
            comptime if Self.is_cg2:
                comptime assert (
                    (Self.EStageElems * Self.mma_bytes) % 1024 == 0
                    and (Self.KVBlockElems * Self.mma_bytes) % 1024 == 0
                    and p_off_bytes % 1024 == 0
                ), "the pair's V blocks are not 1 KiB aligned"
                # The epilogue's output staging reuses the Z region
                # (`out_smem`): pin the fit, so a staging growth fails
                # here rather than overrunning into the V ring.
                comptime _out_stage_bytes = (
                    DecodeOutProducer[
                        Self.output_type, Self.config
                    ].num_out_stages
                    * DecodeOutProducer[
                        Self.output_type, Self.config
                    ].blocks_per_stage
                    * DecodeOutProducer[
                        Self.output_type, Self.config
                    ].out_stage_elems
                    * size_of[Self.output_type]()
                )
                comptime assert (
                    _out_stage_bytes
                    <= Self.KVStageElems * Self.num_mma_stages * Self.mma_bytes
                ), "the epilogue's output staging no longer fits the Z ring"
            debug_assert(
                (Int(q_smem) & 1023) == 0,
                "dynamic SMEM base is not 1 KiB aligned",
            )

        # Output SMEM reuses the SW64 KV region (same as the single-buffer
        # native FP8 kernel and the BF16 kernel).
        var out_smem = kv_smem.unsafe_bitcast[Scalar[Self.output_type]]()

        var max_smem = (
            e_smem.unsafe_offset(
                Self.e_ring_elems + Self.p_ring_elems
            ) if Self.is_cg2 else kv_smem.unsafe_offset(
                Self.KVStageElems * Self.num_mma_stages + Self.p_ring_elems
            )
        ).unsafe_bitcast[Scalar[Self.AccumType]]()
        var li_smem = max_smem.unsafe_offset(2 * WARPGROUP_SIZE)
        # One e8m0 scale slot per K/V stage (`kv_convert`, else empty).
        var scale_smem = (max_smem.unsafe_offset(Self.ml_elems)).unsafe_bitcast[
            UInt8
        ]()

        var mbar_base: MBarType = (
            (scale_smem.unsafe_offset(Self.scale_ring_bytes))
            .unsafe_bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        # The pair's barrier carve (see `num_carve_mbars_pair` for the
        # layout). Z-full, the S/P full barriers and the O-free barrier
        # live on the leader alone; everything else is per CTA, arrived
        # by the leader's multicast MMA commits.
        # The carve count (`num_carve_mbars_pair`) gives every pair ring
        # `num_stages` stages.
        comptime assert (
            Self.z_stages == Self.num_stages
        ), "the pair's Z ring must be as deep as its other rings"
        var z_tx = mbar_base.unsafe_offset(1)
        var z_empty = z_tx.unsafe_offset(Self.z_stages)
        var z_full = z_empty.unsafe_offset(Self.z_stages)
        # The stride from `z_full` to the S ring: z_full's own span plus
        # the V ring's v_tx/v_full/v_free triple (D each).
        comptime _pair_stage_bars = 4 * Self.num_stages
        var v_tx = z_full.unsafe_offset(Self.z_stages)
        var v_full = v_tx.unsafe_offset(Self.num_stages)
        var v_free = v_full.unsafe_offset(Self.num_stages)
        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=2 * 4,
        ](z_full.unsafe_offset(_pair_stage_bars))
        var p_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=2 * 4,
            num_consumer=1,
        ](s_bars.end())
        var idx_bars = Self.IdxBars(p_bars.end())
        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=2 * 4
        ](idx_bars.end())
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](o_bars.end())
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](c_bars.end())
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
            cta_group=Self.config.cta_group,
        ](corr_done_bars.end())
        var mbar_end: MBarType = corr_done_bars.end().unsafe_offset(
            out_pipeline.num_mbars()
        )
        # The layout above must end exactly at the carve's barrier count:
        # a base that disagrees with the count is silent corruption. mbar_q,
        # the two Z frees, the V ring stride, the S/P/index rings, then the
        # O/C/corr_done/out pipelines.
        comptime assert (
            1
            + 2 * Self.z_stages
            + _pair_stage_bars
            + 4 * Self.num_stages
            + 2 * Self.idx_stages
            + 4
            + 2
            + 4
            + Int(out_pipeline.num_mbars())
            == Self.num_carve_mbars_pair()
        ), "the paired barrier layout disagrees with the carve count"
        var ptr_tmem_addr = mbar_end.unsafe_bitcast[UInt32]()
        var idx_smem_base = (ptr_tmem_addr.unsafe_offset(1)).unsafe_bitcast[
            Int32
        ]()
        comptime if Self.idx_smem_align_pad > 0:
            idx_smem_base = idx_smem_base.unsafe_offset(
                ((16 - (Int(idx_smem_base) & 15)) & 15) // 4
            )
        # Never used under the pair (`kv_convert` is excluded); the
        # index producer only reads it through the scale staging path.
        var kv_pipeline_gather_dummy = Self.GatherPipe(v_free)

        var warp_idx_c2 = UInt32(warp_id[broadcast=True]())
        var rank = block_rank_in_cluster()
        if warp_idx_c2 == 8:
            if elect() != 0:
                mbar_q[].init(1)
                comptime for i in range(Self.z_stages):
                    # This CTA's own gather completion: four warps'
                    # expect-arrives plus their TMA bytes.
                    z_tx[i].init(
                        Int32(4 + (WARPGROUP_SIZE if Self.is_nvfp4_kv else 0))
                    )
                    # Freed by the leader's PV multicast commit.
                    z_empty[i].init(1)
                    # One arrival per CTA once its Z landed: warp 8
                    # forwards it (`forward_pair_landings`), or WG3 under
                    # NVFP4 after its dequant.
                    z_full[i].init(2)
                comptime for i in range(Self.num_stages):
                    # This CTA's V gather completion, like z_tx.
                    v_tx[i].init(
                        Int32(4 + (WARPGROUP_SIZE if Self.is_nvfp4_kv else 0))
                    )
                    # One arrival per CTA once its V landed, like
                    # z_full; the leader's PV waits it.
                    v_full[i].init(2)
                    # Freed by the leader's PV multicast commit.
                    v_free[i].init(1)
                s_bars.init()
                p_bars.init()
                idx_bars.init()
                o_bars.init()
                c_bars.init()
                corr_done_bars.init()
                out_pipeline.init()
                stamp_cta_event_here(CTA_INITS_UNFENCED)
                fence_mbarrier_init()
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
                comptime if Self.has_extra_kv:
                    extra_k_tma.prefetch_descriptor()
                stamp_cta_event_here(CTA_INITS)
        elif warp_idx_c2 == 9:
            tcgen05_alloc[Int32(Self.config.cta_group)](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
            stamp_cta_event_warp(CTA_ALLOC)
        # Zero the NoPE tail before the warpgroups read it. The fence
        # publishes the generic-proxy stores to the async proxy for the
        # tcgen05 read.
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            Self.zero_input_tail(
                q_smem.as_unsafe_any_origin(),
                kv_smem.as_unsafe_any_origin(),
            )
        stamp_cta_event(CTA_ZEROED)
        stamp_park(stamp_addr)
        barrier()
        stamp_cta_event(CTA_SETUP)
        stamp_cta_keys(offset_position.num_keys)
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            fence_async_view_proxy()
        # Order the pair: the barrier inits must be visible before either
        # CTA arrives on a peer barrier. All 16 warps arrive here, before
        # the role split. No release is needed: the mbarrier-init fence
        # publishes the inits, the pair's forward arrives publish
        # everything else the peer reads, and the NoPE tail zeros reach the
        # leader's MMA through the barrier above and the first z_full
        # forward, which the leader's QK waits. The warps whose roles touch
        # no peer state wait at the end of their roles instead
        # (`defers_cluster_wait`).
        cluster_arrive_relaxed()
        if not Self.defers_cluster_wait(warp_idx_c2):
            cluster_wait()
        stamp_cta_event(CTA_SYNC_IN)

        comptime num_softmax_warps = 4
        if warp_idx_c2 < UInt32(num_softmax_warps):
            warpgroup_reg_alloc[num_reg_softmax]()

            var attn_sink_log2 = Float32(min_or_neg_inf[.float32]())
            comptime if Self.has_attn_sink:
                var row = thread_idx.x & (Self.config.BM - 1)
                var head_idx_local = (
                    decode_head_tile[Self.config]() * Self.config.BM + row
                )
                if head_idx_local < Self.config.num_q_heads:
                    attn_sink_log2 = attn_sink_ptr.value()[head_idx_local].cast[
                        DType.float32
                    ]() * Float32(log2e)

            comptime if Self.MaskType.get_type_name() == "CausalMask" and not Self.fold_shared_index:
                if not logical_indices:
                    offset_position.assert_row_packed[Self.sparse_row](
                        d_indices, indices_stride, selection_lag
                    )
            var sm_offset_position = offset_position
            sm_offset_position.sparse_topk = Int(
                opaque_nonneg_i32(Int32(offset_position.sparse_topk))
            )
            Self.Common_MLA_Op.Softmax[
                native_fp8=Self.mma_bytes == 1,
                mask_physical_sentinels=Self.skip_masked_tiles,
                bf16_p_ring=Self.mma_bytes == 2,
                num_sp_stages=Self.num_stages,
                num_barrier_warps=2 * 4,
                has_attn_sink=Self.has_attn_sink,
                fold_q=Self.fold_shared_index,
                q_len_fold=Self.q_len_fold,
                sparse_row=Self.sparse_row,
            ](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                p_smem.unsafe_bitcast[
                    Scalar[Self.Common_MLA_Op.q_type]
                ]().as_unsafe_any_origin(),
                max_smem.as_unsafe_any_origin(),
                li_smem.as_unsafe_any_origin(),
                out_smem.as_unsafe_any_origin(),
                c_bars,
                corr_done_bars,
                out_pipeline,
                sm_offset_position,
                scale,
                mask,
                prompt_idx=UInt32(offset_position.batch_idx),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
                attn_sink_log2=attn_sink_log2,
                logical_indices=logical_indices,
                physical_sentinel_indices=d_indices,
                logical_indices_stride=indices_stride,
                logical_indices_len=sm_offset_position.sparse_topk,
                selection_lag=selection_lag,
            )
        elif warp_idx_c2 >= 4 and warp_idx_c2 < 8:
            comptime if num_reg_correction < 128:
                warpgroup_reg_dealloc[num_reg_correction]()
            else:
                warpgroup_reg_alloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction[num_o_warps=2 * 4](
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        elif warp_idx_c2 >= 8 and warp_idx_c2 < 12:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx_c2 == 8:
                Self.load_q_pair(
                    q_tma,
                    q_smem.as_unsafe_any_origin(),
                    mbar_q,
                    rank,
                    offset_position,
                )
                comptime if not Self.is_nvfp4_kv:
                    Self.forward_pair_landings(
                        z_tx, z_full, v_tx, v_full, offset_position
                    )
            elif warp_idx_c2 == 9:
                if rank == 0:
                    Self.mmaQK_pair(
                        ptr_tmem_addr[0],
                        q_smem.as_unsafe_any_origin(),
                        kv_smem.as_unsafe_any_origin(),
                        mbar_q,
                        z_full,
                        z_empty,
                        s_bars,
                        offset_position,
                    )
            elif warp_idx_c2 == 10:
                if rank == 0:
                    Self.mmaPV_pair(
                        ptr_tmem_addr[0],
                        e_smem.as_unsafe_any_origin(),
                        p_smem.as_unsafe_any_origin(),
                        p_bars,
                        o_bars,
                        v_full,
                        v_free,
                        offset_position,
                    )
            elif warp_idx_c2 == 11:
                stamp_cta_event_warp(CTA_INDEX_START)
                var batch_d_indices_w11 = (
                    d_indices.unsafe_value().unsafe_offset(
                        (offset_position.q_token_idx * indices_stride)
                    )
                )
                var batch_extra_d_indices_w11 = extra_d_indices
                comptime if Self.has_extra_kv:
                    batch_extra_d_indices_w11 = (
                        extra_d_indices.unsafe_value().unsafe_offset(
                            (offset_position.q_token_idx * extra_indices_stride)
                        )
                    )
                Self.idx_producer(
                    idx_bars,
                    idx_smem_base,
                    kv_lut,
                    batch_d_indices_w11,
                    topk,
                    offset_position,
                    num_orig_blocks,
                    extra_kv_lut,
                    batch_extra_d_indices_w11,
                    extra_topk,
                    kv_pipeline_gather_dummy,
                    scale_smem.as_unsafe_any_origin(),
                    scales_ptr,
                    extra_scales_ptr,
                    topk_stride,
                )
                Self.Common_MLA_Op.store[
                    fold_q=Self.fold_shared_index,
                    q_len_fold=Self.q_len_fold,
                ](
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    offset_position,
                )
        else:
            comptime if num_reg_reswizzle > 128:
                warpgroup_reg_alloc[num_reg_reswizzle]()
            else:
                warpgroup_reg_dealloc[num_reg_reswizzle]()
            comptime if Self.is_nvfp4_kv:
                Self.gather_pair_nvfp4(
                    k_tma,
                    packed_v_tma,
                    kv_smem.as_unsafe_any_origin(),
                    e_smem.as_unsafe_any_origin(),
                    z_tx,
                    z_empty,
                    z_full,
                    v_tx,
                    v_free,
                    v_full,
                    idx_bars,
                    idx_smem_base,
                    scales_ptr,
                    rank,
                    offset_position.num_keys_this_split,
                )
            else:
                Self.gather_pair(
                    k_tma,
                    kv_smem.as_unsafe_any_origin(),
                    e_smem.as_unsafe_any_origin(),
                    z_tx,
                    z_empty,
                    v_tx,
                    v_free,
                    idx_bars,
                    idx_smem_base,
                    topk,
                    extra_k_tma,
                    rank,
                    offset_position,
                )
        if Self.defers_cluster_wait(warp_idx_c2):
            cluster_wait()
        # Warp 9's tail (its last-phase waits, then the TMEM dealloc) holds
        # the CTA's retire, so it stages one barrier stage per lane, and the
        # TMEM base, in registers while the CTA barrier waits, then polls
        # them all at once.
        var tail_wait = UInt32(0)
        var tail_tmem = UInt32(0)
        if warp_idx_c2 == 9:
            tail_wait = opaque_u32(
                Self.pair_tail_lane_wait(
                    z_empty,
                    v_free,
                    s_bars,
                    p_bars,
                    o_bars,
                    offset_position.num_keys_this_split,
                )
            )
            tail_tmem = opaque_u32(ptr_tmem_addr[0])
        barrier()
        stamp_cta_event(CTA_ROLES_DONE)

        comptime if _pdl_on:
            launch_dependent_grids()

        # Peer SMEM must outlive every remote write and arrive, so before
        # the pair's TMEM dealloc the dealloc warp observes every inbound
        # arrive's last phase (CUTLASS's producer tail). Those include the
        # leader's last QK and PV commits, which complete only after their
        # MMAs have written this CTA's TMEM; the observation is a thread
        # sync, so `tcgen05.fence::after_thread_sync` orders the dealloc
        # after it.
        if warp_idx_c2 == 9:
            _wait_parity_word(tail_wait)
            syncwarp()
            tcgen05_fence_after()
            stamp_cta_late(CTA_LATE_INBOUND)
        if warp_idx_c2 == 9:
            tcgen05_release_allocation_lock[Int32(Self.config.cta_group)]()
            stamp_cta_late(CTA_LATE_RELEASED)
            tcgen05_dealloc[Int32(Self.config.cta_group)](
                tail_tmem, Self.config.sm100_tmem_cols
            )
            stamp_cta_late(CTA_LATE_DEALLOC)
        stamp_cta_exit()
        return

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(extra_k_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    def kernel(
        q_tma: QOTMATile[
            dtype=Self.mma_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.operand_swizzle,
        ],
        # The K/V gather4 TMA: one INT64 SWIZZLE_NONE box per row under the
        # linear front end, SW128 column blocks under the direct gather.
        k_tma: Self.KTMA,
        o_tma: ORaggedTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType=Self.ValidLengthType,
            MaskType=Self.MaskType,
            SplitAccumType=Self.SplitAccumType,
            prepared_bounds=Self.prepared_bounds,
        ],
        d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        indices_stride_dev: Int32,
        # Logical sparse indices (same layout as `d_indices`, -1 padding) for
        # position-based causal masking. See mla_decode_utils.mojo.
        logical_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        topk_lengths: Self.TopkLengthsPtrType,
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
        attn_sink_ptr: Self.AttnSinkPtrType,
        extra_k_tma: Self.KTMA,
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        extra_topk_lengths: Self.TopkLengthsPtrType,
        extra_indices_stride_dev: Int32,
        extra_scales_ptr: OptionalReg[Pointer[Float32, MutAnyOrigin]],
        scalar_args: Self.ScalarArgsType,
        # Positions between the token each sparse row was computed for and
        # the token that reads it.
        selection_lag_dev: Int32,
        stamp_addr: UInt64,
    ):
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_Sparse_QKV_FP8 supports NullMask and CausalMask"
            " only. Sliding-window is not yet wired for sparse."
        )
        comptime assert (
            not Self.fold_shared_index
            or Self.config.num_q_heads * Self.q_len_fold <= Self.config.BM
        ), "fold_shared_index requires num_q_heads * q_len_fold <= BM."
        comptime assert not Self.config.decode_layout_g or (
            Self.config.MMA_M == 32 and Self.config.BN_QK == 64
        ), "Sparse Layout G requires MMA_M == 32 and BN_QK == 64."
        comptime assert not Self.direct_gather or (
            Self.config.decode_layout_g
            or Self.config.cta_group == 2
            or Self.mma_bytes == 2
            or (
                (_MLA_SPARSE_DIRECT_GATHER_E != 0 or Self.config.BN_QK == 128)
                and not Self.p_in_kv
            )
        ), (
            "The FP8 direct gather needs Layout G, the pair, or a separate P"
            " ring."
        )
        comptime assert (
            not Self.is_cg2 or _MLA_SPARSE_CG2 != 0
        ), "the CTA pair is behind MLA_SPARSE_CG2"
        comptime assert not Self.is_cg2 or (
            Self.config.cta_group == 2
            and not Self.config.decode_layout_g
            and not Self.kv_convert
            and not Self.p_in_kv
            and not Self.fold_shared_index
            and Self.config.num_q_heads == 2 * Self.config.BM
        ), "the pair needs the FP8 Layout E direct-gather form at 128 heads"
        comptime assert not Self.p_in_kv or (
            Self.mma_bytes == 2
        ), "P in the K/V stage needs the rope block of a 2-byte operand"
        comptime assert (
            Self.config.operand_size == Self.mma_bytes
        ), "the config's operand size is not Q's"
        comptime assert not Self.kv_convert or (
            (Self.mma_bytes == 2 or Self.nvfp4_ring)
            and size_of[Self.kv_type]() == 1
            and not Self.direct_gather
            and Self.num_stages == Self.num_mma_stages
        ), "kv_convert converts an FP8 or NVFP4 ring in place"
        comptime assert not Self.p_in_kv or (
            Self.num_stages == Self.num_mma_stages
        ), "P in the K/V stage needs the S/P and K/V rings in lockstep"
        comptime assert not (
            Self.fold_shared_index and Self.has_extra_kv
        ), "fold_shared_index does not support extra always-attend KV cache."
        comptime assert not (
            Self.is_nvfp4_kv and Self.has_extra_kv
        ), "an NVFP4 KV cache does not support an extra always-attend KV cache."
        comptime assert (
            not Self.is_nvfp4_kv or Self.physical_indices
        ), "an NVFP4 KV cache needs the indexer's physical and scale rows"
        comptime assert not Self.is_nvfp4_kv or (
            Self.mma_bytes == 1 and (not Self.direct_gather or Self.is_cg2)
        ), "an NVFP4 KV cache takes the FP8 linear front end or ring"
        comptime assert not Self.is_nvfp4_kv or (
            (Self.config.BN_QK == 64 or Self.config.BN_QK == 128)
            and Self.config.input_q_depth % 64 == 0
            and Self.nvfp4_sf_offset % 16 == 0
            and Self.nvfp4_sf_offset
            + Self.config.BN_QK * Self.nvfp4_sf_row_bytes
            <= Self.LinStageElems
            or Self.is_cg2
        ), (
            "NVFP4 staging requires 64- or 128-row packed tiles whose scales"
            " fit a stage."
        )

        # The converter holds a row's overlay blocks across its barrier, so
        # it takes the correction warpgroup's share. Each warpgroup grows
        # or shrinks from the 128-register launch share, so which of
        # `setmaxnreg.inc` and `.dec` it issues follows its budget.
        comptime num_reg_softmax = 192
        # The linear NVFP4 dequant holds a row's packed chunks across their
        # loads; below 80 registers WG3 spills, and up to 152 it still gains
        # (measured on B200), while correction needs 72.
        comptime nvfp4_linear = Self.is_nvfp4_kv and not Self.kv_convert
        comptime num_reg_correction = (
            72 if Self.kv_convert else (104 if nvfp4_linear else 184)
        )
        comptime num_reg_other = 64
        comptime num_reg_reswizzle = (
            184 if Self.kv_convert else (152 if nvfp4_linear else 72)
        )
        comptime assert (
            num_reg_softmax
            + num_reg_correction
            + num_reg_other
            + num_reg_reswizzle
            <= 512
        ), "the warpgroup register budgets overrun the register file"

        stamp_cta_entry()
        comptime _pdl_on = Self.config.decoding_warp_split_k or _MLA_PDL_ALWAYS
        var indices_stride = Int(indices_stride_dev)
        var topk_stride = indices_stride // Self.sparse_row_parts
        var extra_indices_stride = Int(extra_indices_stride_dev)
        var batch_size: Int
        var q_max_seq_len: Int
        comptime if Self.prepared_bounds:
            batch_size = mla_decode_pack.batch_size()
            q_max_seq_len = mla_decode_pack.q_max_seq_len()
        else:
            var dynamic_args = rebind[
                TileTensor[
                    DType.int64,
                    RowMajorLayout[ComptimeInt[3]],
                    MutAnyOrigin,
                    Engine=Self.Engine,
                ]
            ](scalar_args)
            batch_size = Int(dynamic_args[0])
            q_max_seq_len = Int(dynamic_args[1])
        var selection_lag = Int(selection_lag_dev)
        var num_partitions = Int(mla_decode_pack.num_partitions)
        var mask = mla_decode_pack.mask
        var valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr

        decode_pdl_wait[_pdl_on]()
        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ](
            kv_lut,
            rebind[
                Pointer[
                    Scalar[Self.ValidLengthType.dtype],
                    ImmutAnyOrigin,
                    address_space=.GENERIC,
                ]
            ](valid_length.value()),
            q_max_seq_len,
            num_partitions,
            batch_size,
            sparse_indices_stride=topk_stride,
            sparse_topk_lengths=topk_lengths,
            sparse_extra_indices_stride=extra_indices_stride,
            sparse_extra_topk_lengths=extra_topk_lengths,
            topk_lengths_per_token=Self.skip_masked_tiles,
        )

        comptime if Self.skip_masked_tiles:
            comptime assert (
                not Self.fold_shared_index
            ), "skip_masked_tiles bounds one token per CTA"
            offset_position.skip_masked_tiles[
                causal=Self.MaskType.get_type_name() == "CausalMask"
            ](num_partitions, indices_stride, topk_lengths)

        var extra_topk: Int = 0
        comptime if Self.has_extra_kv:
            comptime if Self.has_variable_topk:
                extra_topk = Int(
                    extra_topk_lengths.value()[Int(offset_position.batch_idx)]
                )
            else:
                extra_topk = extra_indices_stride
        # Main-segment slot count: the clamped topk, tile-padded when extra KV
        # follows it (see `OffsetPosition.sparse_topk`).
        var topk = offset_position.num_keys - extra_topk

        @inline(.always)
        def _role_position[
            use_metadata: Bool = True
        ](
            role_batch_size: Int,
            role_metadata: SharedMemPointer[Int64],
        ) {
            imm
        } -> type_of(offset_position):
            comptime if Self.role_remat and use_metadata:
                var pos = offset_position
                var slot = Int(llvm_opaque_tid() >> 7) * 2
                var token_base = Int(
                    role_metadata.unsafe_load[volatile=True, invariant=False](
                        slot
                    )
                )
                pos.q_token_idx = token_base + pos.seq_idx
                pos.q_row_offset = (
                    token_base * Self.config.num_q_heads
                    + decode_head_tile[Self.config]() * Self.config.BM
                    + pos.seq_idx * Self.config.num_q_heads
                )
                pos.num_keys_this_split = Int(
                    role_metadata.unsafe_load[volatile=True, invariant=False](
                        slot + 1
                    )
                )
                return pos
            else:
                var role_stride = Int(opaque_i64(Int64(indices_stride_dev)))
                var role_np = Int(
                    opaque_i64(Int64(mla_decode_pack.num_partitions))
                )
                var pos = OffsetPosition[
                    Self.config,
                    Self.KVLUTType,
                    Self.ragged,
                    Self._is_cache_length_accurate,
                    Self.ValidLengthType,
                    Self.config.decoding_warp_split_k,
                    sparse=True,
                    has_extra_kv=Self.has_extra_kv,
                    TopkLengthsPtrType=Self.TopkLengthsPtrType,
                ](
                    kv_lut,
                    rebind[
                        Pointer[
                            Scalar[Self.ValidLengthType.dtype],
                            ImmutAnyOrigin,
                            address_space=.GENERIC,
                        ]
                    ](valid_length.value()),
                    q_max_seq_len,
                    role_np,
                    role_batch_size,
                    sparse_indices_stride=role_stride // Self.sparse_row_parts,
                    sparse_topk_lengths=topk_lengths,
                    sparse_extra_indices_stride=extra_indices_stride,
                    sparse_extra_topk_lengths=extra_topk_lengths,
                    topk_lengths_per_token=Self.skip_masked_tiles,
                    role_block_x=_role_block_x(),
                    role_block_z=_role_block_z(),
                )
                comptime if Self.skip_masked_tiles:
                    pos.skip_masked_tiles[
                        causal=Self.MaskType.get_type_name() == "CausalMask"
                    ](role_np, role_stride, topk_lengths)
                return pos

        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)

        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                Self.pdl_early_exit_all_q(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                )
                return

        comptime if Self.ragged and (
            Self.config.decoding_warp_split_k or _MLA_RAGGED_EXIT_BREAK != 2
        ):
            if offset_position.seq_idx >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    Self.pdl_early_exit_all_q(
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                    )
                return

        comptime if Self.skip_masked_tiles and not Self.config.decoding_warp_split_k:
            if offset_position.num_keys == 0:
                Self.Common_MLA_Op.store_empty(
                    o_tma, offset_position.out_row_offset
                )
                # With an LSE output (decode context parallelism) an empty
                # shard also writes `-inf`, so a cross-rank merge weights it 0.
                comptime if not Self.SplitAccumType.is_null:
                    Self.Common_MLA_Op.store_empty_lse(
                        offset_position.q_token_idx, lse_accum_split_ptr
                    )
                return

        stamp_cta_event(CTA_OFFSETS)
        var q_smem = external_memory[
            Scalar[Self.mma_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        # Linear (SWIZZLE_NONE) gather destination -- the re-swizzle WG's
        # source, a slot the size of an SW64 stage. Empty under the direct
        # gather.
        var kv_smem_linear = q_smem.unsafe_offset(
            Self.BlockElems * Self.NumQSmemBlocks
        )
        # SW64 destination the native FP8 MMA reads -- byte-identical role
        # to the single-buffer kernel's `kv_smem`, one ring deeper.
        comptime kv_linear_elems = Self.kv_linear_elems
        var kv_smem = kv_smem_linear.unsafe_offset(kv_linear_elems)
        comptime if Self.kv_convert:
            kv_smem_linear = kv_smem.unsafe_offset(Self.cvt_src_offset)
        comptime if Self.q_in_tmem:
            comptime assert (
                Self.num_mma_stages > Self.num_stages
            ), "Q in TMEM lands in the extra MMA stage"
            q_smem = kv_smem.unsafe_offset(
                Self.KVStageElems * (Self.num_mma_stages - 1)
            )
        # The pair's V ring sits between the Z ring and the P ring;
        # `out_smem` still reuses the Z region at the epilogue.
        var e_smem = kv_smem.unsafe_offset(
            Self.KVStageElems * Self.num_mma_stages
        )
        var p_smem = e_smem.unsafe_offset(
            Self.e_ring_elems
        ) if Self.is_cg2 else kv_smem.unsafe_offset(
            Self.KVStageElems * Self.num_mma_stages
        )
        comptime if Self.p_in_kv:
            p_smem = kv_smem.unsafe_offset(Self.NumVBlocks * Self.KVBlockElems)
        comptime if Self.kv_sw128:
            # An SW128 operand's swizzle phase comes from its address (the
            # TMA writes and the MMA descriptors assume base offset 0). The
            # offsets are carve constants; the dynamic SMEM base is declared
            # only 128 B aligned, so its check stays a debug assert.
            comptime kv_off_bytes = (
                Self.BlockElems * Self.NumQSmemBlocks + kv_linear_elems
            ) * Self.mma_bytes
            comptime assert (
                kv_off_bytes % 1024 == 0
                and (Self.KVStageElems * Self.mma_bytes) % 1024 == 0
            ), "SW128 K/V stage offset is not 1 KiB aligned"
            comptime p_off_bytes = (
                kv_off_bytes
                + Self.KVStageElems * Self.num_mma_stages * Self.mma_bytes
                + (Self.e_ring_elems * Self.mma_bytes if Self.is_cg2 else 0)
            )
            comptime assert (
                Self.config.BK_PV != 128 or p_off_bytes % 1024 == 0
            ), "SW128 P stage offset is not 1 KiB aligned"
            comptime if Self.is_cg2:
                comptime assert (
                    (Self.EStageElems * Self.mma_bytes) % 1024 == 0
                    and (Self.KVBlockElems * Self.mma_bytes) % 1024 == 0
                    and p_off_bytes % 1024 == 0
                ), "the pair's E blocks are not 1 KiB aligned"
            debug_assert(
                (Int(q_smem) & 1023) == 0,
                "dynamic SMEM base is not 1 KiB aligned",
            )

        # Output SMEM reuses the SW64 KV region (same as the single-buffer
        # native FP8 kernel and the BF16 kernel).
        var out_smem = kv_smem.unsafe_bitcast[Scalar[Self.output_type]]()

        var max_smem = (
            e_smem.unsafe_offset(
                Self.e_ring_elems + Self.p_ring_elems
            ) if Self.is_cg2 else kv_smem.unsafe_offset(
                Self.KVStageElems * Self.num_mma_stages + Self.p_ring_elems
            )
        ).unsafe_bitcast[Scalar[Self.AccumType]]()
        var li_smem = max_smem.unsafe_offset(2 * WARPGROUP_SIZE)
        # One e8m0 scale slot per K/V stage (`kv_convert`, else empty).
        var scale_smem = (max_smem.unsafe_offset(Self.ml_elems)).unsafe_bitcast[
            UInt8
        ]()

        var mbar_base: MBarType = (
            (scale_smem.unsafe_offset(Self.scale_ring_bytes))
            .unsafe_bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        var mbar_kv_gather_base: MBarType = mbar_base.unsafe_offset(1)
        # Load (TMA, 1 leader thread) -> re-swizzle WG (128 threads, all
        # independently release).
        var kv_pipeline_gather = Self.GatherPipe(mbar_kv_gather_base)
        var mbar_kv_mma_base: MBarType = mbar_kv_gather_base.unsafe_offset(
            kv_pipeline_gather.num_mbars()
        )
        # Re-swizzle WG (128 threads, all independently commit) -> mmaQK +
        # mmaPV (2 consumers) -- same consumer shape as the single-buffer
        # kernel's own KV pipeline, just producer side is now a warpgroup
        # instead of a single TMA leader thread.
        var kv_pipeline_mma = Self.MmaPipe(mbar_kv_mma_base)
        var mbar_base2: MBarType = mbar_kv_mma_base.unsafe_offset(
            kv_pipeline_mma.num_mbars()
        )

        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base2)
        var mbar_base3: MBarType = s_bars.end()

        var p_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base3)
        var mbar_base4: MBarType = p_bars.end()

        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](mbar_base4)
        var mbar_base5: MBarType = o_bars.end()

        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base5)
        var mbar_base6: MBarType = c_bars.end()

        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base6)
        var mbar_base7: MBarType = corr_done_bars.end()

        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
            cta_group=Self.config.cta_group,
        ](mbar_base7)
        var mbar_base8: MBarType = mbar_base7.unsafe_offset(
            out_pipeline.num_mbars()
        )

        # idx_bars: warp 11 produces; the gather consumes (warp 8's 32
        # lanes under the linear front end, WG3's four elected lanes under
        # the direct gather). The ring is `idx_stages` deep.
        var idx_bars = Self.IdxBars(mbar_base8)
        var cvt_blk_bars: MBarType = idx_bars.end()
        var mbar_base9: MBarType = cvt_blk_bars.unsafe_offset(
            Self.cvt_blk_mbars
        )

        var ptr_tmem_addr = mbar_base9.unsafe_bitcast[UInt32]()
        var idx_smem_base = (ptr_tmem_addr.unsafe_offset(1)).unsafe_bitcast[
            Int32
        ]()
        comptime if Self.idx_smem_align_pad > 0:
            idx_smem_base = idx_smem_base.unsafe_offset(
                ((16 - (Int(idx_smem_base) & 15)) & 15) // 4
            )
        var role_metadata = (
            idx_smem_base.unsafe_offset(Self.idx_stages * Self.idx_stage_slots)
        ).unsafe_bitcast[Int64]()
        comptime if Self.role_remat:
            comptime assert Self.config.num_threads == 512
            comptime assert (
                Self.idx_stages * Self.idx_stage_slots * size_of[Int32]()
            ) % 16 == 0

        var warp_idx = UInt32(warp_id[broadcast=True]())
        var is_leader = elect() != 0

        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                comptime if not Self.direct_gather:
                    kv_pipeline_gather.init()
                kv_pipeline_mma.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                corr_done_bars.init()
                out_pipeline.init()
                idx_bars.init()
                comptime for i in range(Self.cvt_blk_only_mbars):
                    cvt_blk_bars[i].init(Int32(WARPGROUP_SIZE))
                comptime if Self.q_in_tmem:
                    cvt_blk_bars[Self.q_cp_mbar_idx].init(1)
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
                comptime if Self.has_extra_kv:
                    extra_k_tma.prefetch_descriptor()
                stamp_cta_event_here(CTA_INITS)
        elif warp_idx == 9:
            tcgen05_alloc[Int32(Self.config.cta_group)](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
            stamp_cta_event_warp(CTA_ALLOC)
        # Zero the NoPE tail before the warpgroups read it. The fence publishes
        # the generic-proxy stores to the async proxy for the tcgen05 read.
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            Self.zero_input_tail(
                q_smem.as_unsafe_any_origin(), kv_smem.as_unsafe_any_origin()
            )
        comptime if Self.role_remat:
            if warp_idx == 8 and is_leader:
                # The setup join publishes one immutable record per warpgroup.
                var token_base = Int64(
                    offset_position.q_token_idx - offset_position.seq_idx
                )
                comptime for wg in range(4):
                    role_metadata[wg * 2] = token_base
                    role_metadata[wg * 2 + 1] = Int64(
                        offset_position.num_keys_this_split
                    )
        stamp_cta_event(CTA_ZEROED)
        stamp_park(stamp_addr)
        barrier()
        stamp_cta_event(CTA_SETUP)
        stamp_cta_keys(offset_position.num_keys)
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            fence_async_view_proxy()

        comptime num_softmax_warps = 4
        if warp_idx < UInt32(num_softmax_warps):
            warpgroup_reg_alloc[num_reg_softmax]()

            var role_position = offset_position
            comptime if Self.role_remat:
                role_position = _role_position(batch_size, role_metadata)

            var attn_sink_log2 = Float32(min_or_neg_inf[.float32]())
            comptime if Self.has_attn_sink:
                # The softmax row this thread owns; lane_id() would fold rows
                # 32-63 onto 0-31 and give them the wrong heads' sinks.
                var row: Int
                comptime if Self.half_row_softmax:
                    row = hr_row(thread_idx.x & 31)
                else:
                    row = thread_idx.x & (Self.config.BM - 1)
                var head_idx_local = (
                    decode_head_tile[Self.config]() * Self.config.BM + row
                )
                if head_idx_local < Self.config.num_q_heads:
                    attn_sink_log2 = attn_sink_ptr.value()[head_idx_local].cast[
                        DType.float32
                    ]() * Float32(log2e)

            comptime if Self.MaskType.get_type_name() == "CausalMask" and not Self.fold_shared_index:
                if not logical_indices:
                    role_position.assert_row_packed[Self.sparse_row](
                        d_indices, indices_stride, selection_lag
                    )
            # sparse_topk is computed before the role split and shared with
            # the other roles; that copy lives in a stack slot that the
            # logical-index path reloads every tile. Re-deriving it here
            # keeps it in softmax registers.
            var sm_role_position = role_position
            comptime if not Self.role_remat:
                sm_role_position.sparse_topk = Int(
                    opaque_nonneg_i32(Int32(role_position.sparse_topk))
                )
            comptime if Self.role_split_remat:
                sm_role_position.num_keys = opaque_split_bound(
                    role_position.num_keys
                )
                sm_role_position.num_keys_this_split = opaque_split_bound(
                    role_position.num_keys_this_split
                )
                sm_role_position.kv_start_row = opaque_split_bound(
                    role_position.kv_start_row
                )
            Self.Common_MLA_Op.Softmax[
                native_fp8=Self.mma_bytes == 1,
                mask_physical_sentinels=Self.skip_masked_tiles,
                bf16_p_ring=Self.mma_bytes == 2 and not Self.p_in_kv,
                num_sp_stages=Self.num_stages,
                has_attn_sink=Self.has_attn_sink,
                fold_q=Self.fold_shared_index,
                q_len_fold=Self.q_len_fold,
                sparse_row=Self.sparse_row,
                corr_smem=Self.corr_smem,
            ](
                _tmem_base(ptr_tmem_addr),
                s_bars,
                p_bars,
                (kv_smem if Self.p_in_kv else p_smem)
                .unsafe_bitcast[Scalar[Self.Common_MLA_Op.q_type]]()
                .as_unsafe_any_origin(),
                max_smem.as_unsafe_any_origin(),
                li_smem.as_unsafe_any_origin(),
                out_smem.as_unsafe_any_origin(),
                c_bars,
                corr_done_bars,
                out_pipeline,
                sm_role_position,
                scale,
                mask,
                prompt_idx=UInt32(role_position.batch_idx),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
                attn_sink_log2=attn_sink_log2,
                logical_indices=logical_indices,
                physical_sentinel_indices=d_indices,
                logical_indices_stride=indices_stride,
                logical_indices_len=sm_role_position.sparse_topk,
                selection_lag=selection_lag,
                corr_scale_smem=(
                    li_smem.unsafe_offset(WARPGROUP_SIZE)
                ).as_unsafe_any_origin(),
            )
        elif warp_idx >= 4 and warp_idx < 8:
            comptime if num_reg_correction < 128:
                warpgroup_reg_dealloc[num_reg_correction]()
            else:
                warpgroup_reg_alloc[num_reg_correction]()
            var role_position = offset_position
            comptime if Self.role_split_remat:
                role_position.num_keys_this_split = opaque_split_bound(
                    role_position.num_keys_this_split
                )
                role_position.kv_start_row = opaque_split_bound(
                    role_position.kv_start_row
                )
            comptime if Self.role_remat:
                role_position = _role_position(batch_size, role_metadata)
                # The record holds the setup's key count, already bounded by
                # the index stride, so this clamp never changes it. It
                # restates the range for the compiler: `corr_countdown`
                # relies on it to keep the trip count in an Int32.
                role_position.num_keys_this_split = min(
                    max(role_position.num_keys_this_split, Int(0)),
                    max(
                        Int(indices_stride_dev) // Self.sparse_row_parts,
                        Int(0),
                    ),
                )
            Self.Common_MLA_Op.Correction[
                corr_smem=Self.corr_smem,
                phase_from_o=Self.corr_phase_from_o,
                countdown=Self.corr_countdown,
            ](
                _tmem_base(ptr_tmem_addr),
                o_bars,
                c_bars,
                corr_done_bars,
                role_position,
                corr_scale_smem=(
                    li_smem.unsafe_offset(WARPGROUP_SIZE)
                ).as_unsafe_any_origin(),
            )
        elif warp_idx >= 8 and warp_idx < 12:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                var role_position = offset_position
                comptime if Self.role_split_remat:
                    role_position.num_keys = opaque_split_bound(
                        role_position.num_keys
                    )
                    role_position.num_keys_this_split = opaque_split_bound(
                        role_position.num_keys_this_split
                    )
                    role_position.kv_start_row = opaque_split_bound(
                        role_position.kv_start_row
                    )
                comptime if Self.role_remat:
                    role_position = _role_position(batch_size, role_metadata)
                var role_topk = topk
                var role_num_orig_blocks = num_orig_blocks
                comptime if Self.role_split_remat:
                    role_topk = opaque_split_bound(role_topk)
                    role_num_orig_blocks = ceildiv(role_topk, Self.config.BN_QK)
                comptime if Self.role_remat:
                    role_topk = role_position.num_keys
                    role_num_orig_blocks = ceildiv(role_topk, Self.config.BN_QK)
                comptime if Self.direct_gather:
                    Self.load_q(
                        q_tma,
                        q_smem.as_unsafe_any_origin(),
                        mbar_q,
                        role_position,
                    )
                else:
                    Self.load(
                        q_tma,
                        k_tma,
                        kv_lut,
                        q_smem.as_unsafe_any_origin(),
                        kv_smem_linear.as_unsafe_any_origin(),
                        mbar_q,
                        kv_pipeline_gather,
                        role_position,
                        idx_bars,
                        idx_smem_base,
                        role_num_orig_blocks,
                        role_topk,
                        extra_k_tma,
                        extra_topk,
                        scales_ptr,
                    )
            elif warp_idx == 9:
                var role_position = offset_position
                comptime if Self.role_split_remat:
                    role_position.num_keys = opaque_split_bound(
                        role_position.num_keys
                    )
                    role_position.num_keys_this_split = opaque_split_bound(
                        role_position.num_keys_this_split
                    )
                    role_position.kv_start_row = opaque_split_bound(
                        role_position.kv_start_row
                    )
                comptime if Self.role_remat:
                    role_position = _role_position(batch_size, role_metadata)
                Self.mmaQK(
                    _tmem_base(ptr_tmem_addr),
                    q_smem.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline_mma,
                    kv_pipeline_gather,
                    cvt_blk_bars,
                    role_position,
                )
            elif warp_idx == 10:
                var role_position = offset_position
                comptime if Self.role_split_remat:
                    role_position.num_keys = opaque_split_bound(
                        role_position.num_keys
                    )
                    role_position.num_keys_this_split = opaque_split_bound(
                        role_position.num_keys_this_split
                    )
                    role_position.kv_start_row = opaque_split_bound(
                        role_position.kv_start_row
                    )
                comptime if Self.role_remat:
                    role_position = _role_position(batch_size, role_metadata)
                Self.mmaPV(
                    _tmem_base(ptr_tmem_addr),
                    kv_smem.as_unsafe_any_origin(),
                    p_smem.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline_mma,
                    kv_pipeline_gather,
                    role_position,
                )
            elif warp_idx == 11:
                var role_position = offset_position
                comptime if Self.role_split_remat:
                    role_position.num_keys = opaque_split_bound(
                        role_position.num_keys
                    )
                    role_position.num_keys_this_split = opaque_split_bound(
                        role_position.num_keys_this_split
                    )
                    role_position.kv_start_row = opaque_split_bound(
                        role_position.kv_start_row
                    )
                comptime if Self.role_remat:
                    role_position = _role_position(batch_size, role_metadata)
                var role_topk = topk
                var role_num_orig_blocks = num_orig_blocks
                comptime if Self.role_split_remat:
                    role_topk = opaque_split_bound(role_topk)
                    role_num_orig_blocks = ceildiv(role_topk, Self.config.BN_QK)
                comptime if Self.role_remat:
                    role_topk = role_position.num_keys
                    role_num_orig_blocks = ceildiv(role_topk, Self.config.BN_QK)
                stamp_cta_event_warp(CTA_INDEX_START)
                var batch_d_indices_w11 = (
                    d_indices.unsafe_value().unsafe_offset(
                        (role_position.q_token_idx * indices_stride)
                    )
                )
                var batch_extra_d_indices_w11 = extra_d_indices
                comptime if Self.has_extra_kv:
                    batch_extra_d_indices_w11 = (
                        extra_d_indices.unsafe_value().unsafe_offset(
                            (role_position.q_token_idx * extra_indices_stride)
                        )
                    )
                Self.idx_producer(
                    idx_bars,
                    idx_smem_base,
                    kv_lut,
                    batch_d_indices_w11,
                    role_topk,
                    role_position,
                    role_num_orig_blocks,
                    extra_kv_lut,
                    batch_extra_d_indices_w11,
                    extra_topk,
                    kv_pipeline_gather,
                    scale_smem.as_unsafe_any_origin(),
                    scales_ptr,
                    extra_scales_ptr,
                    topk_stride,
                )
                Self.Common_MLA_Op.store[
                    fold_q=Self.fold_shared_index,
                    q_len_fold=Self.q_len_fold,
                ](
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    role_position,
                )
        else:
            # WG3 (warps 12-15): the K/V front end -- the direct gather, the
            # re-swizzle into SW64, or the in-place convert.
            comptime if num_reg_reswizzle > 128:
                warpgroup_reg_alloc[num_reg_reswizzle]()
            else:
                warpgroup_reg_dealloc[num_reg_reswizzle]()
            var role_position = offset_position
            comptime if Self.role_split_remat:
                role_position.num_keys_this_split = opaque_split_bound(
                    role_position.num_keys_this_split
                )
                role_position.kv_start_row = opaque_split_bound(
                    role_position.kv_start_row
                )
            comptime if Self.role_remat:
                role_position = _role_position[use_metadata=False](
                    batch_size, role_metadata
                )
            comptime if Self.direct_gather:
                Self.gather_sw64(
                    k_tma,
                    kv_smem.as_unsafe_any_origin(),
                    kv_pipeline_mma,
                    role_position,
                    idx_bars,
                    idx_smem_base,
                    topk,
                    extra_k_tma,
                )
            elif Self.nvfp4_ring:
                comptime if Self.config.BN_QK == 128:
                    Self.convertNVFP4Wide(
                        kv_smem_linear.as_unsafe_any_origin(),
                        kv_smem.as_unsafe_any_origin(),
                        kv_pipeline_gather,
                        kv_pipeline_mma,
                        cvt_blk_bars,
                        role_position,
                    )
                else:
                    Self.convertNVFP4(
                        kv_smem_linear.as_unsafe_any_origin(),
                        kv_smem.as_unsafe_any_origin(),
                        kv_pipeline_gather,
                        kv_pipeline_mma,
                        cvt_blk_bars,
                        role_position,
                    )
            elif Self.kv_convert:
                Self.convertFP8(
                    kv_smem_linear.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    scale_smem.as_unsafe_any_origin(),
                    kv_pipeline_gather,
                    kv_pipeline_mma,
                    cvt_blk_bars,
                    role_position,
                )
            elif Self.is_nvfp4_kv:
                Self.reswizzleNVFP4(
                    kv_smem_linear.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    kv_pipeline_gather,
                    kv_pipeline_mma,
                    cvt_blk_bars,
                    role_position,
                )
            else:
                Self.reswizzleFP8(
                    kv_smem_linear.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    kv_pipeline_gather,
                    kv_pipeline_mma,
                    cvt_blk_bars,
                    role_position,
                )
        barrier()
        stamp_cta_event(CTA_ROLES_DONE)

        comptime if _pdl_on:
            launch_dependent_grids()

        var dealloc_warp_idx = warp_idx
        comptime if Self.corr_phase_from_o:
            dealloc_warp_idx = llvm_opaque_tid() >> 5
        if dealloc_warp_idx == 9:
            tcgen05_release_allocation_lock[Int32(Self.config.cta_group)]()
            tcgen05_dealloc[Int32(Self.config.cta_group)](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )
            stamp_cta_late(CTA_LATE_DEALLOC)

        stamp_cta_exit()

    @staticmethod
    @inline(.always)
    def _load_raw_indices(
        d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        indices_base: Int,
        topk: UInt32,
    ) -> StaticTuple[Int32, Self.config.BN_QK // 32]:
        """Loads this lane's raw indices of the tile at `indices_base`."""
        var lane: Int
        comptime if Self.role_local_lane:
            lane = Int(llvm_opaque_tid()) & 31
        else:
            lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1
        var raw = StaticTuple[Int32, Self.config.BN_QK // 32]()
        comptime for row_pass in range(Self.config.BN_QK // 32):
            var idx_pos = UInt32(indices_base + lane + row_pass * 32)
            var clamped_pos = min(idx_pos, max_idx)
            raw[row_pass] = d_indices.unsafe_value()[Int(clamped_pos)]
        return raw

    @staticmethod
    @inline(.always)
    def _store_index_rows(
        raw: StaticTuple[Int32, Self.config.BN_QK // 32],
        idx_smem: SharedMemPointer[Int32],
        kv_lut: Self.KVLUTType,
        batch_idx: UInt32,
    ):
        """Translates raw indices to gather rows and stores them."""
        var lane: Int
        comptime if Self.role_local_lane:
            lane = Int(llvm_opaque_tid()) & 31
        else:
            lane = thread_idx.x & 31
        comptime for row_pass in range(Self.config.BN_QK // 32):
            idx_smem[lane + row_pass * 32] = sparse_kv_row[
                Self.physical_indices
            ](kv_lut, batch_idx, raw[row_pass])

    @staticmethod
    @inline(.always)
    def _transform_indices_to_smem(
        d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        idx_smem: SharedMemPointer[Int32],
        indices_base: Int,
        kv_lut: Self.KVLUTType,
        topk: UInt32,
        batch_idx: UInt32,
        scale_rows_offset: Int,
    ):
        var lane: Int
        comptime if Self.role_local_lane:
            lane = Int(llvm_opaque_tid()) & 31
        else:
            lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1

        comptime for row_pass in range(Self.config.BN_QK // 32):
            var row_in_tile = lane + row_pass * 32
            var idx_pos = UInt32(indices_base + row_in_tile)
            var clamped_pos = min(idx_pos, max_idx)
            var raw_index = d_indices.unsafe_value()[Int(clamped_pos)]
            idx_smem[row_in_tile] = sparse_kv_row[Self.physical_indices](
                kv_lut, batch_idx, raw_index
            )
            comptime if Self.is_nvfp4_kv:
                # Already the scale pool's row (`-1` for an unused slot), see
                # `PagedKVScaleRemap`.
                idx_smem[
                    Self.config.BN_QK + row_in_tile
                ] = d_indices.unsafe_value()[
                    scale_rows_offset + Int(clamped_pos)
                ]

    @staticmethod
    @inline(.always)
    def idx_producer(
        idx_bars: Self.IdxBars,
        idx_smem_base: SharedMemPointer[Int32],
        kv_lut: Self.KVLUTType,
        d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        topk: Int,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
        num_orig_blocks: Int,
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[Pointer[Int32, MutAnyOrigin]],
        extra_topk: Int,
        kv_pipeline_gather: Self.GatherPipe,
        scale_smem: SharedMemPointer[UInt8],
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
        extra_scales_ptr: OptionalReg[Pointer[Float32, MutAnyOrigin]],
        scale_rows_offset: Int,
    ):
        # The extra stream's caller (latent sparse attention) passes flat
        # physical rows, so it cannot share a logical selection's decoding.
        comptime assert not (
            not Self.physical_indices and Self.has_extra_kv
        ), "logical sparse indices do not support an extra KV stream"
        comptime if Self.idx_prefetch:
            # One bulk L2 prefetch of this split's whole index range, so the
            # per-tile loads below hit L2 once the gathers have evicted the
            # indexer's output. A hint: both ends round down to 16 B absolute
            # boundaries (the row of an odd top-k is not aligned), so the
            # range stays inside the 16 B-aligned index tensor. It reads only
            # arguments: a block that names this function's locals changes
            # the codegen of forms without the prefetch, even compiled out.
            if elect() != 0:
                var row = d_indices.unsafe_value()
                var start = Int(offset_position.kv_start_row)
                var lo = Int(row.unsafe_offset(start))
                var hi = Int(
                    row.unsafe_offset(
                        min(
                            start + offset_position.num_keys_this_split,
                            offset_position.sparse_topk,
                        )
                    )
                )
                var lo16 = lo & ~15
                var hi16 = hi & ~15
                if hi16 > lo16:
                    cp_async_bulk_prefetch(
                        (
                            row.unsafe_bitcast[UInt8]().unsafe_offset(
                                (lo16 - Int(row))
                            )
                        ).unsafe_bitcast[Int32](),
                        Int32(hi16 - lo16),
                    )
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var idx_prod = idx_bars.producer()
        # Blockwise `kv_convert`: this warp also stages each tile's scales
        # into its K/V stage's slot, as 32 of the slot's producers.
        var scale_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](
            kv_pipeline_gather,
            scale_smem.unsafe_bitcast[Scalar[Self.mma_type]](),
        )
        var orig_topk_u32 = UInt32(offset_position.sparse_topk)
        var batch_idx = UInt32(offset_position.batch_idx)

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        var first_tile_from_orig = num_orig_tiles > 0
        var orig_indices_base = Int(offset_position.kv_start_row)
        # Stage index comes from idx_prod's own ring counter (PipelineState,
        # any depth), not a hand-rolled toggle.

        comptime if Self.idx_prefetch:
            # The next tile's raw indices are loaded before this tile's
            # commit, so their latency overlaps the wait for the next slot.
            # Positions past the list clamp to its last entry, so the load
            # one tile past the end stays in bounds.
            var _sb = stamp_begin[5]()
            if first_tile_from_orig:
                var raw = Self._load_raw_indices(
                    d_indices, orig_indices_base, orig_topk_u32
                )
                var t: Int = 0
                while True:
                    if t > 0:
                        idx_prod.acquire()
                    stamp_tile[5](_sb, t, 0)
                    var raw_next = Self._load_raw_indices(
                        d_indices,
                        orig_indices_base + Self.config.BN_QK,
                        orig_topk_u32,
                    )
                    var idx_smem = idx_smem_base.unsafe_offset(
                        idx_prod.state.index() * (UInt32(Self.idx_stage_slots))
                    )
                    comptime if _MLA_P6_BREAK == 8:
                        Self._store_index_rows(
                            raw_next, idx_smem, kv_lut, batch_idx
                        )
                    else:
                        Self._store_index_rows(raw, idx_smem, kv_lut, batch_idx)
                    idx_prod.commit()
                    comptime if Self.kv_convert and Self.blockwise:
                        Self._stage_scales(
                            scale_prod, scale_smem, scales_ptr, idx_smem
                        )
                    stamp_tile[5](_sb, t, 1)
                    orig_indices_base += Self.config.BN_QK
                    raw = raw_next
                    t += 1
                    if t >= num_orig_tiles:
                        break
            stamp_finish[5](_sb, num_k_tiles)
        else:
            if first_tile_from_orig:
                var idx_smem = idx_smem_base.unsafe_offset(
                    idx_prod.state.index() * UInt32(Self.idx_stage_slots)
                )
                Self._transform_indices_to_smem(
                    d_indices,
                    idx_smem,
                    orig_indices_base,
                    kv_lut,
                    orig_topk_u32,
                    batch_idx,
                    scale_rows_offset,
                )
                idx_prod.commit()
                comptime if Self.kv_convert and Self.blockwise:
                    Self._stage_scales(
                        scale_prod, scale_smem, scales_ptr, idx_smem
                    )
                orig_indices_base += Self.config.BN_QK

            var remaining_orig = (
                num_orig_tiles - 1 if first_tile_from_orig else 0
            )
            var t: Int = 0
            var _sb = stamp_begin[5]()
            while t < remaining_orig:
                idx_prod.acquire()
                stamp_tile[5](_sb, t + 1, 0)
                var idx_smem = idx_smem_base.unsafe_offset(
                    idx_prod.state.index() * UInt32(Self.idx_stage_slots)
                )
                Self._transform_indices_to_smem(
                    d_indices,
                    idx_smem,
                    orig_indices_base,
                    kv_lut,
                    orig_topk_u32,
                    batch_idx,
                    scale_rows_offset,
                )
                idx_prod.commit()
                comptime if Self.kv_convert and Self.blockwise:
                    Self._stage_scales(
                        scale_prod, scale_smem, scales_ptr, idx_smem
                    )
                stamp_tile[5](_sb, t + 1, 1)
                orig_indices_base += Self.config.BN_QK
                t += 1
            stamp_finish[5](_sb, num_k_tiles)

        comptime if Self.has_extra_kv:
            var extra_topk_u32 = UInt32(extra_topk)
            var extra_indices_base = max(
                0, Int(offset_position.kv_start_row) - topk
            )
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                var idx_smem = idx_smem_base.unsafe_offset(
                    idx_prod.state.index() * UInt32(Self.idx_stage_slots)
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                    batch_idx,
                    scale_rows_offset,
                )
                idx_prod.commit()
                comptime if Self.kv_convert and Self.blockwise:
                    Self._stage_scales(
                        scale_prod,
                        scale_smem,
                        extra_scales_ptr.unsafe_value(),
                        idx_smem,
                    )
                extra_indices_base += Self.config.BN_QK
                num_extra_tiles -= 1

            var te: Int = 0
            while te < num_extra_tiles:
                idx_prod.acquire()
                var idx_smem = idx_smem_base.unsafe_offset(
                    idx_prod.state.index() * UInt32(Self.idx_stage_slots)
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                    batch_idx,
                    scale_rows_offset,
                )
                idx_prod.commit()
                comptime if Self.kv_convert and Self.blockwise:
                    Self._stage_scales(
                        scale_prod,
                        scale_smem,
                        extra_scales_ptr.unsafe_value(),
                        idx_smem,
                    )
                extra_indices_base += Self.config.BN_QK
                te += 1

    @staticmethod
    @inline(.always)
    def _stage_scales(
        mut scale_prod: DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ],
        scale_smem: SharedMemPointer[UInt8],
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
        idx_smem: SharedMemPointer[Int32],
    ):
        """Stages a tile's scales into its K/V stage's scale slot.

        Reads the gather rows this warp just wrote to `idx_smem`, each lane
        the two rows it wrote, so program order alone orders the read after
        the write. The loads issue before the slot acquire, so their latency
        overlaps the wait. An unused slot (`-1`) gathers zeros, so its scale
        is 1 rather than a load from before the pool.
        """
        comptime scales_per_token = Self.config.scales_per_token
        var lane = thread_idx.x & 31
        comptime staged_dtype = (
            DType.uint16 if Self.config.fp8_scale_bf16 else DType.uint8
        )
        var vals = StaticTuple[Scalar[staged_dtype], 2 * scales_per_token]()
        comptime for row_pass in range(2):
            var gmem_row = Int(idx_smem[lane + row_pass * 32])
            comptime for s in range(scales_per_token):
                var fp32_val = Float32(1.0)
                if gmem_row >= 0:
                    fp32_val = scales_ptr[gmem_row * scales_per_token + s]
                comptime if Self.config.fp8_scale_bf16:
                    vals[row_pass * scales_per_token + s] = bitcast[.uint16](
                        fp32_val.cast[.bfloat16]()
                    ).cast[staged_dtype]()
                else:
                    vals[row_pass * scales_per_token + s] = bitcast[.uint8](
                        fp32_val.cast[.float8_e8m0fnu]()
                    ).cast[staged_dtype]()

        scale_prod.acquire()
        var stage = scale_smem.unsafe_offset(
            scale_prod.stage_index() * UInt32(Self.config.scale_smem_per_stage)
        )
        var stage_values = stage.unsafe_bitcast[Scalar[staged_dtype]]()
        comptime for row_pass in range(2):
            var smem_off = (lane + row_pass * 32) * scales_per_token
            comptime for s in range(scales_per_token):
                stage_values[smem_off + s] = vals[
                    row_pass * scales_per_token + s
                ]
        _ = scale_prod.producer_mbar()[].arrive()
        scale_prod.commit_step()

    @staticmethod
    @inline(.always)
    def load(
        q_tma: QOTMATile[
            dtype=Self.mma_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.operand_swizzle,
        ],
        k_tma: Self.KTMA,
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        mbar_q: MBarType,
        kv_pipeline_gather: Self.GatherPipe,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
        idx_bars: Self.IdxBars,
        idx_smem_base: SharedMemPointer[Int32],
        num_orig_blocks: Int,
        topk: Int,
        extra_k_tma: Self.KTMA,
        extra_topk: Int,
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        # Match the Q TMA's transfer size (the row as stored), not the SMEM width.
        expect_bytes_pred(
            mbar_q,
            Int32(Self.config.BM * Self.config.input_q_depth * Self.mma_bytes),
            elect_mask,
        )
        if is_leader:
            comptime q_elems = type_of(q_tma).tile_shape.element_types[
                0
            ].static_value * type_of(q_tma).tile_shape.element_types[
                1
            ].static_value
            comptime q_tt_layout = tt_row_major[q_elems]()
            var q_smem_tensor = TileTensor[
                Self.mma_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](q_smem.unsafe_bitcast[Scalar[Self.mma_type]](), q_tt_layout)
            q_tma.async_copy(q_smem_tensor, mbar_q[], (0, row))

        # The row as stored: BN_QK * (input_q_depth / 8) INT64 bytes.
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var first_tile_from_orig = num_orig_tiles > 0
        var idx_cons = idx_bars.consumer()

        if first_tile_from_orig:
            Self._load_one_tile(
                kv_prod,
                is_leader,
                k_tma,
                idx_cons,
                idx_smem_base,
                kv_bytes,
                scales_ptr,
            )
            # Wait here so Q load overlaps with the first KV transfer.
            mbar_q[].wait(0)

        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        Self._load_tile_range(
            kv_prod,
            is_leader,
            k_tma,
            idx_cons,
            idx_smem_base,
            remaining_orig,
            scales_ptr,
        )

        comptime if Self.has_extra_kv:
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                Self._load_one_tile(
                    kv_prod,
                    is_leader,
                    extra_k_tma,
                    idx_cons,
                    idx_smem_base,
                    kv_bytes,
                    scales_ptr,
                )
                # First tile comes from extra_kv since orig was empty.
                mbar_q[].wait(0)
                num_extra_tiles -= 1

            Self._load_tile_range(
                kv_prod,
                is_leader,
                extra_k_tma,
                idx_cons,
                idx_smem_base,
                num_extra_tiles,
                scales_ptr,
            )

    @staticmethod
    @inline(.always)
    def load_q_pair(
        q_tma: QOTMATile[
            dtype=Self.mma_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.operand_swizzle,
        ],
        q_smem: SharedMemPointer[Scalar[Self.mma_type]],
        mbar_q: MBarType,
        rank: UInt32,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Loads this CTA's Q tile; warp 8's only work under the pair.

        Each CTA TMA-loads its own head tile (the grid's head tile equals the
        cluster rank), routing the completion to the leader's `mbar_q`. Only
        the leader arms it (both CTAs' bytes) and only the leader's QK waits
        it; the cluster sync after the init orders the arm before either
        CTA's TMA can complete.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return
        var elect_mask = elect()
        var row: Int = offset_position.q_row_offset
        if rank == 0:
            expect_bytes_pred(
                mbar_q,
                Int32(
                    2
                    * Self.config.BM
                    * Self.config.input_q_depth
                    * Self.mma_bytes
                ),
                elect_mask,
            )
        if elect_mask != 0:
            comptime q_elems = type_of(q_tma).tile_shape.element_types[
                0
            ].static_value * type_of(q_tma).tile_shape.element_types[
                1
            ].static_value
            comptime q_tt_layout = tt_row_major[q_elems]()
            var q_smem_tensor = TileTensor[
                Self.mma_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](q_smem.unsafe_bitcast[Scalar[Self.mma_type]](), q_tt_layout)
            q_tma.async_copy[cta_group=2](q_smem_tensor, mbar_q[], (0, row))

    @staticmethod
    @inline(.always)
    def defers_cluster_wait(warp_idx: UInt32) -> Bool:
        """Whether a warp's role touches no peer state before it ends.

        The index producer and the gather lanes read global memory and
        write and wait only this CTA's SMEM and barriers; the leader's
        commits reach their free barriers only after the leader's own
        entry wait, which this CTA's arrive gates.
        """
        if warp_idx == 11:
            return True
        return warp_idx >= 12

    @staticmethod
    @inline(.always)
    def pair_tail_lane_wait(
        z_empty: MBarType,
        v_free: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=2 * 4,
        ],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=2 * 4,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=2 * 4
        ],
        num_keys: Int,
    ) -> UInt32:
        """Returns this lane's share of the pair's producer tail as one
        word.

        The tail waits each barrier's last phase that no role waits. The
        leader's QK and PV commits free Z, V and P in both CTAs, and only
        tile t + D waits a free, so each stage's last free lands
        unobserved; the follower's S and O releases reach the leader's
        free barriers likewise. Every other inbound arrive is waited by
        the role that consumes it. Each role has already waited the
        next-to-last completion, so the parity wait cannot alias.

        One barrier stage per lane, so a single parity poll covers all of
        them; lanes `[0, D)` take `z_empty`, then `v_free`, the P frees
        and, in the leader, the S frees and the two O frees. The word is
        the barrier's shared address times 2 plus the parity to wait, or 0
        for a lane with nothing to wait.
        """
        comptime D = Self.num_stages
        comptime num_atoms = Self.config.depth // Self.config.MMA_PV_N
        comptime assert 4 * D + 2 <= WARP_SIZE, "one barrier stage per lane"
        var num_k_tiles = ceildiv(num_keys, Self.config.BN_QK)
        var lane = Int(lane_id())
        var num_lanes = 4 * D + 2 if block_rank_in_cluster() == 0 else 3 * D
        if lane >= num_lanes:
            return 0
        var bar = z_empty
        var stage = lane % D
        var stages = D
        var completions = num_k_tiles
        if lane >= 4 * D:
            bar = o_bars.mbar_base.unsafe_offset(2)
            stage = lane - 4 * D
            stages = 2
            completions = num_k_tiles * num_atoms
        elif lane >= 3 * D:
            bar = s_bars.mbar_base.unsafe_offset(D)
        elif lane >= 2 * D:
            bar = p_bars.mbar_base.unsafe_offset(D)
        elif lane >= D:
            bar = v_free
        var c = (completions - stage + stages - 1) // stages
        if c <= 0:
            return 0
        return UInt32(Int(bar.unsafe_offset(stage))) * 2 + UInt32((c - 1) & 1)

    @staticmethod
    @inline(.always)
    def gather_pair(
        k_tma: Self.KTMA,
        z_smem: SharedMemPointer[Scalar[Self.mma_type]],
        e_smem: SharedMemPointer[Scalar[Self.mma_type]],
        z_tx: MBarType,
        z_empty: MBarType,
        v_tx: MBarType,
        v_free: MBarType,
        idx_bars: Self.IdxBars,
        idx_smem_base: SharedMemPointer[Int32],
        topk: Int,
        extra_k_tma: Self.KTMA,
        rank: UInt32,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """WG3 under the pair: gather this CTA's key half of K, then all
        pair keys of its V depth slice, with Z one tile ahead of V.

        Each of the four warps gathers its 16 rows of the CTA's key half
        straight into its Z slot (the direct SW128 gather), and `z_full`
        (leader-local, one arrival per CTA) gates the QK, which reads both
        CTAs' Z. Each CTA also gathers ALL pair keys, in its own V depth
        slice of every PV atom, into its V ring, which the PV reads. The
        own half's V bytes are gathered redundantly, which buys a plain
        producer ring with no cross-CTA copies. Z feeds QK alone, so the
        QK commit frees the Z slot early.

        Step t issues Z(t), whose only gates are its slot's QK(t - D)
        commit and the index slot, then waits V(t - 1)'s slot free (the
        PV(t - 1 - D) commit) and issues V(t - 1). No gather lane waits a
        landing: warp 8 forwards both (`forward_pair_landings`). So QK(t)
        waits neither PV(t - D) nor V, and the K ring runs at the QK
        pace.

        Two index cursors: the head waits Z's slot, the tail reads V's
        quads from the slot Z read one step earlier and then releases
        it. The index ring is `idx_stages` deep (`_SPARSE_PAIR_IDX_STAGES`),
        so both slots can be held at once, and the index producer refills
        the tail's slot after V's issue, `idx_stages - 1` tiles before Z
        needs it.

        A lane that waits gates an arrival of the ring it waits on (Z
        and V issues, index releases), so no ring can lap a waiter.
        """
        comptime D = Self.num_stages
        comptime DZ = Self.z_stages
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        comptime box_w = Self.kv_gather4_box_w
        comptime assert (
            box_w == Self.KVBlockCols
        ), "the pair's gather box is one K/V SMEM column block"
        comptime C = Self.CtaKeys // 16
        comptime NUM_CG = ceildiv(
            min(Self.config.input_q_depth, Self.z_row_elems), box_w
        )
        comptime warp_bytes = C * 4 * NUM_CG * box_w * size_of[Self.kv_type]()
        comptime num_atoms = Self.config.depth // Self.config.MMA_PV_N
        comptime vgather_warp_bytes = (
            2
            * C
            * 4
            * num_atoms
            * Self.v_boxes
            * box_w
            * size_of[Self.kv_type]()
        )
        comptime half_chunks = Self.CtaKeys // 4

        var idx_head = idx_bars.consumer()
        var idx_tail = idx_head
        var wq = UInt32(warp_id[broadcast=True]()) & 3
        if elect() == 0:
            return
        var z_chunk0 = Int(wq) * C
        var k_rank = 0 if _MLA_CG2_SLICE_BREAK == 2 else Int(rank)
        var v_rank = 0 if _MLA_CG2_SLICE_BREAK == 1 else Int(rank)
        var idx_chunk0 = z_chunk0 + k_rank * half_chunks
        comptime z_stage_elems: UInt32 = UInt32(Self.KVStageElems)
        comptime e_stage_elems: UInt32 = UInt32(Self.EStageElems)

        var t: Int = 0
        var _sb = stamp_begin[6]()
        while t <= num_k_tiles:
            if t < num_k_tiles:
                var s = t % DZ
                var n = t // DZ
                if t >= DZ:
                    z_empty[s].wait(UInt32((n - 1) & 1))
                idx_head.wait()
                stamp_tile[6](_sb, t, 0)
                var idx_u8 = (
                    idx_smem_base.unsafe_offset(
                        idx_head.state.index() * UInt32(Self.config.BN_QK)
                    )
                ).unsafe_bitcast[UInt8]()
                var k_mbar = z_tx.unsafe_offset(UInt32(s))
                k_mbar[].expect_bytes(Int32(warp_bytes))
                var stage = (
                    z_smem.unsafe_offset(UInt32(s) * z_stage_elems)
                ).unsafe_bitcast[Scalar[Self.k_tma_dtype]]()
                var rows = StaticTuple[SIMD[DType.uint32, 4], C]()
                comptime for j in range(C):
                    rows[j] = ld_shared_v4_u32[ordered=True](
                        idx_u8, (idx_chunk0 + j) * 16
                    )
                var cg_p = stage.unsafe_offset(UInt32(z_chunk0 * 4 * box_w))
                comptime if Self.has_extra_kv:
                    if t < num_orig_tiles:
                        Self._issue_pair_gather(k_tma, cg_p, k_mbar, rows)
                    else:
                        Self._issue_pair_gather(extra_k_tma, cg_p, k_mbar, rows)
                else:
                    Self._issue_pair_gather(k_tma, cg_p, k_mbar, rows)
                idx_head.state.step()
            if t > 0:
                var u = t - 1
                var s = u % D
                var n = u // D
                if u >= D:
                    v_free[s].wait(UInt32((n - 1) & 1))
                stamp_tile[6](_sb, u, 2)
                var idx_u8 = (
                    idx_smem_base.unsafe_offset(
                        idx_tail.state.index() * UInt32(Self.config.BN_QK)
                    )
                ).unsafe_bitcast[UInt8]()
                var v_mbar = v_tx.unsafe_offset(UInt32(s))
                v_mbar[].expect_bytes(Int32(vgather_warp_bytes))
                var v_stage = (
                    e_smem.unsafe_offset(UInt32(s) * e_stage_elems)
                ).unsafe_bitcast[Scalar[Self.k_tma_dtype]]()
                comptime for h in range(2):
                    var vrows = StaticTuple[SIMD[DType.uint32, 4], C]()
                    comptime for j in range(C):
                        vrows[j] = ld_shared_v4_u32[ordered=True](
                            idx_u8,
                            (h * half_chunks + z_chunk0 + j) * 16,
                        )
                    comptime for a in range(num_atoms):
                        comptime for c in range(Self.v_boxes):
                            var v_p = v_stage.unsafe_offset(
                                UInt32(
                                    Self.v_block_offset[h, a, c]()
                                    + z_chunk0 * 4 * box_w
                                )
                            )
                            var v_col = Int32(
                                a * Self.config.MMA_PV_N
                                + v_rank * (Self.config.MMA_PV_N // 2)
                                + c * box_w
                            )
                            comptime if Self.has_extra_kv:
                                if u < num_orig_tiles:
                                    Self._issue_pair_columns(
                                        k_tma, v_p, v_mbar, vrows, v_col
                                    )
                                else:
                                    Self._issue_pair_columns(
                                        extra_k_tma, v_p, v_mbar, vrows, v_col
                                    )
                            else:
                                Self._issue_pair_columns(
                                    k_tma, v_p, v_mbar, vrows, v_col
                                )
                idx_tail.release()
                stamp_tile[6](_sb, u, 1)
            t += 1
        stamp_finish[6](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def gather_pair_nvfp4(
        k_tma: Self.KTMA,
        v_tma: Self.PackedVTMA,
        z_smem: SharedMemPointer[Scalar[Self.mma_type]],
        e_smem: SharedMemPointer[Scalar[Self.mma_type]],
        z_tx: MBarType,
        z_empty: MBarType,
        z_full: MBarType,
        v_tx: MBarType,
        v_free: MBarType,
        v_full: MBarType,
        idx_bars: Self.IdxBars,
        idx_smem: SharedMemPointer[Int32],
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
        rank: UInt32,
        num_keys: Int,
    ):
        """Gathers packed Z and projected V into independent stage overlays."""
        comptime Z_PACKED = Self.CtaKeys * Self.nvfp4_row_bytes
        comptime Z_OVERLAY = Self.KVStageElems - Self.CtaKeys * (
            Self.nvfp4_row_bytes + Self.nvfp4_sf_row_bytes
        )
        comptime V_PACKED = Self.config.BN_QK * 128
        comptime V_OVERLAY = Self.EStageElems - Self.config.BN_QK * 144
        comptime sw = make_swizzle[
            Self.mma_type, TensorMapSwizzle.SWIZZLE_128B
        ]()
        var tid = thread_idx.x & 127
        var warp = tid // 32
        var leader = elect() != 0
        var head = idx_bars.consumer()
        var tail = head
        var sf_gmem = scales_ptr.unsafe_bitcast[UInt8]()
        var k_rank = 0 if _MLA_CG2_SLICE_BREAK == 2 else Int(rank)
        var v_rank = 0 if (
            _MLA_CG2_SLICE_BREAK == 1 or (_MLA_NVFP4_BREAK & 4) != 0
        ) else Int(rank)
        var tiles = ceildiv(num_keys, Self.config.BN_QK)
        for t in range(tiles + 1):
            if t < tiles:
                var slot = t % Self.z_stages
                if t >= Self.z_stages:
                    z_empty[slot].wait(UInt32((t // Self.z_stages - 1) & 1))
                head.wait()
                var indices = idx_smem.unsafe_offset(
                    head.state.index() * UInt32(Self.idx_stage_slots)
                )
                var dst = z_smem.unsafe_offset(slot * Self.KVStageElems)
                var src = dst.unsafe_bitcast[UInt8]().unsafe_offset(Z_OVERLAY)
                if leader:
                    z_tx[slot].expect_bytes(Int32(16 * Self.nvfp4_row_bytes))
                    comptime for chunk in range(4):
                        var rows = ld_shared_v4_u32[ordered=True](
                            indices.unsafe_bitcast[UInt8](),
                            (k_rank * 64 + warp * 16 + chunk * 4) * 4,
                        )
                        k_tma.async_copy_gather4[
                            eviction_policy=CacheEviction.EVICT_LAST
                        ](
                            TileTensor(
                                src.unsafe_bitcast[
                                    Scalar[Self.k_tma_dtype]
                                ]().unsafe_offset(
                                    (warp * 16 + chunk * 4) * Self.kv_row_words
                                ),
                                tt_row_major[4, Self.kv_row_words](),
                            ),
                            z_tx[slot],
                            Int32(0),
                            Int32(rows[0]),
                            Int32(rows[1]),
                            Int32(rows[2]),
                            Int32(rows[3]),
                        )
                var row = tid // 2
                var sf_row = indices[Self.config.BN_QK + k_rank * 64 + row]
                comptime if (_MLA_NVFP4_BREAK & 2) != 0:
                    sf_row = 0
                var live = sf_row != -1
                async_copy[16, fill=UInt8(0)](
                    sf_gmem.unsafe_offset(
                        (Int(bitcast[DType.uint32, 1](sf_row)) if live else 0)
                        * 32
                        + (tid % 2) * 16
                    ).unsafe_address_space_cast[.GLOBAL](),
                    src.unsafe_offset(Z_PACKED + row * 32 + (tid % 2) * 16),
                    src_size=Int32(16) if live else Int32(0),
                )
                async_copy_arrive[noinc=True](z_tx.unsafe_offset(slot))
                head.state.step()
                z_tx[slot].wait(UInt32((t // Self.z_stages) & 1))
                var cp = tid % 2
                var held = StaticTuple[SIMD[DType.uint32, 4], 18]()
                var sf_a = ld_shared_v4_u32(src, Z_PACKED + row * 32)
                var sf_b = ld_shared_v4_u32(src, Z_PACKED + row * 32 + 16)
                comptime for i in range(8):
                    var q = ld_shared_v4_u32(
                        src, row * Self.nvfp4_row_bytes + (cp + 2 * i) * 16
                    )
                    var sf = sf_a[i] if i < 4 else sf_b[i - 4]
                    var out = Self._nvfp4_chunk(
                        q, sf, sf, UInt32(2 * cp), UInt32(2 * cp + 1)
                    )
                    held[2 * i] = out[0]
                    held[2 * i + 1] = out[1]
                comptime for r in range(2):
                    held[16 + r] = ld_shared_v4_u32(
                        src,
                        row * Self.nvfp4_row_bytes + 256 + (cp + 2 * r) * 16,
                    )
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                comptime for i in range(8):
                    comptime for g in range(2):
                        st_shared_v4_b32_at_bf16_elem_off[
                            out_dtype=Self.mma_type
                        ](
                            dst.unsafe_offset((i // 2) * Self.KVBlockElems),
                            sw(row * 128 + (i % 2) * 64 + cp * 32 + 16 * g),
                            held[2 * i + g],
                        )
                comptime for r in range(2):
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                        dst.unsafe_offset(4 * Self.KVBlockElems),
                        sw(row * 128 + (cp + 2 * r) * 16),
                        held[16 + r],
                    )
                fence_async_view_proxy()
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                if tid == 0:
                    z_full[slot].arrive_cluster(UInt32(0), UInt32(1))
            if t > 0:
                var u = t - 1
                var slot = u % Self.num_stages
                if u >= Self.num_stages:
                    v_free[slot].wait(UInt32((u // Self.num_stages - 1) & 1))
                var indices = idx_smem.unsafe_offset(
                    tail.state.index() * UInt32(Self.idx_stage_slots)
                )
                var dst = e_smem.unsafe_offset(slot * Self.EStageElems)
                var src = dst.unsafe_bitcast[UInt8]().unsafe_offset(V_OVERLAY)
                if leader:
                    v_tx[slot].expect_bytes(Int32(32 * 128))
                    comptime for atom in range(2):
                        comptime for chunk in range(8):
                            var rows = ld_shared_v4_u32[ordered=True](
                                indices.unsafe_bitcast[UInt8](),
                                (warp * 32 + chunk * 4) * 4,
                            )
                            v_tma.async_copy_gather4[
                                eviction_policy=CacheEviction.EVICT_LAST
                            ](
                                TileTensor(
                                    src.unsafe_bitcast[
                                        Scalar[Self.PackedVTMA.dtype]
                                    ]().unsafe_offset(
                                        atom * 128 * 8
                                        + (warp * 32 + chunk * 4) * 8
                                    ),
                                    tt_row_major[4, 8](),
                                ),
                                v_tx[slot],
                                Int32(atom * 16 + v_rank * 8),
                                Int32(rows[0]),
                                Int32(rows[1]),
                                Int32(rows[2]),
                                Int32(rows[3]),
                            )
                var sf_row = indices[Self.config.BN_QK + tid]
                comptime if (_MLA_NVFP4_BREAK & 2) != 0:
                    sf_row = 0
                var live = sf_row != -1
                comptime for atom in range(2):
                    async_copy[8, fill=UInt8(0)](
                        sf_gmem.unsafe_offset(
                            (
                                Int(
                                    bitcast[DType.uint32, 1](sf_row)
                                ) if live else 0
                            )
                            * 32
                            + atom * 16
                            + v_rank * 8
                        ).unsafe_address_space_cast[.GLOBAL](),
                        src.unsafe_offset(V_PACKED + atom * 128 * 8 + tid * 8),
                        src_size=Int32(8) if live else Int32(0),
                    )
                async_copy_arrive[noinc=True](v_tx.unsafe_offset(slot))
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                if leader:
                    tail.release()
                else:
                    tail.state.step()
                # All row and scale indices have been consumed before reuse.
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                v_tx[slot].wait(UInt32((u // Self.num_stages) & 1))
                var held = StaticTuple[SIMD[DType.uint32, 4], 16]()
                comptime for atom in range(2):
                    var sf = (
                        src.unsafe_offset(V_PACKED + atom * 128 * 8 + tid * 8)
                        .unsafe_bitcast[UInt32]()
                        .unsafe_load[width=2, alignment=8]()
                    )
                    comptime for chunk in range(4):
                        var q = ld_shared_v4_u32(
                            src, atom * 128 * 64 + tid * 64 + chunk * 16
                        )
                        var word = sf[chunk // 2]
                        var out = Self._nvfp4_chunk(
                            q,
                            word,
                            word,
                            UInt32((chunk % 2) * 2),
                            UInt32((chunk % 2) * 2 + 1),
                        )
                        held[atom * 8 + chunk * 2] = out[0]
                        held[atom * 8 + chunk * 2 + 1] = out[1]
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                comptime for atom in range(2):
                    comptime for chunk in range(4):
                        comptime for g in range(2):
                            var off = (
                                (2 * (tid // 64) + atom) * Self.KVBlockElems
                            ) + sw((tid % 64) * 128 + chunk * 32 + 16 * g)
                            st_shared_v4_b32_at_bf16_elem_off[
                                out_dtype=Self.mma_type
                            ](dst, off, held[atom * 8 + chunk * 2 + g])
                fence_async_view_proxy()
                named_barrier[Int32(WARPGROUP_SIZE)](3)
                if tid == 0:
                    v_full[slot].arrive_cluster(UInt32(0), UInt32(1))

    @staticmethod
    @inline(.always)
    def forward_pair_landings(
        z_tx: MBarType,
        z_full: MBarType,
        v_tx: MBarType,
        v_full: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Warp 8 under the pair: forwards this CTA's Z
        and V landings to the leader's `z_full` and `v_full`.

        In the gather lanes, a landing wait stalls the next issue behind
        the TMA latency. The order is the issue order, Z(t + 1) before
        V(t), so a wait never blocks on a later landing. Neither barrier
        can lap the wait: Z(t + D) needs QK(t), and V(t + D) needs PV(t),
        each gated on the forward it follows.
        """
        comptime D = Self.num_stages
        comptime DZ = Self.z_stages
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if elect() == 0:
            return
        var t: Int = 0
        while t <= num_k_tiles:
            if t < num_k_tiles:
                z_tx[t % DZ].wait(UInt32((t // DZ) & 1))
                z_full[t % DZ].arrive_cluster(UInt32(0), UInt32(1))
            if t > 0:
                var u = t - 1
                v_tx[u % D].wait(UInt32((u // D) & 1))
                v_full[u % D].arrive_cluster(UInt32(0), UInt32(1))
            t += 1

    @staticmethod
    @inline(.always)
    def _issue_pair_gather(
        cur_k_tma: Self.KTMA,
        z_rows: SharedMemPointer[Scalar[Self.k_tma_dtype]],
        k_mbar: MBarType,
        rows: StaticTuple[SIMD[DType.uint32, 4], Self.CtaKeys // 16],
    ):
        """Issues one warp's gather4s of its rows of this CTA's key half,
        one SW128 column block at a time, starting at `z_rows`."""
        comptime box_w = Self.kv_gather4_box_w
        comptime C = Self.CtaKeys // 16
        comptime NUM_CG = ceildiv(
            min(Self.config.input_q_depth, Self.z_row_elems), box_w
        )
        var cg_p = z_rows
        var col = Int32(0)
        comptime for cg in range(NUM_CG):
            comptime for j in range(C):
                var r = rows[j].cast[DType.int32]()
                cur_k_tma.async_copy_gather4[
                    eviction_policy=CacheEviction.EVICT_LAST
                ](
                    TileTensor(
                        cg_p.unsafe_offset(j * 4 * box_w),
                        tt_row_major[4, box_w](),
                    ),
                    k_mbar[],
                    col,
                    r[0],
                    r[1],
                    r[2],
                    r[3],
                )
            cg_p = cg_p.unsafe_offset(UInt32(Self.CtaKeys * box_w))
            col += Int32(box_w)

    @staticmethod
    @inline(.always)
    def _issue_pair_columns(
        cur_k_tma: Self.KTMA,
        rg_rows: SharedMemPointer[Scalar[Self.k_tma_dtype]],
        z1_mbar: MBarType,
        rows: StaticTuple[SIMD[DType.uint32, 4], Self.CtaKeys // 16],
        col: Int32,
    ):
        """Issues one warp's gather4s: its 16 rows, one 128-column block
        at column `col` of the row as stored, starting at `rg_rows`. The
        V gather calls it once per (source column, destination block)."""
        comptime box_w = Self.kv_gather4_box_w
        comptime C = Self.CtaKeys // 16
        var rg_p = rg_rows
        comptime for j in range(C):
            var r = rows[j].cast[DType.int32]()
            cur_k_tma.async_copy_gather4[
                eviction_policy=CacheEviction.EVICT_LAST
            ](
                TileTensor(
                    rg_p.unsafe_offset(j * 4 * box_w), tt_row_major[4, box_w]()
                ),
                z1_mbar[],
                col,
                r[0],
                r[1],
                r[2],
                r[3],
            )

    @staticmethod
    @inline(.always)
    def wait_q_pair_landed(mbar_q: MBarType):
        """Waits the pair's Q landing: both CTAs' head tiles of one cluster
        TMA.

        `load_q_pair` has each CTA's warp 8 issue its own head tile into that
        CTA's `q_smem` and counts both copies' bytes into the leader's
        `mbar_q`, and the pair's first QK reads Q from *both* CTAs' SMEM. So
        this wait is what orders that MMA after both halves land; without it
        the MMA reads whichever Q rows have not been written yet.
        """
        mbar_q[].wait(0)

    @staticmethod
    @inline(.always)
    def mmaQK_pair(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.mma_type]],
        z_smem: SharedMemPointer[Scalar[Self.mma_type]],
        mbar_q: MBarType,
        z_full: MBarType,
        z_empty: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=2 * 4,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """The pair's QK^T MMA: the leader's warp 9 only.

        One `cta_group::2` MMA per tile over both CTAs' Q (each CTA's 64
        head rows) and both CTAs' Z (each CTA's 64-key half), writing the
        pair's 128-key S accumulator into both CTAs' TMEM at the slot's
        64-column stride. The S-full commit is multicast so each CTA's
        softmax reads its own TMEM copy. V lives in its own ring, so the
        same commit is Z's last reader and frees the Z slot.
        """
        comptime DZ = Self.z_stages
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var s_stride = UInt32(comptime (Self.config.s_slot_stride()))
        var elect_mask = elect()

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(z_smem)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.mma_bytes

        Self.wait_q_pair_landed(mbar_q)
        var t: Int = 0
        var _sb = stamp_begin[3]()
        while t < num_k_tiles:
            s_prod.acquire()
            stamp_tile[3](_sb, t, 0)
            var slot_idx: UInt32 = s_prod.slot_index()
            var z_slot = t % DZ
            # Both CTAs' Z is gathered.
            z_full[z_slot].wait(UInt32((t // DZ) & 1))
            stamp_tile[3](_sb, t, 1)
            Self.UMMAQKTSS.mma[stage_idx=0](
                a=q_descriptor,
                b=k_descriptor + UInt32(z_slot * stage_stride_in_bytes),
                c=s0_tmem + slot_idx * s_stride,
                c_scale=UInt32(0),
                elect=elect_mask,
            )
            tcgen05_fence_before()
            if elect_mask != 0:
                mma_arrive_multicast[cta_group=2](
                    s_prod.pipe.producer_mbar()[].unsafe_ptr(), 0b11
                )
                # V lives in its own ring, so this commit is Z's last
                # reader: free the slot here and the K ring cycles at the
                # QK pace, not the PV pace.
                mma_arrive_multicast[cta_group=2](
                    z_empty[z_slot].unsafe_ptr(), 0b11
                )
            s_prod.pipe.step()
            stamp_tile[3](_sb, t, 2)
            t += 1
        stamp_finish[3](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def mmaPV_pair(
        tmem_addr: UInt32,
        e_smem: SharedMemPointer[Scalar[Self.mma_type]],
        p_smem: SharedMemPointer[Scalar[Self.mma_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=2 * 4,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=2 * 4
        ],
        v_full: MBarType,
        v_free: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """The pair's PV MMA: the leader's warp 10 only.

        Per tile and PV atom (half of V's 512 depth columns), one
        `cta_group::2` MMA per key half over both CTAs' P (each CTA's 64
        head rows of that half's 64 keys) and the cross-half V operand,
        accumulating into this CTA's half of the O atom in the same TMEM
        columns the unpaired kernel uses.

        It reads the V ring, each CTA holding all pair keys in its own
        half of every atom's depth columns. One leader-local `v_full` per
        stage (one arrival per CTA, after its V gather landed) gates it,
        with no arming. The QK commit frees Z, so this commit frees the P
        slot and the V slot only.
        """
        comptime D = Self.num_stages
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(e_smem)
        comptime num_atoms = Self.config.depth // Self.config.MMA_PV_N
        comptime o_round_cols = Self.config.o_round_cols()
        # P: one 64-key SW64 block per half, BM rows tall.
        comptime p_half_stride_in_bytes = Self.BlockElems * Self.mma_bytes
        comptime p_stage_stride_in_bytes = (Self.PStageElems * Self.mma_bytes)
        comptime v_stage_stride_in_bytes = Self.EStageElems * Self.mma_bytes

        var t: Int = 0
        var c_scale: UInt32 = 0
        var _sb = stamp_begin[4]()
        while t < num_k_tiles:
            var s = t % D
            var n = t // D
            # Both CTAs' V gathers landed (count 2, one arrival per CTA).
            v_full[s].wait(UInt32(n & 1))
            var p_slot_index = p_cons.wait()
            stamp_tile[4](_sb, t, 1)
            comptime for a in range(num_atoms):
                o_prod.acquire()
                comptime if a == 0:
                    stamp_tile[4](_sb, t, 2)
                # A 2-byte V block spans the whole 64-key pair tile, so one
                # MMA of BK_PV = 64 keys covers both halves.
                comptime pv_halves = 2 if Self.mma_bytes == 1 else 1
                comptime for h in range(pv_halves):
                    comptime b_off = (
                        Self.v_block_offset[h, a, 0]() * Self.mma_bytes
                    )
                    Self.UMMAPVSS.mma[stage_idx=0](
                        a=p_descriptor
                        + p_slot_index * UInt32(p_stage_stride_in_bytes)
                        + UInt32(h * p_half_stride_in_bytes),
                        b=v_descriptor
                        + UInt32(s) * UInt32(v_stage_stride_in_bytes)
                        + UInt32(b_off),
                        c=o_tmem + UInt32(a * o_round_cols),
                        c_scale=(c_scale if h == 0 else UInt32(1)),
                        elect=elect_mask,
                    )
                if elect_mask != 0:
                    mma_arrive_multicast[cta_group=2](
                        o_prod.pipe.producer_mbar()[].unsafe_ptr(), 0b11
                    )
                o_prod.pipe.step()
            tcgen05_fence_before()
            # Free the P slot and the V slot on both CTAs (the QK commit
            # frees Z). The commits track every MMA above, so the freed
            # buffers' last readers are done.
            if elect_mask != 0:
                mma_arrive_multicast[cta_group=2](
                    p_cons.pipe.consumer_mbar()[].unsafe_ptr(), 0b11
                )
                mma_arrive_multicast[cta_group=2](v_free[s].unsafe_ptr(), 0b11)
            p_cons.pipe.step()
            stamp_tile[4](_sb, t, 3)
            if t == 0:
                c_scale = 1
            t += 1
        stamp_finish[4](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def _issue_kv_gather(
        cur_k_tma: Self.KTMA,
        kv_stage_ptr: SharedMemPointer[Scalar[Self.mma_type]],
        k_mbar: MBarType,
        idx_smem: SharedMemPointer[Int32],
        is_leader: Bool,
    ):
        """Issues a tile's gather4 copies."""
        comptime box_w = Self.kv_gather4_box_w
        comptime num_chunks = Self.config.BN_QK // 4
        comptime if Self.gather4_idx_batch > 0:
            comptime assert (
                box_w == Self.kv_row_words
            ), "per-chunk gather4 assumes one column group per row"
            comptime B = min(Self.gather4_idx_batch, num_chunks)
            comptime assert num_chunks % B == 0
            if is_leader:
                var idx_u8 = idx_smem.unsafe_bitcast[UInt8]()
                comptime for b in range(num_chunks // B):
                    var rows = StaticTuple[SIMD[DType.uint32, 4], B]()
                    comptime for j in range(B):
                        rows[j] = ld_shared_v4_u32(idx_u8, (b * B + j) * 16)
                    comptime for j in range(B):
                        comptime c = b * B + j
                        var dst = TileTensor(
                            kv_stage_ptr.unsafe_bitcast[
                                Scalar[Self.k_tma_dtype]
                            ]().unsafe_offset(c * 4 * box_w),
                            tt_row_major[4, box_w](),
                        )
                        var r = rows[j].cast[DType.int32]()
                        cur_k_tma.async_copy_gather4[
                            eviction_policy=CacheEviction.EVICT_LAST
                        ](dst, k_mbar[], Int32(0), r[0], r[1], r[2], r[3])
        else:
            if is_leader:
                comptime for c in range(num_chunks):
                    var rows = ld_shared_v4_u32(
                        idx_smem.unsafe_bitcast[UInt8](), c * 16
                    ).cast[DType.int32]()
                    cur_k_tma.async_copy_gather4[
                        eviction_policy=CacheEviction.EVICT_LAST
                    ](
                        TileTensor(
                            kv_stage_ptr.unsafe_bitcast[
                                Scalar[Self.k_tma_dtype]
                            ]().unsafe_offset(c * 4 * box_w),
                            tt_row_major[4, box_w](),
                        ),
                        k_mbar[],
                        Int32(0),
                        rows[0],
                        rows[1],
                        rows[2],
                        rows[3],
                    )

    @staticmethod
    @inline(.always)
    def _expect_kv_bytes(k_mbar: MBarType, kv_bytes: Int, is_leader: Bool):
        expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))

    @staticmethod
    @inline(.always)
    def _load_one_tile(
        mut kv_prod: DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ],
        is_leader: Bool,
        cur_k_tma: Self.KTMA,
        mut idx_cons: ConsumerPipeline[Self.idx_stages],
        idx_smem_base: SharedMemPointer[Int32],
        kv_bytes: Int,
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
    ):
        var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
        var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

        idx_cons.wait()
        var idx_smem = idx_smem_base.unsafe_offset(
            idx_cons.state.index() * UInt32(Self.idx_stage_slots)
        )

        Self._expect_kv_bytes(k_mbar, kv_bytes, is_leader)
        Self._issue_kv_gather(
            cur_k_tma, kv_stage_ptr, k_mbar, idx_smem, is_leader
        )
        comptime if Self.is_nvfp4_kv:
            Self._stage_nvfp4_scales(kv_stage_ptr, idx_smem, scales_ptr, k_mbar)

        idx_cons.release()
        kv_prod.commit_step()

    @staticmethod
    @inline(.always)
    def _load_tile_range(
        mut kv_prod: DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ],
        is_leader: Bool,
        cur_k_tma: Self.KTMA,
        mut idx_cons: ConsumerPipeline[Self.idx_stages],
        idx_smem_base: SharedMemPointer[Int32],
        num_tiles: Int,
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
    ):
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var t: Int = 0
        var _sb = stamp_begin[2]()
        while t < num_tiles:
            kv_prod.acquire[qk_stage=0]()
            stamp_tile[2](_sb, t + 1, 0)

            var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

            idx_cons.wait()
            stamp_tile[2](_sb, t + 1, 1)
            var idx_smem = idx_smem_base.unsafe_offset(
                idx_cons.state.index() * UInt32(Self.idx_stage_slots)
            )

            Self._expect_kv_bytes(k_mbar, kv_bytes, is_leader)
            Self._issue_kv_gather(
                cur_k_tma, kv_stage_ptr, k_mbar, idx_smem, is_leader
            )
            comptime if Self.is_nvfp4_kv:
                Self._stage_nvfp4_scales(
                    kv_stage_ptr, idx_smem, scales_ptr, k_mbar
                )

            idx_cons.release()
            kv_prod.commit_step()
            stamp_tile[2](_sb, t + 1, 2)
            t += 1
        stamp_finish[2](_sb, num_tiles + 1)

    @staticmethod
    @inline(.always)
    def reswizzleFP8(
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem_sw64: SharedMemPointer[Scalar[Self.mma_type]],
        kv_pipeline_gather: Self.GatherPipe,
        kv_pipeline_mma: Self.MmaPipe,
        aux_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Re-swizzles one KV tile per iteration: FP8 linear -> FP8 SW64.

        A pure layout permute into a separate destination ring, so every
        column block is read, then written, independently. GROUP_SIZE=4:
        128 threads form 32 groups of 2 rows each, and 9 column iterations
        of 16 bytes per thread cover the row as stored.
        """
        comptime assert (
            Self.mma_bytes == 1 and size_of[Self.kv_type]() == 1
        ), "the re-swizzle front end copies FP8 bytes"
        comptime assert (
            Self.config.BN_QK == 64
        ), "the re-swizzle maps 64-key tiles: rows g and g + 32"
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime fp8_row_stride: Int = Self.config.input_q_depth
        comptime GROUP_SIZE: Int = 4
        comptime NUM_GROUPS: Int = WARPGROUP_SIZE // GROUP_SIZE  # 32
        # input_q_depth / (4 threads * 16 bytes) column iterations.
        comptime COLS_PER_GROUP: Int = fp8_row_stride // (GROUP_SIZE * 16)

        var kv_gather_cons = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem_sw64)

        var lane: Int = thread_idx.x & 0x7F
        var group_idx: Int = lane // GROUP_SIZE  # 0..31
        var idx_in_group: Int = lane % GROUP_SIZE  # 0..3

        var row_0: Int = 0 * NUM_GROUPS + group_idx
        var row_1: Int = 1 * NUM_GROUPS + group_idx

        # Linear-source read base offsets (bytes == elements for FP8).
        var fp8_base_0: Int = row_0 * fp8_row_stride + idx_in_group * 16
        var fp8_base_1: Int = row_1 * fp8_row_stride + idx_in_group * 16

        # SW64-destination swizzled element offsets. One 16-byte read maps
        # to exactly one 16-byte write: the SW64 swizzle's `base` (untouched
        # low bits) covers 16 FP8 elements (log2_floor(16 // 1 byte) = 4),
        # so a 16-byte-aligned source chunk always lands at a 16-byte
        # -aligned (permuted) destination chunk -- no finer-grained scatter.
        var col: Int = idx_in_group * 16
        var sw_off_0: Int = Self.sw_fp8(row_0 * SW64_COLS + col)
        var sw_off_1: Int = Self.sw_fp8(row_1 * SW64_COLS + col)

        var tile_idx: Int = 0
        var _sb = stamp_begin[6]()
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            stamp_tile[6](_sb, tile_idx, 0)
            kv_mma_prod.acquire()
            comptime if Self.q_in_tmem:
                # The last MMA stage holds Q until the TMEM copy reads it.
                if tile_idx == Self.num_mma_stages - 1:
                    aux_bars[Self.q_cp_mbar_idx].wait(0)
            stamp_tile[6](_sb, tile_idx, 1)

            var src_u8 = kv_gather_cons.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()

            comptime for c in range(COLS_PER_GROUP):
                var q0 = ld_shared_v4_u32(
                    src_u8, fp8_base_0 + c * GROUP_SIZE * 16
                )
                var q1 = ld_shared_v4_u32(
                    src_u8, fp8_base_1 + c * GROUP_SIZE * 16
                )
                var dst_block = dst.unsafe_offset(c * Self.KVBlockElems)
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_off_0, q0
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_off_1, q1
                )

            fence_async_view_proxy()
            kv_gather_cons.release_all()
            kv_mma_prod.commit_all()
            stamp_tile[6](_sb, tile_idx, 2)
            tile_idx += 1
        stamp_finish[6](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def _stage_nvfp4_scales(
        kv_stage_ptr: SharedMemPointer[Scalar[Self.mma_type]],
        idx_smem: SharedMemPointer[Int32],
        scales_ptr: Pointer[Float32, origin=MutAnyOrigin],
        k_mbar: MBarType,
    ):
        """Copies a tile's E4M3 scales next to its packed rows in the stage.

        Runs on all 32 lanes of the load warp, two lanes per row: each issues
        four 16-byte `cp.async` copies straight into the stage and then makes
        the gather barrier track them, so the warp does not wait for the
        scales to land before it moves on. `idx_smem` holds the value-pool TMA
        rows, then the scale-pool rows the indexer resolved through the scales
        lookup table; a row's scales sit at `scale_row * nvfp4_sf_row_bytes`.
        An unused slot (-1) copies zero bytes and is zero-filled, and its
        TMA-zeroed packed row dequantizes to zeros.
        """
        comptime assert (
            Self.nvfp4_sf_row_bytes == 32
        ), "two lanes per scale row, 16 B each"
        var sf_gmem = scales_ptr.unsafe_bitcast[UInt8]()
        var sf_smem = kv_stage_ptr.unsafe_bitcast[UInt8]().unsafe_offset(
            Self.nvfp4_sf_offset
        )
        var lane = thread_idx.x & 31
        var half = lane % 2
        comptime RP = Self.config.BN_QK // 16
        comptime for rp in range(RP):
            var r = rp * 16 + lane // 2
            # Unsigned scale-pool row, -1 for an unused slot; see
            # `_remap_scale_one`.
            var row = idx_smem[Self.config.BN_QK + r]
            comptime if (_MLA_NVFP4_BREAK & 2) != 0:
                row = 0
            var live = row != -1
            var src = sf_gmem.unsafe_offset(
                (Int(bitcast[DType.uint32, 1](row)) if live else 0)
                * Self.nvfp4_sf_row_bytes
                + half * 16
            )
            var dst = sf_smem.unsafe_offset(
                r * Self.nvfp4_sf_row_bytes + half * 16
            )
            async_copy[16, fill=UInt8(0)](
                src.unsafe_address_space_cast[.GLOBAL](),
                dst,
                src_size=Int32(16) if live else Int32(0),
            )
        async_copy_arrive[noinc=True](k_mbar)

    @staticmethod
    @inline(.always)
    def _nvfp4_chunk(
        q: SIMD[DType.uint32, 4],
        sf_even: UInt32,
        sf_odd: UInt32,
        sel0: UInt32,
        sel1: UInt32,
    ) -> StaticTuple[SIMD[DType.uint32, 4], 2]:
        """Dequantizes one 16-byte packed chunk (its two scale groups) to 32
        e4m3 bytes. `sel0` / `sel1` pick the groups' scale bytes from
        `sf_even` and `sf_odd` (see `reswizzleNVFP4`)."""
        var out = StaticTuple[SIMD[DType.uint32, 4], 2]()
        comptime if _MLA_NVFP4_DEQUANT == 0:
            comptime for g in range(2):
                var sf_byte = (
                    byte_permute(sf_even, sf_odd, sel1 if g == 1 else sel0)
                    & 0xFF
                )
                var scale = bitcast[DType.float8_e4m3fn, 1](
                    SIMD[DType.uint8, 1](UInt8(sf_byte))
                ).cast[DType.float32]()[0]
                var words = SIMD[DType.uint32, 2](q[2 * g], q[2 * g + 1])
                out[g] = nvfp4_dequant_group(
                    bitcast[DType.uint8, 8](words), scale
                )
        else:
            var sf_pair = e4m3x2_to_f16x2(
                byte_permute(sf_even, sf_odd, sel0 | (sel1 << 4))
            )
            comptime for g in range(2):
                var sf2 = byte_permute(
                    sf_pair, sf_pair, UInt32(0x3232 if g == 1 else 0x1010)
                )
                out[g] = nvfp4_dequant_group_f16[_MLA_NVFP4_DEQUANT == 1](
                    q[2 * g], q[2 * g + 1], sf2
                )
        return out

    @staticmethod
    @inline(.always)
    def reswizzleNVFP4(
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem_sw64: SharedMemPointer[Scalar[Self.mma_type]],
        kv_pipeline_gather: Self.GatherPipe,
        kv_pipeline_mma: Self.MmaPipe,
        aux_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Dequantizes one packed-NVFP4 tile per iteration to the SW64 FP8 tile.

        The product `e2m1 * sf` is rounded once to e4m3 and written straight to
        the swizzled destination, so the MMA sees the same bytes an FP8 cache
        holding the dequantized values would give it. The rope dims after the
        packed latent are FP8 already and are copied unchanged.

        Shared memory serves a 16 B access one quarter-warp (eight lanes) at
        a time, so the lanes of a quarter must hit eight different 16 B bank
        slots, on both the loads and the swizzled stores. A thread owns one
        row and the chunks (16 packed bytes: 32 elements, two scale groups) of
        one parity. A quarter covers four adjacent rows by the two chunks of
        one column block: rows differ in parity, and a row pair `cg` takes the
        chunks of block `i ^ cg` at step `i`, which moves its loads by two
        slots and, through the SW64 swizzle's dependence on `row // 2`, its
        stores by one.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime BN_QK: Int = Self.config.BN_QK
        comptime LATENT_DIM: Int = (
            Self.config.input_q_depth - NVFP4_MLA_ROPE_DIM
        )
        # Packed bytes of the latent, where the FP8 rope bytes start.
        comptime ROPE_OFF: Int = LATENT_DIM // 2
        # Column blocks of 64 FP8 columns the latent spans.
        comptime LATENT_BLOCKS: Int = LATENT_DIM // SW64_COLS
        # A chunk is 16 packed bytes: 32 elements, two scale groups.
        comptime QCHUNKS: Int = Self.nvfp4_sf_row_bytes // 2
        comptime ITEMS: Int = BN_QK * QCHUNKS // WARPGROUP_SIZE
        comptime assert (
            BN_QK == 64
            and Self.KVBlockCols == SW64_COLS
            and QCHUNKS == 16
            and ITEMS == 8
            and NVFP4_MLA_ROPE_DIM == 64
        )

        var kv_gather_cons = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem_sw64)

        var tid: Int = thread_idx.x & 0x7F
        var j: Int = (tid % WARP_SIZE) % 8
        var cp: Int = j % 2
        var cg: Int = (j // 2) % 2
        var row: Int = (
            16 * (tid // WARP_SIZE)
            + 4 * ((tid % WARP_SIZE) // 8)
            + (2 * cg + j // 4)
        )
        var pk_base: Int = row * Self.nvfp4_row_bytes + cp * 16
        var sf_base: Int = Self.nvfp4_sf_offset + row * Self.nvfp4_sf_row_bytes
        # `byte_permute` selectors for the scale byte of the chunk's group 0 / 1
        # among the words of blocks `i` and `i ^ 1` (bytes 0..3 and 4..7).
        var sel0: UInt32 = UInt32(4 * cg + 2 * cp)
        var sel1: UInt32 = UInt32(4 * cg + 2 * cp + 1)
        var d0: Int = Self.sw_fp8(row * BN_QK + cp * 32)
        var d1: Int = Self.sw_fp8(row * BN_QK + cp * 32 + 16)
        # The rope chunk `rc` of a row lands at column `LATENT_DIM + 16 * rc`.
        var rope_chunk0: Int = cp + 2 * cg
        var rope_chunk1: Int = cp + 2 * (1 - cg)
        var rope_src0: Int = (
            row * Self.nvfp4_row_bytes + ROPE_OFF + 16 * rope_chunk0
        )
        var rope_src1: Int = (
            row * Self.nvfp4_row_bytes + ROPE_OFF + 16 * rope_chunk1
        )
        var rope_dst0: Int = Self.sw_fp8(row * BN_QK + 16 * rope_chunk0)
        var rope_dst1: Int = Self.sw_fp8(row * BN_QK + 16 * rope_chunk1)

        var tile_idx: Int = 0
        var _sb = stamp_begin[6]()
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            stamp_tile[6](_sb, tile_idx, 0)
            kv_mma_prod.acquire()
            comptime if Self.q_in_tmem:
                # The last MMA stage holds Q until the TMEM copy reads it.
                if tile_idx == Self.num_mma_stages - 1:
                    aux_bars[Self.q_cp_mbar_idx].wait(0)
            stamp_tile[6](_sb, tile_idx, 1)

            var src_u8 = kv_gather_cons.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()

            # The row's scales: word `b` holds the two groups of each of the
            # chunks of block `b`; the lane's chunk of block `i ^ cg` is the
            # `cp`-th half of word `i ^ cg`.
            var sf_a = ld_shared_v4_u32(src_u8, sf_base)
            var sf_b = ld_shared_v4_u32(src_u8, sf_base + 16)
            var qs = StaticTuple[SIMD[DType.uint32, 4], ITEMS]()
            comptime if _MLA_NVFP4_BATCH != 0:
                comptime for i in range(ITEMS):
                    var blk = i + cg if i % 2 == 0 else i - cg
                    qs[i] = ld_shared_v4_u32(src_u8, pk_base + 32 * blk)
            comptime for i in range(ITEMS):
                var blk = i + cg if i % 2 == 0 else i - cg
                var q: SIMD[DType.uint32, 4]
                comptime if _MLA_NVFP4_BATCH != 0:
                    q = qs[i]
                else:
                    q = ld_shared_v4_u32(src_u8, pk_base + 32 * blk)
                var sf_even = sf_a[i] if i < 4 else sf_b[i - 4]
                var sf_odd = sf_a[i ^ 1] if i < 4 else sf_b[(i - 4) ^ 1]
                var dst_block = dst.unsafe_offset(blk * Self.KVBlockElems)
                var out = Self._nvfp4_chunk(q, sf_even, sf_odd, sel0, sel1)
                comptime for g in range(2):
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                        dst_block, d1 if g == 1 else d0, out[g]
                    )

            var rope_block = dst.unsafe_offset(
                LATENT_BLOCKS * Self.KVBlockElems
            )
            comptime for r in range(2):
                var rope = ld_shared_v4_u32(
                    src_u8, rope_src1 if r == 1 else rope_src0
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    rope_block, rope_dst1 if r == 1 else rope_dst0, rope
                )

            fence_async_view_proxy()
            kv_gather_cons.release_all()
            kv_mma_prod.commit_all()
            stamp_tile[6](_sb, tile_idx, 2)
            tile_idx += 1
        stamp_finish[6](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def convertNVFP4Wide(
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_pipeline_gather: Self.GatherPipe,
        kv_pipeline_mma: Self.MmaPipe,
        cvt_blk_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Dequantizes both 64-row bands before overwriting the packed overlay.
        """
        comptime assert (
            _MLA_NVFP4_WIDE_EARLY_STORE == 0 or _MLA_NVFP4_WIDE_EARLY_STORE == 1
        ), "MLA_NVFP4_WIDE_EARLY_STORE is 0 or 1"
        comptime if _MLA_NVFP4_WIDE_EARLY_STORE != 0:
            Self.convertNVFP4WideEarlyStore(
                kv_smem_linear,
                kv_smem,
                kv_pipeline_gather,
                kv_pipeline_mma,
                cvt_blk_bars,
                offset_position,
            )
            return
        comptime assert Self.config.BN_QK == 128 and Self.KVBlockCols == 128
        comptime LATENT = Self.config.input_q_depth - NVFP4_MLA_ROPE_DIM
        comptime assert (
            LATENT == 512 and Self.nvfp4_sf_row_bytes == 32
        ), "the wide NVFP4 ring requires the 512-latent, 64-rope cache"
        comptime BLOCKS = LATENT // Self.KVBlockCols
        comptime CHUNKS = LATENT // 32
        comptime HELD_PER_BAND = CHUNKS + 2
        comptime sw = make_swizzle[
            Self.mma_type, TensorMapSwizzle.SWIZZLE_128B
        ]()
        var gather = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var output = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
            stage_elems=Self.KVStageElems,
        ](kv_pipeline_mma, kv_smem)
        var tid = thread_idx.x & 127
        var j = (tid % 32) % 8
        var cp = j % 2
        var cg = (j // 2) % 2
        var row0 = 16 * (tid // 32) + 4 * ((tid % 32) // 8) + 2 * cg + j // 4
        var tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        for tile in range(tiles):
            gather.wait()
            output.acquire()
            var src = gather.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = output.stage_base_ptr()
            var held = StaticTuple[SIMD[DType.uint32, 4], 2 * HELD_PER_BAND]()
            comptime for band in range(2):
                var row = row0 + 64 * band
                var sf = Self.nvfp4_sf_offset + row * Self.nvfp4_sf_row_bytes
                var sf_a = ld_shared_v4_u32(src, sf)
                var sf_b = ld_shared_v4_u32(src, sf + 16)
                comptime for i in range(CHUNKS // 2):
                    var blk = i + cg if i % 2 == 0 else i - cg
                    var q = ld_shared_v4_u32(
                        src, row * Self.nvfp4_row_bytes + cp * 16 + 32 * blk
                    )
                    var sf_even = sf_a[i] if i < 4 else sf_b[i - 4]
                    var sf_odd = sf_a[i ^ 1] if i < 4 else sf_b[(i - 4) ^ 1]
                    var out = Self._nvfp4_chunk(
                        q,
                        sf_even,
                        sf_odd,
                        UInt32(4 * cg + 2 * cp),
                        UInt32(4 * cg + 2 * cp + 1),
                    )
                    held[band * HELD_PER_BAND + 2 * i] = out[0]
                    held[band * HELD_PER_BAND + 2 * i + 1] = out[1]
                comptime for r in range(2):
                    var rc = cp + 2 * (cg if r == 0 else 1 - cg)
                    held[band * HELD_PER_BAND + CHUNKS + r] = ld_shared_v4_u32(
                        src, row * Self.nvfp4_row_bytes + LATENT // 2 + 16 * rc
                    )
            # Both bands may overwrite packed bytes belonging to other threads.
            named_barrier[Int32(WARPGROUP_SIZE)](3)
            gather.release_all()
            comptime for block in range(BLOCKS + 1):
                comptime for band in range(2):
                    var row = row0 + 64 * band
                    var dst_block = dst.unsafe_offset(
                        UInt32(block * Self.KVBlockElems)
                    )
                    comptime if block < BLOCKS:
                        comptime for half in range(2):
                            comptime i = 2 * block + half
                            var col = (
                                (cg if half == 0 else 1 - cg) * 64
                            ) + cp * 32
                            comptime for g in range(2):
                                st_shared_v4_b32_at_bf16_elem_off[
                                    out_dtype=Self.mma_type
                                ](
                                    dst_block,
                                    sw(row * 128 + col + 16 * g),
                                    held[
                                        (
                                            0 if (_MLA_NVFP4_BREAK & 1)
                                            != 0 else band
                                        )
                                        * HELD_PER_BAND
                                        + 2 * i
                                        + g
                                    ],
                                )
                    else:
                        comptime for r in range(2):
                            var rc = cp + 2 * (cg if r == 0 else 1 - cg)
                            st_shared_v4_b32_at_bf16_elem_off[
                                out_dtype=Self.mma_type
                            ](
                                dst_block,
                                sw(row * 128 + 16 * rc),
                                held[
                                    (0 if (_MLA_NVFP4_BREAK & 1) != 0 else band)
                                    * HELD_PER_BAND
                                    + CHUNKS
                                    + r
                                ],
                            )
                fence_async_view_proxy()
                _ = cvt_blk_bars[
                    output.stage_index() * UInt32(Self.NumKVBlocks)
                    + UInt32(block)
                ].arrive()
            output.commit_all()

    @staticmethod
    @inline(.always)
    def convertNVFP4WideEarlyStore(
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_pipeline_gather: Self.GatherPipe,
        kv_pipeline_mma: Self.MmaPipe,
        cvt_blk_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Stores nonoverlapping blocks early and holds the overlay's blocks."""
        comptime assert Self.config.BN_QK == 128 and Self.KVBlockCols == 128
        comptime LATENT = Self.config.input_q_depth - NVFP4_MLA_ROPE_DIM
        comptime assert (
            LATENT == 512 and Self.nvfp4_sf_row_bytes == 32
        ), "the wide NVFP4 ring requires the 512-latent, 64-rope cache"
        comptime BLOCKS = LATENT // Self.KVBlockCols
        comptime CHUNKS = LATENT // 32
        comptime FREE_BLOCKS = min(
            BLOCKS, Self.cvt_src_offset // Self.KVBlockElems
        )
        comptime FREE_CHUNKS = 2 * FREE_BLOCKS
        comptime HELD_CHUNKS = CHUNKS // 2 - FREE_CHUNKS
        comptime HELD_PER_BAND = 2 * HELD_CHUNKS + 2
        comptime assert (
            Self.mma_bytes == 1
            and Self.num_stages == Self.num_mma_stages
            and Self.LinStageElems == Self.KVStageElems
            and FREE_BLOCKS > 0
            and FREE_BLOCKS * Self.KVBlockElems <= Self.cvt_src_offset
            and Self.cvt_src_offset
            + Self.config.BN_QK
            * (Self.nvfp4_row_bytes + Self.nvfp4_sf_row_bytes)
            <= Self.KVStageElems
        ), "early stores must stay below the same stage's packed overlay"
        comptime sw = make_swizzle[
            Self.mma_type, TensorMapSwizzle.SWIZZLE_128B
        ]()
        var gather = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var output = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
            stage_elems=Self.KVStageElems,
        ](kv_pipeline_mma, kv_smem)
        var tid = thread_idx.x & 127
        var j = (tid % 32) % 8
        var cp = j % 2
        var cg = (j // 2) % 2
        var row0 = 16 * (tid // 32) + 4 * ((tid % 32) // 8) + 2 * cg + j // 4
        var tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        for tile in range(tiles):
            gather.wait()
            output.acquire()
            var src = gather.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = output.stage_base_ptr()
            var held = StaticTuple[SIMD[DType.uint32, 4], 2 * HELD_PER_BAND]()
            # Scale words of each band: `sf[2 * band]` and `sf[2 * band + 1]`.
            var sf = StaticTuple[SIMD[DType.uint32, 4], 4]()
            comptime ITEMS_PER_BAND = CHUNKS // 2
            comptime for k in range(2 * ITEMS_PER_BAND):
                comptime band = k // ITEMS_PER_BAND
                comptime i = k % ITEMS_PER_BAND
                var row = row0 + 64 * band
                var src_row = row0 + 64 * (
                    0 if (_MLA_NVFP4_BREAK & 1) != 0 else band
                )
                comptime if i == 0:
                    var sf_off = (
                        Self.nvfp4_sf_offset + src_row * Self.nvfp4_sf_row_bytes
                    )
                    sf[2 * band] = ld_shared_v4_u32(src, sf_off)
                    sf[2 * band + 1] = ld_shared_v4_u32(src, sf_off + 16)
                var blk = i + cg if i % 2 == 0 else i - cg
                var q = ld_shared_v4_u32(
                    src, src_row * Self.nvfp4_row_bytes + cp * 16 + 32 * blk
                )
                var sf_even = (
                    sf[2 * band][i] if i < 4 else sf[2 * band + 1][i - 4]
                )
                var sf_odd = (
                    sf[2 * band][i ^ 1] if i
                    < 4 else sf[2 * band + 1][(i - 4) ^ 1]
                )
                var out = Self._nvfp4_chunk(
                    q,
                    sf_even,
                    sf_odd,
                    UInt32(4 * cg + 2 * cp),
                    UInt32(4 * cg + 2 * cp + 1),
                )
                comptime if i < FREE_CHUNKS:
                    var dst_block = dst.unsafe_offset(
                        (i // 2) * Self.KVBlockElems
                    )
                    var col = (cg if i % 2 == 0 else 1 - cg) * 64 + cp * 32
                    comptime for g in range(2):
                        st_shared_v4_b32_at_bf16_elem_off[
                            out_dtype=Self.mma_type
                        ](dst_block, sw(row * 128 + col + 16 * g), out[g])
                    # Both orders reach band 1 of an odd item after every
                    # other store of its block.
                    comptime if band == 1 and i % 2 == 1:
                        # Each arrival follows this thread's stores for
                        # both 64-row bands of the complete block.
                        fence_async_view_proxy()
                        _ = cvt_blk_bars[
                            output.stage_index() * UInt32(Self.NumKVBlocks)
                            + UInt32(i // 2)
                        ].arrive()
                else:
                    held[band * HELD_PER_BAND + 2 * (i - FREE_CHUNKS)] = out[0]
                    held[
                        band * HELD_PER_BAND + 2 * (i - FREE_CHUNKS) + 1
                    ] = out[1]
                comptime if i == ITEMS_PER_BAND - 1:
                    comptime for r in range(2):
                        var rc = cp + 2 * (cg if r == 0 else 1 - cg)
                        held[
                            band * HELD_PER_BAND + 2 * HELD_CHUNKS + r
                        ] = ld_shared_v4_u32(
                            src,
                            src_row * Self.nvfp4_row_bytes
                            + LATENT // 2
                            + 16 * rc,
                        )
            # The remaining blocks overlap other threads' packed source bytes.
            named_barrier[Int32(WARPGROUP_SIZE)](3)
            gather.release_all()
            comptime for block in range(FREE_BLOCKS, BLOCKS + 1):
                comptime for band in range(2):
                    var row = row0 + 64 * band
                    var dst_block = dst.unsafe_offset(block * Self.KVBlockElems)
                    comptime if block < BLOCKS:
                        comptime for half in range(2):
                            comptime i = 2 * block + half
                            var col = (
                                (cg if half == 0 else 1 - cg) * 64
                            ) + cp * 32
                            comptime for g in range(2):
                                st_shared_v4_b32_at_bf16_elem_off[
                                    out_dtype=Self.mma_type
                                ](
                                    dst_block,
                                    sw(row * 128 + col + 16 * g),
                                    held[
                                        (
                                            0 if (_MLA_NVFP4_BREAK & 1)
                                            != 0 else band
                                        )
                                        * HELD_PER_BAND
                                        + 2 * (i - FREE_CHUNKS)
                                        + g
                                    ],
                                )
                    else:
                        comptime for r in range(2):
                            var rc = cp + 2 * (cg if r == 0 else 1 - cg)
                            st_shared_v4_b32_at_bf16_elem_off[
                                out_dtype=Self.mma_type
                            ](
                                dst_block,
                                sw(row * 128 + 16 * rc),
                                held[
                                    (0 if (_MLA_NVFP4_BREAK & 1) != 0 else band)
                                    * HELD_PER_BAND
                                    + 2 * HELD_CHUNKS
                                    + r
                                ],
                            )
                fence_async_view_proxy()
                _ = cvt_blk_bars[
                    output.stage_index() * UInt32(Self.NumKVBlocks)
                    + UInt32(block)
                ].arrive()
            output.commit_all()

    @staticmethod
    @inline(.always)
    def convertNVFP4(
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_pipeline_gather: Self.GatherPipe,
        kv_pipeline_mma: Self.MmaPipe,
        cvt_blk_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """WG3 under the NVFP4 ring: dequantizes each packed tile in place.

        The tile's packed rows and scales (`reswizzleNVFP4`'s source layout)
        sit at the top of its own SW128 FP8 stage. The thread mapping is
        `reswizzleNVFP4`'s, which also keeps the SW128 stores free of bank
        conflicts, and item `i` lands in SW128 block `i // 2`. The blocks
        below the source are stored and published to the QK MMA on
        `cvt_blk_bars` as they convert; the rest wait in registers until
        every thread has read its packed bytes.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime BN_QK: Int = Self.config.BN_QK
        comptime LATENT_DIM: Int = (
            Self.config.input_q_depth - NVFP4_MLA_ROPE_DIM
        )
        comptime ROPE_OFF: Int = LATENT_DIM // 2
        comptime LATENT_BLOCKS: Int = LATENT_DIM // Self.KVBlockCols
        comptime QCHUNKS: Int = Self.nvfp4_sf_row_bytes // 2
        comptime ITEMS: Int = BN_QK * QCHUNKS // WARPGROUP_SIZE
        comptime num_free_blocks: Int = Self.cvt_src_offset // (
            Self.KVBlockElems
        )
        comptime free_items: Int = 2 * num_free_blocks
        comptime num_held: Int = 2 * (ITEMS - free_items)
        comptime assert (
            BN_QK == 64
            and Self.KVBlockCols == 2 * SW64_COLS
            and QCHUNKS == 16
            and ITEMS == 8
            and NVFP4_MLA_ROPE_DIM == 64
            and Self.NumKVBlocks == LATENT_BLOCKS + 1
            and free_items <= ITEMS
        )
        comptime sw = make_swizzle[
            Self.mma_type, TensorMapSwizzle.SWIZZLE_128B
        ]()

        var kv_gather_cons = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
            stage_elems=Self.KVStageElems,
        ](kv_pipeline_mma, kv_smem)

        var tid: Int = thread_idx.x & 0x7F
        var j: Int = (tid % WARP_SIZE) % 8
        var cp: Int = j % 2
        var cg: Int = (j // 2) % 2
        var row: Int = (
            16 * (tid // WARP_SIZE)
            + 4 * ((tid % WARP_SIZE) // 8)
            + (2 * cg + j // 4)
        )
        var pk_base: Int = row * Self.nvfp4_row_bytes + cp * 16
        var sf_base: Int = Self.nvfp4_sf_offset + row * Self.nvfp4_sf_row_bytes
        var sel0: UInt32 = UInt32(4 * cg + 2 * cp)
        var sel1: UInt32 = UInt32(4 * cg + 2 * cp + 1)
        # Item `i` converts the chunk of SW64-wide block `i ^ cg`, which is
        # half `(i % 2) ^ cg` of SW128 block `i // 2`.
        var row_off: Int = row * Self.KVBlockCols
        var de0: Int = sw(row_off + cg * SW64_COLS + cp * 32)
        var de1: Int = sw(row_off + cg * SW64_COLS + cp * 32 + 16)
        var do0: Int = sw(row_off + (1 - cg) * SW64_COLS + cp * 32)
        var do1: Int = sw(row_off + (1 - cg) * SW64_COLS + cp * 32 + 16)
        var rope_chunk0: Int = cp + 2 * cg
        var rope_chunk1: Int = cp + 2 * (1 - cg)
        var rope_src0: Int = (
            row * Self.nvfp4_row_bytes + ROPE_OFF + 16 * rope_chunk0
        )
        var rope_src1: Int = (
            row * Self.nvfp4_row_bytes + ROPE_OFF + 16 * rope_chunk1
        )
        var rope_dst0: Int = sw(row_off + 16 * rope_chunk0)
        var rope_dst1: Int = sw(row_off + 16 * rope_chunk1)

        var tile_idx: Int = 0
        var _sb = stamp_begin[6]()
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            stamp_tile[6](_sb, tile_idx, 0)
            kv_mma_prod.acquire()
            stamp_tile[6](_sb, tile_idx, 1)

            var src_u8 = kv_gather_cons.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()
            var blk_bar0 = cvt_blk_bars.unsafe_offset(
                kv_mma_prod.stage_index() * UInt32(Self.NumKVBlocks)
            )

            var sf_a = ld_shared_v4_u32(src_u8, sf_base)
            var sf_b = ld_shared_v4_u32(src_u8, sf_base + 16)
            var held = StaticTuple[SIMD[DType.uint32, 4], num_held + 2]()
            var qs = StaticTuple[SIMD[DType.uint32, 4], ITEMS]()
            comptime if _MLA_NVFP4_BATCH != 0:
                comptime for i in range(ITEMS):
                    var blk = i + cg if i % 2 == 0 else i - cg
                    qs[i] = ld_shared_v4_u32(src_u8, pk_base + 32 * blk)
                held[num_held] = ld_shared_v4_u32(src_u8, rope_src0)
                held[num_held + 1] = ld_shared_v4_u32(src_u8, rope_src1)
            comptime for i in range(ITEMS):
                var blk = i + cg if i % 2 == 0 else i - cg
                var q: SIMD[DType.uint32, 4]
                comptime if _MLA_NVFP4_BATCH != 0:
                    q = qs[i]
                else:
                    q = ld_shared_v4_u32(src_u8, pk_base + 32 * blk)
                var sf_even = sf_a[i] if i < 4 else sf_b[i - 4]
                var sf_odd = sf_a[i ^ 1] if i < 4 else sf_b[(i - 4) ^ 1]
                var out = Self._nvfp4_chunk(q, sf_even, sf_odd, sel0, sel1)
                comptime if i < free_items:
                    var dst_block = dst.unsafe_offset(
                        (i // 2) * Self.KVBlockElems
                    )
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                        dst_block, do0 if i % 2 == 1 else de0, out[0]
                    )
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                        dst_block, do1 if i % 2 == 1 else de1, out[1]
                    )
                    comptime if i % 2 == 1:
                        fence_async_view_proxy()
                        _ = blk_bar0[i // 2].arrive()
                else:
                    held[2 * (i - free_items)] = out[0]
                    held[2 * (i - free_items) + 1] = out[1]
            comptime if _MLA_NVFP4_BATCH == 0:
                held[num_held] = ld_shared_v4_u32(src_u8, rope_src0)
                held[num_held + 1] = ld_shared_v4_u32(src_u8, rope_src1)

            # Every packed byte of the slot is read: its FP8 blocks may land.
            named_barrier[Int32(WARPGROUP_SIZE)](3)
            kv_gather_cons.release_all()

            comptime for i in range(free_items, ITEMS):
                var dst_block = dst.unsafe_offset((i // 2) * Self.KVBlockElems)
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block,
                    do0 if i % 2 == 1 else de0,
                    held[2 * (i - free_items)],
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block,
                    do1 if i % 2 == 1 else de1,
                    held[2 * (i - free_items) + 1],
                )
                comptime if i % 2 == 1:
                    fence_async_view_proxy()
                    _ = blk_bar0[i // 2].arrive()
            var rope_block = dst.unsafe_offset(
                LATENT_BLOCKS * Self.KVBlockElems
            )
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                rope_block, rope_dst0, held[num_held]
            )
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                rope_block, rope_dst1, held[num_held + 1]
            )
            fence_async_view_proxy()
            _ = blk_bar0[LATENT_BLOCKS].arrive()

            kv_mma_prod.commit_all()
            stamp_tile[6](_sb, tile_idx, 2)
            tile_idx += 1
        stamp_finish[6](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def _cvt_chunk(
        q: SIMD[DType.uint32, 4],
        scale_stage: SharedMemPointer[UInt8],
        row: Int,
        fp8_col: Int,
    ) -> StaticTuple[SIMD[DType.uint32, 4], 2]:
        """Converts 16 FP8 bytes of `row` at `fp8_col` to two BF16 chunks,
        scaled when blockwise. 16 bytes never straddle a scale block (32
        columns or wider)."""
        var p = cvt_fp8x16_from_u32x4_to_bf16x16_packed_2xu32x4[
            fp8_dtype=Self.kv_type, out_dtype=Self.mma_type
        ](q)
        comptime if Self.blockwise:
            var scale_idx = (
                row * Self.config.scales_per_token
                + fp8_col // Self.config.scale_block_size
            )
            var sc: UInt32
            comptime if Self.config.fp8_scale_bf16:
                var bits = UInt32(
                    scale_stage.unsafe_bitcast[UInt16]()[scale_idx]
                )
                sc = bits | (bits << 16)
            else:
                sc = e8m0_to_bf16_broadcast(scale_stage[scale_idx])
            p[0] = hmul2_bf16x8_by_scalar[Self.mma_type](p[0], sc)
            p[1] = hmul2_bf16x8_by_scalar[Self.mma_type](p[1], sc)
        return p

    @staticmethod
    @inline(.always)
    def convertFP8(
        kv_smem_linear: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        scale_smem: SharedMemPointer[UInt8],
        kv_pipeline_gather: Self.GatherPipe,
        kv_pipeline_mma: Self.MmaPipe,
        cvt_blk_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """WG3 under `kv_convert`: converts each FP8 tile in place to BF16.

        The tile's FP8 rows (linear, `input_q_depth` bytes each) sit in the
        upper half of its own BF16 stage. Thread mapping as in
        `reswizzleFP8`: 32 groups of 4 threads, rows `g` and `g + 32`, 16 B
        per thread per 64-column block. Blocks below the FP8 rows are stored
        as they convert; the rest wait in registers until every thread has
        read its FP8 bytes, after which the slot's gather side is released.
        Each block is published to the QK MMA on `cvt_blk_bars` as it lands;
        the PV MMA waits for the whole tile.
        """
        comptime assert (
            Self.config.BN_QK == 64
        ), "the in-place convert maps 64-key tiles: rows g and g + 32"
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime GROUP_SIZE: Int = 4
        comptime NUM_GROUPS: Int = WARPGROUP_SIZE // GROUP_SIZE
        comptime fp8_row_stride: Int = Self.config.input_q_depth
        # One 64-column BF16 block per 64 FP8 bytes of the row as stored.
        comptime num_blocks: Int = fp8_row_stride // (GROUP_SIZE * 16)
        comptime block_cols = Self.KVBlockCols
        comptime assert block_cols == GROUP_SIZE * 16
        comptime num_free_blocks = (Self.KVStageElems // 2) // (
            Self.KVBlockElems
        )
        comptime num_held_blocks = num_blocks - num_free_blocks
        comptime sw = make_swizzle[
            Self.mma_type, TensorMapSwizzle.SWIZZLE_128B
        ]()

        var kv_gather_cons = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.gather_producers,
            num_consumer=Self.gather_consumers,
            num_stages=Self.num_stages,
            stage_elems=Self.LinStageElems,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)

        var lane: Int = thread_idx.x & 0x7F
        var group_idx: Int = lane // GROUP_SIZE
        var idx_in_group: Int = lane % GROUP_SIZE
        var row_0: Int = group_idx
        var row_1: Int = NUM_GROUPS + group_idx
        var fp8_base_0: Int = row_0 * fp8_row_stride + idx_in_group * 16
        var fp8_base_1: Int = row_1 * fp8_row_stride + idx_in_group * 16
        var col: Int = idx_in_group * 16
        var sw_0a: Int = sw(row_0 * block_cols + col)
        var sw_0b: Int = sw(row_0 * block_cols + col + 8)
        var sw_1a: Int = sw(row_1 * block_cols + col)
        var sw_1b: Int = sw(row_1 * block_cols + col + 8)

        var tile_idx: Int = 0
        var _sb = stamp_begin[6]()
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            stamp_tile[6](_sb, tile_idx, 0)
            kv_mma_prod.acquire()
            stamp_tile[6](_sb, tile_idx, 1)

            var src_u8 = kv_gather_cons.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()
            var blk_bar0 = cvt_blk_bars.unsafe_offset(
                kv_mma_prod.stage_index() * UInt32(Self.NumKVBlocks)
            )
            var scale_stage = scale_smem.unsafe_offset(
                kv_gather_cons.stage_index()
                * UInt32(Self.config.scale_smem_per_stage)
            )

            comptime for c in range(num_free_blocks):
                var fp8_col = c * GROUP_SIZE * 16 + col
                var p0 = Self._cvt_chunk(
                    ld_shared_v4_u32(src_u8, fp8_base_0 + c * GROUP_SIZE * 16),
                    scale_stage,
                    row_0,
                    fp8_col,
                )
                var p1 = Self._cvt_chunk(
                    ld_shared_v4_u32(src_u8, fp8_base_1 + c * GROUP_SIZE * 16),
                    scale_stage,
                    row_1,
                    fp8_col,
                )
                var dst_block = dst.unsafe_offset(c * Self.KVBlockElems)
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_0a, p0[0]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_0b, p0[1]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_1a, p1[0]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_1b, p1[1]
                )
                fence_async_view_proxy()
                _ = blk_bar0[c].arrive()

            var held = StaticTuple[SIMD[DType.uint32, 4], 4 * num_held_blocks]()
            comptime for h in range(num_held_blocks):
                comptime c = num_free_blocks + h
                var fp8_col = c * GROUP_SIZE * 16 + col
                var p0 = Self._cvt_chunk(
                    ld_shared_v4_u32(src_u8, fp8_base_0 + c * GROUP_SIZE * 16),
                    scale_stage,
                    row_0,
                    fp8_col,
                )
                var p1 = Self._cvt_chunk(
                    ld_shared_v4_u32(src_u8, fp8_base_1 + c * GROUP_SIZE * 16),
                    scale_stage,
                    row_1,
                    fp8_col,
                )
                held[4 * h] = p0[0]
                held[4 * h + 1] = p0[1]
                held[4 * h + 2] = p1[0]
                held[4 * h + 3] = p1[1]

            # Every FP8 byte of the slot is read: its BF16 blocks may land.
            named_barrier[Int32(WARPGROUP_SIZE)](3)
            kv_gather_cons.release_all()

            comptime for h in range(num_held_blocks):
                var dst_block = dst.unsafe_offset(
                    (num_free_blocks + h) * Self.KVBlockElems
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_0a, held[4 * h]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_0b, held[4 * h + 1]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_1a, held[4 * h + 2]
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_1b, held[4 * h + 3]
                )
                fence_async_view_proxy()
                _ = blk_bar0[num_free_blocks + h].arrive()
            # A NoPE row leaves tail blocks the FP8 rows overlap, so the
            # once-per-CTA zeroing does not survive the next gather.
            comptime for c in range(num_blocks, Self.NumKVBlocks):
                var dst_block = dst.unsafe_offset(c * Self.KVBlockElems)
                comptime zero = SIMD[DType.uint32, 4](0)
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_0a, zero
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_0b, zero
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_1a, zero
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.mma_type](
                    dst_block, sw_1b, zero
                )
                fence_async_view_proxy()
                _ = blk_bar0[c].arrive()

            kv_mma_prod.commit_all()
            stamp_tile[6](_sb, tile_idx, 2)
            tile_idx += 1
        stamp_finish[6](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def load_q(
        q_tma: QOTMATile[
            dtype=Self.mma_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.operand_swizzle,
        ],
        q_smem: SharedMemPointer[Scalar[Self.mma_type]],
        mbar_q: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        """Loads Q; under the direct gather this is warp 8's only work."""
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return
        var elect_mask = elect()
        var row: Int = offset_position.q_row_offset
        expect_bytes_pred(
            mbar_q,
            Int32(Self.config.BM * Self.config.input_q_depth * Self.mma_bytes),
            elect_mask,
        )
        if elect_mask != 0:
            comptime q_elems = type_of(q_tma).tile_shape.element_types[
                0
            ].static_value * type_of(q_tma).tile_shape.element_types[
                1
            ].static_value
            comptime q_tt_layout = tt_row_major[q_elems]()
            var q_smem_tensor = TileTensor[
                Self.mma_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](q_smem.unsafe_bitcast[Scalar[Self.mma_type]](), q_tt_layout)
            q_tma.async_copy(q_smem_tensor, mbar_q[], (0, row))

    @staticmethod
    @inline(.always)
    def gather_sw64(
        k_tma: Self.KTMA,
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_pipeline_mma: Self.MmaPipe,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
        idx_bars: Self.IdxBars,
        idx_smem_base: SharedMemPointer[Int32],
        topk: Int,
        extra_k_tma: Self.KTMA,
    ):
        """WG3 under the direct gather: TMA gathers each tile straight into
        the SW64 ring.

        Warp `w` owns rows `[w * BN_QK / 4, (w + 1) * BN_QK / 4)` of every
        tile; its elected lane arms the slot's full barrier for its bytes
        (count 4, one arrival per warp), loads its index quads and issues one
        FP8 SW64 gather4 per 4-row chunk and 64-column group, at the k-major
        SW64 offset the K and V descriptors read. A warp cannot run a ring
        ahead: refilling slot s for tile t + M waits on PV(t), which needs all
        four arrivals for tile t.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        var kv_prod = DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
            stage_elems=Self.KVStageElems if Self.kv_sw128 else 0,
        ](kv_pipeline_mma, kv_smem)
        var idx_cons = idx_bars.consumer()
        var wq = UInt32(warp_id[broadcast=True]()) & 3
        var is_leader = elect() != 0

        Self._gather_sw64_range(
            kv_prod,
            idx_cons,
            idx_smem_base,
            k_tma,
            num_orig_tiles,
            wq,
            is_leader,
            idx_bars.end(),
        )
        comptime if Self.has_extra_kv:
            Self._gather_sw64_range(
                kv_prod,
                idx_cons,
                idx_smem_base,
                extra_k_tma,
                num_k_tiles - num_orig_tiles,
                wq,
                is_leader,
                idx_bars.end(),
            )

    @staticmethod
    @inline(.always)
    def _gather_sw64_range(
        mut kv_prod: DecodeKVProducer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
            stage_elems=Self.KVStageElems if Self.kv_sw128 else 0,
        ],
        mut idx_cons: ConsumerPipeline[Self.idx_stages],
        idx_smem_base: SharedMemPointer[Int32],
        cur_k_tma: Self.KTMA,
        num_tiles: Int,
        wq: UInt32,
        is_leader: Bool,
        aux_bars: MBarType,
    ):
        comptime box_w = Self.kv_gather4_box_w
        comptime assert (
            box_w == Self.KVBlockCols
        ), "direct gather box is one K/V SMEM column block"
        # 4-row chunks per warp, and column groups per row as stored.
        comptime C = Self.config.BN_QK // 16
        comptime NUM_CG = ceildiv(Self.config.input_q_depth, box_w)
        comptime skip_rope = _MLA_P6_BREAK == 5
        comptime cg_loaded = NUM_CG - 1 if skip_rope else NUM_CG
        comptime warp_bytes = (
            C * 4 * cg_loaded * box_w * size_of[Self.kv_type]()
        )
        # Index quads loaded back to back before their gathers.
        comptime IB = min(C, 4)

        var t: Int = 0
        var _sb = stamp_begin[6]()
        while t < num_tiles:
            kv_prod.acquire()
            comptime if Self.q_in_tmem:
                # The last MMA stage holds Q until the TMEM copy reads it; a
                # later wait on the completed phase returns at once.
                if kv_prod.stage_index() == UInt32(Self.num_mma_stages - 1):
                    aux_bars[Self.q_cp_mbar_idx].wait(0)
            stamp_tile[6](_sb, t, 0)
            idx_cons.wait()
            stamp_tile[6](_sb, t, 1)
            if is_leader:
                var k_mbar = kv_prod.producer_mbar()
                k_mbar[].expect_bytes(Int32(warp_bytes))
                var stage = kv_prod.stage_base_ptr().unsafe_bitcast[
                    Scalar[Self.k_tma_dtype]
                ]()
                var idx_u8 = (
                    idx_smem_base.unsafe_offset(
                        idx_cons.state.index() * UInt32(Self.config.BN_QK)
                    )
                ).unsafe_bitcast[UInt8]()
                var chunk0 = Int(wq) * C
                # A runtime loop over index batches: unrolled, 128-key tiles
                # keep every batch's addresses live and spill.
                for b in range(C // IB):
                    var rows = StaticTuple[SIMD[DType.uint32, 4], IB]()
                    comptime for j in range(IB):
                        rows[j] = ld_shared_v4_u32[ordered=True](
                            idx_u8, (chunk0 + b * IB + j) * 16
                        )
                    # The last batch's quads are in registers: release the
                    # index slot before issuing its gathers, so warp 11 can
                    # refill it during the issue span.
                    if b == C // IB - 1:
                        _ = idx_cons.consumer_mbar()[].arrive()
                    # Column groups outer and rolled, row quads inner:
                    # IB operand tuples per trip, each stepped in place,
                    # so one tuple's release after its UTMALDG overlaps
                    # the other quads' issue. Unrolled, every gather gets
                    # its own uniform-register tuple and ptxas spills them;
                    # `num_tiles >> 40` (always 0) is a trip-count term
                    # neither LLVM nor ptxas can fold. The 32-bit trip
                    # counter keeps the loop test to one compare.
                    var cg_p = stage.unsafe_offset(
                        (chunk0 + b * IB) * 4 * box_w
                    )
                    var col = Int32(0)
                    var n_cg32 = UInt32(cg_loaded) + UInt32(num_tiles >> 40)
                    var k = UInt32(0)
                    while k < n_cg32:
                        # ptxas otherwise unrolls this loop 8x in the
                        # non-split kernel and spills the tuples.
                        inlined_assembly[
                            '.pragma "nounroll";',
                            NoneType,
                            constraints="",
                            has_side_effect=True,
                        ]()
                        comptime for j in range(IB):
                            var r = rows[j].cast[DType.int32]()
                            cur_k_tma.async_copy_gather4[
                                eviction_policy=CacheEviction.EVICT_LAST
                            ](
                                TileTensor(
                                    cg_p.unsafe_offset(j * 4 * box_w),
                                    tt_row_major[4, box_w](),
                                ),
                                k_mbar[],
                                col,
                                r[0],
                                r[1],
                                r[2],
                                r[3],
                            )
                        cg_p = cg_p.unsafe_offset(Self.config.BN_QK * box_w)
                        col += Int32(box_w)
                        k += 1
            idx_cons.state.step()
            kv_prod.commit_step()
            stamp_tile[6](_sb, t, 2)
            t += 1
        stamp_finish[6](_sb, num_tiles)

    @staticmethod
    @inline(.always)
    def cp_h64_q192_to_tmem(
        q_smem: SharedMemPointer[Scalar[Self.mma_type]], tmem_addr: UInt32
    ):
        comptime assert Self.h64_q192_tmem
        comptime assert Self.num_stages == 2 and Self.num_mma_stages == 2
        comptime assert Self.KVBlockCols == 64 and Self.config.BK_QK == 576
        comptime assert (
            Self.config.TMEM_S0 + Self.num_stages * Self.config.s_slot_stride()
            <= Self.config.TMEM_Q_LO
            and Self.config.TMEM_Q_LO + 192 <= Self.config.sm100_tmem_cols
        ), "the h64 Q192 operand overlaps a live TMEM region"
        var desc = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        # Persistent Q SMEM keeps this copy independent of the KV gather.
        comptime for block in range(6):
            comptime col = Self.config.TMEM_Q_LO + block * 32
            comptime for sub in range(8):
                comptime off = ((block + 3) * Self.BlockElems + sub * 8) * 2
                tcgen05_cp[
                    cta_group=1,
                    datapaths=64,
                    bits=128,
                    multicast="warpx2::02_13",
                ](
                    tmem_addr + UInt32(col + sub * 4),
                    (desc + UInt32(off)).descriptor(),
                )

    @staticmethod
    @inline(.always)
    def mma_h64_q192_tmem(
        q: MMASmemDescriptorPair,
        k: MMASmemDescriptorPair,
        scores: UInt32,
        tmem_addr: UInt32,
        elect_mask: Int32,
    ):
        comptime assert Self.h64_q192_tmem
        bulk_mma_ws[
            Self.UMMAQKTSS.mma_kind,
            Self.mma_type,
            Self.mma_type,
            a_BMN=Self.config.BM,
            a_BK=Self.config.BK_QK,
            a_swizzle=Self.operand_swizzle,
            a_is_k_major=True,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.z_row_elems,
            b_swizzle=Self.kv_swizzle,
            b_is_k_major=True,
            num_k_mmas=12,
            operand_size=2,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAQKTSS.UMMAInstDesc, q, k, scores, UInt32(0), elect_mask)
        comptime for part in range(2):
            comptime depth = 192 if part == 0 else 448
            comptime q_col = (Self.config.TMEM_Q_LO + (0 if part == 0 else 128))
            # Descriptor offsets are whole SW128 depth blocks, preserving
            # the source swizzle base and its row stride.
            comptime k_bytes = depth // 64 * Self.KVBlockElems * 2
            bulk_mma_ws_ts[
                Self.UMMAQKTSS.mma_kind,
                Self.mma_type,
                b_BMN=Self.config.BN_QK,
                b_BK=Self.z_row_elems,
                b_swizzle=Self.kv_swizzle,
                b_is_k_major=True,
                num_k_mmas=16 if part == 0 else 8,
                operand_size=2,
                tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            ](
                Self.UMMAQKTSS.UMMAInstDesc,
                tmem_addr + UInt32(q_col),
                k + UInt32(k_bytes),
                scores,
                UInt32(1),
                elect_mask,
            )

    @staticmethod
    @inline(.always)
    def cp_q_to_tmem_full(q: MMASmemDescriptorPair, tmem_addr: UInt32):
        """Copies Q's nine SW64 blocks to TMEM columns `TMEM_Q_FULL` on,
        16 per block, in the layout the TS QK contraction reads."""
        comptime assert Self.q_in_tmem
        comptime assert (
            Self.config.TMEM_S0 + Self.num_stages * Self.config.s_slot_stride()
            <= Self.config.TMEM_Q_FULL
            and Self.config.TMEM_Q_FULL + Self.config.TMEM_Q_FULL_COLS
            <= Self.config.tmem_corr_scale
            and Self.NumQKBlocks * 16 == Self.config.TMEM_Q_FULL_COLS
        ), "Q's TMEM columns overlap a live region"
        comptime for qb in range(Self.NumQKBlocks):
            comptime for sub in range(SW64_COLS // 16):
                tcgen05_cp[
                    cta_group=1,
                    datapaths=64,
                    bits=128,
                    multicast="warpx2::02_13",
                ](
                    tmem_addr
                    + UInt32(Self.config.TMEM_Q_FULL + qb * 16 + sub * 4),
                    (
                        q
                        + UInt32(
                            # Negative controls swap Q blocks 0 and 1:
                            # `MLA_Q_TMEM=2` in the NVFP4 kernel, `=3` in
                            # the FP8 one.
                            (
                                (qb ^ 1) if qb < 2
                                and _MLA_Q_TMEM
                                == (2 if Self.is_nvfp4_kv else 3) else qb
                            )
                            * Self.BlockElems
                            + sub * 16
                        )
                    ).descriptor(),
                )

    @staticmethod
    @inline(.always)
    def mma_q_tmem(
        k: MMASmemDescriptorPair,
        scores: UInt32,
        tmem_addr: UInt32,
        elect_mask: Int32,
    ):
        """Issues the QK contraction with Q from TMEM (`q_in_tmem`)."""
        comptime assert Self.q_in_tmem
        bulk_mma_ws_ts[
            Self.UMMAQKTSS.mma_kind,
            Self.mma_type,
            b_BMN=Self.config.BN_QK,
            b_BK=Self.UMMAQKTSS.K_BK,
            b_swizzle=Self.kv_swizzle,
            b_is_k_major=True,
            num_k_mmas=Self.UMMAQKTSS.num_k_mmas,
            operand_size=1,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
            mma_k=Self.UMMAQKTSS.MMA_K,
        ](
            Self.UMMAQKTSS.UMMAInstDesc,
            tmem_addr + UInt32(Self.config.TMEM_Q_FULL),
            k,
            scores,
            UInt32(0),
            elect_mask,
        )

    @staticmethod
    @inline(.always)
    def mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.mma_type]],
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline_mma: Self.MmaPipe,
        kv_pipeline_gather: Self.GatherPipe,
        cvt_blk_bars: MBarType,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.s_slot_stride())

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = (Self.KVStageElems * Self.mma_bytes)

        mbar_q[].wait(0)
        comptime if Self.h64_q192_tmem:
            tcgen05_fence_after()
            if elect_mask != 0:
                Self.cp_h64_q192_to_tmem(q_smem, tmem_addr)
        # A walk shorter than the MMA ring never refills the last stage,
        # where Q sits, so it reads Q from SMEM throughout and skips the
        # copy, which would otherwise queue ahead of its first QK and PV.
        var q_from_tmem = False
        comptime if Self.q_in_tmem:
            tcgen05_fence_after()
            q_from_tmem = num_k_tiles >= Self.num_mma_stages
        var tile_idx: Int = 0
        var _sb = stamp_begin[3]()
        while tile_idx < num_k_tiles:
            s_prod.acquire()
            stamp_tile[3](_sb, tile_idx, 0)
            var slot_idx: UInt32 = s_prod.slot_index()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            comptime if Self.kv_convert:
                # Each block issues as soon as WG3 publishes it. Slot s is
                # filled at tiles s, s + num_stages, ...; its n-th fill
                # completes phase n & 1.
                var k_slot_index = kv_cons.stage_index[qk_stage=0]()
                var b_desc = k_descriptor + k_slot_index * UInt32(
                    stage_stride_in_bytes
                )
                var blk_bar0 = cvt_blk_bars.unsafe_offset(
                    k_slot_index * UInt32(Self.NumKVBlocks)
                )
                var blk_parity = UInt32((tile_idx // Self.num_stages) & 1)
                comptime for blk in range(Self.NumKVBlocks):
                    blk_bar0[blk].wait(blk_parity)
                    comptime if Self.nvfp4_ring:
                        # The FP8 contraction's 18 k-mmas do not split
                        # evenly over 5 SW128 blocks; the last block holds
                        # the rope and the row's padding.
                        Self.UMMAQKTSS.mma_cols[
                            col_start=blk * Self.KVBlockCols,
                            cols=Self.KVBlockCols,
                        ](
                            a=q_descriptor,
                            b=b_desc,
                            c=s_tmem_slot,
                            c_scale=UInt32(0),
                            elect=elect_mask,
                        )
                    else:
                        Self.UMMAQKTSS.mma_block[
                            block_idx=blk, num_blocks=Self.NumKVBlocks
                        ](
                            a=q_descriptor,
                            b=b_desc,
                            c=s_tmem_slot,
                            c_scale=UInt32(0),
                            elect=elect_mask,
                        )
                tcgen05_fence_before()
                s_prod.commit_mma(elect_mask)
                elect_mma_arrive(
                    kv_pipeline_gather.consumer_mbar[0](k_slot_index),
                    elect_mask,
                )
            else:
                kv_cons.wait[qk_stage=0]()
                stamp_tile[3](_sb, tile_idx, 1)
                var k_slot_index = kv_cons.stage_index[qk_stage=0]()
                comptime if Self.h64_q192_tmem:
                    Self.mma_h64_q192_tmem(
                        q_descriptor,
                        k_descriptor
                        + k_slot_index * UInt32(stage_stride_in_bytes),
                        s_tmem_slot,
                        tmem_addr,
                        elect_mask,
                    )
                elif Self.q_in_tmem:
                    if tile_idx == 0 or not q_from_tmem:
                        # The first QK reads Q from SMEM, so it does not
                        # wait for the copy, which follows tile 0's commits
                        # (a commit tracks every earlier tcgen05 op); every
                        # later QK reads TMEM.
                        Self.UMMAQKTSS.mma[stage_idx=0](
                            a=q_descriptor,
                            b=k_descriptor
                            + k_slot_index * UInt32(stage_stride_in_bytes),
                            c=s_tmem_slot,
                            c_scale=UInt32(0),
                            elect=elect_mask,
                        )
                    else:
                        Self.mma_q_tmem(
                            k_descriptor
                            + k_slot_index * UInt32(stage_stride_in_bytes),
                            s_tmem_slot,
                            tmem_addr,
                            elect_mask,
                        )
                else:
                    Self.UMMAQKTSS.mma[stage_idx=0](
                        a=q_descriptor,
                        b=k_descriptor
                        + k_slot_index * UInt32(stage_stride_in_bytes),
                        c=s_tmem_slot,
                        c_scale=UInt32(0),
                        elect=elect_mask,
                    )
                tcgen05_fence_before()
                s_prod.commit_mma(elect_mask)
            kv_cons.release[qk_stage=0](elect_mask)
            comptime if Self.q_in_tmem:
                if tile_idx == 0 and q_from_tmem:
                    if elect_mask != 0:
                        Self.cp_q_to_tmem_full(q_descriptor, tmem_addr)
                    # Completes once the copy has read Q's SMEM, the last
                    # MMA stage.
                    elect_mma_arrive(
                        cvt_blk_bars.unsafe_offset(Self.q_cp_mbar_idx),
                        elect_mask,
                    )
            stamp_tile[3](_sb, tile_idx, 2)
            tile_idx += 1
        stamp_finish[3](_sb, num_k_tiles)

    @staticmethod
    @inline(.always)
    def mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.mma_type]],
        p_smem: SharedMemPointer[Scalar[Self.mma_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline_mma: Self.MmaPipe,
        kv_pipeline_gather: Self.GatherPipe,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            TopkLengthsPtrType=Self.TopkLengthsPtrType,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime s_stride = UInt32(Self.config.s_slot_stride())
        var kv_cons = DecodeKVConsumer[
            Self.mma_type,
            Self.config,
            num_producer=Self.mma_producers,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.KVBlockCols
        comptime o_round_cols = Self.config.o_round_cols()
        comptime kv_stage_stride_in_bytes = (Self.KVStageElems * Self.mma_bytes)
        comptime p_stage_stride_in_bytes = (
            Self.p_stage_stride * Self.mma_bytes
        )
        comptime block_stride_in_bytes = (Self.KVBlockElems * Self.mma_bytes)

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        var _sb = stamp_begin[4]()
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            stamp_tile[4](_sb, tile_idx, 0)
            var p_slot_index = p_cons.wait()
            stamp_tile[4](_sb, tile_idx, 1)
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            comptime for block in range(0, Self.NumVBlocks, block_step):
                o_prod.acquire()
                comptime if block == 0:
                    stamp_tile[4](_sb, tile_idx, 2)
                comptime for band in range(
                    Self.config.BN_QK // Self.config.BK_PV
                ):
                    Self.UMMAPVSS.mma[stage_idx=0](
                        a=p_descriptor
                        + p_slot_index * UInt32(p_stage_stride_in_bytes)
                        + UInt32(
                            band * Self.config.BM * SW64_COLS * Self.mma_bytes
                        ),
                        b=v_descriptor
                        + v_slot_index * UInt32(kv_stage_stride_in_bytes)
                        + UInt32(block * block_stride_in_bytes)
                        + UInt32(
                            band * SW64_COLS * Self.KVBlockCols * Self.mma_bytes
                        ),
                        c=o_tmem
                        + UInt32(block // block_step) * UInt32(o_round_cols),
                        c_scale=c_scale if band == 0 else UInt32(1),
                        elect=elect_mask,
                    )
                o_prod.commit_mma(elect_mask)
            p_cons.release_mma(elect_mask)

            comptime if Self.kv_convert:
                elect_mma_arrive(
                    kv_pipeline_gather.consumer_mbar[0](v_slot_index),
                    elect_mask,
                )
            kv_cons.release[qk_stage=0](elect_mask)
            tcgen05_fence_before()

            stamp_tile[4](_sb, tile_idx, 3)
            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
        stamp_finish[4](_sb, num_k_tiles)


def sparse_qkv_fp8_carve_bytes(
    config: MLA_SM100_Decode_Config,
    idx_smem_align_pad: Int = 12,
    sparse_row_parts: Int = 1,
    corr_smem: Bool = False,
    role_metadata_bytes: Int = 0,
) -> Int:
    """Returns the end of `MLA_SM100_Decode_Sparse_QKV_FP8`'s SMEM carve.

    Recomputes `MLA_SM100_Decode_Sparse_QKV_FP8.carve_bytes` from the config
    alone, for host-side pins; the launcher checks the two agree at compile
    time.

    Args:
        config: The kernel's config (`config_g` at the launcher).
        idx_smem_align_pad: The kernel's worst-case idx_smem alignment pad.
        sparse_row_parts: Index slots per key in a stage of `idx_smem` (2
            for an NVFP4 cache's value and scale rows, else 1).
        corr_smem: Whether the form keeps the correction warpgroup's SMEM.
        role_metadata_bytes: Dedicated metadata appended after the index ring.

    Returns:
        The carve end in bytes from the dynamic SMEM base.
    """
    var n = config.num_kv_stages
    var m = config.num_kv_mma_stages
    if config.cta_group == 2:
        # The pair's carve: Q, the per-CTA Z ring (M stages), the V ring
        # (N stages), the P ring, max/li, the exact barrier count (12 per
        # stage), the TMEM pointer, the idx alignment pad and the index
        # ring.
        var cta_keys = config.BN_QK // config.cta_group
        var z_stage = cta_keys * config.kv_row_elems()
        var e_stage = cta_keys * config.padded_depth
        var stage_bars = 12
        var p_ring = config.BM * config.BN_QK * n
        var operand_bytes = (
            config.BM * config.padded_q_depth
            + z_stage * m
            + e_stage * n
            + p_ring
        ) * config.operand_size
        var out_blocks = config.depth // (config.BN_PV // 4)
        var blocks_per_stage = (
            2 if (config.MMA_PV_N // 2) // (config.BN_PV // 4) != 0 else 1
        )
        var out_mbars = 2 * (out_blocks // blocks_per_stage)
        var idx_n = _SPARSE_PAIR_IDX_STAGES
        var mbars = 1 + stage_bars * n + 2 * (idx_n - n) + 4 + 2 + 4 + out_mbars
        return (
            operand_bytes
            + 3 * WARPGROUP_SIZE * 4
            + mbars * config.mbar_size
            + 4
            + idx_smem_align_pad
            + idx_n * config.BN_QK * sparse_row_parts * 4
        )
    var kv_stage = config.BN_QK * config.kv_row_elems()
    # A packed NVFP4 linear slot (`MLA_NVFP4_PACKED_LINEAR`) is smaller than
    # a full stage; the gate mirrors the kernel's `nvfp4_packed_linear`.
    var linear = 0 if config.direct_gather or config.kv_convert else (
        (
            config.gather_stage_bytes if (
                sparse_row_parts == 2 and config.gather_stage_bytes > 0
            ) else kv_stage
        )
        * n
    )
    var scale_ring = config.scale_smem_per_stage * n if config.kv_convert else 0
    var p_ring = 0 if config.p_in_kv() else config.BM * config.BN_QK * n
    var q_bytes = 0 if config.q_in_tmem else config.BM * config.padded_q_depth
    var operand_bytes = (
        q_bytes + linear + kv_stage * m + p_ring
    ) * config.operand_size
    var f32_bytes = (4 if corr_smem else 3) * WARPGROUP_SIZE * 4
    var out_blocks = config.depth // (config.BN_PV // 4)
    var blocks_per_stage = (
        2 if (config.MMA_PV_N // 2) // (config.BN_PV // 4) != 0 else 1
    )
    var out_mbars = 2 * (out_blocks // blocks_per_stage)
    var mbars = 1 + 2 * n + 2 * m + 2 * n + 2 * n + 10 + out_mbars + 2 * n
    if config.kv_convert:
        mbars += config.kv_row_elems() // config.kv_block_cols() * n
    if config.q_in_tmem:
        mbars += 1
    return (
        operand_bytes
        + f32_bytes
        + scale_ring
        + mbars * config.mbar_size
        + 4
        + idx_smem_align_pad
        + n * config.BN_QK * sparse_row_parts * 4
        + role_metadata_bytes
    )
