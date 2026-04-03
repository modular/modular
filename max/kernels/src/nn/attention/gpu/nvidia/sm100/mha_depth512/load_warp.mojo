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
"""TMA load warp logic for depth=512 pair-CTA SM100 attention.

Each CTA in the pair loads its own half of K/V data into its local SMEM.
The pair-CTA MMA instruction reads from both SMs' SMEM to combine the halves.

K is split along BN rows: even CTA loads K[0:BN//2, :], odd loads K[BN//2:BN, :].
V is split into V_lo and V_hi (separate pipeline slots, each [BK1, ov_depth//4]):
  V_lo: even loads V[:, 0:ov_depth//4], odd loads V[:, ov_depth//4:ov_depth//2]
  V_hi: even loads V[:, ov_depth//2:3*ov_depth//4], odd loads V[:, 3*ov_depth//4:ov_depth]
Q is per-CTA: even loads Q[0:64, :], odd loads Q[64:128, :].

All TMA loads use async_multicast_load_3d[cta_group=2] with a per-CTA mask.
The cta_group=2 ensures the leader CTA's barrier tracks byte arrivals from both
CTAs. Only the leader CTA calls expect_bytes and wait.

Mask computations use PairBM (BM*2=128) so both CTAs make identical skip
decisions. If one CTA skips a tile and the other doesn't, barriers desync.
"""

from std.sys import size_of
from std.gpu.primitives.cluster import block_rank_in_cluster
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from layout.tma_async import SharedMemBarrier
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem
from .barriers import Depth512MBars
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    StagedPipeline,
)
from nn.attention.gpu.nvidia.sm90.attention import (
    KVTMATile,
    MHAPosition,
    OptionalPointer,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from nn.attention.mha_utils import OptionallyStaticInt, _is_decoding
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple


@always_inline
def depth512_load[
    KVLUTType: MHAOperand,
    MaskType: MHAMask,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
](
    smem: Depth512AttentionSMem[config=config],
    score_row: UInt32,
    num_keys: UInt32,
    seq_info: SeqInfo,
    max_seq_len: MaxSeqLenType,
    mask: MaskType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BM=config.BM,
        depth=config.qk_depth,
        group=config.group,
        decoding=False,
        num_qk_stages=config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=config.BN // 2,
        BK=config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=config.BK1,
        BK=config.ov_depth // 4,
    ],
    kv_lut: KVLUTType,
):
    comptime assert KVLUTType.dtype == config.qkv_dtype
    comptime qkv_type = KVLUTType.dtype
    comptime BM = config.BM
    comptime BN = config.BN
    comptime BK0 = config.BK0
    comptime BK1 = config.BK1
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime num_kv_stages = config.num_kv_stages
    comptime group = config.group
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group = config.cta_group
    comptime qkv_size = size_of[qkv_type]()

    # Full pair-CTA M dimension for mask computations.
    # CRITICAL: Both CTAs must use the same M=128 so they make identical
    # skip/load decisions. If one CTA skips a tile and the other doesn't,
    # pipeline barriers desync and the kernel hangs.
    comptime PairBM = BM * 2

    comptime PositionType = MHAPosition[
        PairBM,
        BN,
        config.qk_depth,
        config.qk_depth,  # padded_qk_depth = qk_depth for depth512
        config.num_q_heads,
        group,
        _is_decoding[MaxSeqLenType](),
    ]

    # ---- CTA identity and multicast mask ------------------------------------

    var cta_rank = block_rank_in_cluster() % 2
    var is_leader = cta_rank == 0
    var local_mask: UInt16 = UInt16(1) << UInt16(cta_rank)

    # ---- TileTensor types for TMA destinations ------------------------------
    # TMA only uses .ptr — flat row_major TileTensor is sufficient.

    comptime q_elems = type_of(q_tma_op).tile_shape[0] * type_of(
        q_tma_op
    ).tile_shape[1] * type_of(q_tma_op).tile_shape[2]
    comptime QType = TileTensor[
        KVLUTType.dtype,
        type_of(tt_row_major[q_elems]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    comptime kv_elems = type_of(k_tma_op).tile_shape[0] * type_of(
        k_tma_op
    ).tile_shape[1] * type_of(k_tma_op).tile_shape[2]
    comptime KVType = TileTensor[
        KVLUTType.dtype,
        type_of(tt_row_major[kv_elems]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ]

    # ---- Byte sizes for expect_bytes ----------------------------------------

    comptime q_stage_elements = BM * BK0
    comptime q_stage_bytes = q_stage_elements * qkv_size
    # Per-CTA bytes: K tile is BN//2 rows, V tile is ov_depth//4 cols.
    # cta_group multiplier below accounts for both CTAs in the cluster.
    comptime k_stage_bytes = (BN // 2) * BK0 * qkv_size
    comptime v_stage_bytes = BK1 * (config.ov_depth // 4) * qkv_size
    comptime qk_expect_bytes = cta_group * (q_stage_bytes + k_stage_bytes)
    comptime k_expect_bytes = cta_group * k_stage_bytes
    comptime v_expect_bytes = cta_group * v_stage_bytes

    # ---- SMEM pointers ------------------------------------------------------

    var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.q_smem()
    )
    var kv_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.kv_smem_base()
    )

    # ---- Pipeline setup -----------------------------------------------------

    var mbars = Depth512MBars[num_kv_stages](smem.mbar_base())
    comptime KVPipeType = StagedPipeline[num_kv_stages, 1]
    var kv_pipeline: KVPipeType = {mbars.get_kv_mbars()}
    kv_pipeline.state._phase = 1  # producer starts at phase 1

    # ---- GMEM coordinates ---------------------------------------------------

    var q_gmem_row: UInt32 = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )
    # Each CTA loads its own BM rows of Q.
    q_gmem_row += UInt32(cta_rank) * UInt32(BM)

    var q_head_idx: UInt32 = seq_info.head_idx
    var kv_head_idx: UInt32 = seq_info.head_idx // UInt32(group)

    e = elect()

    var kv_row: UInt32 = mask.start_column[PairBM, BN, page_size](score_row)
    var kv_gmem_row: UInt32 = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
    var iter_count: UInt32 = (
        mask.last_masked_set_end[PairBM, BN, page_size](score_row, num_keys) - 1
    )

    # CTA-specific offsets for K rows and V depth columns.
    # With cta_group=2, each CTA loads ov_depth//4 V columns.
    # V_lo: leader [0, ov_depth/4), peer [ov_depth/4, ov_depth/2)
    # V_hi: leader [ov_depth/2, 3*ov_depth/4), peer [3*ov_depth/4, ov_depth)
    var k_row_offset: UInt32 = UInt32(cta_rank) * UInt32(BN // 2)
    var v_lo_col_offset = Int(cta_rank) * (config.ov_depth // 4)
    var v_hi_col_offset = config.ov_depth // 2 + Int(cta_rank) * (
        config.ov_depth // 4
    )

    # Mask check uses PairBM (128) so both CTAs make identical skip decisions.
    # If one CTA skips a tile and the other doesn't, the pipeline barriers
    # desync and the kernel hangs or produces wrong results.
    comptime check_mask = mask.nonfull_sets[PairBM, BN]()[
        0
    ] == TileMaskStatus.UNKNOWN_MASK

    # ---- Peeled first iteration: Q + K depth stages -------------------------
    # Q is loaded once (4 depth stages) co-arrived with K depth stages on the
    # same barriers. This allows MMA to start Q@K' as soon as the first
    # Q+K stage is ready.

    comptime for qk_stage in range(num_qk_stages):
        comptime d_idx = qk_stage * BK0
        var mbar = kv_pipeline.producer_mbar()

        # Leader CTA: set expected bytes for both CTAs.
        if is_leader:
            if e != 0:
                mbar[].expect_bytes(Int32(qk_expect_bytes))

        # Both CTAs: load Q depth stage.
        if e != 0:
            q_tma_op.async_multicast_load_3d[cta_group](
                QType(
                    q_smem + q_stage_elements * qk_stage,
                    tt_row_major[q_elems](),
                ),
                mbar[],
                (d_idx, Int(q_head_idx), Int(q_gmem_row)),
                local_mask,
            )

        # Both CTAs: load K depth stage (each CTA loads its BN//2 half).
        if e != 0:
            k_tma_op.async_multicast_load_3d[cta_group](
                KVType(
                    kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
                    tt_row_major[kv_elems](),
                ),
                mbar[],
                (
                    d_idx,
                    Int(kv_head_idx),
                    Int(kv_gmem_row + k_row_offset),
                ),
                local_mask,
            )
        kv_pipeline.state.step()

    # ---- Peeled first iteration: V_lo BN stages ------------------------------
    # V_lo and V_hi occupy separate pipeline slots, each [BK1, ov_depth//4].

    comptime for pv_stage in range(num_pv_stages):
        kv_pipeline.producer_acquire()
        var mbar = kv_pipeline.producer_mbar()

        if is_leader:
            if e != 0:
                mbar[].expect_bytes(Int32(v_expect_bytes))

        if e != 0:
            v_tma_op.async_multicast_load_3d[cta_group](
                KVType(
                    kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
                    tt_row_major[kv_elems](),
                ),
                mbar[],
                (
                    v_lo_col_offset,
                    Int(kv_head_idx),
                    Int(kv_gmem_row) + pv_stage * BK1,
                ),
                local_mask,
            )
        kv_pipeline.state.step()

    # ---- Peeled first iteration: V_hi BN stages ------------------------------

    comptime for pv_stage in range(num_pv_stages):
        kv_pipeline.producer_acquire()
        var mbar = kv_pipeline.producer_mbar()

        if is_leader:
            if e != 0:
                mbar[].expect_bytes(Int32(v_expect_bytes))

        if e != 0:
            v_tma_op.async_multicast_load_3d[cta_group](
                KVType(
                    kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
                    tt_row_major[kv_elems](),
                ),
                mbar[],
                (
                    v_hi_col_offset,
                    Int(kv_head_idx),
                    Int(kv_gmem_row) + pv_stage * BK1,
                ),
                local_mask,
            )
        kv_pipeline.state.step()

    # ---- Main KV producer loop ----------------------------------------------

    while iter_count != 0:
        iter_count -= 1
        kv_row += UInt32(BN)

        # Mask check: skip fully-masked tiles.
        # CRITICAL: Uses PairBM (BM*2=128), NOT per-CTA BM=64.
        # Both CTAs must make identical skip/load decisions to stay
        # synchronized, since the pipeline barriers and MMA are coordinated
        # across the pair. If one CTA skips and the other doesn't, barriers
        # desync and the kernel hangs.
        comptime if check_mask:
            if (
                mask.status(
                    Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                    Index[dtype=DType.int32](PairBM, BN),
                )
                == TileMaskStatus.FULL_MASK
            ):
                continue

        kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)

        # ---- K depth stages (num_qk_stages loads) ----
        comptime for qk_stage in range(num_qk_stages):
            comptime d_idx = qk_stage * BK0
            kv_pipeline.producer_acquire()
            var mbar = kv_pipeline.producer_mbar()

            if is_leader:
                if e != 0:
                    mbar[].expect_bytes(Int32(k_expect_bytes))

            if e != 0:
                k_tma_op.async_multicast_load_3d[cta_group](
                    KVType(
                        kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
                        tt_row_major[kv_elems](),
                    ),
                    mbar[],
                    (
                        d_idx,
                        Int(kv_head_idx),
                        Int(kv_gmem_row + k_row_offset),
                    ),
                    local_mask,
                )
            kv_pipeline.state.step()

        # ---- V_lo BN stages ----
        comptime for pv_stage in range(num_pv_stages):
            kv_pipeline.producer_acquire()
            var mbar = kv_pipeline.producer_mbar()

            if is_leader:
                if e != 0:
                    mbar[].expect_bytes(Int32(v_expect_bytes))

            if e != 0:
                v_tma_op.async_multicast_load_3d[cta_group](
                    KVType(
                        kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
                        tt_row_major[kv_elems](),
                    ),
                    mbar[],
                    (
                        v_lo_col_offset,
                        Int(kv_head_idx),
                        Int(kv_gmem_row) + pv_stage * BK1,
                    ),
                    local_mask,
                )
            kv_pipeline.state.step()

        # ---- V_hi BN stages ----
        comptime for pv_stage in range(num_pv_stages):
            kv_pipeline.producer_acquire()
            var mbar = kv_pipeline.producer_mbar()

            if is_leader:
                if e != 0:
                    mbar[].expect_bytes(Int32(v_expect_bytes))

            if e != 0:
                v_tma_op.async_multicast_load_3d[cta_group](
                    KVType(
                        kv_smem + kv_pipeline.state.index() * UInt32(kv_elems),
                        tt_row_major[kv_elems](),
                    ),
                    mbar[],
                    (
                        v_hi_col_offset,
                        Int(kv_head_idx),
                        Int(kv_gmem_row) + pv_stage * BK1,
                    ),
                    local_mask,
                )
            kv_pipeline.state.step()
