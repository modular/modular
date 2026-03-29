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
"""TMA load warp logic for FA4 (SM100 Flash Attention)."""

from std.sys import size_of
from std.gpu.memory import CacheEviction
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    KProducerPipeline,
    VProducerPipeline,
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
from .smem import SM100AttentionSMem


@always_inline
def fa4_load[
    KVLUTType: MHAOperand,
    MaskType: MHAMask,
    config: FA4Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
](
    smem: SM100AttentionSMem[config],
    score_row: UInt32,
    num_keys: UInt32,
    seq_info: SeqInfo,
    max_seq_len: MaxSeqLenType,
    mask: MaskType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BM=config.BM // 2,
        depth=config.qk_depth,
        group=config.group,
        decoding=False,
        num_qk_stages=config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=config.BN,
        BK=config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN=config.BN,
        BK=config.padded_ov_depth,
    ],
    kv_lut: KVLUTType,
):
    comptime assert KVLUTType.dtype == config.qkv_dtype
    comptime qkv_type = KVLUTType.dtype
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime group = config.group
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null

    var mbars = smem.misc_mbars()

    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.qk_depth,
        config.padded_qk_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    comptime KPipeType = KProducerPipeline[KVLUTType.dtype, config]
    comptime VPipeType = VProducerPipeline[KVLUTType.dtype, config]

    # If two-qo, we produce qkv in a pattern of
    # q0 & k0, q1, v0, k1, v1, k2, v2...
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

    var kv_head_idx: UInt32 = seq_info.head_idx // UInt32(group)

    var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
        smem.q_smem()
    )
    comptime q_elements = HalfBM * config.BK0
    comptime assert q_elements == q_elems
    comptime q_bytes = size_of[qkv_type]() * q_elements

    var q_gmem_row: UInt32 = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )
    var q_head_idx: UInt32 = seq_info.head_idx
    e = elect()

    var kv_row: UInt32 = mask.start_column[BM, BN, page_size](score_row)
    var kv_gmem_row: UInt32 = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
    var iter_count: UInt32 = (
        mask.last_masked_set_end[BM, BN, page_size](score_row, num_keys) - 1
    )

    comptime if config.use_fused_kv:
        # ---- Fused KV mode ----
        # Single StagedPipeline with alternating K and V stages.
        # Stages: K0, V0, K1, V1, ...
        # For MHA: padded_qk_depth == padded_ov_depth, rope_depth == 0.
        # num_qk_stages=1 in fused mode.

        var kv_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.k_smem_base()
        )
        comptime kv_stage_elems = config.padded_ov_depth * BN
        comptime k_elems = type_of(k_tma_op).tile_shape[0] * type_of(
            k_tma_op
        ).tile_shape[1] * type_of(k_tma_op).tile_shape[2]
        comptime KType = TileTensor[
            KVLUTType.dtype,
            type_of(tt_row_major[k_elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime v_elems = type_of(v_tma_op).tile_shape[0] * type_of(
            v_tma_op
        ).tile_shape[1] * type_of(v_tma_op).tile_shape[2]
        comptime VType = TileTensor[
            KVLUTType.dtype,
            type_of(tt_row_major[v_elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime k_bytes = config.BK0 * BN * size_of[qkv_type]()
        comptime v_bytes = config.padded_ov_depth * BN * size_of[qkv_type]()
        comptime qk_fused_bytes = k_bytes + q_bytes

        comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
        var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}
        kv_pipeline.state._phase = 1  # producer starts at phase 1

        # ---- Peeled: K0 + Q0 on same barrier ----
        var k0_mbar = kv_pipeline.producer_mbar()
        if e != 0:
            k0_mbar[].expect_bytes(Int32(qk_fused_bytes))
        # Copy Q0
        if e != 0:
            q_tma_op.async_copy[eviction_policy=CacheEviction.EVICT_FIRST](
                QType(q_smem, tt_row_major[q_elems]()),
                k0_mbar[],
                StaticTuple[UInt32, 3](0, q_head_idx, q_gmem_row),
            )
        # Copy K0
        if e != 0:
            k_tma_op.async_copy(
                KType(
                    kv_smem
                    + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                    tt_row_major[k_elems](),
                ),
                k0_mbar[],
                StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
            )
        kv_pipeline.state.step()  # step -> stage 1

        # ---- Q1 (separate barrier) ----
        q_gmem_row += UInt32(HalfBM)
        var q1_mbar = mbars.q1_wait_mbar()
        if e != 0:
            q1_mbar[0].expect_bytes(Int32(q_bytes))
        if e != 0:
            comptime q1_smem_offset = q_elements * 1  # num_qk_stages=1
            q_tma_op.async_copy(
                QType(q_smem + q1_smem_offset, tt_row_major[q_elems]()),
                q1_mbar[0],
                StaticTuple[UInt32, 3](0, q_head_idx, q_gmem_row),
            )

        # ---- V0 ----
        kv_pipeline.producer_acquire()
        var v0_mbar = kv_pipeline.producer_mbar()
        if e != 0:
            v0_mbar[].expect_bytes(Int32(v_bytes))
        if e != 0:
            v_tma_op.async_copy(
                VType(
                    kv_smem
                    + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                    tt_row_major[v_elems](),
                ),
                v0_mbar[],
                StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
            )
        kv_pipeline.state.step()

        comptime check_mask = mask.nonfull_sets[BM, BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        # ---- KV producer loop ----
        while iter_count != 0:
            iter_count -= 1
            kv_row += UInt32(config.BN)

            comptime if check_mask:
                if (
                    mask.status(
                        Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                        Index[dtype=DType.int32](BM, BN),
                    )
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)

            # Produce Kn
            kv_pipeline.producer_acquire()
            var kn_mbar = kv_pipeline.producer_mbar()
            if e != 0:
                kn_mbar[].expect_bytes(Int32(k_bytes))
            if e != 0:
                k_tma_op.async_copy(
                    KType(
                        kv_smem
                        + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                        tt_row_major[k_elems](),
                    ),
                    kn_mbar[],
                    StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
                )
            kv_pipeline.state.step()

            # Produce Vn
            kv_pipeline.producer_acquire()
            var vn_mbar = kv_pipeline.producer_mbar()
            if e != 0:
                vn_mbar[].expect_bytes(Int32(v_bytes))
            if e != 0:
                v_tma_op.async_copy(
                    VType(
                        kv_smem
                        + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                        tt_row_major[v_elems](),
                    ),
                    vn_mbar[],
                    StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
                )
            kv_pipeline.state.step()

    else:
        # ---- Split KV mode (original) ----

        comptime qk_bytes = KPipeType.bytes + q_bytes
        var k_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            smem.v_smem_base()
        )
        var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
        var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

        var mbark0: KPipeType.KPairType

        mbark0 = pipeline_k.get_k[qk_stage=0]()  # no wait
        # copy q0
        if e != 0:
            # Q0
            mbark0.mbar[].expect_bytes(Int32(qk_bytes))
        # copy q0
        if e != 0:
            q_tma_op.async_copy[eviction_policy=CacheEviction.EVICT_FIRST](
                QType(q_smem, tt_row_major[q_elems]()),
                mbark0.mbar[],
                StaticTuple[UInt32, 3](0, q_head_idx, q_gmem_row),
            )
        # copy k0
        if e != 0:  # K0
            k_tma_op.async_copy(
                mbark0.smem,
                mbark0.mbar[],
                StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
            )

        comptime for qk_stage in range(1, config.num_qk_stages):
            comptime d_idx = qk_stage * config.BK0
            mbark = pipeline_k.get_k[qk_stage=qk_stage]()  # no wait
            if e != 0:
                mbark.mbar[].expect_bytes(Int32(qk_bytes))
            if e != 0:
                q_tma_op.async_copy[eviction_policy=CacheEviction.EVICT_FIRST](
                    QType(
                        q_smem + q_elements * qk_stage, tt_row_major[q_elems]()
                    ),
                    mbark.mbar[],
                    StaticTuple[UInt32, 3](
                        UInt32(d_idx), q_head_idx, q_gmem_row
                    ),
                )
            if e != 0:
                k_tma_op.async_copy(
                    mbark.smem,
                    mbark.mbar[],
                    StaticTuple[UInt32, 3](
                        UInt32(d_idx), kv_head_idx, kv_gmem_row
                    ),
                )

        pipeline_k.commit_step()
        # Q1
        q_gmem_row += UInt32(HalfBM)
        var q1_mbar = mbars.q1_wait_mbar()

        comptime for qk_stage in range(config.num_qk_stages):
            comptime q_smem_offset = q_elements * (
                config.num_qk_stages + qk_stage
            )
            comptime d_idx = qk_stage * config.BK0
            if e != 0:
                q1_mbar[qk_stage].expect_bytes(Int32(q_bytes))
            if e != 0:
                q_tma_op.async_copy(
                    QType(q_smem + q_smem_offset, tt_row_major[q_elems]()),
                    q1_mbar[qk_stage],
                    StaticTuple[UInt32, 3](
                        UInt32(d_idx), q_head_idx, q_gmem_row
                    ),
                )
        # copy v0
        mbarv0 = pipeline_v.get_v(e)
        if e != 0:
            v_tma_op.async_copy(
                mbarv0.smem,
                mbarv0.mbar[],
                StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
            )
        pipeline_v.commit_step()
        comptime check_mask = mask.nonfull_sets[BM, BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK
        # kv producer loop
        while iter_count != 0:
            iter_count -= 1
            kv_row += UInt32(config.BN)

            comptime if check_mask:
                if (
                    mask.status(
                        Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                        Index[dtype=DType.int32](BM, BN),
                    )
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)

            # produce k
            comptime for k_stage in range(config.num_qk_stages):
                pipeline_k.acquire_k[qk_stage=k_stage]()
                mbarkn = pipeline_k.get_k[qk_stage=k_stage](e)
                comptime d_idx = k_stage * config.BK0
                if e != 0:
                    k_tma_op.async_copy(
                        mbarkn.smem,
                        mbarkn.mbar[],
                        StaticTuple[UInt32, 3](
                            UInt32(d_idx), kv_head_idx, kv_gmem_row
                        ),
                    )
            pipeline_k.commit_step()

            pipeline_v.acquire_v()
            mbarvn = pipeline_v.get_v(e)
            if e != 0:
                v_tma_op.async_copy(
                    mbarvn.smem,
                    mbarvn.mbar[],
                    StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
                )
            pipeline_v.commit_step()
