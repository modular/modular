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

from sys import size_of
from gpu.memory import CacheEviction
from layout.layout import Layout
from layout.tma_async import SharedMemBarrier
from nn.fa4_config import FA4Config, EnableForcedOrdering
from nn.sm100_attention_utils import (
    SharedMemPointer,
    SharedMemLT,
    FA4MiscMBars,
    elect,
    TMAProducerPipeline,
    KProducerPipeline,
    VProducerPipeline,
)
from nn.mha_fa3_utils import (
    KVTMATile,
    MHAPosition,
    OptionalPointer,
    PositionSummary,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_tile_scheduler import SeqInfo
from nn.mha_utils import OptionallyStaticInt, _is_decoding
from utils.index import Index
from utils.static_tuple import StaticTuple


@always_inline
fn fa4_load[
    KVLUTType: MHAOperand,
    MaskType: MHAMask,
    config: FA4Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
](
    mbars: FA4MiscMBars[
        num_qk_stages = config.num_qk_stages,
        num_pv_stages = config.num_pv_stages,
        num_kv_stages = config.num_kv_stages,
        separate_kv=True,
        use_order_barriers=EnableForcedOrdering,
    ],
    score_row: UInt32,
    num_keys: UInt32,
    seq_info: SeqInfo,
    max_seq_len: MaxSeqLenType,
    mask: MaskType,
    q_tma_op: QTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BM = config.BM // 2,
        depth = config.depth,
        group = config.group,
        decoding=False,
        num_qk_stages = config.num_qk_stages,
    ],
    k_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN = config.BN,
        BK = config.BK0,
    ],
    v_tma_op: KVTMATile[
        KVLUTType.dtype,
        config.swizzle_mode,
        BN = config.BN,
        BK = config.padded_depth,
    ],
    kv_lut: KVLUTType,
):
    comptime qkv_type = KVLUTType.dtype
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime group = config.group
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null

    # Offset calculations
    comptime q_offset: Int32 = 0
    comptime kv_offset: Int32 = q_offset + Int32(
        config.BM * config.padded_depth
    )
    comptime correction_offset: Int32 = (
        kv_offset
        + Int32(2 * config.num_kv_stages * config.padded_depth * config.BN)
    ) * Int32(size_of[qkv_type]()) // Int32(size_of[DType.float32]())
    comptime mbar_offset = (correction_offset + Int32(config.BM)) * Int32(
        size_of[DType.float32]()
    ) // Int32(size_of[SharedMemBarrier]())

    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.depth,
        config.padded_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    comptime KPipeType = KProducerPipeline[KVLUTType.dtype, config]
    comptime VPipeType = VProducerPipeline[KVLUTType.dtype, config]

    # If two-qo, we produce qkv in a pattern of
    # q0 & k0, q1, v0, k1, v1, k2, v2...
    comptime SMemTensorLT[layout: Layout] = SharedMemLT[KVLUTType.dtype, layout]
    comptime QType = SMemTensorLT[type_of(q_tma_op).layout]
    comptime KType = SMemTensorLT[type_of(k_tma_op).layout]
    comptime VType = SMemTensorLT[type_of(v_tma_op).layout]

    var kv_head_idx: UInt32 = seq_info.head_idx // UInt32(group)

    var q_smem: SharedMemPointer[Scalar[KVLUTType.dtype]] = (
        mbars.mbar_base - mbar_offset
    ).bitcast[Scalar[qkv_type]]() + q_offset
    comptime q_elements = HalfBM * config.BK0
    comptime assert q_elements == QType.layout.size()
    comptime q_bytes = size_of[qkv_type]() * q_elements
    comptime qk_bytes = KPipeType.bytes + q_bytes
    var k_smem = q_smem + config.BM * config.padded_depth
    var v_smem = (
        k_smem + (config.BN * config.padded_depth) * config.num_kv_stages
    )
    var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
    var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

    var mbark0: KPipeType.KPairType

    mbark0 = pipeline_k.get_k[qk_stage=0]()  # no wait
    var q_gmem_row: UInt32 = PositionType.get_q_gmem_row[ragged=ragged](
        seq_info, max_seq_len
    )
    var q_head_idx: UInt32 = seq_info.head_idx
    e = elect()
    # copy q0
    if e != 0:
        # Q0
        mbark0.mbar[].expect_bytes(Int32(qk_bytes))
    # copy q0
    if e != 0:
        q_tma_op.async_copy[eviction_policy = CacheEviction.EVICT_FIRST](
            QType(q_smem),
            mbark0.mbar[],
            StaticTuple[UInt32, 3](0, q_head_idx, q_gmem_row),
        )
    var kv_row: UInt32 = mask.start_column[BM, BN, page_size](score_row)
    var kv_gmem_row: UInt32 = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
    var iter_count: UInt32 = (
        mask.last_masked_set_end[BM, BN, page_size](score_row, num_keys) - 1
    )
    # copy k0
    if e != 0:  # K0
        k_tma_op.async_copy(
            mbark0.smem,
            mbark0.mbar[],
            StaticTuple[UInt32, 3](0, kv_head_idx, kv_gmem_row),
        )

    @parameter
    for qk_stage in range(1, config.num_qk_stages):
        comptime d_idx = qk_stage * config.BK0
        mbark = pipeline_k.get_k[qk_stage=qk_stage]()  # no wait
        if e != 0:
            mbark.mbar[].expect_bytes(Int32(qk_bytes))
        if e != 0:
            q_tma_op.async_copy[eviction_policy = CacheEviction.EVICT_FIRST](
                QType(q_smem + q_elements * qk_stage),
                mbark.mbar[],
                StaticTuple[UInt32, 3](UInt32(d_idx), q_head_idx, q_gmem_row),
            )
        if e != 0:
            k_tma_op.async_copy(
                mbark.smem,
                mbark.mbar[],
                StaticTuple[UInt32, 3](UInt32(d_idx), kv_head_idx, kv_gmem_row),
            )

    pipeline_k.commit_step()
    # Q1
    q_gmem_row += UInt32(HalfBM)
    var q1_mbar = mbars.q1_wait_mbar()

    @parameter
    for qk_stage in range(config.num_qk_stages):
        comptime q_smem_offset = q_elements * (config.num_qk_stages + qk_stage)
        comptime d_idx = qk_stage * config.BK0
        if e != 0:
            q1_mbar[qk_stage].expect_bytes(Int32(q_bytes))
        if e != 0:
            q_tma_op.async_copy(
                QType(q_smem + q_smem_offset),
                q1_mbar[qk_stage],
                StaticTuple[UInt32, 3](UInt32(d_idx), q_head_idx, q_gmem_row),
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

        @parameter
        if check_mask:
            if (
                mask.status(
                    Index[dtype = DType.int32](Int(score_row), Int(kv_row)),
                    Index[dtype = DType.int32](BM, BN),
                )
                == TileMaskStatus.FULL_MASK
            ):
                continue
        kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)

        # produce k
        @parameter
        for k_stage in range(config.num_qk_stages):
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
