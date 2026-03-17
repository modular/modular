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

from std.sys import size_of
from std.math import ceildiv
from nn.mha_operand import MHAOperand
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.fa4_config import FA4Config
from nn.sm100_attention_utils import (
    KVPipeline,
    SharedMemPointer,
    elect,
    MBarType,
    ProducerPipeline,
    elect_mma_arrive,
)
from nn.mha import q_num_matrix_view_rows
from nn.mha_fa3_utils import (
    get_seq_info,
    KVTMATile,
    get_seq_info,
    KVTMATile,
    kv_coord,
    NonNullPointer,
    NullPointer,
    Pack,
    produce,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
    TMATensorTile,
    create_tensor_tile,
)
from layout.tile_tensor import TileTensor
from layout.tile_layout import row_major
from layout.coord import Idx, Coord
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.swizzle import make_swizzle

import std.gpu.primitives.warp as warp
from std.gpu.globals import WARP_SIZE
from std.gpu.memory import AddressSpace, external_memory
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    block_idx,
    warp_id,
)
from nn.mha_utils import (
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
)
from std.gpu.compute.arch.tcgen05 import *
from linalg.arch.sm100.mma import smem_descriptor
from std.utils.static_tuple import StaticTuple
from std.utils.index import Index
from kv_cache.types import swizzle_granularity, padded_depth

from nn.mla_prefill_sm100_utils import (
    MLAConfig,
    SM100MLA,
    MLAPositionSummary,
    MLAKVProducerPipeline,
    split_smem,
)


@always_inline
fn q_scale_tma[
    dtype: DType, //, BM: Int
](
    ctx: DeviceContext,
    q_scale_tensor: LayoutTensor[dtype, ...],
    out tma: TMATensorTile[dtype, 2, Index(1, BM), Index(1, BM)],
) raises:
    var num_elements = q_scale_tensor.size()
    debug_assert(num_elements % 4 == 0, "num_elements must be divisible by 4")
    var tensor = TileTensor(
        q_scale_tensor.ptr, row_major(Coord(Idx[1](), Idx(num_elements)))
    )

    return create_tensor_tile[
        Index(1, BM),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __desc_shape=Index(1, BM),
    ](ctx, tensor.to_layout_tensor())


@fieldwise_init
struct WarpRole(Equatable, TrivialRegisterPassable):
    var _role: Int32
    comptime Softmax0 = Self(0)
    comptime Softmax1 = Self(1)
    comptime Correction = Self(2)
    comptime MMA = Self(3)
    comptime Load = Self(4)
    comptime Empty = Self(6)

    @always_inline
    fn __eq__(self, other: Int) -> Bool:
        return self == Self(Int32(other))


fn warp_idx_to_role(warp_idx: UInt32) -> WarpRole:
    var wg_idx = warp_idx // 4
    if wg_idx == 0:
        return WarpRole.Softmax0
    elif wg_idx == 1:
        return WarpRole.Softmax1
    elif wg_idx == 2:
        return WarpRole.Correction
    elif warp_idx == 12:
        return WarpRole.MMA
    elif warp_idx == 13:
        return WarpRole.Load
    else:
        return WarpRole.Empty


struct MLASmemStorage[
    qkv_dtype: DType, rope_dtype: DType, num_mbars: Int, config: MLAConfig
]:
    comptime q_nope_bytes = Self.config.BM * Self.config.nope_depth * size_of[
        Self.qkv_dtype
    ]()
    comptime q_rope_bytes = Self.config.BM * Self.config.rope_depth * size_of[
        Self.rope_dtype
    ]()
    comptime q_bytes = Self.q_nope_bytes + Self.q_rope_bytes

    comptime num_kv_stages = Self.config.num_kv_stages * Self.config.num_qk_stages

    comptime kv_nope_bytes = Self.config.nope_depth * Self.config.BN * size_of[
        Self.qkv_dtype
    ]() * Self.num_kv_stages
    comptime kv_rope_bytes = Self.config.rope_depth * Self.config.BN * size_of[
        Self.rope_dtype
    ]() * Self.num_kv_stages
    comptime kv_bytes = Self.kv_nope_bytes + Self.kv_rope_bytes

    comptime q_scale_bytes = Self.config.BM * size_of[DType.float32]()
    comptime k_scale_bytes = Self.config.BN * size_of[DType.float32]()

    comptime correction_smem_size = Self.config.correction_smem_elements()

    var q_smem: InlineArray[Scalar[DType.uint8], Self.q_bytes]
    var kv_smem: InlineArray[Scalar[DType.uint8], Self.kv_bytes]
    var q_scale_smem: InlineArray[Scalar[DType.uint8], Self.q_scale_bytes]
    var k_scale_smem: InlineArray[Scalar[DType.uint8], Self.k_scale_bytes]
    var correction_smem: InlineArray[Float32, Self.correction_smem_size]
    var mbar_base: InlineArray[SharedMemBarrier, Self.num_mbars]
    var tmem_addr: InlineArray[UInt32, 1]


__extension SM100MLA:
    @staticmethod
    @__llvm_arg_metadata(q_nope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(q_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(q_scale_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_nope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_scale_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=Int(1))
    fn mla_prefill_kernel_per_token_scale(
        q_nope_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.nope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_rope_tma_op: QTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.rope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_scale_tma_op: TMATensorTile[
            Self.KVLUTType.scale_dtype,
            2,
            Index(1, Self.config.BM),
            Index(1, Self.config.BM),
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.rope_depth,
        ],
        k_scale_tma_op: TMATensorTile[
            Self.KVLUTType.scale_dtype,
            2,
            Index(1, Self.config.BN),
            Index(1, Self.config.BN),
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.output_swizzle_mode,
            BM=Self.config.BM // 2,
            BN=Self.nope_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        scale: Float32,
        batch_size: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            Self.SinkType,
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
    ):
        comptime assert Self.MMA_M == 64 or Self.MMA_M == 128
        comptime assert Self.config.supported(), (
            "depth = "
            + String(Self.config.depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used)
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        mask = pack.mask
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        comptime num_qo = Self.config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        comptime assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        smem_ptr = external_memory[
            Scalar[DType.uint8],
            address_space=AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        comptime SmemStorageType = MLASmemStorage[
            Self.qkv_type,
            Self.KRopeType.dtype,
            Int(Self.MiscMBarsType.num_mbars()),
            Self.config,
        ]
        ref smem_storage = smem_ptr.bitcast[SmemStorageType]()[]
        var q_smem = smem_storage.q_smem.unsafe_ptr().bitcast[
            Scalar[Self.qkv_type]
        ]()
        var kv_smem = smem_storage.kv_smem.unsafe_ptr().bitcast[
            Scalar[Self.qkv_type]
        ]()
        var q_scale_smem = smem_storage.q_scale_smem.unsafe_ptr().bitcast[
            Scalar[Self.KVLUTType.scale_dtype]
        ]()
        var k_scale_smem = smem_storage.k_scale_smem.unsafe_ptr().bitcast[
            Scalar[Self.KVLUTType.scale_dtype]
        ]()
        var correction_smem = smem_storage.correction_smem.unsafe_ptr()
        var mbar_base = smem_storage.mbar_base.unsafe_ptr()
        var ptr_tmem_addr = smem_storage.tmem_addr.unsafe_ptr()

        # All barriers are managed by misc_mbars (S/C/order/Q1Sync/KV/O)
        var misc_mbars: Self.MiscMBarsType = {mbar_base}

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 96
        comptime num_reg_other = 48
        comptime num_reg_empty = 24

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp.broadcast(warp_id()))
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/KV/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
        elif warp_idx == 1:
            tcgen05_alloc[Self.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_nope_tma_op.prefetch_descriptor()
            if e != 0:
                q_rope_tma_op.prefetch_descriptor()
            if e != 0:
                q_scale_tma_op.prefetch_descriptor()
            if e != 0:
                k_nope_tma_op.prefetch_descriptor()
            if e != 0:
                k_scale_tma_op.prefetch_descriptor()
            if e != 0:
                k_rope_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        barrier()

        var role = warp_idx_to_role(warp_idx)

        # warp group partitioning
        # Two QO:
        if role == WarpRole.Softmax0 or role == WarpRole.Softmax1:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            Self.softmax[apply_scale=True](
                ptr_tmem_addr[0],
                warp_idx,
                misc_mbars,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_type](),
                max_seq_len.as_uint32(),
                ragged_tma_store,
                q_smem.bitcast[Scalar[Self.output_type]](),
                correction_smem,
                q_scale_smem,
                k_scale_smem,
            )

        elif role == WarpRole.Correction:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)
            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            Self.correction(
                ptr_tmem_addr[0],
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                mask,
                correction_smem,
            )
        elif role == WarpRole.Load:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            Self.load(
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                seq_info,
                max_seq_len,
                mask,
                q_nope_tma_op,
                q_rope_tma_op,
                q_scale_tma_op,
                k_nope_tma_op,
                k_rope_tma_op,
                k_scale_tma_op,
                v_tma_op,
                kv_lut,
                k_rope_lut,
                q_smem,
                q_scale_smem,
                k_scale_smem,
            )

        elif role == WarpRole.MMA:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                tcgen05_release_allocation_lock[Self.cta_group]()
                tcgen05_dealloc[Self.cta_group](
                    ptr_tmem_addr[0], Self.config.sm100_tmem_cols
                )
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            Self.mma(
                ptr_tmem_addr[0],
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
            )
        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    fn load(
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_nope_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.nope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_rope_tma_op: QTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.rope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_scale_tma_op: TMATensorTile[
            Self.KVLUTType.scale_dtype,
            2,
            Index(1, Self.config.BM),
            Index(1, Self.config.BM),
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.rope_depth,
        ],
        k_scale_tma_op: TMATensorTile[
            Self.KVLUTType.scale_dtype,
            2,
            Index(1, Self.config.BN),
            Index(1, Self.config.BN),
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        q_scale_smem: SharedMemPointer[Scalar[Self.KVLUTType.scale_dtype]],
        k_scale_smem: SharedMemPointer[Scalar[Self.KVLUTType.scale_dtype]],
    ):
        comptime KVPipeType = MLAKVProducerPipeline[
            Self.KVLUTType.dtype,
            KRopeType.dtype,
            Self.KVLUTType.scale_dtype,
            Self.config,
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        # TMA only uses .ptr — flat row_major TileTensor is sufficient.
        comptime _smem_tt[dtype: DType, elems: Int] = TileTensor[
            dtype,
            type_of(row_major[elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime q_nope_elems = type_of(q_nope_tma_op).tile_shape[0] * type_of(
            q_nope_tma_op
        ).tile_shape[1]
        comptime q_rope_elems = type_of(q_rope_tma_op).tile_shape[0] * type_of(
            q_rope_tma_op
        ).tile_shape[1]
        comptime q_scale_elems = type_of(q_scale_tma_op).tile_shape[
            0
        ] * type_of(q_scale_tma_op).tile_shape[1]
        comptime k_scale_elems = type_of(k_scale_tma_op).tile_shape[
            0
        ] * type_of(k_scale_tma_op).tile_shape[1]
        comptime QNopeType = _smem_tt[Self.KVLUTType.dtype, q_nope_elems]
        comptime QRopeType = _smem_tt[Self.KRopeType.dtype, q_rope_elems]
        comptime QScaleType = _smem_tt[
            Self.KVLUTType.scale_dtype, q_scale_elems
        ]
        comptime KScaleType = _smem_tt[
            Self.KVLUTType.scale_dtype, k_scale_elems
        ]

        var k_rope_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)
        var kv_head_idx: UInt32 = seq_info.head_idx

        comptime q_nope_elements = (
            Self.config.BM // 2
        ) * Self.config.nope_depth
        comptime q_rope_elements = (
            Self.config.BM // 2
        ) * Self.config.rope_depth
        comptime q_nope_bytes = q_nope_elements * size_of[Self.qkv_type]()
        comptime q_rope_bytes = q_rope_elements * size_of[
            Self.KRopeType.dtype
        ]()
        comptime q_bytes = q_nope_bytes + q_rope_bytes
        comptime q_scale_bytes = Self.config.BM * size_of[DType.float32]()
        comptime k_scale_bytes = Self.config.BN * size_of[DType.float32]()
        comptime v_scale_bytes = k_scale_bytes

        q_smem_addr = q_smem.bitcast[UInt8]()

        q0_nope_smem = q_smem_addr.bitcast[Scalar[Self.qkv_type]]()
        q0_rope_smem = (q_smem_addr + q_nope_bytes).bitcast[
            Scalar[Self.KRopeType.dtype]
        ]()

        q1_nope_smem = (q_smem_addr + q_bytes).bitcast[Scalar[Self.qkv_type]]()
        q1_rope_smem = (q_smem_addr + q_bytes + q_nope_bytes).bitcast[
            Scalar[Self.KRopeType.dtype]
        ]()

        kv_smem = (q_smem_addr + 2 * q_bytes).bitcast[Scalar[Self.qkv_type]]()
        # Construct KV pipeline from unified barrier storage
        var kv_pipeline_arg = Self.KVPipelineType(mbars.get_kv_mbars())
        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        var mbark0: KVPipeType.KPairType

        mbark0 = pipeline_kv.get_k[qk_stage=0, expect=False]()  # no wait
        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )
        var q_head_idx: UInt32 = seq_info.head_idx
        elect = elect() != 0

        # copy q0
        if elect:
            # Q0
            mbark0.mbar[].expect_bytes(
                Int32(
                    pipeline_kv.k_bytes
                    + k_scale_bytes
                    + q_bytes
                    + q_scale_bytes
                )
            )
            q_nope_tma_op.async_copy(
                QNopeType(q0_nope_smem, row_major[q_nope_elems]()),
                mbark0.mbar[],
                q_coord[
                    depth=Self.nope_depth,
                    decoding=False,
                ](q_gmem_row, q_head_idx),
            )
            q_rope_tma_op.async_copy(
                QRopeType(q0_rope_smem, row_major[q_rope_elems]()),
                mbark0.mbar[],
                q_coord[
                    depth=Self.rope_depth,
                    decoding=False,
                ](q_gmem_row, q_head_idx),
            )
            q_scale_tma_op.async_copy(
                QScaleType(q_scale_smem, row_major[q_scale_elems]()),
                mbark0.mbar[],
                (Int(q_gmem_row), 0),
            )
            # print(seq_info.prompt_idx, q_gmem_row)
        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
        var k_rope_gmem_row: UInt32 = k_rope_lut.row_idx(
            seq_info.prompt_idx, kv_row
        )
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )
        # copy k0
        k_smem, k_rope_smem = split_smem[
            KVPipeType.k_nope_tma_layout,
            KVPipeType.k_rope_tma_layout,
            Self.KVLUTType.dtype,
            KRopeType.dtype,
        ](mbark0.smem)
        if elect:
            # K0
            k_nope_tma_op.async_copy(
                k_smem,
                mbark0.mbar[],
                kv_coord[depth=Self.nope_depth](kv_gmem_row, kv_head_idx),
            )
            # K0 rope
            var k_rope_coord = kv_coord[depth=Self.rope_depth](
                k_rope_gmem_row, k_rope_head_idx
            )
            k_rope_coord[0] = UInt32(
                Self.cache_depth - Self.rope_depth
            )  # only load last 64 head_dims

            k_rope_tma_op.async_copy(
                k_rope_smem,
                mbark0.mbar[],
                k_rope_coord,
            )
            k_scale_tma_op.async_copy(
                KScaleType(k_scale_smem, row_major[k_scale_elems]()),
                mbark0.mbar[],
                (Int(kv_gmem_row), 0),
            )

        pipeline_kv.commit_kv_step()
        if elect:
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].expect_bytes(Int32(q_bytes))
            # Q1
            q_nope_tma_op.async_copy(
                QNopeType(q1_nope_smem, row_major[q_nope_elems]()),
                q1_mbar[0],
                q_coord[
                    depth=Self.nope_depth,
                    decoding=False,
                ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
            )
            q_rope_tma_op.async_copy(
                QRopeType(q1_rope_smem, row_major[q_rope_elems]()),
                q1_mbar[0],
                q_coord[
                    depth=Self.rope_depth,
                    decoding=False,
                ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
            )
        # copy v0
        if elect:
            mbarv0 = pipeline_kv.get_v[qk_stage=0]()
            v_tma_op.async_copy(
                mbarv0.smem,
                mbarv0.mbar[],
                kv_coord[depth=Self.nope_depth](kv_gmem_row, kv_head_idx),
            )

        pipeline_kv.commit_kv_step()
        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        # kv producer loop
        while iter_count != 0:
            iter_count -= 1
            kv_row += UInt32(Self.config.BN)

            comptime if check_mask:
                if (
                    Self.mask_status(mask, score_row, kv_row)
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
            k_rope_gmem_row = k_rope_lut.row_idx(seq_info.prompt_idx, kv_row)
            # produce k
            pipeline_kv.acquire_kv()
            if elect:
                mbarkn = pipeline_kv.get_k[qk_stage=0]()
                k_smem_n, k_rope_smem_n = split_smem[
                    KVPipeType.k_nope_tma_layout,
                    KVPipeType.k_rope_tma_layout,
                    Self.KVLUTType.dtype,
                    KRopeType.dtype,
                ](mbarkn.smem)

                k_nope_tma_op.async_copy(
                    k_smem_n,
                    mbarkn.mbar[],
                    kv_coord[depth=Self.nope_depth](kv_gmem_row, kv_head_idx),
                )
                # K rope
                var k_rope_coord = kv_coord[depth=Self.rope_depth](
                    k_rope_gmem_row, k_rope_head_idx
                )
                k_rope_coord[0] = UInt32(
                    Self.cache_depth - Self.rope_depth
                )  # only load last 64 head_dims
                k_rope_tma_op.async_copy(
                    k_rope_smem_n,
                    mbarkn.mbar[],
                    k_rope_coord,
                )
                k_scale_tma_op.async_copy(
                    KScaleType(k_scale_smem, row_major[k_scale_elems]()),
                    mbarkn.mbar[],
                    (Int(kv_gmem_row), 0),
                )

            pipeline_kv.commit_kv_step()
            pipeline_kv.acquire_kv()
            if elect:
                mbarvn = pipeline_kv.get_v[qk_stage=0]()
                v_tma_op.async_copy(
                    mbarvn.smem,
                    mbarvn.mbar[],
                    kv_coord[depth=Self.nope_depth](kv_gmem_row, kv_head_idx),
                )

            pipeline_kv.commit_kv_step()

    @staticmethod
    @always_inline
    fn mma(
        tmem_addr: UInt32,
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        s1_tmem = tmem_addr + UInt32(Self.config.TMEM_S1)
        o0_tmem = tmem_addr + UInt32(Self.config.TMEM_O0)
        o1_tmem = tmem_addr + UInt32(Self.config.TMEM_O1)

        # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
        var pipeline_s0 = mbars.producer_s0()
        var pipeline_s1 = mbars.producer_s1()
        # Keep consumer pointers for acquire operations (shared phase tracking)
        consumer_s0 = pipeline_s0.consumer_mbar_base
        consumer_s1 = pipeline_s1.consumer_mbar_base

        # O pipelines (producer side only; consumer wait is merged into S barriers)
        var pipeline_o0 = mbars.producer_o0()
        var pipeline_o1 = mbars.producer_o1()

        comptime q_nope_bytes = (
            Self.config.BM // 2
        ) * Self.config.nope_depth * size_of[Self.KVLUTType.dtype]()
        comptime q_rope_bytes = (
            Self.config.BM // 2
        ) * Self.config.rope_depth * size_of[Self.KRopeType.dtype]()
        comptime q_half_bytes = q_nope_bytes + q_rope_bytes
        q0 = Self.descriptor_q(q_smem)
        q0_rope_smem = (q_smem.bitcast[UInt8]() + q_nope_bytes).bitcast[
            Scalar[Self.KRopeType.dtype]
        ]()
        q0_rope = Self.descriptor_q_rope(q0_rope_smem)

        q1 = q0 + UInt32(q_half_bytes)
        q1_rope_smem = (
            q_smem.bitcast[UInt8]() + q_half_bytes + q_nope_bytes
        ).bitcast[Scalar[Self.KRopeType.dtype]]()
        q1_rope = Self.descriptor_q_rope(q1_rope_smem)
        kv_smem = (q_smem.bitcast[UInt8]() + 2 * q_half_bytes).bitcast[
            Scalar[Self.KVLUTType.dtype]
        ]()

        # MLA uses a shared KVPipeline where K and V alternate states.
        # Create K and V smem descriptors with MLA-specific dimensions.
        comptime k_nope_bytes = Self.config.nope_depth * Self.config.BN * size_of[
            Self.KVLUTType.dtype
        ]()
        comptime k_rope_bytes = Self.config.rope_depth * Self.config.BN * size_of[
            Self.KRopeType.dtype
        ]()
        comptime full_kv_bytes = UInt32(k_nope_bytes + k_rope_bytes)

        var k_rope_smem = (kv_smem.bitcast[UInt8]() + k_nope_bytes).bitcast[
            Scalar[Self.KRopeType.dtype]
        ]()
        var k_rope_smem_descriptor = smem_descriptor[
            BMN=Self.config.BN,
            BK=Self.config.rope_depth,
            swizzle_mode=Self.config.rope_swizzle_mode,
            is_k_major=True,
        ](k_rope_smem)
        var k_smem_descriptor = smem_descriptor[
            BMN=Self.config.BN,
            BK=Self.config.nope_depth,
            swizzle_mode=Self.config.qkv_swizzle_mode,
            is_k_major=True,
        ](kv_smem)
        var v_smem_descriptor = smem_descriptor[
            BMN=Self.nope_depth,
            BK=Self.config.BK1,
            swizzle_mode=Self.config.qkv_swizzle_mode,
            is_k_major=False,
        ](kv_smem)

        # Construct KV pipeline from unified barrier storage (consumer starts phase=0)
        var pipeline = Self.KVPipelineType(mbars.get_kv_mbars())

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        # Q_0 @ K_0'
        pipeline.consumer_wait[0]()  # wait K0
        k0 = k_smem_descriptor + full_kv_bytes * pipeline.state.index()
        k0_rope = (
            k_rope_smem_descriptor + full_kv_bytes * pipeline.state.index()
        )
        e = elect()
        Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
        Self.UMMA0RopeType.mma(q0_rope, k0_rope, s0_tmem, elect=e, c_scale=1)
        pipeline_s0.commit_mma(e)

        # Q_1 @ K_0'
        mbars.q1_wait_mbar()[0].wait()  # wait on Q1
        Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
        Self.UMMA0RopeType.mma(q1_rope, k0_rope, s1_tmem, elect=e, c_scale=1)
        pipeline_s1.commit_mma(e)

        pipeline.consumer_release[0](e)  # release K0, step to V state

        # Wait V0
        pipeline.consumer_wait[0]()  # wait V0
        vlatest = v_smem_descriptor + full_kv_bytes * pipeline.state.index()
        _ = consumer_s0[].wait(0)
        Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0)
        pipeline_o0.commit_mma(e)
        var phase: UInt32 = 0

        var c_scale: UInt32 = 0

        while iter_count != 0:
            iter_count -= 1
            # Save V release index and step to next K position
            var v_release_index = pipeline.state.index()
            pipeline.state.step()

            # Q_0 @ K_n'
            pipeline.consumer_wait[0]()  # wait Kn
            kn = k_smem_descriptor + full_kv_bytes * pipeline.state.index()
            kn_rope = (
                k_rope_smem_descriptor + full_kv_bytes * pipeline.state.index()
            )
            Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
            Self.UMMA0RopeType.mma(
                q0_rope, kn_rope, s0_tmem, elect=e, c_scale=1
            )
            pipeline_s0.commit_mma(e)

            # O_1 + P_1 @ V_{n-1}
            _ = consumer_s1[].wait(phase)
            Self.UMMA1Type.mma(
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
            pipeline_o1.commit_mma(e)
            c_scale = 1
            # Release old V (deferred release, no step)
            elect_mma_arrive(pipeline.consumer_mbar[0](v_release_index), e)

            # Q_1 @ K_n'
            Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
            Self.UMMA0RopeType.mma(
                q1_rope, kn_rope, s1_tmem, elect=e, c_scale=1
            )
            pipeline_s1.commit_mma(e)
            phase ^= 1

            pipeline.consumer_release[0](e)  # release K, step to V state

            # O_0 + P_0 @ V_n
            pipeline.consumer_wait[0]()  # wait Vn
            vlatest = v_smem_descriptor + full_kv_bytes * pipeline.state.index()
            _ = consumer_s0[].wait(phase)
            Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1)
            pipeline_o0.commit_mma(e)

        _ = consumer_s1[].wait(phase)
        Self.UMMA1Type.mma(s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale)
        pipeline_o1.commit_mma(e)


@always_inline
fn mla_sm100_prefill_per_token_scale[
    output_type: DType,
    q_type: DType,
    rope_type: DType,
    scale_dtype: DType,
    KType: MHAOperand,
    VType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
](
    output: LayoutTensor[output_type, address_space=AddressSpace.GENERIC, ...],
    q_nope: LayoutTensor[q_type, _, address_space=AddressSpace.GENERIC, ...],
    q_rope: LayoutTensor[rope_type, _, address_space=AddressSpace.GENERIC, ...],
    q_scale: LayoutTensor[
        scale_dtype, _, address_space=AddressSpace.GENERIC, ...
    ],
    k_nope: KType,
    k_rope: KRopeType,
    v: VType,
    mask_functor: MaskType,
    valid_length: LayoutTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        rope_type == KRopeType.dtype
    ), "q_rope and k_rope must have the same dtype"

    comptime fa4_config = MLAConfig(
        num_q_heads=Int(config.num_heads),
        group=group,
        depth=q_depth,
        qkv_dtype_size=size_of[q_type](),
        rope_dtype_size=size_of[KRopeType.dtype](),
        output_dtype_size=size_of[output_type](),
        page_size=KType.page_size,
    )

    var num_rows_q = q_num_matrix_view_rows(q_nope)

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.BM // 2,
        BN=fa4_config.nope_depth,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q, middle_dim=fa4_config.num_q_heads
    )

    q_nope_tma_op = q_tma[
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.nope_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
    ](
        ctx,
        q_nope.ptr,
        num_rows_q,
    )

    q_rope_tma_op = q_tma[
        fa4_config.rope_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.rope_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
    ](
        ctx,
        q_rope.ptr,
        num_rows_q,
    )

    q_scale_tma_op = q_scale_tma[BM=fa4_config.BM](ctx, q_scale)

    # [batch_size * num_keys, num_heads, kv_depth]
    k_nope_tma_op = k_nope.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        depth=fa4_config.nope_depth,
    ](ctx)

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    k_rope_tma_op = k_rope.create_tma_tile[
        fa4_config.rope_swizzle_mode,
        BN=fa4_config.BN,
        depth=cache_depth,
        BK=fa4_config.rope_depth,
    ](ctx)

    k_scale_tma_op = k_nope.create_scale_tma_tile[fa4_config.BN](ctx)

    # [batch_size * num_keys, num_heads, kv_depth]

    v_tma_op = v.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        depth=fa4_config.nope_depth,
    ](ctx)

    _mla_prefill_sm100_valid_length_dispatch[
        fa4_config=fa4_config,
        cache_depth=cache_depth,
        _ndbuffer_mha_operand=_ndbuffer_mha_operand,
    ](
        ragged_tma_store,
        q_nope_tma_op,
        q_rope_tma_op,
        q_scale_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        k_scale_tma_op,
        v_tma_op,
        k_nope,
        k_rope,
        mask_functor,
        valid_length,
        max_prompt_len,
        scale,
        batch_size,
        ctx,
    )


@always_inline
fn _mla_prefill_sm100_valid_length_dispatch[
    KType: MHAOperand,
    VType: MHAOperand,
    output_type: DType,
    q_type: DType,
    q_scale_type: DType,
    rope_type: DType,
    MaskType: MHAMask,
    KRopeType: MHAOperand,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    fa4_config: MLAConfig,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
](
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.BM // 2,
        BN=fa4_config.nope_depth,
    ],
    q_nope_tma_op: QTMATile[
        q_type,
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.nope_depth,
        group=fa4_config.group,
        decoding=False,
    ],
    q_rope_tma_op: QTMATile[
        rope_type,
        fa4_config.rope_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.rope_depth,
        group=fa4_config.group,
        decoding=False,
    ],
    q_scale_tma_op: TMATensorTile[
        q_scale_type,
        2,
        Index(1, fa4_config.BM),
        Index(1, fa4_config.BM),
    ],
    k_nope_tma_op: KVTMATile[
        KType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        BK=padded_depth[
            KType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    k_rope_tma_op: KVTMATile[
        KRopeType.dtype,
        fa4_config.rope_swizzle_mode,
        BN=fa4_config.BN,
        BK=fa4_config.rope_depth,
    ],
    k_scale_tma_op: TMATensorTile[
        KType.scale_dtype,
        2,
        Index(1, fa4_config.BN),
        Index(1, fa4_config.BN),
    ],
    v_tma_op: KVTMATile[
        VType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        BK=padded_depth[
            VType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    kv_lut: KType,
    k_rope_lut: KRopeType,
    mask_functor: MaskType,
    valid_length: LayoutTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime SchedulerType = TransientScheduler[
        UInt32(fa4_config.BM),
        UInt32(fa4_config.num_q_heads),
        flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
    ]
    comptime ValidLengthType = NonNullPointer[DType.uint32]
    comptime SinkType = NullPointer[output_type]
    comptime KVRowOffsetsType = NullPointer[DType.uint32]
    comptime PartitionType = NoPartition[DType.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    comptime SM100MLAType = SM100MLA[
        KType,
        KRopeType,
        output_type,
        MaskType,
        SchedulerType,
        fa4_config,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
        _ndbuffer_mha_operand,
    ]

    comptime kernel = SM100MLAType.mla_prefill_kernel_per_token_scale

    comptime PackType = Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
    ]

    var pack: PackType = {
        mask_functor,
        SchedulerType(),
        valid_len,
        SinkType(),
        KVRowOffsetsType(),
        max_prompt_len,
        PartitionType(),
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_prompt_len.as_uint32(), UInt32(fa4_config.BM)
    )
    var num_blocks: UInt32 = (
        max_num_prompt_tiles * PartitionType().num_partitions()
    )

    comptime num_threads = fa4_config.num_threads
    comptime smem_use = size_of[
        MLASmemStorage[
            SM100MLAType.qkv_type,
            SM100MLAType.KRopeType.dtype,
            Int(SM100MLAType.MiscMBarsType.num_mbars()),
            fa4_config,
        ]
    ]()

    ctx.enqueue_function[kernel, kernel](
        q_nope_tma_op,
        q_rope_tma_op,
        q_scale_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        k_scale_tma_op,
        v_tma_op,
        ragged_tma_store,
        kv_lut,
        k_rope_lut,
        scale,
        UInt32(batch_size),
        pack,
        grid_dim=SchedulerType.grid_dim(UInt32(batch_size), num_blocks),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )
