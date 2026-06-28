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
"""Per-token-scale MLA prefill kernel.

Thin dispatch wrapper that creates scale TMA tiles and calls the generic
MLA kernel's per-token-scale variant (mla_prefill_kernel_per_token_scale).
"""

from std.sys import size_of
from std.math import ceildiv
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.gpu.nvidia.common import (
    get_seq_info,
    kv_coord,
    KVTMATile,
    NonNullPointer,
    NullPointer,
    Pack,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    create_tensor_tile,
    RaggedTMA3DTile,
    SharedMemBarrier,
    TMATensorTile,
)
from layout import TileTensor
from layout.tile_tensor import TileTensor
from layout.tile_layout import row_major as tt_row_major
from layout.coord import Idx, Coord
from layout.layout_tensor import LayoutTensor
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier, thread_idx, warp_id
from std.gpu.primitives.warp import broadcast
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from std.gpu.compute.arch.tcgen05 import tcgen05_alloc
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.utils.index import Index
from nn.attention.gpu.mha import q_num_matrix_view_rows
from nn.attention.gpu.nvidia.sm100.smem import SM100AttentionSMem
from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    expect_bytes_pred,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    o_store_tma_blocks_per_op,
    PagedRowIndices,
    SharedMemPointer,
    StagedPipeline,
    VProducerPipeline,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_generic import (
    warp_idx_to_role,
    WarpRole,
)
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.attention.mha_utils import (
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
)
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.mla_prefill_utils import (
    MLAConfig,
    MLAKVLayouts,
    MLAPositionSummary,
    SM100MLA,
    split_smem,
)


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
    @__llvm_arg_metadata(k_nope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(q_scale_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_scale_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
    @__name(
        t"sm100_mla_prefill_per_token_scale_{Self.qkv_dtype}_{Self.output_dtype}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def mla_prefill_kernel_per_token_scale(
        q_nope_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            # `BM // num_qo` = 128 in both modes (one of two Q halves in
            # 2Q, the single full-BM Q tile in 1Q), so the TMA-op type
            # folds across the 1Q/2Q configs.
            BM=Self.config.q_tile_rows(),
            depth=Self.config.nope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_rope_tma_op: QTMATile[
            config.rope_gmem_dtype,
            Self.config.rope_gmem_swizzle_mode,
            BM=Self.config.q_tile_rows(),
            depth=Self.config.rope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        q_scale_tma_op: TMATensorTile[
            config.scale_dtype,
            2,
            # Per-Q-tile box (128 in both modes): 2Q issues two TMAs
            # (one per Q half, see the with_q / Q1 sites in
            # `load_per_token_scale`); 1Q issues one. Keeps the TMA-op
            # type folding across the 1Q/2Q configs.
            Index(1, Self.config.q_tile_rows()),
            Index(1, Self.config.q_tile_rows()),
        ],
        k_scale_tma_op: TMATensorTile[
            config.scale_dtype,
            2,
            Index(1, kv_sub_tile_rows(Self.config.BN, Self.page_size)),
            Index(1, kv_sub_tile_rows(Self.config.BN, Self.page_size)),
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_dtype,
            Self.config.output_swizzle_mode,
            # `// fa4_config.num_qo` instead of `// 2`: matches the
            # fa4_softmax / fa4_lse_combine_write signature so both the
            # 1Q and 2Q instantiations type-check under the unified
            # signature (128 in both modes).
            BM=Self.config.fa4_config.BM // Self.config.fa4_config.num_qo,
            BN=Self.config.fa4_config.ov_depth,
            middle_dim=Self.config.num_q_heads,
            group=config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
            # Concrete value: a GPU-kernel entry (and the device impl it calls)
            # must be a fully-bound function. Matches the created store.
            tma_blocks_per_op=o_store_tma_blocks_per_op[
                Self.output_dtype,
                Self.config.output_swizzle_mode,
                Self.config.fa4_config.ov_depth,
                config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
                depth_splits=2,
            ](),
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
        # Thin entrypoint — see `mla_prefill_kernel_generic` for the full
        # rationale. Compute `SeqInfo` once (blockIdx-derived, warp-uniform)
        # and route per-tile work whose remaining valid rows fit a single 1Q
        # tile to the cheaper 1Q body when this 2Q config admits a
        # type-compatible 1Q variant AND the mask needs no runtime FULL_MASK
        # slow path. `seq_info.prompt_offset` is a multiple of the 2Q `BM`
        # (256) — also a valid 1Q tile boundary (128).
        var seq_info: SeqInfo = get_seq_info[
            Self.BM,
            Self.num_q_heads,
            Self.MaskType.get_type_name() == "CausalMask",
        ](batch_size, pack.max_seq_len, pack.valid_length, pack.partition)

        comptime cfg_1q = Self.config.switch_1q_config()
        comptime check_mask = (
            Self.MaskType.nonfull_sets[cfg_1q.BM, cfg_1q.BN]()[0]
            == TileMaskStatus.UNKNOWN_MASK
        )
        comptime if Self.config.can_switch_to_1q() and not check_mask:
            comptime Kernel1Q = SM100MLA[
                Self.KVLUTType,
                Self.KRopeType,
                Self.output_dtype,
                Self.MaskType,
                Self.SchedulerType,
                cfg_1q,
                Self.ValidLengthType,
                Self.SinkType,
                Self.KVRowOffsetsType,
                Self.MaxSeqLenType,
                Self.PartitionType,
                Self._ndbuffer_mha_operand,
            ]
            # All eight TMA-op types fold between the 2Q and 1Q configs (Q
            # nope/rope and the q_scale box use `BM // num_qo` = 128 in both;
            # K_nope/K_rope/V/k_scale and the ragged store are
            # BM-independent), but the parser sees distinct parameter
            # expressions, so `rebind`.
            comptime QNope1Q = QTMATile[
                Kernel1Q.KVLUTType.dtype,
                Kernel1Q.config.qkv_swizzle_mode,
                BM=Kernel1Q.config.q_tile_rows(),
                depth=Kernel1Q.config.nope_depth,
                group=Kernel1Q.config.group,
                decoding=False,
            ]
            comptime QRope1Q = QTMATile[
                Kernel1Q.config.rope_gmem_dtype,
                Kernel1Q.config.rope_gmem_swizzle_mode,
                BM=Kernel1Q.config.q_tile_rows(),
                depth=Kernel1Q.config.rope_depth,
                group=Kernel1Q.config.group,
                decoding=False,
            ]
            comptime KNope1Q = KVTMATile[
                Kernel1Q.KVLUTType.dtype,
                Kernel1Q.config.qkv_swizzle_mode,
                BN=kv_sub_tile_rows(Kernel1Q.config.BN, Kernel1Q.page_size),
                BK=Kernel1Q.nope_depth,
            ]
            comptime KRope1Q = KVTMATile[
                Kernel1Q.KRopeType.dtype,
                Kernel1Q.config.rope_gmem_swizzle_mode,
                BN=kv_sub_tile_rows(
                    Kernel1Q.config.BN, Kernel1Q.KRopeType.page_size
                ),
                BK=Kernel1Q.rope_depth,
            ]
            comptime V1Q = KVTMATile[
                Kernel1Q.KVLUTType.dtype,
                Kernel1Q.config.qkv_swizzle_mode,
                BN=kv_sub_tile_rows(Kernel1Q.config.BN, Kernel1Q.page_size),
                BK=Kernel1Q.nope_depth,
            ]
            comptime QScale1Q = TMATensorTile[
                Kernel1Q.config.scale_dtype,
                2,
                Index(1, Kernel1Q.config.q_tile_rows()),
                Index(1, Kernel1Q.config.q_tile_rows()),
            ]
            comptime KScale1Q = TMATensorTile[
                Kernel1Q.config.scale_dtype,
                2,
                Index(
                    1, kv_sub_tile_rows(Kernel1Q.config.BN, Kernel1Q.page_size)
                ),
                Index(
                    1, kv_sub_tile_rows(Kernel1Q.config.BN, Kernel1Q.page_size)
                ),
            ]
            comptime O1Q = RaggedTMA3DTile[
                Kernel1Q.output_dtype,
                Kernel1Q.config.output_swizzle_mode,
                BM=Kernel1Q.config.fa4_config.BM
                // Kernel1Q.config.fa4_config.num_qo,
                BN=Kernel1Q.config.fa4_config.ov_depth,
                middle_dim=Kernel1Q.config.num_q_heads,
                group=Kernel1Q.config.fa4_config.group if Kernel1Q.config.fa4_config.fuse_gqa else 1,
                # Rebind target: must match the batched store the kernel built.
                tma_blocks_per_op=o_store_tma_blocks_per_op[
                    Kernel1Q.output_dtype,
                    Kernel1Q.config.output_swizzle_mode,
                    Kernel1Q.config.fa4_config.ov_depth,
                    Kernel1Q.config.fa4_config.group if Kernel1Q.config.fa4_config.fuse_gqa else 1,
                    depth_splits=2,
                ](),
            ]
            if broadcast(seq_info.seq_len - seq_info.prompt_offset) <= UInt32(
                Kernel1Q.BM
            ):
                Kernel1Q._kernel_impl_per_token_scale(
                    rebind[QNope1Q](q_nope_tma_op),
                    rebind[QRope1Q](q_rope_tma_op),
                    rebind[KNope1Q](k_nope_tma_op),
                    rebind[KRope1Q](k_rope_tma_op),
                    rebind[V1Q](v_tma_op),
                    rebind[QScale1Q](q_scale_tma_op),
                    rebind[KScale1Q](k_scale_tma_op),
                    rebind[O1Q](ragged_tma_store),
                    kv_lut,
                    k_rope_lut,
                    scale,
                    pack,
                    seq_info,
                )
                return
        Self._kernel_impl_per_token_scale(
            q_nope_tma_op,
            q_rope_tma_op,
            k_nope_tma_op,
            k_rope_tma_op,
            v_tma_op,
            q_scale_tma_op,
            k_scale_tma_op,
            ragged_tma_store,
            kv_lut,
            k_rope_lut,
            scale,
            pack,
            seq_info,
        )

    @staticmethod
    @always_inline
    def _kernel_impl_per_token_scale(
        q_nope_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.q_tile_rows(),
            depth=Self.config.nope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_rope_tma_op: QTMATile[
            config.rope_gmem_dtype,
            Self.config.rope_gmem_swizzle_mode,
            BM=Self.config.q_tile_rows(),
            depth=Self.config.rope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        q_scale_tma_op: TMATensorTile[
            config.scale_dtype,
            2,
            Index(1, Self.config.q_tile_rows()),
            Index(1, Self.config.q_tile_rows()),
        ],
        k_scale_tma_op: TMATensorTile[
            config.scale_dtype,
            2,
            Index(1, kv_sub_tile_rows(Self.config.BN, Self.page_size)),
            Index(1, kv_sub_tile_rows(Self.config.BN, Self.page_size)),
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_dtype,
            Self.config.output_swizzle_mode,
            BM=Self.config.fa4_config.BM // Self.config.fa4_config.num_qo,
            BN=Self.config.fa4_config.ov_depth,
            middle_dim=Self.config.num_q_heads,
            group=config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
            # Concrete value: a GPU-kernel entry (and the device impl it calls)
            # must be a fully-bound function. Matches the created store.
            tma_blocks_per_op=o_store_tma_blocks_per_op[
                Self.output_dtype,
                Self.config.output_swizzle_mode,
                Self.config.fa4_config.ov_depth,
                config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
                depth_splits=2,
            ](),
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        scale: Float32,
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            Self.SinkType,
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
        seq_info: SeqInfo,
    ):
        comptime assert (
            Self.config.supported()
        ), Self.config.fa4_config.description()
        comptime assert Self.KRopeType.dtype == config.rope_gmem_dtype
        comptime assert not Self.SchedulerType.may_advance

        mask = pack.mask
        max_seq_len = pack.max_seq_len

        # Matches the thin entrypoint's switch predicate; see the generic
        # kernel for the rationale. A 1Q instantiation has
        # `can_switch_to_1q() == False`, so it stays False.
        comptime _cfg_1q = Self.config.switch_1q_config()
        comptime _check_mask = (
            Self.MaskType.nonfull_sets[_cfg_1q.BM, _cfg_1q.BN]()[0]
            == TileMaskStatus.UNKNOWN_MASK
        )
        comptime output_nonempty = (
            Self.config.can_switch_to_1q() and not _check_mask
        )

        comptime SmemType = SM100AttentionSMem[Self.config.fa4_config]
        var attn_smem = SmemType()
        var misc_mbars = attn_smem.misc_mbars()
        var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.q_smem()
        )
        var k_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.v_smem_base()
        )
        var rope_smem = rebind[SharedMemPointer[Scalar[Self.KRopeType.dtype]]](
            attn_smem.rope_smem_base()
        )
        var q_scale_smem = rebind[SharedMemPointer[Scalar[config.scale_dtype]]](
            attn_smem.q_scale_smem()
        )
        var k_scale_smem = rebind[SharedMemPointer[Scalar[config.scale_dtype]]](
            attn_smem.k_scale_smem()
        )
        var ptr_tmem_addr = attn_smem.tmem_addr_ptr()

        comptime num_reg_softmax = 184
        comptime num_reg_correction = 96
        comptime num_reg_other = 48
        comptime num_reg_empty = 24

        comptime assert not Self.PartitionType.do_partition

        # Initialize barriers and tmem address
        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
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
                k_nope_tma_op.prefetch_descriptor()
            if e != 0:
                k_rope_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()
            if e != 0:
                q_scale_tma_op.prefetch_descriptor()
            if e != 0:
                k_scale_tma_op.prefetch_descriptor()

        barrier()

        var tmem_addr = ptr_tmem_addr[0]
        var role = warp_idx_to_role(warp_idx)

        if role == WarpRole.Softmax0 or role == WarpRole.Softmax1:
            warpgroup_reg_alloc[num_reg_softmax]()

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            fa4_softmax[
                Self.KVLUTType,
                Self.config.fa4_config,
                Self.ValidLengthType,
                NullPointer[Self.output_dtype],
                False,
                Self.MaxSeqLenType,
                output_nonempty=output_nonempty,
            ](
                attn_smem,
                tmem_addr,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_dtype](),
                max_seq_len.as_uint32(),
                ragged_tma_store,
                NullPointer[Self.output_dtype](),
                NonNullPointer[config.scale_dtype, AddressSpace.SHARED](
                    rebind[
                        UnsafePointer[
                            Scalar[config.scale_dtype],
                            ImmutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                        ]
                    ](q_scale_smem)
                ),
                NonNullPointer[config.scale_dtype, AddressSpace.SHARED](
                    rebind[
                        UnsafePointer[
                            Scalar[config.scale_dtype],
                            ImmutAnyOrigin,
                            address_space=AddressSpace.SHARED,
                        ]
                    ](k_scale_smem)
                ),
            )

        elif role == WarpRole.Correction:
            warpgroup_reg_dealloc[num_reg_correction]()

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            fa4_correction[
                Self.config.fa4_config,
                Self.page_size,
            ](
                attn_smem,
                tmem_addr,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        elif role == WarpRole.Load:
            warpgroup_reg_dealloc[num_reg_other]()

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            Self.load_per_token_scale(
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                seq_info,
                max_seq_len,
                mask,
                q_nope_tma_op,
                q_rope_tma_op,
                k_nope_tma_op,
                k_rope_tma_op,
                v_tma_op,
                q_scale_tma_op,
                k_scale_tma_op,
                kv_lut,
                k_rope_lut,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
                q_scale_smem,
                k_scale_smem,
            )

        elif role == WarpRole.MMA:
            warpgroup_reg_dealloc[num_reg_other]()

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            Self.mma(
                tmem_addr,
                misc_mbars,
                seq_info.prompt_idx,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
            )
        else:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    def load_per_token_scale(
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_nope_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.q_tile_rows(),
            depth=Self.config.nope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        q_rope_tma_op: QTMATile[
            config.rope_gmem_dtype,
            Self.config.rope_gmem_swizzle_mode,
            BM=Self.config.q_tile_rows(),
            depth=Self.config.rope_depth,
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        q_scale_tma_op: TMATensorTile[
            config.scale_dtype,
            2,
            Index(1, Self.config.q_tile_rows()),
            Index(1, Self.config.q_tile_rows()),
        ],
        k_scale_tma_op: TMATensorTile[
            config.scale_dtype,
            2,
            Index(1, kv_sub_tile_rows(Self.config.BN, Self.page_size)),
            Index(1, kv_sub_tile_rows(Self.config.BN, Self.page_size)),
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KRopeType.dtype]],
        q_scale_smem: SharedMemPointer[Scalar[config.scale_dtype]],
        k_scale_smem: SharedMemPointer[Scalar[config.scale_dtype]],
    ):
        """Load warp logic with per-token scale TMA loading.

        Structurally identical to Self.load() but adds q_scale and k_scale
        TMA loading on K barriers. q_scale halves accompany their Q-half
        TMAs (K0 barrier for Q0, Q1Sync for Q1; single issue in 1Q);
        k_scale is loaded on every K barrier with staged buffer indexing.
        """
        comptime num_qo = Self.config.num_qo()
        # 1Q does not implement mid-range FULL_MASK tile skipping (the
        # `check_mask` slow path that e.g. MaterializedMask requires):
        # the load/mma/softmax warps would disagree on tile counts.
        # Dispatch must route such masks to 2Q. Range-bounded
        # early-skipping (e.g. sliding-window via `start_column` /
        # `last_masked_set_end`) is supported.
        comptime if num_qo == 1:
            comptime assert not (
                mask.nonfull_sets[Self.BM, Self.BN]()[0]
                == TileMaskStatus.UNKNOWN_MASK
            ), (
                "1Q MLA prefill does not support masks requiring runtime"
                " FULL_MASK checks"
            )
        comptime KVPipeType = MLAKVLayouts[
            Self.KVLUTType.dtype,
            Self.KRopeType.dtype,
            DType.invalid,
            Self.config,
        ]

        comptime SMemTensorLT[elems: Int] = TileTensor[
            Self.KVLUTType.dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime RopeMemTensorLT[elems: Int] = TileTensor[
            config.rope_gmem_dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime q_nope_elems = type_of(q_nope_tma_op).tile_shape[0] * type_of(
            q_nope_tma_op
        ).tile_shape[1] * type_of(q_nope_tma_op).tile_shape[2]
        comptime QNopeType = SMemTensorLT[q_nope_elems]
        comptime q_rope_elems = type_of(q_rope_tma_op).tile_shape[0] * type_of(
            q_rope_tma_op
        ).tile_shape[1] * type_of(q_rope_tma_op).tile_shape[2]
        comptime QRopeType = RopeMemTensorLT[q_rope_elems]
        comptime ScaleSmemLT[elems: Int] = TileTensor[
            config.scale_dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime q_scale_elems_tma = type_of(q_scale_tma_op).tile_shape[
            0
        ] * type_of(q_scale_tma_op).tile_shape[1]
        comptime QScaleSmemType = ScaleSmemLT[q_scale_elems_tma]
        comptime k_scale_elems_tma = type_of(k_scale_tma_op).tile_shape[
            0
        ] * type_of(k_scale_tma_op).tile_shape[1]
        comptime KScaleSmemType = ScaleSmemLT[k_scale_elems_tma]
        comptime q_scale_elems = Self.config.BM
        comptime k_scale_elems = Self.config.BN

        var k_rope_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)
        var kv_head_idx: UInt32 = seq_info.head_idx

        # Per-TMA-call element counts: one of two Q halves in 2Q, the
        # single full-BM Q tile in 1Q — 128 rows in both modes.
        comptime q_nope_elements = (
            Self.config.q_tile_rows()
        ) * Self.config.nope_depth
        comptime q_rope_elements = (
            Self.config.q_tile_rows()
        ) * Self.config.rope_depth
        comptime q_nope_bytes = size_of[Self.qkv_dtype]() * q_nope_elements
        comptime q_rope_bytes = size_of[
            config.rope_gmem_dtype
        ]() * q_rope_elements
        comptime q_bytes = q_nope_bytes + q_rope_bytes
        # Per-TMA-issue q_scale bytes (one Q half in 2Q, the full BM=128
        # tile in 1Q — same value either way).
        comptime q_scale_bytes = (Self.config.q_tile_rows()) * size_of[
            config.scale_dtype
        ]()
        comptime k_scale_bytes = Self.config.BN * size_of[config.scale_dtype]()

        # Sub-tile paging: when page_size < BN, each BN-row load is split
        # into num_kv_pages sub-tile loads of kv_sub_BN rows each.
        comptime kv_sub_BN = kv_sub_tile_rows(Self.config.BN, Self.page_size)
        comptime num_kv_pages = kv_num_sub_tiles(Self.config.BN, Self.page_size)
        comptime rope_sub_BN = kv_sub_tile_rows(
            Self.config.BN, Self.KRopeType.page_size
        )
        comptime num_rope_pages = kv_num_sub_tiles(
            Self.config.BN, Self.KRopeType.page_size
        )
        comptime PagedRows = PagedRowIndices[Self.config.BN, Self.page_size]
        comptime RopePagedRows = PagedRowIndices[
            Self.config.BN, Self.KRopeType.page_size
        ]

        # Alignment of `kv_row` produced by mask-driven iteration.
        comptime base_alignment: Int = Self.MaskType.start_column_alignment[
            Self.BM, Self.BN, Self.page_size
        ]()

        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )
        var q_head_idx: UInt32 = seq_info.head_idx
        e = elect()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](seq_info.prompt_idx, score_row)
        var paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
            seq_info.prompt_idx, kv_row
        )
        var rope_paged_rows = k_rope_lut.populate[
            Self.config.BN, base_alignment
        ](seq_info.prompt_idx, kv_row)
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                seq_info.prompt_idx, score_row, num_keys
            )
            - 1
        )

        # Partial-page handling: when page_size < BN, runtime-bound the
        # K_nope/V/K_rope/k_scale sub-tile loops via the
        # `needs_partial=True` overloads (mirrors FA4 `load_warp.mojo`
        # and the analogous `mla_prefill_generic.mojo` /
        # `mla_prefill_blockscale.mojo` fixes). K_nope/V/k_scale use
        # `Self.page_size`; K_rope uses `KRopeType.page_size`. Flags
        # computed independently so the iter_count peel below correctly
        # gates on either page table needing a runtime bound.
        comptime needs_partial_kv = (
            Self.page_size > 0 and Self.page_size < Self.config.BN
        )
        comptime needs_partial_rope = (
            Self.KRopeType.page_size > 0
            and Self.KRopeType.page_size < Self.config.BN
        )
        comptime needs_partial = needs_partial_kv or needs_partial_rope

        # Per-sub-page byte sizes for partial expect_bytes_pred. Unlike
        # blockscale, per-token-scale has no CVT pipeline — all bytes
        # (K_nope + K_rope + k_scale, plus Q + q_scale on the prologue
        # only) go on the K barrier. q_scale/q bytes are constant and
        # NOT partial-aware.
        comptime k_nope_bytes_pp = (
            Self.nope_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )
        comptime k_rope_bytes_pp = (
            Self.rope_depth * rope_sub_BN * size_of[Self.KRopeType.dtype]()
        )
        comptime v_bytes_pp = (
            Self.nope_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )
        comptime k_scale_bytes_pp = kv_sub_BN * size_of[config.scale_dtype]()
        # Full-tile byte counts (when no partial bound applies).
        comptime k_rope_full_bytes = (
            Self.rope_depth * Self.config.BN * size_of[Self.KRopeType.dtype]()
        )
        comptime kv_data_full_bytes = (
            Self.nope_depth * Self.config.BN * size_of[Self.qkv_dtype]()
        )

        @parameter
        @always_inline
        def _k_num_valid_pages(current_kv_row: UInt32) -> UInt32:
            """Valid K_nope/V/k_scale sub-tile pages at `current_kv_row`."""
            if current_kv_row >= num_keys:
                return UInt32(0)
            return min(
                UInt32(num_kv_pages),
                UInt32(ceildiv(Int(num_keys - current_kv_row), Int(kv_sub_BN))),
            )

        @parameter
        @always_inline
        def _rope_num_valid_pages(current_kv_row: UInt32) -> UInt32:
            """Valid K_rope sub-tile pages at `current_kv_row`."""
            if current_kv_row >= num_keys:
                return UInt32(0)
            return min(
                UInt32(num_rope_pages),
                UInt32(
                    ceildiv(Int(num_keys - current_kv_row), Int(rope_sub_BN))
                ),
            )

        # ---- Mode-shared sub-tile constants ----
        # The K_rope sub-tile shape (`rope_depth * rope_sub_BN`) is
        # identical in fused-KV and split-KV mode; only the smem
        # destination differs. Hoist so the unified `_produce_k_rope`
        # closure works for both modes.
        comptime k_rope_sub_elems = Self.rope_depth * rope_sub_BN
        comptime KRopeSubType = TileTensor[
            Self.KRopeType.dtype,
            type_of(tt_row_major[k_rope_sub_elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]

        # K_rope shared closure. Bytes are accounted by the caller on
        # the K barrier (no separate CVT mbar — unlike blockscale).
        @parameter
        @always_inline
        def _produce_k_rope[
            partial: Bool,
        ](
            rope_pages: type_of(rope_paged_rows),
            kv_row_base: UInt32,
            smem_base_ptr: SharedMemPointer[Scalar[Self.KRopeType.dtype]],
            mbar: SharedMemPointer[SharedMemBarrier],
            rope_nvp: UInt32 = UInt32(num_rope_pages),
        ):
            """K_rope sub-tile TMA. `partial=True` early-returns at
            `_p == rope_nvp`. Bytes are accounted by the caller on
            `mbar`.
            """
            comptime for _p in range(num_rope_pages):
                comptime if partial:
                    if UInt32(_p) == rope_nvp:
                        return
                # Belt-and-suspenders: post-fix this should be
                # unreachable on every config.
                debug_assert(
                    kv_row_base + UInt32(_p * rope_sub_BN) < num_keys,
                    (
                        "MLA per_token_scale K_rope sub-tile TMA OOB"
                        " after partial bound: kv_row_base="
                    ),
                    kv_row_base,
                    " _p=",
                    _p,
                    " rope_sub_BN=",
                    rope_sub_BN,
                    " num_keys=",
                    num_keys,
                    " rope_nvp=",
                    rope_nvp,
                    " partial=",
                    partial,
                )
                var k_rope_coord = kv_coord[depth=Self.rope_depth,](
                    rope_pages.get_row(UInt32(_p * rope_sub_BN)),
                    k_rope_head_idx,
                )
                k_rope_coord[0] = UInt32(Self.cache_depth - Self.rope_depth)
                k_rope_tma_op.async_copy_elect(
                    KRopeSubType(
                        smem_base_ptr + _p * k_rope_sub_elems,
                        tt_row_major[k_rope_sub_elems](),
                    ),
                    mbar[],
                    k_rope_coord,
                    e,
                )

        # k_scale shared closure. Bytes accounted by caller on `mbar`.
        @parameter
        @always_inline
        def _produce_k_scale[
            partial: Bool,
        ](
            k_pages: type_of(paged_rows),
            kv_row_base: UInt32,
            k_scale_smem_ptr: SharedMemPointer[Scalar[config.scale_dtype]],
            mbar: SharedMemPointer[SharedMemBarrier],
            k_nvp: UInt32 = UInt32(num_kv_pages),
        ):
            """k_scale sub-tile TMA. `partial=True` early-returns at
            `_p == k_nvp`. Bytes are accounted by the caller on `mbar`.
            """
            comptime for _p in range(num_kv_pages):
                comptime if partial:
                    if UInt32(_p) == k_nvp:
                        return
                debug_assert(
                    kv_row_base + UInt32(_p * kv_sub_BN) < num_keys,
                    (
                        "MLA per_token_scale k_scale sub-tile TMA OOB"
                        " after partial bound: kv_row_base="
                    ),
                    kv_row_base,
                    " _p=",
                    _p,
                    " kv_sub_BN=",
                    kv_sub_BN,
                    " num_keys=",
                    num_keys,
                    " k_nvp=",
                    k_nvp,
                    " partial=",
                    partial,
                )
                k_scale_tma_op.async_copy_elect(
                    KScaleSmemType(
                        k_scale_smem_ptr + _p * k_scale_elems_tma,
                        tt_row_major[k_scale_elems_tma](),
                    ),
                    mbar[],
                    (Int(k_pages.get_row(UInt32(_p * kv_sub_BN))), 0),
                    e,
                )

        # V shared closure. V is on its own barrier in both modes
        # (kv_pipeline.producer_mbar() in fused-KV,
        # pipeline_v.get_tile().mbar in split-KV), so this closure
        # emits its own partial-aware `expect_bytes_pred` directly.
        @parameter
        @always_inline
        def _produce_v[
            partial: Bool,
        ](
            paged: type_of(paged_rows),
            mbar: SharedMemPointer[SharedMemBarrier],
            smem_ptr: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
            v_nvp: UInt32 = UInt32(num_kv_pages),
        ):
            """V tile production at `mbar` / `smem_ptr`. Emits partial-
            aware `expect_bytes_pred` so the caller doesn't need to
            track V byte count.
            """
            var v_bytes_local: Int32
            comptime if partial:
                v_bytes_local = Int32(v_bytes_pp) * Int32(v_nvp)
            else:
                v_bytes_local = Int32(kv_data_full_bytes)
            expect_bytes_pred(mbar, v_bytes_local, e)
            paged.tma_copy_v[needs_partial=partial](
                v_tma_op,
                smem_ptr,
                mbar[],
                kv_head_idx=kv_head_idx,
                elect=e,
                num_valid_pages=v_nvp,
            )

        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        comptime if Self.config.fa4_config.use_fused_kv:
            # ---- Fused KV mode with per-token scale ----
            comptime kv_stage_elems = (
                Self.config.fa4_config.padded_ov_depth * Self.config.BN
            )
            comptime rope_stage_elems = (
                Self.config.rope_depth * Self.config.BN
            )

            comptime KVPipeProdType = StagedPipeline[
                Self.config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeProdType = {mbars.get_k_mbars()}
            kv_pipeline.state._phase = 1

            var rope_idx: UInt32 = 0
            comptime num_rope_bufs = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )
            comptime num_k_scale_bufs = UInt32(
                Self.config.fa4_config.num_k_scale_bufs()
            )
            var k_scale_idx: UInt32 = 0

            @parameter
            @always_inline
            def _fused_kv_buffer_ptr() -> (
                SharedMemPointer[Scalar[Self.KVLUTType.dtype]]
            ):
                """Current kv_pipeline buffer slot (shared K_nope/V)."""
                return k_smem_base + kv_pipeline.state.index() * UInt32(
                    kv_stage_elems
                )

            @parameter
            @always_inline
            def _fused_rope_smem_ptr() -> (
                SharedMemPointer[Scalar[Self.KRopeType.dtype]]
            ):
                """Current rope buffer slot.

                `rope_smem_base` is already typed as
                `SharedMemPointer[Scalar[Self.KRopeType.dtype]]` (no
                bitcast needed unlike blockscale's BF16 buffer).
                """
                return rope_smem_base + rope_idx * UInt32(rope_stage_elems)

            @parameter
            @always_inline
            def _fused_k_scale_smem_ptr() -> (
                SharedMemPointer[Scalar[config.scale_dtype]]
            ):
                """Current k_scale buffer slot."""
                return k_scale_smem + k_scale_idx * UInt32(k_scale_elems)

            @parameter
            @always_inline
            def _produce_k_fused[
                partial: Bool,
                with_q: Bool = False,
            ](
                paged: type_of(paged_rows),
                rope_paged: type_of(rope_paged_rows),
                kv_row_local: UInt32,
                mbar: type_of(kv_pipeline.producer_mbar()),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q + q_scale (if `with_q`) + K_nope + K_rope + k_scale
                onto `mbar`.

                In per-token-scale, all bytes go on the same K barrier
                (no separate CVT mbar like blockscale). q + q_scale are
                only loaded with `with_q=True` (prologue); main-loop /
                peeled-last skip them.
                """
                var qk_bytes: Int32 = Int32(
                    q_bytes + q_scale_bytes
                ) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                    qk_bytes += Int32(k_rope_bytes_pp) * Int32(rope_nvp)
                    qk_bytes += Int32(k_scale_bytes_pp) * Int32(k_nvp)
                else:
                    qk_bytes += Int32(kv_data_full_bytes)
                    qk_bytes += Int32(k_rope_full_bytes)
                    qk_bytes += Int32(k_scale_bytes)
                expect_bytes_pred(mbar, qk_bytes, e)

                comptime if with_q:
                    q_nope_tma_op.async_copy_elect(
                        QNopeType(q_smem, tt_row_major[q_nope_elems]()),
                        mbar[],
                        q_coord[
                            depth=Self.config.nope_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )
                    q_rope_tma_op.async_copy_elect(
                        QRopeType(
                            (
                                q_smem.bitcast[Scalar[DType.uint8]]()
                                + q_nope_bytes
                            ).bitcast[Scalar[config.rope_gmem_dtype]](),
                            tt_row_major[q_rope_elems](),
                        ),
                        mbar[],
                        q_coord[
                            depth=Self.config.rope_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )
                paged.tma_copy_k[needs_partial=partial](
                    k_nope_tma_op,
                    _fused_kv_buffer_ptr(),
                    mbar[],
                    kv_head_idx=kv_head_idx,
                    elect=e,
                    k_num_valid_pages=k_nvp,
                )
                _produce_k_rope[partial=partial](
                    rope_paged,
                    kv_row_local,
                    _fused_rope_smem_ptr(),
                    mbar,
                    rope_nvp,
                )
                comptime if with_q:
                    q_scale_tma_op.async_copy_elect(
                        QScaleSmemType(
                            q_scale_smem, tt_row_major[q_scale_elems_tma]()
                        ),
                        mbar[],
                        (Int(q_gmem_row), 0),
                        e,
                    )
                _produce_k_scale[partial=partial](
                    paged,
                    kv_row_local,
                    _fused_k_scale_smem_ptr(),
                    mbar,
                    k_nvp,
                )

            comptime if num_qo == 1:
                # ---- 1Q fused-KV producer ----
                # MMA consumes K_e, K_o, V_e, V_o per logical iter;
                # produce in matching slot order (mirrors the generic
                # MLA / load_warp.mojo 1Q fused producers). No FULL_MASK
                # skipping here (see the `check_mask` assert at the top
                # of `load_per_token_scale`). Q + q_scale ride the
                # peeled K_e[0] barrier (with_q=True); there is no Q1.

                # Per-tile emit helpers bundling the KV producer-
                # pipeline lifecycle (acquire / mbar / produce /
                # rope+k_scale buffer cycle / step). The first peeled K
                # slot passes `acquire=False` (initial producer
                # phase = 1).
                @parameter
                @always_inline
                def _emit_k_1q[
                    partial: Bool,
                    with_q: Bool = False,
                    acquire: Bool = True,
                ](
                    paged: type_of(paged_rows),
                    rope_paged: type_of(rope_paged_rows),
                    kv_row_local: UInt32,
                    k_nvp: UInt32,
                    rope_nvp: UInt32,
                ):
                    comptime if acquire:
                        kv_pipeline.producer_acquire()
                    var mbar = kv_pipeline.producer_mbar()
                    _produce_k_fused[partial=partial, with_q=with_q](
                        paged,
                        rope_paged,
                        kv_row_local,
                        mbar,
                        k_nvp,
                        rope_nvp,
                    )
                    rope_idx = (rope_idx + 1) % num_rope_bufs
                    k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
                    kv_pipeline.state.step()

                @parameter
                @always_inline
                def _emit_v_1q[
                    partial: Bool
                ](paged: type_of(paged_rows), v_nvp: UInt32):
                    kv_pipeline.producer_acquire()
                    var mbar = kv_pipeline.producer_mbar()
                    _produce_v[partial=partial](
                        paged, mbar, _fused_kv_buffer_ptr(), v_nvp
                    )
                    kv_pipeline.state.step()

                # T is the total K-tile count (iter_count was peeled by
                # 1 at the top of `load_per_token_scale`).
                var T: UInt32 = iter_count + UInt32(1)
                var k_nvp_0 = _k_num_valid_pages(kv_row)
                var rope_nvp_0 = _rope_num_valid_pages(kv_row)

                # T == 1 fast path: produce K_e[0] (with Q + q_scale) +
                # V_e[0] only.
                if T == UInt32(1):
                    _emit_k_1q[
                        partial=needs_partial, with_q=True, acquire=False
                    ](paged_rows, rope_paged_rows, kv_row, k_nvp_0, rope_nvp_0)
                    _emit_v_1q[partial=needs_partial](paged_rows, k_nvp_0)
                    return

                # ---- Peel (T >= 2): K_e[0], K_o[0], V_e[0], V_o[0] ----
                var kv_row_o = kv_row + UInt32(Self.config.BN)
                var k_nvp_o: UInt32 = UInt32(num_kv_pages)
                var rope_nvp_o: UInt32 = UInt32(num_rope_pages)
                comptime if needs_partial:
                    k_nvp_o = _k_num_valid_pages(kv_row_o)
                    rope_nvp_o = _rope_num_valid_pages(kv_row_o)
                var paged_rows_o = kv_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row_o)
                var rope_paged_rows_o = k_rope_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row_o)

                # K_e[0] with Q + q_scale (initial slot; no acquire).
                _emit_k_1q[partial=needs_partial, with_q=True, acquire=False](
                    paged_rows, rope_paged_rows, kv_row, k_nvp_0, rope_nvp_0
                )
                # K_o[0]
                _emit_k_1q[partial=needs_partial](
                    paged_rows_o,
                    rope_paged_rows_o,
                    kv_row_o,
                    k_nvp_o,
                    rope_nvp_o,
                )
                # V_e[0] / V_o[0] (reuse rows)
                _emit_v_1q[partial=needs_partial](paged_rows, k_nvp_0)
                _emit_v_1q[partial=needs_partial](paged_rows_o, k_nvp_o)

                # ---- Loop bookkeeping (see generic MLA 1Q producer) ----
                var main_iters_1q: UInt32 = (T - UInt32(2)) >> UInt32(1)
                var has_tail: Bool = (T & UInt32(1)) == UInt32(1)
                var has_peeled_last_full: Bool = False
                comptime if needs_partial:
                    if not has_tail and main_iters_1q > UInt32(0):
                        main_iters_1q -= UInt32(1)
                        has_peeled_last_full = True

                # ---- Main loop (full tiles) ----
                while main_iters_1q != UInt32(0):
                    main_iters_1q -= UInt32(1)
                    kv_row += UInt32(2 * Self.config.BN)
                    kv_row_o = kv_row + UInt32(Self.config.BN)

                    # K_e[n] (full)
                    paged_rows = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    rope_paged_rows = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    _emit_k_1q[partial=False](
                        paged_rows,
                        rope_paged_rows,
                        kv_row,
                        UInt32(num_kv_pages),
                        UInt32(num_rope_pages),
                    )
                    # K_o[n] (full)
                    paged_rows_o = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row_o)
                    rope_paged_rows_o = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row_o)
                    _emit_k_1q[partial=False](
                        paged_rows_o,
                        rope_paged_rows_o,
                        kv_row_o,
                        UInt32(num_kv_pages),
                        UInt32(num_rope_pages),
                    )
                    # V_e[n] / V_o[n] (full, reuse rows)
                    _emit_v_1q[partial=False](paged_rows, UInt32(num_kv_pages))
                    _emit_v_1q[partial=False](
                        paged_rows_o, UInt32(num_kv_pages)
                    )

                # ---- Tail K_e + V_e (T odd, any needs_partial) ----
                if has_tail:
                    kv_row += UInt32(2 * Self.config.BN)
                    var k_nvp_t: UInt32 = UInt32(num_kv_pages)
                    var rope_nvp_t: UInt32 = UInt32(num_rope_pages)
                    comptime if needs_partial:
                        k_nvp_t = _k_num_valid_pages(kv_row)
                        rope_nvp_t = _rope_num_valid_pages(kv_row)
                    paged_rows = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    rope_paged_rows = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    _emit_k_1q[partial=needs_partial](
                        paged_rows, rope_paged_rows, kv_row, k_nvp_t, rope_nvp_t
                    )
                    _emit_v_1q[partial=needs_partial](paged_rows, k_nvp_t)

                # ---- Peeled-last full pair (needs_partial, T even) ----
                comptime if needs_partial:
                    if has_peeled_last_full:
                        kv_row += UInt32(2 * Self.config.BN)
                        kv_row_o = kv_row + UInt32(Self.config.BN)
                        var k_nvp_pe = _k_num_valid_pages(kv_row)
                        var rope_nvp_pe = _rope_num_valid_pages(kv_row)
                        var k_nvp_po = _k_num_valid_pages(kv_row_o)
                        var rope_nvp_po = _rope_num_valid_pages(kv_row_o)
                        # K_e (partial)
                        paged_rows = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        rope_paged_rows = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        _emit_k_1q[partial=True](
                            paged_rows,
                            rope_paged_rows,
                            kv_row,
                            k_nvp_pe,
                            rope_nvp_pe,
                        )
                        # K_o (partial)
                        paged_rows_o = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row_o)
                        rope_paged_rows_o = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row_o)
                        _emit_k_1q[partial=True](
                            paged_rows_o,
                            rope_paged_rows_o,
                            kv_row_o,
                            k_nvp_po,
                            rope_nvp_po,
                        )
                        # V_e / V_o (partial, reuse rows)
                        _emit_v_1q[partial=True](paged_rows, k_nvp_pe)
                        _emit_v_1q[partial=True](paged_rows_o, k_nvp_po)
            else:
                # ---- 2Q fused-KV producer (original) ----

                # ---- Peeled: K0 + Q0 + q_scale + k_scale[0] ----
                var k0_mbar = kv_pipeline.producer_mbar()
                var k_nvp_0 = _k_num_valid_pages(kv_row)
                var rope_nvp_0 = _rope_num_valid_pages(kv_row)
                _produce_k_fused[partial=needs_partial, with_q=True](
                    paged_rows,
                    rope_paged_rows,
                    kv_row,
                    k0_mbar,
                    k_nvp_0,
                    rope_nvp_0,
                )
                rope_idx = (rope_idx + 1) % num_rope_bufs
                k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
                kv_pipeline.state.step()

                # ---- Q1 + q_scale half 1 (separate barrier) ----
                q_gmem_row += UInt32(Self.config.BM // 2)
                var q1_mbar = mbars.q1_wait_mbar()
                expect_bytes_pred(q1_mbar, Int32(q_bytes + q_scale_bytes), e)
                # Q1 nope/rope — elect-predicated in-PTX via `_elect`.
                q_nope_tma_op.async_copy_elect(
                    QNopeType(
                        (
                            q_smem.bitcast[Scalar[DType.uint8]]()
                            + q_nope_bytes
                            + q_rope_bytes
                        ).bitcast[Scalar[Self.qkv_dtype]](),
                        tt_row_major[q_nope_elems](),
                    ),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.config.nope_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                    e,
                )
                q_rope_tma_op.async_copy_elect(
                    QRopeType(
                        (
                            q_smem.bitcast[Scalar[DType.uint8]]()
                            + q_nope_bytes
                            + q_rope_bytes
                            + q_nope_bytes
                        ).bitcast[Scalar[config.rope_gmem_dtype]](),
                        tt_row_major[q_rope_elems](),
                    ),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.config.rope_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                    e,
                )
                # Q1's q_scale half: rows [BM//2, BM) at smem offset
                # BM//2 (q_gmem_row was advanced above). Visibility for
                # softmax WG1 follows the q1_mbar -> MMA -> S1-commit
                # chain, same as Q1 itself.
                q_scale_tma_op.async_copy_elect(
                    QScaleSmemType(
                        q_scale_smem + (Self.config.BM // 2),
                        tt_row_major[q_scale_elems_tma](),
                    ),
                    q1_mbar[0],
                    (Int(q_gmem_row), 0),
                    e,
                )

                # ---- V0 (reuses paged_rows from K0) ----
                kv_pipeline.producer_acquire()
                var v0_mbar = kv_pipeline.producer_mbar()
                _produce_v[partial=needs_partial](
                    paged_rows, v0_mbar, _fused_kv_buffer_ptr(), k_nvp_0
                )
                kv_pipeline.state.step()

                # ---- KV producer loop ----
                # Main body always issues full tiles (partial=False). When
                # needs_partial, peel off the last iteration so its
                # populate/TMAs can be runtime-bounded.
                var main_iters = iter_count
                comptime if needs_partial:
                    if main_iters > 0:
                        main_iters -= 1
                while main_iters != 0:
                    main_iters -= 1
                    kv_row += UInt32(Self.config.BN)

                    comptime if check_mask:
                        if (
                            Self.mask_status(
                                mask, seq_info.prompt_idx, score_row, kv_row
                            )
                            == TileMaskStatus.FULL_MASK
                        ):
                            continue
                    paged_rows = kv_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)
                    rope_paged_rows = k_rope_lut.populate[
                        Self.config.BN, base_alignment
                    ](seq_info.prompt_idx, kv_row)

                    # Produce K_nope_n + K_rope_n + k_scale (full sub-tiles)
                    kv_pipeline.producer_acquire()
                    var kn_mbar = kv_pipeline.producer_mbar()
                    _produce_k_fused[partial=False](
                        paged_rows, rope_paged_rows, kv_row, kn_mbar
                    )
                    rope_idx = (rope_idx + 1) % num_rope_bufs
                    k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
                    kv_pipeline.state.step()

                    # Produce Vn (full)
                    kv_pipeline.producer_acquire()
                    var vn_mbar = kv_pipeline.producer_mbar()
                    _produce_v[partial=False](
                        paged_rows, vn_mbar, _fused_kv_buffer_ptr()
                    )
                    kv_pipeline.state.step()

                # ---- Peeled last iteration (partial-page bound) ----
                comptime if needs_partial:
                    if iter_count > 0:
                        kv_row += UInt32(Self.config.BN)
                        var _skip_last = False
                        comptime if check_mask:
                            if (
                                Self.mask_status(
                                    mask,
                                    seq_info.prompt_idx,
                                    score_row,
                                    kv_row,
                                )
                                == TileMaskStatus.FULL_MASK
                            ):
                                _skip_last = True
                        if not _skip_last:
                            # Re-populate BOTH LUTs at the new kv_row.
                            paged_rows = kv_lut.populate[
                                Self.config.BN, base_alignment
                            ](seq_info.prompt_idx, kv_row)
                            rope_paged_rows = k_rope_lut.populate[
                                Self.config.BN, base_alignment
                            ](seq_info.prompt_idx, kv_row)
                            var k_nvp_last = _k_num_valid_pages(kv_row)
                            var rope_nvp_last = _rope_num_valid_pages(kv_row)
                            # Kn (partial) + K_rope_n (partial) + k_scale_n
                            kv_pipeline.producer_acquire()
                            var kn_mbar_last = kv_pipeline.producer_mbar()
                            _produce_k_fused[partial=needs_partial](
                                paged_rows,
                                rope_paged_rows,
                                kv_row,
                                kn_mbar_last,
                                k_nvp_last,
                                rope_nvp_last,
                            )
                            rope_idx = (rope_idx + 1) % num_rope_bufs
                            k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
                            kv_pipeline.state.step()
                            # Vn (partial)
                            kv_pipeline.producer_acquire()
                            var vn_mbar_last = kv_pipeline.producer_mbar()
                            _produce_v[partial=needs_partial](
                                paged_rows,
                                vn_mbar_last,
                                _fused_kv_buffer_ptr(),
                                k_nvp_last,
                            )
                            kv_pipeline.state.step()

        else:
            # ---- Split KV mode with per-token scale ----
            comptime VPipeType = VProducerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            var k_pipeline = StagedPipeline[
                Self.config.num_kv_stages, Self.config.num_qk_stages
            ](mbars.get_k_mbars())
            k_pipeline.state._phase = 1
            var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem_base}

            # K stage may contain mixed dtypes (e.g. FP8 nope + BF16 rope).
            # Compute byte size then convert to qkv_dtype element count.
            comptime k_stage_bytes = (
                Self.config.fa4_config.padded_ov_depth
                * Self.config.BN
                * Self.qkv_dt_size
                + Self.config.rope_depth
                * Self.config.BN
                * Self.config.rope_mma_dtype_size
            )
            comptime k_elements_per_stage = k_stage_bytes // Self.qkv_dt_size
            comptime num_k_scale_bufs = UInt32(
                Self.config.fa4_config.num_k_scale_bufs()
            )
            var k_scale_idx: UInt32 = 0

            # Sub-tile element counts for split-mode paged KV loads.
            # k_rope_split_sub == k_rope_sub_elems (just renamed for
            # split-mode readability) since both modes write
            # `rope_depth * rope_sub_BN` per sub-page.
            comptime k_nope_split_sub = Self.nope_depth * kv_sub_BN
            comptime k_rope_split_sub = Self.rope_depth * rope_sub_BN
            comptime KNopeSplitSub = SMemTensorLT[k_nope_split_sub]

            @parameter
            @always_inline
            def _split_v_smem_ptr(
                pair: type_of(pipeline_v.get_tile[qk_stage=0]()),
            ) -> SharedMemPointer[Scalar[Self.KVLUTType.dtype]]:
                """V destination smem ptr for split-KV V pipeline pair.

                Note we switched from `pipeline_v.get_v(e)` (which auto-
                emits a fixed-size `expect_bytes`) to
                `pipeline_v.get_tile[qk_stage=0]()` so the unified
                `_produce_v` closure can emit a partial-aware
                `expect_bytes_pred` itself.
                """
                return rebind[SharedMemPointer[Scalar[Self.KVLUTType.dtype]]](
                    pair.smem.ptr
                )

            @parameter
            @always_inline
            def _produce_k_split[
                partial: Bool,
                with_q: Bool = False,
            ](
                paged: type_of(paged_rows),
                rope_paged: type_of(rope_paged_rows),
                kv_row_local: UInt32,
                mbar: type_of(k_pipeline.producer_mbar[qk_stage=0]()),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q + q_scale (if `with_q`) + K_nope + K_rope + k_scale
                onto `mbar`.

                Per-token-scale split-KV bundles K_rope onto the SAME K
                barrier (no separate CVT mbar like blockscale). Includes
                the `split_smem` decomposition into K_nope and K_rope
                smem regions.
                """
                var qk_bytes: Int32 = Int32(
                    q_bytes + q_scale_bytes
                ) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                    qk_bytes += Int32(k_rope_bytes_pp) * Int32(rope_nvp)
                    qk_bytes += Int32(k_scale_bytes_pp) * Int32(k_nvp)
                else:
                    qk_bytes += Int32(kv_data_full_bytes)
                    qk_bytes += Int32(k_rope_full_bytes)
                    qk_bytes += Int32(k_scale_bytes)
                expect_bytes_pred(mbar, qk_bytes, e)

                comptime if with_q:
                    q_nope_tma_op.async_copy_elect(
                        QNopeType(q_smem, tt_row_major[q_nope_elems]()),
                        mbar[],
                        q_coord[
                            depth=Self.config.nope_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )
                    q_rope_tma_op.async_copy_elect(
                        QRopeType(
                            (
                                q_smem.bitcast[Scalar[DType.uint8]]()
                                + q_nope_bytes
                            ).bitcast[Scalar[config.rope_gmem_dtype]](),
                            tt_row_major[q_rope_elems](),
                        ),
                        mbar[],
                        q_coord[
                            depth=Self.config.rope_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )

                var smem_ptr = k_smem_base + k_pipeline.state.index() * UInt32(
                    k_elements_per_stage
                )
                k_nope_smem_local, k_rope_smem_local = split_smem[
                    KVPipeType.k_nope_tma_layout,
                    KVPipeType.k_rope_tma_layout,
                    Self.KVLUTType.dtype,
                    Self.KRopeType.dtype,
                ](
                    SMemTensorLT[KVPipeType.k_tma_layout](
                        smem_ptr, tt_row_major[KVPipeType.k_tma_layout]()
                    )
                )
                paged.tma_copy_k[needs_partial=partial](
                    k_nope_tma_op,
                    rebind[SharedMemPointer[Scalar[Self.KVLUTType.dtype]]](
                        k_nope_smem_local.ptr
                    ),
                    mbar[],
                    kv_head_idx=kv_head_idx,
                    elect=e,
                    k_num_valid_pages=k_nvp,
                )
                _produce_k_rope[partial=partial](
                    rope_paged,
                    kv_row_local,
                    rebind[SharedMemPointer[Scalar[Self.KRopeType.dtype]]](
                        k_rope_smem_local.ptr
                    ),
                    mbar,
                    rope_nvp,
                )
                comptime if with_q:
                    q_scale_tma_op.async_copy_elect(
                        QScaleSmemType(
                            q_scale_smem, tt_row_major[q_scale_elems_tma]()
                        ),
                        mbar[],
                        (Int(q_gmem_row), 0),
                        e,
                    )
                _produce_k_scale[partial=partial](
                    paged,
                    kv_row_local,
                    k_scale_smem + k_scale_idx * UInt32(k_scale_elems),
                    mbar,
                    k_nvp,
                )

            # ---- K0 + Q0 + q_scale + k_scale[0] (combined K barrier) ----
            var k0_mbar = k_pipeline.producer_mbar[qk_stage=0]()
            var k_nvp_0 = _k_num_valid_pages(kv_row)
            var rope_nvp_0 = _rope_num_valid_pages(kv_row)
            _produce_k_split[partial=needs_partial, with_q=True](
                paged_rows,
                rope_paged_rows,
                kv_row,
                k0_mbar,
                k_nvp_0,
                rope_nvp_0,
            )
            k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
            k_pipeline.state.step()

            # ---- Q1 + q_scale half 1 (separate barrier) ----
            # Skipped in 1Q: the peeled K0 issue above (with_q=True)
            # already loaded the full BM-row Q tile + q_scale on the K
            # mbar.
            comptime if num_qo == 2:
                q_gmem_row += UInt32(Self.config.BM // 2)
                var q1_mbar = mbars.q1_wait_mbar()
                expect_bytes_pred(q1_mbar, Int32(q_bytes + q_scale_bytes), e)
                # Q1 nope/rope — elect-predicated in-PTX via `_elect`.
                q_nope_tma_op.async_copy_elect(
                    QNopeType(
                        (
                            q_smem.bitcast[Scalar[DType.uint8]]()
                            + q_nope_bytes
                            + q_rope_bytes
                        ).bitcast[Scalar[Self.qkv_dtype]](),
                        tt_row_major[q_nope_elems](),
                    ),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.config.nope_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                    e,
                )
                q_rope_tma_op.async_copy_elect(
                    QRopeType(
                        (
                            q_smem.bitcast[Scalar[DType.uint8]]()
                            + q_nope_bytes
                            + q_rope_bytes
                            + q_nope_bytes
                        ).bitcast[Scalar[config.rope_gmem_dtype]](),
                        tt_row_major[q_rope_elems](),
                    ),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.config.rope_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                    e,
                )
                # Q1's q_scale half: rows [BM//2, BM) at smem offset
                # BM//2 (q_gmem_row was advanced above).
                q_scale_tma_op.async_copy_elect(
                    QScaleSmemType(
                        q_scale_smem + (Self.config.BM // 2),
                        tt_row_major[q_scale_elems_tma](),
                    ),
                    q1_mbar[0],
                    (Int(q_gmem_row), 0),
                    e,
                )

            # ---- V0 (reuses paged_rows from K0) ----
            var mbarv0 = pipeline_v.get_tile[qk_stage=0]()
            _produce_v[partial=needs_partial](
                paged_rows, mbarv0.mbar, _split_v_smem_ptr(mbarv0), k_nvp_0
            )
            pipeline_v.commit_step()

            # ---- KV producer loop ----
            # Main body always issues full tiles (partial=False). When
            # needs_partial, peel off the last iteration so its
            # populate/TMAs can be runtime-bounded.
            var main_iters = iter_count
            comptime if needs_partial:
                if main_iters > 0:
                    main_iters -= 1
            while main_iters != 0:
                main_iters -= 1
                kv_row += UInt32(Self.config.BN)

                comptime if check_mask:
                    if (
                        Self.mask_status(
                            mask, seq_info.prompt_idx, score_row, kv_row
                        )
                        == TileMaskStatus.FULL_MASK
                    ):
                        continue
                paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
                    seq_info.prompt_idx, kv_row
                )
                rope_paged_rows = k_rope_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row)

                # produce k (full sub-tile loops for paged KV)
                k_pipeline.producer_acquire[qk_stage=0]()
                var kn_mbar = k_pipeline.producer_mbar[qk_stage=0]()
                _produce_k_split[partial=False](
                    paged_rows, rope_paged_rows, kv_row, kn_mbar
                )
                k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
                k_pipeline.state.step()
                # produce v (reuses paged_rows)
                pipeline_v.acquire_v()
                var mbarvn = pipeline_v.get_tile[qk_stage=0]()
                _produce_v[partial=False](
                    paged_rows, mbarvn.mbar, _split_v_smem_ptr(mbarvn)
                )
                pipeline_v.commit_step()

            # ---- Peeled last iteration (partial-page bound) ----
            comptime if needs_partial:
                if iter_count > 0:
                    kv_row += UInt32(Self.config.BN)
                    var _skip_last = False
                    comptime if check_mask:
                        if (
                            Self.mask_status(
                                mask, seq_info.prompt_idx, score_row, kv_row
                            )
                            == TileMaskStatus.FULL_MASK
                        ):
                            _skip_last = True
                    if not _skip_last:
                        # Re-populate BOTH LUTs at the new kv_row.
                        paged_rows = kv_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        rope_paged_rows = k_rope_lut.populate[
                            Self.config.BN, base_alignment
                        ](seq_info.prompt_idx, kv_row)
                        var k_nvp_last = _k_num_valid_pages(kv_row)
                        var rope_nvp_last = _rope_num_valid_pages(kv_row)
                        # produce k (partial)
                        k_pipeline.producer_acquire[qk_stage=0]()
                        var kn_mbar_last = k_pipeline.producer_mbar[
                            qk_stage=0
                        ]()
                        _produce_k_split[partial=needs_partial](
                            paged_rows,
                            rope_paged_rows,
                            kv_row,
                            kn_mbar_last,
                            k_nvp_last,
                            rope_nvp_last,
                        )
                        k_scale_idx = (k_scale_idx + 1) % num_k_scale_bufs
                        k_pipeline.state.step()
                        # produce v (partial)
                        pipeline_v.acquire_v()
                        var mbarvn_last = pipeline_v.get_tile[qk_stage=0]()
                        _produce_v[partial=needs_partial](
                            paged_rows,
                            mbarvn_last.mbar,
                            _split_v_smem_ptr(mbarvn_last),
                            k_nvp_last,
                        )
                        pipeline_v.commit_step()


@always_inline
def q_scale_tma[
    dtype: DType, //, BM: Int
](
    ctx: DeviceContext,
    q_scale_tensor: LayoutTensor[dtype, ...],
    out tma: TMATensorTile[dtype, 2, Index(1, BM), Index(1, BM)],
) raises:
    var num_elements = q_scale_tensor.size()
    debug_assert(num_elements % 4 == 0, "num_elements must be divisible by 4")
    var tensor = TileTensor(
        q_scale_tensor.ptr, tt_row_major(Coord(Idx[1], num_elements))
    )

    return create_tensor_tile[
        Index(1, BM),
        swizzle_mode=TensorMapSwizzle.SWIZZLE_NONE,
        __desc_shape=Index(1, BM),
    ](ctx, tensor.to_layout_tensor())


def mla_sm100_prefill_per_token_scale[
    output_dtype: DType,
    q_dtype: DType,
    rope_dtype: DType,
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
    output: TileTensor[
        mut=True, output_dtype, address_space=AddressSpace.GENERIC, ...
    ],
    q_nope: TileTensor[q_dtype, address_space=AddressSpace.GENERIC, ...],
    q_rope: LayoutTensor[
        rope_dtype, _, address_space=AddressSpace.GENERIC, ...
    ],
    q_scale: LayoutTensor[
        scale_dtype, _, address_space=AddressSpace.GENERIC, ...
    ],
    k_nope: KType,
    k_rope: KRopeType,
    v: VType,
    mask_functor: MaskType,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        rope_dtype == KRopeType.dtype
    ), "q_rope and k_rope must have the same dtype"
    comptime assert KType.dtype == VType.dtype

    comptime fa4_config = MLAConfig[
        q_dtype,
        rope_mma_dtype=rope_dtype,
        rope_gmem_dtype=rope_dtype,
        scale_dtype=scale_dtype,
    ](
        num_q_heads=config.num_heads,
        group=group,
        depth=q_depth,
        page_size=KType.page_size,
    )
    comptime assert fa4_config.supported()

    var num_rows_q = q_num_matrix_view_rows(q_nope)

    # Batched O store: half-depth box (depth_splits=2, shared with the 1Q
    # combine) for SWIZZLE_NONE group==1; 0 (per-block) otherwise.
    comptime store_blocks_per_op = o_store_tma_blocks_per_op[
        output_dtype,
        fa4_config.output_swizzle_mode,
        fa4_config.fa4_config.ov_depth,
        1,
        depth_splits=2,
    ]()
    comptime RaggedStoreType = RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // fa4_config.fa4_config.num_qo,
        BN=fa4_config.fa4_config.ov_depth,
        middle_dim=fa4_config.num_q_heads,
        tma_blocks_per_op=store_blocks_per_op,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q
    )

    q_nope_tma_op = q_tma[
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.q_tile_rows(),
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
        fa4_config.rope_gmem_swizzle_mode,
        BM=fa4_config.q_tile_rows(),
        depth=fa4_config.rope_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
    ](
        ctx,
        q_rope.ptr,
        num_rows_q,
    )

    # Per-Q-tile box (128 in both 1Q/2Q modes); 2Q issues two TMAs.
    q_scale_tma_op = q_scale_tma[BM=fa4_config.q_tile_rows()](ctx, q_scale)

    k_nope_tma_op = k_nope.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KType.page_size),
        depth=fa4_config.nope_depth,
    ](ctx)

    k_rope_tma_op = k_rope.create_tma_tile[
        fa4_config.rope_gmem_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KRopeType.page_size),
        depth=cache_depth,
        BK=fa4_config.rope_depth,
    ](ctx)

    k_scale_tma_op = k_nope.create_scale_tma_tile[
        kv_sub_tile_rows(fa4_config.BN, KType.page_size)
    ](ctx)

    v_tma_op = v.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KType.page_size),
        depth=fa4_config.nope_depth,
    ](ctx)

    comptime ValidLengthType = NonNullPointer[DType.uint32]
    comptime SinkType = NullPointer[output_dtype]
    comptime KVRowOffsetsType = NullPointer[DType.uint32]
    comptime PartitionType = NoPartition[DType.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    # Launch the kernel built from `cfg` (the 2Q `fa4_config` or its 1Q
    # variant). All TMA-op types fold to identical values for both
    # configs (Q nope/rope TMAs, q_scale box, and ragged store use
    # `BM // num_qo` = 128 in both modes; K/V/rope/k_scale TMA shapes
    # are BM-independent), so they are passed through unchanged.
    @parameter
    @always_inline
    def _launch[cfg: MLAConfig]() raises:
        comptime assert cfg.supported(), cfg.fa4_config.description()
        comptime SchedulerType = TransientScheduler[
            UInt32(cfg.BM),
            UInt32(cfg.num_q_heads),
            flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
        ]

        comptime SM100MLAType = SM100MLA[
            KType,
            KRopeType,
            output_dtype,
            MaskType,
            SchedulerType,
            cfg,
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
            max_prompt_len.as_uint32(), UInt32(cfg.BM)
        )
        var num_blocks: UInt32 = (
            max_num_prompt_tiles * PartitionType().num_partitions()
        )

        comptime num_threads = cfg.num_threads
        # When the launched (2Q) kernel may dispatch to the 1Q body at
        # runtime, it builds the 1Q `SM100AttentionSMem` over the same
        # dynamic-smem region, so reserve the max of both footprints.
        comptime smem_use = cfg.launch_smem_used()

        ctx.enqueue_function[kernel](
            q_nope_tma_op,
            q_rope_tma_op,
            k_nope_tma_op,
            k_rope_tma_op,
            v_tma_op,
            q_scale_tma_op,
            k_scale_tma_op,
            ragged_tma_store,
            k_nope,
            k_rope,
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

    # --- 1Q / 2Q dispatch (see the generic MLA dispatch for details) ---
    comptime cfg_1q = fa4_config.with_num_qo(1)
    comptime can_use_1q: Bool = (
        cfg_1q.supported()
        and cfg_1q.fa4_config.supported()
        and mask_functor.nonfull_sets[cfg_1q.BM, cfg_1q.BN]()[0]
        != TileMaskStatus.UNKNOWN_MASK
    )
    comptime if can_use_1q:
        if fa4_config.prefer_1q(
            max_prompt_len.as_uint32(),
            UInt32(PartitionType().num_partitions()),
            UInt32(batch_size),
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
        ):
            _launch[cfg_1q]()
        else:
            _launch[fa4_config]()
    else:
        _launch[fa4_config]()
