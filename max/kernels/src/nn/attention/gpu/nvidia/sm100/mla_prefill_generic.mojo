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
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    expect_bytes_pred,
    KConsumerPipeline,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    PagedRowIndices,
    SharedMemPointer,
    StagedPipeline,
    VConsumerPipeline,
    VProducerPipeline,
)
from nn.attention.gpu.mha import q_num_matrix_view_rows
from nn.attention.gpu.nvidia.sm90.attention import (
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
from layout.tma_async import RaggedTMA3DTile, SharedMemBarrier
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from std.gpu.memory import AddressSpace
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier, thread_idx, warp_id
from nn.attention.mha_utils import (
    _is_decoding,
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
)
from std.gpu.compute.arch.tcgen05 import *
from linalg.arch.sm100.mma import smem_descriptor
from std.utils.static_tuple import StaticTuple
from kv_cache.types import padded_depth

from nn.attention.gpu.nvidia.sm100.mla_prefill_utils import (
    MLAConfig,
    MLAKVLayouts,
    MLAPositionSummary,
    SM100MLA,
    split_smem,
)
from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.nvidia.sm100.smem import SM100AttentionSMem


@fieldwise_init
struct WarpRole(Equatable, TrivialRegisterPassable):
    var _role: Int32
    comptime Softmax0 = Self(0)
    comptime Softmax1 = Self(1)
    comptime Correction = Self(2)
    comptime MMA = Self(3)
    comptime Load = Self(4)
    comptime Empty = Self(5)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self == Self(Int32(other))


def warp_idx_to_role(warp_idx: UInt32) -> WarpRole:
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


__extension SM100MLA:
    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_nope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
    @__name(
        t"sm100_mla_prefill_generic_{Self.qkv_dtype}_{Self.output_dtype}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
        mangle=True,
    )
    def mla_prefill_kernel_generic(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.BK0,
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
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_dtype,
            Self.config.output_swizzle_mode,
            BM=Self.config.fa4_config.BM // 2,
            BN=Self.config.fa4_config.ov_depth,
            group=config.fa4_config.group if config.fa4_config.fuse_gqa else 1,
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
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), (
            "depth = "
            + String(Self.config.qk_depth)
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
        var rope_smem = rebind[SharedMemPointer[Scalar[KRopeType.dtype]]](
            attn_smem.rope_smem_base()
        )
        var ptr_tmem_addr = attn_smem.tmem_addr_ptr()

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 96
        comptime num_reg_other = 48
        comptime num_reg_empty = 24

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp_id[broadcast=True]())
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
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_nope_tma_op.prefetch_descriptor()
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
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

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
            ](
                attn_smem,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_dtype](),
                max_seq_len.as_uint32(),
                ragged_tma_store,
                NullPointer[Self.output_dtype](),
            )

        elif role == WarpRole.Correction:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

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
            fa4_correction[
                Self.config.fa4_config,
                Self.page_size,
            ](
                attn_smem,
                pos.score_row,
                pos.num_keys,
                mask,
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
                q_tma_op,
                k_nope_tma_op,
                k_rope_tma_op,
                v_tma_op,
                kv_lut,
                k_rope_lut,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
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
                k_smem,
                v_smem,
                rope_smem,
            )
        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    def load[
        KRopeType: MHAOperand
    ](
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.BK0,  # padded depth -> 192
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
            KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, KRopeType.page_size),
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
            BK=Self.nope_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[KRopeType.dtype]],
    ):
        comptime KVPipeType = MLAKVLayouts[
            Self.KVLUTType.dtype,
            KRopeType.dtype,
            DType.invalid,
            Self.config,
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        comptime SMemTensorLT[elems: Int] = TileTensor[
            Self.KVLUTType.dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime q_elems = type_of(q_tma_op).tile_shape[0] * type_of(
            q_tma_op
        ).tile_shape[1] * type_of(q_tma_op).tile_shape[2]
        comptime QType = SMemTensorLT[q_elems]

        var k_rope_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)
        var kv_head_idx: UInt32 = seq_info.head_idx

        comptime q_elements = (Self.config.BM // 2) * Self.config.BK0
        comptime q_bytes = size_of[Self.qkv_dtype]() * q_elements

        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )
        var q_head_idx: UInt32 = seq_info.head_idx
        e = elect()

        # Sub-tile paging: when page_size < BN, each BN-row load is split
        # into num_kv_pages sub-tile loads of kv_sub_BN rows each.
        comptime kv_sub_BN = kv_sub_tile_rows(Self.config.BN, Self.page_size)
        comptime num_kv_pages = kv_num_sub_tiles(Self.config.BN, Self.page_size)
        comptime rope_sub_BN = kv_sub_tile_rows(
            Self.config.BN, KRopeType.page_size
        )
        comptime num_rope_pages = kv_num_sub_tiles(
            Self.config.BN, KRopeType.page_size
        )
        comptime PagedRows = PagedRowIndices[Self.config.BN, Self.page_size]
        comptime RopePagedRows = PagedRowIndices[
            Self.config.BN, KRopeType.page_size
        ]

        # Alignment of `kv_row` produced by mask-driven iteration.
        comptime base_alignment: Int = Self.MaskType.start_column_alignment[
            Self.BM, Self.BN, Self.page_size
        ]()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        var paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
            seq_info.prompt_idx, kv_row
        )
        var rope_paged_rows = k_rope_lut.populate[
            Self.config.BN, base_alignment
        ](seq_info.prompt_idx, kv_row)
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        # Partial-page handling: when page_size < BN, runtime-bound the
        # K_nope/V/K_rope sub-tile loops via the `needs_partial=True`
        # overloads (mirrors FA4 `load_warp.mojo`). K_nope/V use
        # `Self.page_size`; K_rope uses `KRopeType.page_size`. Compute
        # the flags independently — the iter_count peel below is gated
        # by either.
        comptime needs_partial_kv = (
            Self.page_size > 0 and Self.page_size < Self.config.BN
        )
        comptime needs_partial_rope = (
            KRopeType.page_size > 0 and KRopeType.page_size < Self.config.BN
        )
        comptime needs_partial = needs_partial_kv or needs_partial_rope

        # Per-sub-page byte sizes for partial expect_bytes_pred.
        comptime k_nope_bytes_pp = (
            Self.nope_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )
        comptime k_rope_bytes_pp = (
            Self.rope_depth * rope_sub_BN * size_of[KRopeType.dtype]()
        )
        comptime v_bytes_pp = (
            Self.nope_depth * kv_sub_BN * size_of[Self.qkv_dtype]()
        )

        @parameter
        @always_inline
        def _k_num_valid_pages(current_kv_row: UInt32) -> UInt32:
            """Valid K_nope/V sub-tile pages at `current_kv_row`."""
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

        # ---- Mode-shared sub-tile constants & closures ----
        # The K_rope sub-tile shape and V tile shape are identical in
        # fused-KV and split-KV modes (only the smem base pointer and
        # pipeline machinery differ). Hoist the constants and the unified
        # `_produce_k_rope` / `_produce_v` closures so both modes share
        # them — mirrors `mla_prefill_blockscale.mojo`'s pattern.
        comptime k_rope_sub_elems = Self.rope_depth * rope_sub_BN
        comptime KRopeSubType = TileTensor[
            KRopeType.dtype,
            type_of(tt_row_major[k_rope_sub_elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        # Full-tile byte counts (no partial bound applies).
        comptime k_nope_full_bytes = (
            Self.nope_depth * Self.config.BN * size_of[Self.qkv_dtype]()
        )
        comptime k_rope_full_bytes = (
            Self.rope_depth * Self.config.BN * size_of[KRopeType.dtype]()
        )
        # V matches K_nope in the qkv dtype; reused for both modes.
        comptime kv_data_full_bytes = k_nope_full_bytes

        @parameter
        @always_inline
        def _produce_k_rope[
            partial: Bool,
        ](
            rope_pages: type_of(rope_paged_rows),
            kv_row_base: UInt32,
            smem_base_ptr: SharedMemPointer[Scalar[KRopeType.dtype]],
            mbar: SharedMemPointer[SharedMemBarrier],
            rope_nvp: UInt32,
        ):
            """K_rope sub-tile TMA into smem starting at `smem_base_ptr`,
            signaling completion on `mbar`.

            Caller is responsible for the `expect_bytes_pred` covering
            the K barrier — K_rope shares the K barrier with K_nope (and
            Q on the prologue) in MLA generic, so the byte total is
            accounted at the call site (`_produce_k_fused` /
            `_produce_k_split`) alongside K_nope and Q.

            `partial=True` early-returns when `_p == rope_nvp`, mirroring
            `PagedRowIndices.tma_copy_k[needs_partial=True]`.
            `partial=False` collapses to the existing fully-unrolled body
            — codegen-identical pre-fix.
            """
            comptime for _p in range(num_rope_pages):
                comptime if partial:
                    if UInt32(_p) == rope_nvp:
                        return
                # Belt-and-suspenders: post-fix this should be
                # unreachable on every config. Kept as a permanent
                # red-test for the partial bound.
                debug_assert(
                    kv_row_base + UInt32(_p * rope_sub_BN) < num_keys,
                    (
                        "MLA K_rope sub-tile TMA OOB after partial"
                        " bound: kv_row_base="
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
            """V tile production at `mbar`/`smem_ptr`.

            Both modes pass the V destination smem pointer directly so
            this closure doesn't need to know about pipeline machinery
            (kv_pipeline vs pipeline_v). `partial=True` runtime-bounds
            the sub-tile loop and accounts only the bytes actually
            delivered.
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

        comptime if Self.config.fa4_config.use_fused_kv:
            # ---- Fused KV mode ----
            # Single StagedPipeline with alternating K_nope and V stages.
            # K_rope stored separately in rope_smem, protected by K barriers.
            # Stages: K_nope0, V0, K_nope1, V1, ...
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
            kv_pipeline.state._phase = 1  # producer starts at phase 1

            # Rope buffer index: cycles through ceildiv(num_kv_stages, 2)
            # independently from the fused KV pipeline, since only K stages
            # (every other fused stage) need rope.
            var rope_idx: UInt32 = 0
            comptime num_rope_bufs = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )

            @parameter
            @always_inline
            def _fused_rope_smem_ptr(
                idx: UInt32,
            ) -> SharedMemPointer[Scalar[KRopeType.dtype]]:
                """Return the K_rope smem base for rope buffer slot `idx`.

                Bitcast preserves parallelism with blockscale's pattern
                where the rope buffer's underlying storage may be a
                different dtype than `KRopeType.dtype`. For generic where
                `rope_smem_base` is already `KRopeType`-typed the bitcast
                is a no-op.
                """
                return rope_smem_base.bitcast[
                    Scalar[KRopeType.dtype]
                ]() + idx * UInt32(rope_stage_elems)

            @parameter
            @always_inline
            def _fused_v_smem_ptr() -> (
                SharedMemPointer[Scalar[Self.KVLUTType.dtype]]
            ):
                """V destination smem ptr at the current KV stage."""
                return k_smem_base + kv_pipeline.state.index() * UInt32(
                    kv_stage_elems
                )

            @parameter
            @always_inline
            def _produce_k_fused[
                partial: Bool,
                with_q: Bool = False,
            ](
                paged: type_of(paged_rows),
                rope_paged: type_of(rope_paged_rows),
                kv_row_local: UInt32,
                rope_idx_local: UInt32,
                mbar: type_of(kv_pipeline.producer_mbar()),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q (if `with_q`) + K_nope + K_rope onto `mbar`.

                Mirrors FA4's `_produce_k_stage` pattern: one helper used
                across prologue (with_q=True), main loop (partial=False),
                and peeled-last (partial=needs_partial). The
                `expect_bytes_pred` accumulator branches on `partial`
                comptime; the runtime barrier hint is one PTX issue.
                """
                var qk_bytes: Int32 = Int32(q_bytes) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                    qk_bytes += Int32(k_rope_bytes_pp) * Int32(rope_nvp)
                else:
                    qk_bytes += Int32(k_nope_full_bytes + k_rope_full_bytes)
                expect_bytes_pred(mbar, qk_bytes, e)

                comptime if with_q:
                    q_tma_op.async_copy_elect(
                        QType(q_smem, tt_row_major[q_elems]()),
                        mbar[],
                        q_coord[
                            depth=Self.qk_depth,
                            decoding=False,
                        ](q_gmem_row, q_head_idx),
                        e,
                    )
                paged.tma_copy_k[needs_partial=partial](
                    k_nope_tma_op,
                    k_smem_base
                    + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                    mbar[],
                    kv_head_idx=kv_head_idx,
                    elect=e,
                    k_num_valid_pages=k_nvp,
                )
                _produce_k_rope[partial=partial](
                    rope_paged,
                    kv_row_local,
                    _fused_rope_smem_ptr(rope_idx_local),
                    mbar,
                    rope_nvp,
                )

            # ---- Peeled: K0 + Q0 on same barrier ----
            var k0_mbar = kv_pipeline.producer_mbar()
            var k_nvp_0 = _k_num_valid_pages(kv_row)
            var rope_nvp_0 = _rope_num_valid_pages(kv_row)
            _produce_k_fused[partial=needs_partial, with_q=True](
                paged_rows,
                rope_paged_rows,
                kv_row,
                rope_idx,
                k0_mbar,
                k_nvp_0,
                rope_nvp_0,
            )
            rope_idx = (rope_idx + 1) % num_rope_bufs
            kv_pipeline.state.step()  # step -> stage 1

            # ---- Q1 (separate barrier) ----
            q_gmem_row += UInt32(Self.config.BM // 2)
            var q1_mbar = mbars.q1_wait_mbar()
            expect_bytes_pred(q1_mbar, Int32(q_bytes), e)
            # Elect-predicated in-PTX via `_elect`; no Mojo `if e != 0:`.
            q_tma_op.async_copy_elect(
                QType(q_smem + q_elements, tt_row_major[q_elems]()),
                q1_mbar[0],
                q_coord[
                    depth=Self.qk_depth,
                    decoding=False,
                ](q_gmem_row, q_head_idx),
                e,
            )

            # ---- V0 (reuses paged_rows from K0) ----
            kv_pipeline.producer_acquire()
            var v0_mbar = kv_pipeline.producer_mbar()
            _produce_v[partial=needs_partial](
                paged_rows, v0_mbar, _fused_v_smem_ptr(), k_nvp_0
            )
            kv_pipeline.state.step()

            comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
                0
            ] == TileMaskStatus.UNKNOWN_MASK

            # ---- KV producer loop ----
            # Main body: always full tiles (partial=False). When
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
                        Self.mask_status(mask, score_row, kv_row)
                        == TileMaskStatus.FULL_MASK
                    ):
                        continue
                paged_rows = kv_lut.populate[Self.config.BN, base_alignment](
                    seq_info.prompt_idx, kv_row
                )
                rope_paged_rows = k_rope_lut.populate[
                    Self.config.BN, base_alignment
                ](seq_info.prompt_idx, kv_row)

                # Produce K_nope_n + K_rope_n (full sub-tile loops)
                kv_pipeline.producer_acquire()
                var kn_mbar = kv_pipeline.producer_mbar()
                _produce_k_fused[partial=False](
                    paged_rows, rope_paged_rows, kv_row, rope_idx, kn_mbar
                )
                rope_idx = (rope_idx + 1) % num_rope_bufs
                kv_pipeline.state.step()

                # Produce Vn (reuses paged_rows)
                kv_pipeline.producer_acquire()
                var vn_mbar = kv_pipeline.producer_mbar()
                _produce_v[partial=False](
                    paged_rows, vn_mbar, _fused_v_smem_ptr()
                )
                kv_pipeline.state.step()

            # ---- Peeled last iteration (partial-page bound) ----
            comptime if needs_partial:
                if iter_count > 0:
                    kv_row += UInt32(Self.config.BN)
                    var _skip_last = False
                    comptime if check_mask:
                        if (
                            Self.mask_status(mask, score_row, kv_row)
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
                        # Kn (partial)
                        kv_pipeline.producer_acquire()
                        var kn_mbar_last = kv_pipeline.producer_mbar()
                        _produce_k_fused[partial=needs_partial](
                            paged_rows,
                            rope_paged_rows,
                            kv_row,
                            rope_idx,
                            kn_mbar_last,
                            k_nvp_last,
                            rope_nvp_last,
                        )
                        rope_idx = (rope_idx + 1) % num_rope_bufs
                        kv_pipeline.state.step()
                        # Vn (partial)
                        kv_pipeline.producer_acquire()
                        var vn_mbar_last = kv_pipeline.producer_mbar()
                        _produce_v[partial=needs_partial](
                            paged_rows,
                            vn_mbar_last,
                            _fused_v_smem_ptr(),
                            k_nvp_last,
                        )
                        kv_pipeline.state.step()

        else:
            # ---- Split KV mode (original) ----

            # Separate K and V pipelines
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

            # Get K0 barrier (no wait needed for first iteration)
            var k0_mbar = k_pipeline.producer_mbar[qk_stage=0]()

            @parameter
            @always_inline
            def _split_v_smem_ptr(
                pair: type_of(pipeline_v.get_tile[qk_stage=0]()),
            ) -> SharedMemPointer[Scalar[Self.KVLUTType.dtype]]:
                """V destination smem ptr for split-KV's V pipeline pair.

                Mirrors blockscale's `_split_v_smem_ptr` so the unified
                `_produce_v` closure can emit a partial-aware
                `expect_bytes_pred` itself (rather than relying on
                `pipeline_v.get_v(e)`'s fixed-size auto-expect).
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
                mbar: type_of(k0_mbar),
                k_nvp: UInt32 = UInt32(num_kv_pages),
                rope_nvp: UInt32 = UInt32(num_rope_pages),
            ):
                """Q (if `with_q`) + K_nope + K_rope onto `mbar` (split-KV).

                Includes the `split_smem` decomposition into K_nope and
                K_rope smem regions so the call sites only need to set
                up the barrier and pass paged-row indices.
                """
                var qk_bytes: Int32 = Int32(q_bytes) if with_q else Int32(0)
                comptime if partial:
                    qk_bytes += Int32(k_nope_bytes_pp) * Int32(k_nvp)
                    qk_bytes += Int32(k_rope_bytes_pp) * Int32(rope_nvp)
                else:
                    qk_bytes += Int32(KVPipeType.k_bytes)
                expect_bytes_pred(mbar, qk_bytes, e)

                comptime if with_q:
                    q_tma_op.async_copy_elect(
                        QType(q_smem, tt_row_major[q_elems]()),
                        mbar[],
                        q_coord[
                            depth=Self.qk_depth,
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
                    KRopeType.dtype,
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
                    rebind[SharedMemPointer[Scalar[KRopeType.dtype]]](
                        k_rope_smem_local.ptr
                    ),
                    mbar,
                    rope_nvp,
                )

            # ---- K0 + Q0 (combined barrier) ----
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
            k_pipeline.state.step()

            # ---- Q1 (separate barrier) ----
            var q1_mbar = mbars.q1_wait_mbar()
            expect_bytes_pred(q1_mbar, Int32(q_bytes), e)
            # Q1 — elect-predicated in-PTX via `_elect`.
            q_tma_op.async_copy_elect(
                QType(q_smem + q_elements, tt_row_major[q_elems]()),
                q1_mbar[0],
                q_coord[
                    depth=Self.qk_depth,
                    decoding=False,
                ](
                    q_gmem_row + UInt32(Self.config.BM // 2),
                    q_head_idx,
                ),
                e,
            )

            # ---- V0 (reuses paged_rows from K0) ----
            var mbarv0 = pipeline_v.get_tile[qk_stage=0]()
            _produce_v[partial=needs_partial](
                paged_rows, mbarv0.mbar, _split_v_smem_ptr(mbarv0), k_nvp_0
            )
            pipeline_v.commit_step()
            comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
                0
            ] == TileMaskStatus.UNKNOWN_MASK

            # kv producer loop. Main body: always full tiles
            # (partial=False). When needs_partial, peel off the last
            # iteration so its populate/TMAs can be runtime-bounded.
            var main_iters = iter_count
            comptime if needs_partial:
                if main_iters > 0:
                    main_iters -= 1
            while main_iters != 0:
                main_iters -= 1
                kv_row += UInt32(Self.config.BN)

                comptime if check_mask:
                    if (
                        Self.mask_status(mask, score_row, kv_row)
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
                            Self.mask_status(mask, score_row, kv_row)
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

    @staticmethod
    @always_inline
    def mma(
        tmem_addr: UInt32,
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KRopeType.dtype]],
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

        comptime q0_size = (Self.config.BM // 2) * Self.config.padded_qk_depth
        comptime q0_bytes = UInt32(q0_size * size_of[Self.KVLUTType.dtype]())
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + q0_bytes

        comptime if Self.config.fa4_config.use_fused_kv:
            # ---- Fused KV mode ----
            # Single StagedPipeline alternating K_nope and V.
            # K_rope is in a separate smem region, protected by the same
            # K barrier (load warp puts both on the same mbarrier).
            # Q@K' = Q_nope@K_nope (c_scale=0) + Q_rope@K_rope (c_scale=1).

            comptime kv_stage_bytes = (
                Self.config.fa4_config.padded_ov_depth
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )
            comptime rope_stage_bytes = (
                Self.config.rope_depth
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )

            # K_nope descriptor: k_major for Q@K_nope'
            kv_desc_k = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.fa4_config.padded_ov_depth,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=True,
            ](k_smem_base)
            # V descriptor: mn_major for P@V
            kv_desc_v = smem_descriptor[
                BMN=Self.config.fa4_config.padded_ov_depth,
                BK=Self.config.BN,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=False,
            ](k_smem_base)
            # K_rope descriptor: k_major for Q@K_rope'
            rope_desc = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.rope_depth,
                swizzle_mode=Self.config.rope_mma_swizzle_mode,
                is_k_major=True,
            ](rope_smem_base)

            # Q_rope offset: Q_nope occupies first MMA_M * padded_v_depth
            # elements in Q's column-major tiled layout.
            comptime q_rope_off = UInt32(Self.q_rope_byte_offset)
            q0_rope = q0 + q_rope_off
            q1_rope = q1 + q_rope_off

            comptime KVPipeType = StagedPipeline[
                Self.config.fa4_config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

            # Rope buffer index: cycles independently through
            # ceildiv(num_kv_stages, 2) indices, one per K tile.
            var rope_idx: UInt32 = 0
            comptime num_rope_bufs = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )

            var iter_count: UInt32 = (
                mask.total_iters[Self.BM, Self.BN, Self.page_size](
                    score_row, num_keys
                )
                - 1
            )

            e = elect()

            # ---- Peeled iteration ----
            # Stage 0 = K0 (K_nope0 + K_rope0)
            kv_pipeline.consumer_wait()
            k0 = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            Self.UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
            r0 = rope_desc + UInt32(rope_stage_bytes) * rope_idx
            Self.UMMA0RopeType.mma[stage_idx=0](
                q0_rope, r0, s0_tmem, elect=e, c_scale=1
            )
            pipeline_s0.commit_mma(e)

            # Q1 @ K0 (wait for Q1 first)
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].wait()
            Self.UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)
            Self.UMMA0RopeType.mma[stage_idx=0](
                q1_rope, r0, s1_tmem, elect=e, c_scale=1
            )
            kv_pipeline.consumer_release(e)  # release K0, step → stage 1
            rope_idx = (rope_idx + 1) % num_rope_bufs
            pipeline_s1.commit_mma(e)

            # Stage 1 = V0
            kv_pipeline.consumer_wait()
            var v_prev_idx: UInt32 = kv_pipeline.state.index()
            v0 = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s0[pv_stage].wait(0)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, v0, o0_tmem, elect=e, c_scale=0
                )
            pipeline_o0.commit_mma(e)
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # ---- Main loop ----
            while iter_count != 0:
                iter_count -= 1

                # Advance past held V to get to next K
                kv_pipeline.state.step()

                # Kn (K_nope_n + K_rope_n)
                kv_pipeline.consumer_wait()
                kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                Self.UMMA0Type.mma[stage_idx=0](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
                rn = rope_desc + UInt32(rope_stage_bytes) * rope_idx
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, rn, s0_tmem, elect=e, c_scale=1
                )
                pipeline_s0.commit_mma(e)

                # P1 @ V_{n-1}
                v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
                c_scale = 1
                kv_pipeline.consumer_release_at(
                    v_prev_idx, e
                )  # release V_{n-1}

                # Q1 @ Kn
                Self.UMMA0Type.mma[stage_idx=0](
                    q1, kn, s1_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q1_rope, rn, s1_tmem, elect=e, c_scale=1
                )
                kv_pipeline.consumer_release(e)  # release Kn, step
                rope_idx = (rope_idx + 1) % num_rope_bufs
                pipeline_s1.commit_mma(e)
                phase ^= 1

                # Vn
                kv_pipeline.consumer_wait()
                v_prev_idx = kv_pipeline.state.index()
                vn = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vn, o0_tmem, elect=e, c_scale=1
                    )
                pipeline_o0.commit_mma(e)

            # ---- Epilogue ----
            v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)
            kv_pipeline.consumer_release_at(v_prev_idx, e)  # release V_last

        else:
            # ---- Split KV mode (original) ----

            # Separate K and V consumer pipelines
            comptime KConType = KConsumerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            comptime VConType = VConsumerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            var pipeline_k: KConType = {mbars.get_k_mbars(), k_smem_base}
            var pipeline_v: VConType = {mbars.get_v_mbars(), v_smem_base}

            # We peel the first iteration, as we want to wait on q1
            var iter_count: UInt32 = (
                mask.total_iters[Self.BM, Self.BN, Self.page_size](
                    score_row, num_keys
                )
                - 1
            )
            comptime if Self.fused_umma0:
                # Q_0 @ K_0'
                pipeline_k.wait_k()
                k0 = pipeline_k.get_k()
                e = elect()
                Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
                pipeline_s0.commit_mma(e)

                # Q_1 @ K_0'
                mbars.q1_wait_mbar()[0].wait()  # wait on Q1
                Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
                pipeline_s1.commit_mma(e)

                pipeline_k.release_k(e)  # release K0

                # Wait V0
                pipeline_v.wait_v()
                vlatest = pipeline_v.get_v()
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(0)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0
                    )
                pipeline_o0.commit_mma(e)
                var phase: UInt32 = 0

                var c_scale: UInt32 = 0

                while iter_count != 0:
                    iter_count -= 1

                    # Q_0 @ K_n'
                    kn = pipeline_k.get_k()
                    pipeline_k.wait_k()
                    Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
                    pipeline_s0.commit_mma(e)

                    # O_1 + P_1 @ V_{n-1}
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s1[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                        )
                    pipeline_o1.commit_mma(e)
                    c_scale = 1
                    pipeline_v.release_v(e)

                    # Q_1 @ K_n'
                    Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
                    pipeline_k.release_k(e)
                    pipeline_s1.commit_mma(e)
                    phase ^= 1

                    # O_0 + P_0 @ V_n
                    vlatest = pipeline_v.get_v()
                    pipeline_v.wait_v()
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s0[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                        )
                    pipeline_o0.commit_mma(e)

                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
            else:
                # ---- Split KV mode with separate nope/rope UMMAs ----
                # K_nope and K_rope have different dtypes (e.g. FP8 + BF16),
                # so Q@K' = Q_nope@K_nope (c_scale=0) + Q_rope@K_rope (c_scale=1).

                # ---- Q descriptor setup ----
                # Q smem uses interleaved layout:
                #   [Q0_nope][Q0_rope][Q1_nope][Q1_rope]
                # Each Q half is q_nope_bytes + q_rope_bytes.
                comptime q_nope_bytes = (
                    (Self.config.BM // 2)
                    * Self.config.fa4_config.padded_ov_depth
                    * Self.qkv_dt_size
                )
                comptime q_rope_bytes = (
                    (Self.config.BM // 2)
                    * Self.rope_depth
                    * Self.config.rope_mma_dtype_size
                )
                comptime q_half_bytes = UInt32(q_nope_bytes + q_rope_bytes)

                q0_nope = Self.descriptor_q(q_smem)
                q0_rope = Self.descriptor_q_rope(
                    (q_smem + q_nope_bytes // Self.qkv_dt_size).bitcast[
                        Scalar[Self.rope_mma_dtype]
                    ]()
                )
                q1_nope = q0_nope + q_half_bytes
                q1_rope = q0_rope + q_half_bytes

                # ---- K descriptor setup ----
                # pipeline_k is used for wait/release only; get_k() is NOT
                # used because its descriptor has BK=padded_qk_depth with a
                # single swizzle, which is wrong for mixed dtypes.
                comptime nope_stage_bytes = (
                    Self.config.fa4_config.padded_ov_depth
                    * Self.config.BN
                    * Self.qkv_dt_size
                )
                comptime k_stage_stride = UInt32(KConType.full_kv_bytes)

                kv_desc_k_nope = smem_descriptor[
                    BMN=Self.config.BN,
                    BK=Self.nope_depth,
                    swizzle_mode=Self.config.qkv_swizzle_mode,
                    is_k_major=True,
                ](k_smem_base)

                kv_desc_k_rope = smem_descriptor[
                    BMN=Self.config.BN,
                    BK=Self.rope_depth,
                    swizzle_mode=Self.config.rope_mma_swizzle_mode,
                    is_k_major=True,
                ](
                    (
                        k_smem_base + nope_stage_bytes // Self.qkv_dt_size
                    ).bitcast[Scalar[Self.rope_mma_dtype]]()
                )

                # ---- Peeled iteration ----
                # Q_0 @ K_0'
                pipeline_k.wait_k()
                var k_idx = pipeline_k.pipeline.state.index()
                k0_nope = kv_desc_k_nope + k_stage_stride * k_idx
                k0_rope = kv_desc_k_rope + k_stage_stride * k_idx
                e = elect()
                Self.UMMA0Type.mma[stage_idx=0](
                    q0_nope, k0_nope, s0_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, k0_rope, s0_tmem, elect=e, c_scale=1
                )
                pipeline_s0.commit_mma(e)

                # Q_1 @ K_0'
                mbars.q1_wait_mbar()[0].wait()
                Self.UMMA0Type.mma[stage_idx=0](
                    q1_nope, k0_nope, s1_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q1_rope, k0_rope, s1_tmem, elect=e, c_scale=1
                )
                pipeline_s1.commit_mma(e)

                pipeline_k.release_k(e)

                # Wait V0
                pipeline_v.wait_v()
                vlatest = pipeline_v.get_v()
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(0)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0
                    )
                pipeline_o0.commit_mma(e)
                var phase: UInt32 = 0
                var c_scale: UInt32 = 0

                # ---- Main loop ----
                while iter_count != 0:
                    iter_count -= 1

                    # Q_0 @ K_n'
                    kn_nope = (
                        kv_desc_k_nope
                        + k_stage_stride * pipeline_k.pipeline.state.index()
                    )
                    kn_rope = (
                        kv_desc_k_rope
                        + k_stage_stride * pipeline_k.pipeline.state.index()
                    )
                    pipeline_k.wait_k()
                    Self.UMMA0Type.mma[stage_idx=0](
                        q0_nope, kn_nope, s0_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q0_rope, kn_rope, s0_tmem, elect=e, c_scale=1
                    )
                    pipeline_s0.commit_mma(e)

                    # O_1 + P_1 @ V_{n-1}
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s1[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                        )
                    pipeline_o1.commit_mma(e)
                    c_scale = 1
                    pipeline_v.release_v(e)

                    # Q_1 @ K_n'
                    Self.UMMA0Type.mma[stage_idx=0](
                        q1_nope, kn_nope, s1_tmem, elect=e, c_scale=0
                    )
                    Self.UMMA0RopeType.mma[stage_idx=0](
                        q1_rope, kn_rope, s1_tmem, elect=e, c_scale=1
                    )
                    pipeline_k.release_k(e)
                    pipeline_s1.commit_mma(e)
                    phase ^= 1

                    # O_0 + P_0 @ V_n
                    vlatest = pipeline_v.get_v()
                    pipeline_v.wait_v()
                    comptime for pv_stage in range(Self.config.num_pv_stages):
                        _ = consumer_s0[pv_stage].wait(phase)
                        Self.UMMA1Type.mma[stage_idx=pv_stage](
                            s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                        )
                    pipeline_o0.commit_mma(e)

                # ---- Epilogue ----
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)


@always_inline
def mla_sm100_prefill_generic[
    output_dtype: DType,
    q_type: DType,
    KVType: MHAOperand,
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
    output: TileTensor[output_dtype, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime fa4_config = MLAConfig[
        q_type, rope_gmem_dtype=KRopeType.dtype, rope_mma_dtype=KRopeType.dtype
    ](
        num_q_heads=config.num_heads,
        group=group,
        depth=q_depth,
        page_size=KVType.page_size,
    )

    var num_rows_q = q_num_matrix_view_rows(q)

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // 2,
        BN=fa4_config.fa4_config.ov_depth,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q, middle_dim=fa4_config.num_q_heads
    )

    q_tma_op = q_tma[
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.qk_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
    ](
        ctx,
        q.ptr,
        num_rows_q,
    )

    # [batch_size * num_keys, num_heads, kv_depth]
    k_nope_tma_op = k.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        depth=fa4_config.nope_depth,
    ](ctx)

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    k_rope_tma_op = k_rope.create_tma_tile[
        fa4_config.rope_gmem_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KRopeType.page_size),
        depth=cache_depth,
        BK=fa4_config.rope_depth,
    ](ctx)

    # [batch_size * num_keys, num_heads, kv_depth]
    v_tma_op = v.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        depth=fa4_config.nope_depth,
    ](ctx)

    _mla_prefill_sm100_valid_length_dispatch[
        fa4_config=fa4_config,
        cache_depth=cache_depth,
        _ndbuffer_mha_operand=_ndbuffer_mha_operand,
    ](
        ragged_tma_store,
        q_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        v_tma_op,
        k,
        k_rope,
        mask_functor,
        valid_length,
        max_prompt_len,
        scale,
        batch_size,
        ctx,
    )


@always_inline
def _mla_prefill_sm100_valid_length_dispatch[
    KVType: MHAOperand,
    output_dtype: DType,
    q_type: DType,
    MaskType: MHAMask,
    KRopeType: MHAOperand,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    fa4_config: MLAConfig,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
](
    ragged_tma_store: RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // 2,
        BN=fa4_config.fa4_config.ov_depth,
    ],
    q_tma_op: QTMATile[
        q_type,
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.qk_depth,
        group=fa4_config.group,
        decoding=False,
    ],
    k_nope_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        BK=padded_depth[
            KVType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    k_rope_tma_op: KVTMATile[
        KRopeType.dtype,
        fa4_config.rope_gmem_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KRopeType.page_size),
        BK=fa4_config.rope_depth,
    ],
    v_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=kv_sub_tile_rows(fa4_config.BN, KVType.page_size),
        BK=padded_depth[
            KVType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    kv_lut: KVType,
    k_rope_lut: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[
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
    comptime SinkType = NullPointer[output_dtype]
    comptime KVRowOffsetsType = NullPointer[DType.uint32]
    comptime PartitionType = NoPartition[DType.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    comptime SM100MLAType = SM100MLA[
        KVType,
        KRopeType,
        output_dtype,
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

    comptime kernel = SM100MLAType.mla_prefill_kernel_generic

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
    comptime smem_use = fa4_config.smem_used

    ctx.enqueue_function[kernel](
        q_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
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
