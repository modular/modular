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

"""Native FP8 sparse MLA decode kernel for SM100 (B200) -- unified gather.

Combines the OLD kernel's efficient contiguous KV gather
(mla_decode_sparse_kv_fp8.mojo: INT64/SWIZZLE_NONE gather4, 16 gather4
instructions/tile at 2304 B/txn) with the native-FP8 compute core
(tcgen05.mma.kind::f8f6f4, no dtype conversion). The two are decoupled by a
re-swizzle staging warpgroup: KV gathers contiguously into a LINEAR FP8
SMEM buffer, then WG3 (repurposing the old kernel's convert-WG role, minus
the dtype cast) permutes it FP8->FP8 into the SW64 byte layout the FP8
tcgen05 operand descriptor expects, into a SECOND SMEM buffer. QK^T and P*V
still run natively in FP8 reading the SW64 buffer -- only the KV
front-end changed; the compute core is byte-for-byte what
MLA_SM100_Decode_Sparse_QKV_FP8 already validated.

Root cause this fixes (see .claude/agent-memory/mojo-kernel-engineer/
mla-decode-sm100-native-fp8-warpgroup-count.md): the native kernel's own
K/Q operand layout mandates SWIZZLE_64B for tcgen05.mma.kind::f8f6f4, and a
gather4 TMA under SW64 has box_width=64B, so a 576-byte row needs
ceildiv(576,64)=9 column-groups x 16 four-row-chunks = 144 gather4
instructions/tile -- 9x the old kernel's 16 instructions/tile (SWIZZLE_NONE,
single 576-byte box) for identical bytes. That per-tile hardware-instruction
inflation, not pipeline depth or warp count, is what NCU's PC-sampled stall
attribution pinned to the KV-gather producer path.

4 warpgroups (back from 3 -- the re-swizzle stage, like the old kernel's
convert stage, needs its own resident warps; it is a pure FP8->FP8 layout
permute, not a dtype conversion):
  WG0 (warps  0-3):   Softmax  (native_fp8=True)
  WG1 (warps  4-7):   Correction
  WG2 (warps  8-11):  Load (warp 8, INT64/SWIZZLE_NONE gather4) / MMA QK
                      (warp 9) / MMA PV (warp 10) / idx_producer + Store
                      (warp 11) -- unchanged roles from the 3-WG kernel.
  WG3 (warps 12-15):  Re-swizzle: read the contiguously-gathered FP8 linearly,
                      write FP8 (unchanged bytes, no cast) into the SW64
                      layout via a manual swizzled SMEM store (all 128
                      threads cooperate, mirroring convertFP8ToBF16's
                      thread mapping but without the dtype cast or the
                      BF16-overlay aliasing -- this uses a separate,
                      non-overlaid destination buffer).

Packed NVFP4 KV (`KVLUTType.dtype == uint8`, two e2m1 per byte plus one E4M3
scale per 16 elements): the same gather lands half-width rows in the linear
stage, the load warp copies each row's scales into the stage's unused upper
bytes, and WG3 dequantizes `e2m1 * sf` to FP8 straight into the SW64 buffer. Q
and both MMAs stay FP8; the FP8 path compiles unchanged.

SMEM layout (FP8, N = num_kv_stages, M = num_kv_mma_stages; expect N ~= 2 at
the target shape, since the linear + SW64 buffer pair costs the same total
bytes/stage as the old kernel's own KV region). The two KV buffers do NOT
share a depth: `PV_REL(i)` frees the SW64 slot that `CVT` reclaims for tile
i + M, and that WAR edge is the back-edge of the loop that sets the kernel's
steady-state rate, so the rate is bounded by (cycle weight) / M. The gather
ring is not on that loop, so leftover SMEM buys depth on the SW64 side only
-- a symmetric extra stage does not fit:
  Q FP8:          64 x 576 x 1 = 36864 bytes         (SWIZZLE_64B, persistent)
  KV linear-in:   N x 64 x 576 x 1 bytes              (SWIZZLE_NONE, gather dest)
  KV SW64-out:    M x 64 x 576 x 1 bytes              (SWIZZLE_64B, MMA operand)
  P stages:       N x 64 x 64 x 1 bytes               (SWIZZLE_64B, separate region)
  max/li:         128 x 4 x 3 = 1536 bytes
  barriers:       (6N+2M+11) fixed + output + idx_bars (2N) barriers
  ptr_tmem_addr:  4 bytes
  idx_smem:       N x 64 x 4 bytes (x2 with NVFP4: the tile's scale rows follow)
"""

from std.collections import OptionalReg
from std.math import ceildiv, clamp
from std.math.constants import log2e
from std.memory import bitcast
from std.sys import get_defined_int, size_of
from std.sys.intrinsics import assume
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx,
    block_idx,
    warp_id,
    lane_id,
    WARP_SIZE,
)
from max.gpu.sync import async_copy_arrive, barrier
from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.primitives.grid_controls import launch_dependent_grids
from max.gpu.intrinsics import (
    byte_permute,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)
from max.gpu.memory import (
    CacheEviction,
    async_copy,
    external_memory,
    fence_async_view_proxy,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.swizzle import make_swizzle
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
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import NullPointer, OptionalPointer
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.nvfp4_kv_dequant import (
    NVFP4_MLA_ROPE_DIM,
    mla_kv_is_nvfp4,
    nvfp4_dequant_group,
    nvfp4_mla_row_bytes,
    nvfp4_mla_scale_row_bytes,
    sparse_kv_gather_row_words,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    expect_bytes_pred,
    SharedMemPointer,
    MBarType,
    ConsumerPipeline,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    SparseRowLayout,
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
)


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
    Engine: TensorEngine = DefaultEngine[element_width=1],
](TrivialRegisterPassable):
    """Sparse MLA decode with native FP8 WGMMA for SM100 (B200), unified gather.

    Gathers KV contiguously (INT64/SWIZZLE_NONE, like
    MLA_SM100_Decode_Sparse_KV_FP8) then re-swizzles FP8->FP8 (no dtype
    conversion) into the SW64 layout the native tcgen05.mma.kind::f8f6f4
    operand expects. QK^T and PV MMA are unchanged from the single-buffer
    native-FP8 kernel this supersedes.
    """

    # Presence of the optional operands is carried by their pointer types.
    comptime has_attn_sink = not Self.AttnSinkPtrType.is_null
    comptime has_variable_topk = not Self.TopkLengthsPtrType.is_null

    comptime kv_type = Self.KVLUTType.dtype
    comptime fp8_type = DType.float8_e4m3fn

    # Packed NVFP4 cache (sm_100): the gather lands half-width e2m1 rows plus
    # their E4M3 scales in the linear stage, and the re-swizzle warpgroup
    # expands them into the FP8 tile the MMA reads. The FP8 path is unchanged.
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
    # The 32 load-warp lanes that stage a tile's scales arrive on the gather
    # barrier next to the TMA's own arrival.
    comptime gather_num_producer = 1 + (32 if Self.is_nvfp4_kv else 0)
    # An NVFP4 cache pages its scales apart from its values, so the indexer
    # writes each sparse index row as `[value rows | scale rows]`, `topk`
    # slots each. A stage of `idx_smem` mirrors it: the tile's value rows,
    # then its scale rows.
    comptime sparse_row_parts = 2 if Self.is_nvfp4_kv else 1
    comptime idx_stage_slots = Self.config.BN_QK * Self.sparse_row_parts
    comptime AccumType = get_accum_type[Self.q_type]()
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime PStageElems = Self.BlockElems
    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )

    # Contiguous gather4 TMA descriptor: INT64, SWIZZLE_NONE, single box
    # covering the full 576-byte row (matches
    # MLA_SM100_Decode_Sparse_KV_FP8 exactly -- 16 gather4 instructions/tile
    # instead of the SW64-fragmented 144/tile).
    comptime kv_gather4_tile_width = sparse_kv_gather_row_words[
        Self.KVLUTType, Self.config.input_q_depth
    ]()
    comptime kv_gather4_box_w = _gather4_box_width[
        DType.int64,
        Self.kv_gather4_tile_width,
        TensorMapSwizzle.SWIZZLE_NONE,
    ]()

    # >0: the elected lane loads the row indices of this many 4-row chunks
    # with back-to-back ld.shared.v4 before issuing their gather4s, so the
    # SMEM latency is paid once per batch instead of once per gather4.
    comptime gather4_idx_batch = get_defined_int[
        "MLA_SPARSE_GATHER4_IDX_BATCH", 8
    ]()
    # Padding that 16-byte aligns idx_smem for ld.shared.v4; the dispatch adds
    # this to the SMEM it reserves. `config.smem_used` ends at the barriers,
    # with 4 bytes for the TMEM pointer that this kernel places after the
    # 2 * num_stages idx barriers instead; idx_smem follows the pointer. The
    # exact value matters: an NVFP4 cache's double-width idx_smem leaves the
    # 3-deep MMA ring no slack for the 12-byte worst case.
    comptime _idx_smem_offset = (
        Self.config.smem_used + 2 * Self.num_stages * Self.config.mbar_size
    )
    comptime idx_smem_align_pad = (
        (16 - Self._idx_smem_offset % 16) % 16 if Self.gather4_idx_batch
        > 0 else 0
    )

    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    comptime num_stages = Self.config.num_kv_stages
    # Depth of the SW64 ring only. `PV_REL(i)` frees the slot that
    # `CVT_SW64_ACQ(i + num_mma_stages)` reclaims, and that WAR edge is the
    # back-edge of the loop that sets this kernel's steady-state rate, so the
    # bound is (cycle weight) / this depth. The linear gather ring is not on
    # that loop and stays at `num_stages`.
    comptime num_mma_stages = Self.config.num_kv_mma_stages

    # The ground-truth SW64 destination layout the MMA operand descriptor
    # already reads today -- the re-swizzle WG's job is to reproduce this
    # exact byte permutation from the contiguously-gathered linear source.
    comptime sw_fp8 = make_swizzle[
        Self.fp8_type, TensorMapSwizzle.SWIZZLE_64B
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
    def zero_input_tail(
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
    ):
        """Zero the SMEM columns past the input row, once per CTA.

        Once is enough without qualification: `p_smem` is its own region here,
        not an alias of the tail, so nothing rewrites these columns.
        """
        comptime tail_block0 = Self.config.input_q_depth // Self.config.BN_QK
        comptime tail_elems = (Self.NumQKBlocks - tail_block0) * Self.BlockElems
        comptime tail_off = tail_block0 * Self.BlockElems
        comptime zero = Scalar[Self.fp8_type](0)

        var tid = thread_idx.x
        for i in range(tid, tail_elems, Self.config.num_threads):
            q_smem[tail_off + i] = zero
        comptime for stage in range(Self.num_mma_stages):
            var base = kv_smem.unsafe_offset(
                stage * Self.KVStageElems
            ).unsafe_offset(tail_off)
            for i in range(tid, tail_elems, Self.config.num_threads):
                base[i] = zero

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
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
        ],
        # Single K gather4 TMA covering the row as stored: INT64,
        # SWIZZLE_NONE, tile_width = input_q_depth / 8 INT64 (matches
        # MLA_SM100_Decode_Sparse_KV_FP8's efficient contiguous gather).
        k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
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
        ],
        d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
        indices_stride_dev: Int32,
        # Logical sparse indices (same layout as `d_indices`, -1 padding) for
        # position-based causal masking. See mla_decode_utils.mojo.
        logical_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
        topk_lengths: Self.TopkLengthsPtrType,
        scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
        attn_sink_ptr: Self.AttnSinkPtrType,
        extra_k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
        extra_topk_lengths: Self.TopkLengthsPtrType,
        extra_indices_stride_dev: Int32,
        extra_scales_ptr: OptionalReg[MutPointer[Float32, MutAnyOrigin]],
        scalar_args: TileTensor[
            .int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
            Engine=Self.Engine,
        ],
        # Positions between the token each sparse row was computed for and
        # the token that reads it.
        selection_lag_dev: Int32,
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
            Self.config.BN_QK == 64
            and Self.config.input_q_depth % 64 == 0
            and Self.nvfp4_sf_offset % 16 == 0
            and Self.nvfp4_sf_offset
            + Self.config.BN_QK * Self.nvfp4_sf_row_bytes
            <= Self.KVStageElems
        ), (
            "NVFP4 staging assumes 64-row tiles whose packed rows and scales"
            " fit a stage."
        )

        # The launcher sizes grid.x to the head tiles. Stating it lets the
        # compiler fold `block_idx.x * BM` instead of keeping a 64-bit copy
        # live from kernel entry to the output store, which spilled.
        assume(
            Int(block_idx.x) < ceildiv(Self.config.num_q_heads, Self.config.BM)
        )

        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 64
        comptime num_reg_reswizzle = 72

        var indices_stride = Int(indices_stride_dev)
        var topk_stride = indices_stride // Self.sparse_row_parts
        var extra_indices_stride = Int(extra_indices_stride_dev)
        var batch_size = Int(scalar_args[0])
        var selection_lag = Int(selection_lag_dev)
        var q_max_seq_len = Int(scalar_args[1])
        var num_partitions = Int(mla_decode_pack.num_partitions)
        var mask = mla_decode_pack.mask
        var valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr

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
                ImmPointer[
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
        )

        var topk: Int
        comptime if Self.has_variable_topk:
            topk = Int(topk_lengths.value()[Int(offset_position.batch_idx)])
        else:
            topk = topk_stride
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
        topk = offset_position.num_keys - extra_topk

        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)

        @inline(.always)
        def _pdl_early_exit_all_q() {imm}:
            comptime if Self.fold_shared_index:
                comptime for q_local in range(Self.q_len_fold):
                    Self.Common_MLA_Op.pdl_early_exit[fold_q=True](
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        batch_size,
                        lse_accum_split_ptr,
                        seq_idx_fold=UInt32(q_local),
                    )
            else:
                Self.Common_MLA_Op.pdl_early_exit(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    batch_size,
                    lse_accum_split_ptr,
                )

        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                _pdl_early_exit_all_q()
                return

        comptime if Self.ragged:
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    _pdl_early_exit_all_q()
                return

        var q_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space=.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        comptime kv_total_stages = Self.config.num_kv_stages

        # Linear (SWIZZLE_NONE) gather destination -- the re-swizzle WG's
        # source. Same per-stage size as the SW64 destination below.
        var kv_smem_linear = q_smem.unsafe_offset(
            Self.BlockElems * Self.NumQKBlocks
        )
        # SW64 destination the native FP8 MMA reads -- byte-identical role
        # to the single-buffer kernel's `kv_smem`, one ring deeper.
        var kv_smem = kv_smem_linear.unsafe_offset(
            Self.KVStageElems * kv_total_stages
        )
        var p_smem = kv_smem.unsafe_offset(
            Self.KVStageElems * Self.num_mma_stages
        )

        # Output SMEM reuses the SW64 KV region (same as the single-buffer
        # native FP8 kernel and the BF16 kernel).
        var out_smem = kv_smem.unsafe_bitcast[Scalar[Self.output_type]]()

        var max_smem = (
            p_smem.unsafe_offset(Self.PStageElems * Self.num_stages)
        ).unsafe_bitcast[Scalar[Self.AccumType]]()
        var li_smem = max_smem.unsafe_offset(2 * WARPGROUP_SIZE)

        var mbar_base: MBarType = (
            (li_smem.unsafe_offset(WARPGROUP_SIZE))
            .unsafe_bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        var mbar_kv_gather_base: MBarType = mbar_base.unsafe_offset(1)
        # Load (TMA, 1 leader thread) -> re-swizzle WG (128 threads, all
        # independently release).
        var kv_pipeline_gather = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_kv_gather_base)
        var mbar_kv_mma_base: MBarType = mbar_kv_gather_base.unsafe_offset(
            kv_pipeline_gather.num_mbars()
        )
        # Re-swizzle WG (128 threads, all independently commit) -> mmaQK +
        # mmaPV (2 consumers) -- same consumer shape as the single-buffer
        # kernel's own KV pipeline, just producer side is now a warpgroup
        # instead of a single TMA leader thread.
        var kv_pipeline_mma = KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
        ](mbar_kv_mma_base)
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
        ](mbar_base7)
        var mbar_base8: MBarType = mbar_base7.unsafe_offset(
            out_pipeline.num_mbars()
        )

        # idx_bars: warp 11 (producer, 32 threads) -> warp 8 (consumer, 32
        # threads). Depth matches Self.num_stages (= config.num_kv_stages),
        # not a fixed 2 -- see the KB entry cited in the module docstring.
        var idx_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages, num_producer=32, num_consumer=32
        ](mbar_base8)
        var mbar_base9: MBarType = idx_bars.end()

        var ptr_tmem_addr = (mbar_base9).unsafe_bitcast[UInt32]()
        var idx_smem_base = ptr_tmem_addr.unsafe_offset(1).unsafe_bitcast[
            Int32
        ]()
        comptime if Self.gather4_idx_batch > 0:
            var align_pad = (16 - (Int(idx_smem_base) & 15)) & 15
            debug_assert(
                align_pad == Self.idx_smem_align_pad,
                "idx_smem is not where the dispatch's SMEM size assumes",
            )
            idx_smem_base = idx_smem_base.unsafe_offset(align_pad // 4)
        comptime idx_smem_stride = Self.config.BN_QK

        var warp_idx = UInt32(warp_id[broadcast=True]())
        var is_leader = elect() != 0

        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                kv_pipeline_gather.init()
                kv_pipeline_mma.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                corr_done_bars.init()
                out_pipeline.init()
                idx_bars.init()
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
                comptime if Self.has_extra_kv:
                    extra_k_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        # Zero the NoPE tail before the warpgroups read it. The fence publishes
        # the generic-proxy stores to the async proxy for the tcgen05 read.
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            Self.zero_input_tail(
                q_smem.as_unsafe_any_origin(), kv_smem.as_unsafe_any_origin()
            )
        barrier()
        comptime if Self.config.input_q_depth < Self.config.padded_q_depth:
            fence_async_view_proxy()

        if warp_idx < 4:
            warpgroup_reg_alloc[num_reg_softmax]()

            var attn_sink_log2 = Float32(min_or_neg_inf[.float32]())
            comptime if Self.has_attn_sink:
                # The softmax row this thread owns; lane_id() would fold rows
                # 32-63 onto 0-31 and give them the wrong heads' sinks.
                var row = thread_idx.x & 0x3F
                var head_idx_local = block_idx.x * Self.config.BM + row
                if head_idx_local < Self.config.num_q_heads:
                    attn_sink_log2 = attn_sink_ptr.value()[head_idx_local].cast[
                        DType.float32
                    ]() * Float32(log2e)

            comptime if Self.MaskType.get_type_name() == "CausalMask" and not Self.fold_shared_index:
                if not logical_indices:
                    offset_position.assert_row_packed[Self.sparse_row](
                        d_indices, indices_stride, selection_lag
                    )
            Self.Common_MLA_Op.Softmax[
                native_fp8=True,
                num_sp_stages=Self.num_stages,
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
                offset_position,
                scale,
                mask,
                prompt_idx=UInt32(offset_position.batch_idx),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
                attn_sink_log2=attn_sink_log2,
                logical_indices=logical_indices,
                logical_indices_stride=indices_stride,
                logical_indices_len=offset_position.sparse_topk,
                selection_lag=selection_lag,
            )
        elif warp_idx >= 4 and warp_idx < 8:
            warpgroup_reg_alloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        elif warp_idx >= 8 and warp_idx < 12:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem.as_unsafe_any_origin(),
                    kv_smem_linear.as_unsafe_any_origin(),
                    mbar_q,
                    kv_pipeline_gather,
                    offset_position,
                    idx_bars,
                    idx_smem_base,
                    num_orig_blocks,
                    topk,
                    extra_k_tma,
                    extra_topk,
                    scales_ptr,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_smem.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline_mma,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem.as_unsafe_any_origin(),
                    p_smem.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline_mma,
                    offset_position,
                )
            elif warp_idx == 11:
                var batch_d_indices_w11 = (
                    d_indices.unsafe_value().unsafe_offset(
                        offset_position.q_token_idx * indices_stride
                    )
                )
                var batch_extra_d_indices_w11 = extra_d_indices
                comptime if Self.has_extra_kv:
                    batch_extra_d_indices_w11 = (
                        extra_d_indices.unsafe_value().unsafe_offset(
                            offset_position.q_token_idx * extra_indices_stride
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
            # WG3 (warps 12-15): re-swizzle staging warpgroup. Reads the
            # contiguously-gathered linear FP8, writes FP8 (no cast) into
            # the SW64 layout mmaQK/mmaPV already expect. Repurposes the
            # old kernel's convert-WG role/register budget without the
            # dtype conversion.
            warpgroup_reg_dealloc[num_reg_reswizzle]()
            comptime if Self.is_nvfp4_kv:
                Self.reswizzleNVFP4(
                    kv_smem_linear.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    kv_pipeline_gather,
                    kv_pipeline_mma,
                    offset_position,
                )
            else:
                Self.reswizzleFP8(
                    kv_smem_linear.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    kv_pipeline_gather,
                    kv_pipeline_mma,
                    offset_position,
                )
        barrier()

        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    @staticmethod
    @inline(.always)
    def _transform_indices_to_smem(
        d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
        idx_smem: SharedMemPointer[Int32],
        indices_base: Int,
        kv_lut: Self.KVLUTType,
        topk: UInt32,
        batch_idx: UInt32,
        scale_rows_offset: Int,
    ):
        var lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1

        comptime for row_pass in range(2):
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
        idx_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        kv_lut: Self.KVLUTType,
        d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
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
        extra_d_indices: OptionalReg[MutPointer[Int32, MutAnyOrigin]],
        extra_topk: Int,
        scale_rows_offset: Int,
    ):
        # The extra stream's caller (latent sparse attention) passes flat
        # physical rows, so it cannot share a logical selection's decoding.
        comptime assert not (
            not Self.physical_indices and Self.has_extra_kv
        ), "logical sparse indices do not support an extra KV stream"
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var idx_prod = idx_bars.producer()
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
            orig_indices_base += Self.config.BN_QK

        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        var t: Int = 0
        while t < remaining_orig:
            idx_prod.acquire()
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
            orig_indices_base += Self.config.BN_QK
            t += 1

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
                extra_indices_base += Self.config.BN_QK
                te += 1

    @staticmethod
    @inline(.always)
    def load(
        q_tma: QOTMATile[
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.input_q_depth,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
        ],
        k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem_linear: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        kv_pipeline_gather: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
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
        idx_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        num_orig_blocks: Int,
        topk: Int,
        extra_k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
        extra_topk: Int,
        scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_prod = DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
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
            Int32(
                Self.config.BM
                * Self.config.input_q_depth
                * Self.fp8_bytes_per_element
            ),
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
                Self.fp8_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=.SHARED,
            ](q_smem.unsafe_bitcast[Scalar[Self.fp8_type]](), q_tt_layout)
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
    def _issue_kv_gather(
        cur_k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
        kv_stage_ptr: SharedMemPointer[Scalar[Self.fp8_type]],
        k_mbar: MBarType,
        idx_smem: SharedMemPointer[Int32],
        is_leader: Bool,
    ):
        comptime box_w = Self.kv_gather4_box_w
        comptime num_chunks = Self.config.BK_PV // 4
        comptime if Self.gather4_idx_batch > 0:
            comptime assert (
                box_w == Self.kv_gather4_tile_width
            ), "per-chunk gather4 assumes one column group per row"
            comptime B = Self.gather4_idx_batch
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
                            kv_stage_ptr.unsafe_bitcast[Int64]().unsafe_offset(
                                c * 4 * box_w
                            ),
                            tt_row_major[4, box_w](),
                        )
                        var r = rows[j].cast[DType.int32]()
                        cur_k_tma.async_copy_gather4[
                            eviction_policy=CacheEviction.EVICT_LAST
                        ](dst, k_mbar[], Int32(0), r[0], r[1], r[2], r[3])
        else:
            if is_leader:
                cur_k_tma.async_copy_gather4_tile[
                    tile_width=Self.kv_gather4_tile_width,
                    eviction_policy=CacheEviction.EVICT_LAST,
                ](
                    kv_stage_ptr.unsafe_bitcast[Int64](),
                    k_mbar[],
                    idx_smem,
                    start_idx=0,
                )

    @staticmethod
    @inline(.always)
    def _stage_nvfp4_scales(
        kv_stage_ptr: SharedMemPointer[Scalar[Self.fp8_type]],
        idx_smem: SharedMemPointer[Int32],
        scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
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
        var lane = Int(thread_idx.x) & 31
        var half = lane % 2
        comptime RP = Self.config.BN_QK // 16
        comptime for rp in range(RP):
            var r = rp * 16 + lane // 2
            var row = idx_smem[Self.config.BN_QK + r]
            var src = sf_gmem.unsafe_offset(
                Int(max(row, Int32(0))) * Self.nvfp4_sf_row_bytes + half * 16
            )
            var dst = sf_smem.unsafe_offset(
                r * Self.nvfp4_sf_row_bytes + half * 16
            )
            async_copy[16, fill=UInt8(0)](
                src.unsafe_address_space_cast[.GLOBAL](),
                dst,
                src_size=Int32(16) if row >= 0 else Int32(0),
            )
        async_copy_arrive[noinc=True](k_mbar)

    @staticmethod
    @inline(.always)
    def _load_one_tile(
        mut kv_prod: DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
        mut idx_cons: ConsumerPipeline[Self.num_stages],
        idx_smem_base: SharedMemPointer[Int32],
        kv_bytes: Int,
        scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    ):
        var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
        var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

        idx_cons.wait()
        var idx_smem = idx_smem_base.unsafe_offset(
            idx_cons.state.index() * UInt32(Self.idx_stage_slots)
        )

        expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))
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
            Self.fp8_type,
            Self.config,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            DType.int64,
            tile_shape=coord[Self.config.BK_PV, Self.kv_gather4_box_w],
            desc_shape=coord[1, Self.kv_gather4_box_w],
        ],
        mut idx_cons: ConsumerPipeline[Self.num_stages],
        idx_smem_base: SharedMemPointer[Int32],
        num_tiles: Int,
        scales_ptr: MutPointer[Float32, origin=MutAnyOrigin],
    ):
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_box_w * size_of[
            DType.int64
        ]()

        var t: Int = 0
        while t < num_tiles:
            kv_prod.acquire[qk_stage=0]()

            var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

            idx_cons.wait()
            var idx_smem = idx_smem_base.unsafe_offset(
                idx_cons.state.index() * UInt32(Self.idx_stage_slots)
            )

            expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))
            Self._issue_kv_gather(
                cur_k_tma, kv_stage_ptr, k_mbar, idx_smem, is_leader
            )
            comptime if Self.is_nvfp4_kv:
                Self._stage_nvfp4_scales(
                    kv_stage_ptr, idx_smem, scales_ptr, k_mbar
                )

            idx_cons.release()
            kv_prod.commit_step()
            t += 1

    @staticmethod
    @inline(.always)
    def reswizzleFP8(
        kv_smem_linear: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem_sw64: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_pipeline_gather: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        """Re-swizzles one KV tile per iteration: FP8 linear -> FP8 SW64.

        Pure layout permute, no dtype cast (unlike the old kernel's
        convertFP8ToBF16). Uses a separate, non-overlaid destination
        buffer, so there is no free/held-block aliasing split to manage --
        every column block is read then written independently.

        Thread mapping mirrors convertFP8ToBF16 exactly (GROUP_SIZE=4,
        128 threads -> 32 groups of 2 rows each, 9 column iterations of
        16 bytes/thread cover the row as stored) so the same
        bank-conflict-reduction rationale applies.
        """
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime BN_QK: Int = Self.config.BN_QK
        comptime fp8_row_stride: Int = Self.config.input_q_depth
        comptime GROUP_SIZE: Int = 4
        comptime NUM_GROUPS: Int = WARPGROUP_SIZE // GROUP_SIZE  # 32
        # input_q_depth / (4 threads * 16 bytes) column iterations.
        comptime COLS_PER_GROUP: Int = fp8_row_stride // (GROUP_SIZE * 16)

        var kv_gather_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
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
        var sw_off_0: Int = Self.sw_fp8(row_0 * BN_QK + col)
        var sw_off_1: Int = Self.sw_fp8(row_1 * BN_QK + col)

        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            kv_mma_prod.acquire()

            var src_u8 = kv_gather_cons.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()

            comptime for c in range(COLS_PER_GROUP):
                var q0 = ld_shared_v4_u32(
                    src_u8, fp8_base_0 + c * GROUP_SIZE * 16
                )
                var q1 = ld_shared_v4_u32(
                    src_u8, fp8_base_1 + c * GROUP_SIZE * 16
                )
                var dst_block = dst.unsafe_offset(c * Self.BlockElems)
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.fp8_type](
                    dst_block, sw_off_0, q0
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.fp8_type](
                    dst_block, sw_off_1, q1
                )

            fence_async_view_proxy()
            kv_gather_cons.release_all()
            kv_mma_prod.commit_all()
            tile_idx += 1

    @staticmethod
    @inline(.always)
    def reswizzleNVFP4(
        kv_smem_linear: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem_sw64: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_pipeline_gather: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        comptime LATENT_BLOCKS: Int = LATENT_DIM // 64
        # A chunk is 16 packed bytes: 32 elements, two scale groups.
        comptime QCHUNKS: Int = Self.nvfp4_sf_row_bytes // 2
        comptime ITEMS: Int = BN_QK * QCHUNKS // WARPGROUP_SIZE
        comptime assert (
            BN_QK == 64
            and QCHUNKS == 16
            and ITEMS == 8
            and NVFP4_MLA_ROPE_DIM == 64
        )

        var kv_gather_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=Self.gather_num_producer,
            num_consumer=WARPGROUP_SIZE,
        ](kv_pipeline_gather, kv_smem_linear)
        var kv_mma_prod = DecodeKVProducer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
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
        while tile_idx < num_k_tiles:
            kv_gather_cons.wait()
            kv_mma_prod.acquire()

            var src_u8 = kv_gather_cons.stage_base_ptr().unsafe_bitcast[UInt8]()
            var dst = kv_mma_prod.stage_base_ptr()

            # The row's scales: word `b` holds the two groups of each of the
            # chunks of block `b`; the lane's chunk of block `i ^ cg` is the
            # `cp`-th half of word `i ^ cg`.
            var sf_a = ld_shared_v4_u32(src_u8, sf_base)
            var sf_b = ld_shared_v4_u32(src_u8, sf_base + 16)
            comptime for i in range(ITEMS):
                var blk = i + cg if i % 2 == 0 else i - cg
                var q = ld_shared_v4_u32(src_u8, pk_base + 32 * blk)
                var sf_even = sf_a[i] if i < 4 else sf_b[i - 4]
                var sf_odd = sf_a[i ^ 1] if i < 4 else sf_b[(i - 4) ^ 1]
                var dst_block = dst.unsafe_offset(blk * Self.BlockElems)
                comptime for g in range(2):
                    var sf_byte = (
                        byte_permute(sf_even, sf_odd, sel1 if g == 1 else sel0)
                        & 0xFF
                    )
                    var scale = bitcast[Self.fp8_type, 1](
                        SIMD[DType.uint8, 1](UInt8(sf_byte))
                    ).cast[DType.float32]()[0]
                    var words = SIMD[DType.uint32, 2](q[2 * g], q[2 * g + 1])
                    st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.fp8_type](
                        dst_block,
                        d1 if g == 1 else d0,
                        nvfp4_dequant_group(
                            bitcast[DType.uint8, 8](words), scale
                        ),
                    )

            var rope_block = dst.unsafe_offset(LATENT_BLOCKS * Self.BlockElems)
            comptime for r in range(2):
                var rope = ld_shared_v4_u32(
                    src_u8, rope_src1 if r == 1 else rope_src0
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype=Self.fp8_type](
                    rope_block, rope_dst1 if r == 1 else rope_dst0, rope
                )

            fence_async_view_proxy()
            kv_gather_cons.release_all()
            kv_mma_prod.commit_all()
            tile_idx += 1

    @staticmethod
    @inline(.always)
    def mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = (
            Self.KVStageElems * Self.fp8_bytes_per_element
        )

        mbar_q[].wait(0)

        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            s_prod.acquire()
            var slot_idx: UInt32 = s_prod.slot_index()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            kv_cons.wait[qk_stage=0]()
            var k_slot_index = kv_cons.stage_index[qk_stage=0]()

            Self.UMMAQKTSS.mma[stage_idx=0](
                a=q_descriptor,
                b=k_descriptor + k_slot_index * UInt32(stage_stride_in_bytes),
                c=s_tmem_slot,
                c_scale=UInt32(0),
                elect=elect_mask,
            )
            tcgen05_fence_before()
            s_prod.commit_mma(elect_mask)
            kv_cons.release[qk_stage=0](elect_mask)
            tile_idx += 1

    @staticmethod
    @inline(.always)
    def mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline_mma: KVPipelineGeneric[
            num_kv_stages=Self.num_mma_stages,
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
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
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[
            Self.fp8_type,
            Self.config,
            num_producer=WARPGROUP_SIZE,
            num_consumer=2,
            num_stages=Self.num_mma_stages,
        ](kv_pipeline_mma, kv_smem)
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        comptime kv_stage_stride_in_bytes = (
            Self.KVStageElems * Self.fp8_bytes_per_element
        )
        comptime p_stage_stride_in_bytes = (
            Self.PStageElems * Self.fp8_bytes_per_element
        )
        comptime block_stride_in_bytes = (
            Self.BlockElems * Self.fp8_bytes_per_element
        )

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            comptime for block in range(0, Self.NumVOBlocks, block_step):
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(p_stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(kv_stage_stride_in_bytes)
                    + UInt32(block * block_stride_in_bytes),
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN_QK // 2),
                    c_scale=c_scale,
                    elect=elect_mask,
                )
                o_prod.commit_mma(elect_mask)
            p_cons.release_mma(elect_mask)

            kv_cons.release[qk_stage=0](elect_mask)
            tcgen05_fence_before()

            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
