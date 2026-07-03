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

from std.math import align_up
from std.sys import simd_width_of, size_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    warp_id,
)
from std.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from std.gpu.primitives.warp import broadcast
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from std.gpu.memory import fence_mbarrier_init
from std.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from layout.tma_async import RaggedTMA3DTile
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import FA4Config, MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    SM100TensorAccumulator,
    elect,
    kv_sub_tile_rows,
    o_store_tma_blocks_per_op,
)
from nn.attention.gpu.nvidia.common import (
    get_seq_info,
    KVTMATile,
    MHAPosition,
    OptionalPointer,
    Pack,
    PositionSummary,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHATileScheduler,
    SeqInfo,
)
from nn.attention.mha_utils import (
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor
from .smem import SM100AttentionSMem
from .softmax_warp import fa4_softmax
from .correction_warp import fa4_correction
from .load_warp import fa4_load
from .mma_warp import fa4_mma


struct SM100MHA2Q[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: FA4Config[KVLUTType.dtype],
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](TrivialRegisterPassable):
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32
    comptime simd_size: Int = simd_width_of[Self.qkv_type]()

    comptime pair_cta: Bool = Self.config.pair_cta
    comptime cta_group: Int = 2 if Self.pair_cta else 1
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.qk_depth
    comptime padded_depth = Self.config.padded_qk_depth
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime fuse_gqa = Self.config.fuse_gqa
    # BM_eff: sequence positions per full tile (BM // group when fusing)
    comptime BM_eff: Int = Self.config.BM_eff()
    # BM_mask: the BM value passed to mask functions.
    # For pair-CTA, use PairBM so both CTAs make identical skip decisions.
    comptime BM_mask: Int = Self.config.PairBM_eff()
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.MMA_M  # 128 single-CTA, 256 pair-CTA
    comptime qo_elements = Self.padded_depth * Self.HalfBM
    comptime qkv_dt_size = size_of[Self.qkv_type]()
    comptime HalfBM = Self.BM // 2

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulator[
        Self.qkv_type,
        Self.accum_type,
        MMA_M=Self.MMA_M,  # 128 single-CTA, 256 pair-CTA
        MMA_N=Self.BN,
        BK=align_up(Self.depth, Self.config.MMA_K),  # BK in memory depth
        a_tmem=False,
        swizzle_a=Self.config.swizzle_mode,
        swizzle_b=Self.config.swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
        cta_group=Self.cta_group,
    ]
    # Second MMA is P@V (V not staged, but P writing can be staged)
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulator[
        Self.qkv_type,
        Self.accum_type,
        MMA_M=Self.MMA_M,
        MMA_N=Self.config.padded_ov_depth,
        BK=Self.BN,
        a_tmem=True,
        swizzle_b=Self.config.swizzle_mode,
        transpose_b=False,
        num_stages=Self.num_pv_stages,
        cta_group=Self.cta_group,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = UInt32(
        Self.swizzle_granularity * Self.config.BN
    )
    comptime qo_bytes: UInt32 = UInt32(Self.qkv_dt_size * Self.qo_elements)
    comptime k_bytes: UInt32 = UInt32(Self.qkv_dt_size) * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = UInt32(
        Self.qkv_dt_size * Self.MMA_K * Self.config.padded_ov_depth
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.padded_qk_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    comptime SmemType = SM100AttentionSMem[Self.config]

    # TMA-op types as seen by `kernel`/`_kernel_impl`. Defined once so the 2Q
    # signatures and the 1Q-variant rebinds (see `kernel`) share a single
    # definition.
    comptime QTMAOpType = QTMATile[
        Self.KVLUTType.dtype,
        Self.config.swizzle_mode,
        BM=Self.config.BM // Self.config.num_qo,
        depth=Self.config.qk_depth,
        group=Self.config.group,
        decoding=False,
        fuse_gqa=Self.fuse_gqa,
        num_qk_stages=Self.config.num_qk_stages,
    ]
    comptime KTMAOpType = KVTMATile[
        Self.KVLUTType.dtype,
        Self.config.swizzle_mode,
        BN=kv_sub_tile_rows(Self.config.k_rows_per_cta(), Self.page_size),
        BK=Self.config.BK0,
    ]
    comptime VTMAOpType = KVTMATile[
        Self.KVLUTType.dtype,
        Self.config.swizzle_mode,
        BN=kv_sub_tile_rows(Self.config.BN, Self.page_size),
        BK=Self.config.v_cols_per_cta(),
    ]
    comptime OTMAStoreType = RaggedTMA3DTile[
        Self.output_type,
        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # Q/K/V/S/P buffers governed by `config.swizzle_mode`).
        TensorMapSwizzle.SWIZZLE_NONE,
        # 2Q: BM=128 (each WG writes one of two Q halves).
        # 1Q: BM=128 (both WGs cover the full BM=128 Q rows and write
        # disjoint depth-column ranges).
        BM=Self.config.BM // Self.config.num_qo,
        BN=Self.config.ov_depth,
        middle_dim=Self.config.num_kv_heads if Self.fuse_gqa else Self.config.num_q_heads,
        group=Self.config.group if Self.fuse_gqa else 1,
        # Batched rank-5 O store (must match dispatch.mojo's store).
        tma_blocks_per_op=o_store_tma_blocks_per_op[
            Self.output_type,
            TensorMapSwizzle.SWIZZLE_NONE,
            Self.config.ov_depth,
            Self.config.group if Self.fuse_gqa else 1,
            depth_splits=2,
        ](),
    ]
    comptime PackType = Pack[
        Self.MaskType,
        Self.SchedulerType,
        Self.ValidLengthType,
        Self.SinkType,
        Self.KVRowOffsetsType,
        Self.MaxSeqLenType,
        Self.PartitionType,
    ]

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](Int32(Self.cta_group), 1, 1)
    )
    @__name(
        t"sm100_mha_{Self.config.num_qo}q_depth{Self.config.qk_depth}_{Self.qkv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        q_tma_op: Self.QTMAOpType,
        k_tma_op: Self.KTMAOpType,
        v_tma_op: Self.VTMAOpType,
        ragged_tma_store: Self.OTMAStoreType,
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Self.PackType,
    ):
        # Thin entrypoint. Compute this tile's `SeqInfo` once and forward it to
        # `_kernel_impl` (all warp groups share it — it is derived from
        # blockIdx, not threadIdx, so it is identical on every lane/warp).
        #
        # When this 2Q config admits a type-compatible 1Q variant
        # (`can_switch_to_1q`), route short sequences through the cheaper 1Q
        # body. The 1Q body covers `Kernel1Q.BM_eff` *sequence positions* per
        # tile at `blockIdx.x == 0` (BM // group when fusing GQA, else BM), so
        # the switch is only correct when the whole prompt fits in that one
        # tile: `seq_len <= Kernel1Q.BM_eff`. Then the prompt is a single tile
        # at `prompt_offset == 0` in BOTH the 1Q and 2Q tilings, where their
        # `SeqInfo` coincide, so passing the 2Q `seq_info` into the 1Q body is
        # correct. `broadcast` (lane-0 shuffle) makes the branch warp-uniform;
        # the decision is the same on all 16 warps because `seq_len` is, so no
        # cross-warp sync is needed. The grid is unchanged (still 2Q-tiled) —
        # a prompt this short occupies only `blockIdx.x == 0`, which the 2Q
        # grid always covers.
        var seq_info: SeqInfo = get_seq_info[
            Self.BM_mask,
            Self.config.num_kv_heads if Self.fuse_gqa else Self.num_q_heads,
            Self.MaskType.get_type_name() == "CausalMask",
            pair_cta=Self.pair_cta,
        ](batch_size, pack.max_seq_len, pack.valid_length, pack.partition)

        comptime if Self.config.can_switch_to_1q():
            comptime Kernel1Q = SM100MHA2Q[
                Self.KVLUTType,
                Self.output_type,
                Self.MaskType,
                Self.SchedulerType,
                Self.config.switch_1q_config(),
                Self.ValidLengthType,
                Self.SinkType,
                Self.KVRowOffsetsType,
                Self._is_cache_length_accurate,
                Self.MaxSeqLenType,
                Self.PartitionType,
            ]
            if broadcast(seq_info.seq_len - seq_info.prompt_offset) <= UInt32(
                Kernel1Q.BM_eff
            ):
                # The TMA ops are typed from the 2Q `Self.config`; the 1Q types
                # fold to identical values (`can_switch_to_1q()` guarantees
                # matching `num_qk_stages` ⇒ matching `BK0`; per-half BM=128,
                # BN/depth/group/swizzle already match), but the parser sees
                # distinct parameter expressions, so `rebind`.
                Kernel1Q._kernel_impl(
                    rebind[Kernel1Q.QTMAOpType](q_tma_op),
                    rebind[Kernel1Q.KTMAOpType](k_tma_op),
                    rebind[Kernel1Q.VTMAOpType](v_tma_op),
                    rebind[Kernel1Q.OTMAStoreType](ragged_tma_store),
                    kv_lut,
                    scale,
                    num_keys_arg,
                    pack,
                    seq_info,
                )
                return
        Self._kernel_impl(
            q_tma_op,
            k_tma_op,
            v_tma_op,
            ragged_tma_store,
            kv_lut,
            scale,
            num_keys_arg,
            pack,
            seq_info,
        )

    @staticmethod
    @always_inline
    def _kernel_impl(
        q_tma_op: Self.QTMAOpType,
        k_tma_op: Self.KTMAOpType,
        v_tma_op: Self.VTMAOpType,
        ragged_tma_store: Self.OTMAStoreType,
        kv_lut: Self.KVLUTType,
        scale: Float32,
        num_keys_arg: UInt32,
        pack: Self.PackType,
        seq_info: SeqInfo,
    ):
        comptime assert (
            Self.MMA_M == 64 or Self.MMA_M == 128 or Self.MMA_M == 256
        )
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
        sink_weights = pack.sink_weights
        kv_input_row_offsets = pack.kv_input_row_offsets
        max_seq_len = pack.max_seq_len

        comptime num_qo = Self.config.num_qo
        # TODO: We may want to support num_qo>2 for depth=64?
        comptime assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        var smem = Self.SmemType()
        var misc_mbars = smem.misc_mbars()

        # Per-warpgroup register allocation, mirroring CUTLASS's
        # `sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`.  Softmax gets
        # the largest slice (192), correction the next (88), and the
        # MMA-leader/other path runs lean (40); inactive warps drop to
        # the minimum (24).  Sum × WG size must stay ≤ the SM register
        # file; bump together if a path starts spilling.
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 88
        comptime num_reg_other = 40

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/K/V/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
        elif warp_idx == 1:
            tcgen05_alloc[Int32(Self.cta_group)](
                smem.tmem_addr_ptr(),
                UInt32(Self.config.sm100_tmem_cols),
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        # Pair-CTA: cluster_sync ensures both CTAs see each other's barriers.
        # Single-CTA: plain barrier suffices.
        comptime if Self.pair_cta:
            fence_mbarrier_init()
            cluster_sync()
        else:
            barrier()

        # Read the TMEM base from SMEM EXACTLY ONCE here, post-barrier (the
        # alloc on warp 1 + this barrier publish it), and carry it by register
        # to every consumer (fa4_softmax / fa4_correction / fa4_mma). Mirrors
        # the depth-512 fix: the consumers must NOT re-read
        # `smem.tmem_addr_ptr()` in their bodies, because an in-body slot reload
        # gated only on a pipeline barrier (not the alloc publish) can observe a
        # stale/pre-alloc value under SM co-residency and feed a garbage TMEM
        # operand to `UTCHMMA`. Same published value -> single-shot bit-identical.
        var tmem_addr: UInt32 = smem.tmem_addr_ptr()[]

        # Programmatic Dependent Launch (PDL).  This is the only point every
        # thread of every CTA reaches before the warp-specialized early
        # returns below (invalid tiles bail in warps 0-13 while warps 14-15
        # fall through), so it is the only divergence-free place to honor the
        # contract that *every* CTA signal launch-dependents — otherwise a
        # back-to-back consumer grid's `wait` hangs (see MLA decode).  The
        # data-independent prologue above (barrier init, tmem alloc, TMA
        # descriptor prefetch) overlaps the predecessor grid's tail; `wait`
        # fences here before the data-dependent Q/K/V loads in `fa4_load`;
        # `launch` lets the successor grid's prologue overlap our compute.
        comptime if MHA_PDL_LEVEL > PDLLevel.OFF:
            wait_on_dependent_grids()
            launch_dependent_grids()

        # warp group partitioning
        # Two QO:
        #
        # Pair-CTA: early returns are replaced with conditional work so that
        # ALL threads always reach the cluster_sync at the bottom.  Without
        # this, invalid tiles cause some warps to return early while warps
        # 14-15 (and any valid warps) block at cluster_sync forever.
        if warp_idx < 8:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()

            comptime if not Self.pair_cta:
                if not seq_info.is_valid():
                    return

            if seq_info.is_valid():
                var pos: PositionSummary = PositionSummary.create[
                    ragged=Self.ragged,
                    _is_cache_length_accurate=Self._is_cache_length_accurate,
                ](
                    kv_lut,
                    seq_info,
                    num_keys_arg,
                    kv_input_row_offsets,
                    max_seq_len,
                )

                fa4_softmax[
                    Self.KVLUTType,
                    Self.config,
                    Self.ValidLengthType,
                    Self.SinkType,
                    Self._is_cache_length_accurate,
                    Self.MaxSeqLenType,
                    # `kernel` routes tiles with
                    # `seq_len - prompt_offset <= Kernel1Q.BM_eff` to the
                    # 1Q body when the switch is compiled in, so the 2Q
                    # body's output halves are always non-empty.
                    output_nonempty=Self.config.can_switch_to_1q(),
                ](
                    smem,
                    tmem_addr,
                    pos.score_row,
                    seq_info,
                    mask,
                    pos.num_keys,
                    scale.cast[Self.accum_type](),
                    max_seq_len.as_uint32(),
                    ragged_tma_store,
                    sink_weights,
                )

        elif warp_idx < 12:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            comptime if not Self.pair_cta:
                if not seq_info.is_valid():
                    return

            if seq_info.is_valid():
                var pos: PositionSummary = PositionSummary.create[
                    ragged=Self.ragged,
                    _is_cache_length_accurate=Self._is_cache_length_accurate,
                ](
                    kv_lut,
                    seq_info,
                    num_keys_arg,
                    kv_input_row_offsets,
                    max_seq_len,
                )
                fa4_correction[
                    Self.config,
                    Self.page_size,
                ](
                    smem,
                    tmem_addr,
                    seq_info.prompt_idx,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                )
        else:
            if warp_idx == 13:  # produce
                warpgroup_reg_dealloc[num_reg_other]()

                comptime if not Self.pair_cta:
                    if not seq_info.is_valid():
                        return

                if seq_info.is_valid():
                    var pos: PositionSummary = PositionSummary.create[
                        ragged=Self.ragged,
                        _is_cache_length_accurate=Self._is_cache_length_accurate,
                    ](
                        kv_lut,
                        seq_info,
                        num_keys_arg,
                        kv_input_row_offsets,
                        max_seq_len,
                    )
                    comptime if not Self.pair_cta:
                        fa4_load[
                            Self.config,
                            ValidLengthType=Self.ValidLengthType,
                            _is_cache_length_accurate=Self._is_cache_length_accurate,
                            is_leader=True,
                        ](
                            smem,
                            pos.score_row,
                            pos.num_keys,
                            seq_info,
                            max_seq_len,
                            mask,
                            q_tma_op,
                            k_tma_op,
                            v_tma_op,
                            kv_lut,
                        )
                    else:
                        var cta_rank = block_rank_in_cluster() % 2
                        if cta_rank == 0:
                            fa4_load[
                                Self.config,
                                ValidLengthType=Self.ValidLengthType,
                                _is_cache_length_accurate=Self._is_cache_length_accurate,
                                is_leader=True,
                            ](
                                smem,
                                pos.score_row,
                                pos.num_keys,
                                seq_info,
                                max_seq_len,
                                mask,
                                q_tma_op,
                                k_tma_op,
                                v_tma_op,
                                kv_lut,
                            )
                        else:
                            fa4_load[
                                Self.config,
                                ValidLengthType=Self.ValidLengthType,
                                _is_cache_length_accurate=Self._is_cache_length_accurate,
                                is_leader=False,
                            ](
                                smem,
                                pos.score_row,
                                pos.num_keys,
                                seq_info,
                                max_seq_len,
                                mask,
                                q_tma_op,
                                k_tma_op,
                                v_tma_op,
                                kv_lut,
                            )

            elif warp_idx == 12:  # Q @ K', P @ V
                warpgroup_reg_dealloc[num_reg_other]()

                comptime if not Self.pair_cta:
                    if not seq_info.is_valid():
                        tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                        tcgen05_dealloc[Int32(Self.cta_group)](
                            tmem_addr, UInt32(Self.config.sm100_tmem_cols)
                        )
                        return
                var execute: Bool = seq_info.is_valid()
                comptime if Self.pair_cta:
                    # ---- Pair-CTA: leader-only guard ----
                    execute &= broadcast(block_rank_in_cluster()) % 2 == 0
                if execute:
                    var pos: PositionSummary = PositionSummary.create[
                        ragged=Self.ragged,
                        _is_cache_length_accurate=Self._is_cache_length_accurate,
                    ](
                        kv_lut,
                        seq_info,
                        num_keys_arg,
                        kv_input_row_offsets,
                        max_seq_len,
                    )
                    fa4_mma[Self.config, page_size=Self.page_size](
                        smem,
                        tmem_addr,
                        seq_info.prompt_idx,
                        pos.score_row,
                        pos.num_keys,
                        mask,
                    )
            else:
                # 24 is the floor for `setmaxnreg.dec` on SM90+ — drop
                # this warpgroup's allocation to the minimum so the
                # active WGs can claim its share of the SM register file.
                warpgroup_reg_dealloc[24]()

        # Pair-CTA: cluster_sync before dealloc so that stmatrix
        # (which uses shared::cluster on SM100) in the peer CTA has
        # finished before either CTA exits and breaks the cluster.
        # All early returns above were converted to fall-through for
        # pair_cta so that every thread reaches this sync point.
        comptime if Self.pair_cta:
            cluster_sync()
            if warp_idx == 0:
                tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                tcgen05_dealloc[Int32(Self.cta_group)](
                    tmem_addr, UInt32(Self.config.sm100_tmem_cols)
                )

    @staticmethod
    @always_inline
    def mask_status(
        mask: Self.MaskType,
        seq_id: UInt32,
        score_row: UInt32,
        kv_row: UInt32,
    ) -> TileMaskStatus:
        return mask.status(
            seq_id,
            Index[dtype=DType.int32](
                Int(score_row),
                Int(kv_row),
            ),
            Index[dtype=DType.int32](Self.BM_mask, Self.BN),
        )

    @staticmethod
    @always_inline
    def descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.BM // 2,
            BK=Self.config.BK0,
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,
        ](q_smem)
