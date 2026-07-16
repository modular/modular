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

from std.bit import prev_power_of_two
from std.collections import OptionalReg
from std.math import ceildiv
from std.gpu.primitives.grid_controls import pdl_launch_attributes
from std.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    Dim,
    FuncAttribute,
)
from nn.attention.gpu.nvidia.common import ImmutTileTensor1D
from layout.tma_async import RaggedTMA3DTile
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.logger import Logger
from nn.attention.gpu.nvidia.sm100.attention import FA4Config, MHA_PDL_LEVEL
from nn.attention.gpu.nvidia.common import (
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    Pack,
    q_tma,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.mha_utils import (
    DynamicInt,
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    StaticInt,
    _is_decoding,
)
from .attention_utils import (
    kv_sub_tile_rows,
    kv_tma_fold_chunks,
    o_store_tma_blocks_per_op,
)
from .kernel import SM100MHA2Q

from std.pathlib import Path

comptime logger = Logger()


@always_inline
def mha_sm100_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
    pair_cta: Bool,
](
    output: DeviceBuffer[output_type],
    q_arg: UnsafePointer[Scalar[q_type], _],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask: MaskType,
    valid_length: UnsafePointer[UInt32, _],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[ImmutTileTensor1D[DType.uint32]],
    batch_size_arg: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[ImmutTileTensor1D[q_type]],
) raises:
    comptime PartitionType = NoPartition[DType.float32]
    comptime partition: PartitionType = PartitionType()
    comptime assert (
        config.dtype == KVType.dtype and config.dtype == q_type
    ), "config, kv, and q types must all match for FA3."
    comptime decoding: Bool = _is_decoding[MaxPromptLenType]()
    comptime assert (
        not decoding
    ), "this implementation does not support decoding"
    comptime fa4_config_2q = FA4Config[KVType.dtype](
        num_q_heads=config.num_heads,
        group=group,
        qk_depth=config.depth,
        ov_depth=config.depth,
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
        is_mla=False,
        pair_cta=pair_cta,
    )
    comptime assert fa4_config_2q.supported(), fa4_config_2q.description()
    var q = rebind[UnsafePointer[Scalar[KVType.dtype], q_arg.origin]](q_arg)

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)

    @parameter
    @always_inline
    def with_fa4_config[
        NumPartitionsType: OptionallyStaticInt,
        //,
        fa4_config: FA4Config[KVType.dtype],
    ](num_partitions: NumPartitionsType) raises:
        # `num_partitions` is dynamic (a runtime cluster size) iff this is the
        # num_q==1 split-K path; for the 2Q config (and its in-kernel 1Q switch)
        # it is a static 1, so the split-K launch/grid/combine fold away and no
        # runtime argument is generated.
        comptime assert (
            Bool(NumPartitionsType.static_value) != fa4_config.splitk_dynamic()
        ), "num_partitions must be dynamic iff config.splitk_dynamic()"
        comptime swizzle_mode = fa4_config.swizzle_mode
        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # Q/K/V/S/P buffers governed by `swizzle_mode`). The softmax warp loads
        # O one-row-per-thread and writes it row-major, avoiding cross-thread
        # shuffles and swizzling while staying bank-conflict-free.
        comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
        comptime BM = fa4_config.BM
        comptime fuse_gqa = fa4_config.fuse_gqa
        comptime num_threads = fa4_config.num_threads
        # `MMA_M // cta_group` is 128 in all three valid configs:
        # 2Q single-CTA (128 // 1), 2Q pair-CTA (256 // 2), 1Q
        # single-CTA (128 // 1). Drives q_tma_op and ragged_tma_store
        # BM under a unified expression.
        comptime BM_per_mma = fa4_config.MMA_M // fa4_config.cta_group()
        comptime assert BM == 128 or BM == 256

        # Batch the O store into one TMA per issuer: the box covers
        # `ceil(n_blocks/2)` swizzle-granularity blocks, so the single-issuer
        # writeback emits 2 pipelined copies and the 1Q combine emits 1 per WG
        # (vs `n_blocks` per-block copies). Fused GQA (group > 1) batches too —
        # the RaggedTMA3DTile (middle_dim, rows) selector merge keeps it within
        # the 5D TMA limit (rank-5; rank-4 for group==1). Only swizzled-output
        # callers fall back to per-block (0). Shared formula keeps this in sync
        # with the kernel param type.
        # 1Q split-K (reduce-scatter): each partition CTA TMA-stores only its OWN
        # depth-column BAND via per-block `async_copy_from_col`, and the band
        # offset `p*ceil(blocks/P)` is not a {0, half} batched-box boundary for
        # P>=4. So the split-K config needs the PER-BLOCK (rank-3) O-store
        # descriptor, NOT the batched one (which only `async_copy_batched` over
        # the two {0, ceil(blocks/2)} halves can drive). `fa4_splitk_combine_write`
        # infers `tma_bpo==0` from this store and takes its per-block path. Every
        # non-split config keeps the batched store (single-issuer/intra-CTA WG
        # combine, where the {0, half} boxes hold).
        comptime store_blocks_per_op = 0 if fa4_config.splitk_partitions > 1 else o_store_tma_blocks_per_op[
            output_type,
            output_swizzle_mode,
            fa4_config.ov_depth,
            fa4_config.group if fuse_gqa else 1,
            depth_splits=2,
        ]()

        comptime RaggedStoreType = RaggedTMA3DTile[
            output_type,
            output_swizzle_mode,
            BM=BM_per_mma,
            BN=fa4_config.ov_depth,
            middle_dim=fa4_config.num_kv_heads if fuse_gqa else fa4_config.num_q_heads,
            group=fa4_config.group if fuse_gqa else 1,
            tma_blocks_per_op=store_blocks_per_op,
        ]

        var ragged_tma_store = RaggedStoreType.create(
            ctx,
            output.unsafe_ptr(),
            rows=num_rows_q,
        )

        q_tma_op = q_tma[
            swizzle_mode,
            BM=BM_per_mma,
            depth=fa4_config.qk_depth,
            q_num_heads=fa4_config.num_q_heads,
            group=fa4_config.group,
            decoding=False,
            fuse_gqa=fuse_gqa,
            num_qk_stages=fa4_config.num_qk_stages,
        ](ctx, q, num_rows_q)
        # Depth-chunk TMA fold (SM100): fold the BK0 (K) / v_cols_per_cta (V)
        # depth chunks into one rank-4 TMA when byte-equivalent. Each
        # `kv_tma_fold_chunks` is the single source of truth shared with the
        # `tma_copy_k` / `tma_copy_v` issue sites in `load_warp.mojo`; the builder
        # and issue site must pass identical args so the baked descriptor rank and
        # issue-coord rank agree.
        #   K: smem_BN == k_rows_per_cta (K's per-CTA tile_rows); box_rows ==
        #      kv_sub_tile_rows(k_rows_per_cta, page_size).
        #   V: smem_BN == BN (V's tile_rows, num_v_sub_tiles == 1); box_rows ==
        #      kv_sub_tile_rows(BN, page_size).
        comptime k_sub_BN = kv_sub_tile_rows(
            fa4_config.k_rows_per_cta(), KVType.page_size
        )
        comptime k_row_major = fa4_config.k_row_major()
        comptime k_fold_chunks = kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=fa4_config.BK0,
            head_size=fa4_config.qk_depth,
            box_rows=k_sub_BN,
            smem_BN=fa4_config.k_rows_per_cta(),
            page_size=KVType.page_size,
            row_major=k_row_major,
        ]()
        # Producer/consumer agreement: if the Q@K' consumer reads the page-dense
        # (row-major) K layout, the producer MUST have actually folded it
        # (`k_fold_chunks >= 2`). `k_row_major()` mirrors this predicate, so a
        # mismatch here means the two drifted.
        comptime assert (not k_row_major) or (
            k_fold_chunks > 1
        ), "k_row_major() implies the K row-major fold; predicate drift"
        k_tma_op = k.create_tma_tile[
            fa4_config.swizzle_mode,
            BN=k_sub_BN,
            depth=fa4_config.qk_depth,
            BK=fa4_config.BK0,
            fold_chunks=k_fold_chunks,
            row_major=k_row_major,
        ](ctx)
        comptime v_sub_BN = kv_sub_tile_rows(fa4_config.BN, KVType.page_size)
        comptime v_row_major = fa4_config.v_row_major()
        comptime v_fold_chunks = kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=fa4_config.v_cols_per_cta(),
            head_size=fa4_config.ov_depth,
            box_rows=v_sub_BN,
            smem_BN=fa4_config.BN,
            page_size=KVType.page_size,
            row_major=v_row_major,
        ]()
        # Producer/consumer agreement: if the P@V consumer reads the page-dense
        # (row-major) V layout, the producer MUST have actually folded it
        # (`v_fold_chunks >= 2`). `v_row_major()` mirrors this predicate, so a
        # mismatch here means the two drifted.
        comptime assert (not v_row_major) or (
            v_fold_chunks > 1
        ), "v_row_major() implies the V row-major fold; predicate drift"
        v_tma_op = v.create_tma_tile[
            fa4_config.swizzle_mode,
            BN=v_sub_BN,
            depth=fa4_config.ov_depth,
            BK=fa4_config.v_cols_per_cta(),
            fold_chunks=v_fold_chunks,
            row_major=v_row_major,
        ](ctx)
        comptime PairBM_eff = fa4_config.PairBM_eff()
        comptime SchedulerType = TransientScheduler[
            UInt32(PairBM_eff),
            UInt32(
                fa4_config.num_kv_heads if fuse_gqa else fa4_config.num_q_heads
            ),
            flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
            pair_cta=fa4_config.pair_cta,
            splitk_partitions=UInt32(fa4_config.splitk_partitions),
        ]
        var scheduler: SchedulerType = SchedulerType()

        @parameter
        @always_inline
        def with_sink[SinkType: OptionalPointer](sink_ptr: SinkType) raises:
            @parameter
            @always_inline
            def with_kv_offsets[
                KVRowOffsetsType: OptionalPointer
            ](kv_row_offsets: KVRowOffsetsType) raises:
                @parameter
                @always_inline
                def with_valid_length[
                    ValidLengthType: OptionalPointer
                ](valid_len: ValidLengthType) raises:
                    # the pack contains all possibly 0-sized objects
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
                        mask,
                        scheduler,
                        valid_len,
                        sink_ptr,
                        kv_row_offsets,
                        max_prompt_len_arg,
                        partition,
                    }

                    var max_num_prompt_tiles: UInt32 = ceildiv(
                        max_prompt_len_arg.as_uint32(), UInt32(PairBM_eff)
                    )
                    var block_x: UInt32 = max_num_prompt_tiles
                    logger.info(
                        "------ Dispatching to SM100 FMHA-",
                        fa4_config.num_q,
                        "Q ------",
                    )
                    logger.info(
                        "QKV Type:",
                        KVType.dtype,
                        "Depth:",
                        fa4_config.qk_depth,
                        "Number of Q // KV Heads:",
                        fa4_config.num_q_heads,
                        "//",
                        fa4_config.num_kv_heads,
                        "Batch Size:",
                        batch_size,
                        "Max Num Prompt Tiles:",
                        max_num_prompt_tiles,
                    )

                    # Covers the in-kernel 1Q/2Q switch: when
                    # `can_switch_to_1q()` the kernel constructs the 1Q smem
                    # layout over the same dynamic smem region, so this is the
                    # max of both footprints (see `FA4Config.launch_smem_used`).
                    comptime smem_use = fa4_config.launch_smem_used()

                    comptime KernelStruct = SM100MHA2Q[
                        KVType,
                        output_type,
                        MaskType,
                        SchedulerType,
                        fa4_config,
                        ValidLengthType,
                        SinkType,
                        KVRowOffsetsType,
                        _is_cache_length_accurate,
                        MaxPromptLenType,
                        PartitionType,
                    ]
                    # num_q==1 split-K launches a runtime-sized cluster via the
                    # metadata-less `kernel_dyncluster` entry (no static
                    # `nvvm.cluster_dim`); every other config uses the static
                    # `kernel` entry. The launch `cluster_dim` below supplies the
                    # cluster size for the dynamic entry.
                    comptime kernel = KernelStruct.kernel_dyncluster if fa4_config.splitk_dynamic() else KernelStruct.kernel

                    var cluster_dim: OptionalReg[Dim] = None
                    # Unifies pair-CTA (cluster_size==2) and num_q==1 split-K
                    # (cluster_size==P). Pair-CTA uses the comptime cluster_size
                    # (matches the static `nvvm.cluster_dim` metadata on the
                    # `kernel` entry). Dynamic split-K uses the RUNTIME launch
                    # `num_partitions` (<= P_MAX); the metadata-less
                    # `kernel_dyncluster` entry reads it via cluster_dim.x, so
                    # one compiled kernel (P_MAX) serves any launch P in {2,4,8}.
                    comptime if fa4_config.cluster_size() > 1:
                        comptime if fa4_config.splitk_dynamic():
                            cluster_dim = Dim(Int(num_partitions), 1, 1)
                        else:
                            cluster_dim = Dim(fa4_config.cluster_size(), 1, 1)
                    comptime name = String(
                        "nq",
                        fa4_config.num_q,
                        "d",
                        fa4_config.qk_depth,
                        "qh",
                        fa4_config.num_q_heads,
                        "kvh",
                        fa4_config.num_kv_heads,
                        ".",
                    )
                    ctx.enqueue_function[
                        kernel,
                        # dump_llvm=Path(String(name, "ll")),
                        # dump_asm=Path(String(name, "ptx")),
                        # _dump_sass=Path(String(name, "as")),
                    ](
                        q_tma_op,
                        k_tma_op,
                        v_tma_op,
                        ragged_tma_store,
                        k,
                        scale,
                        batch_size,
                        max_cache_valid_length,
                        pack,
                        grid_dim=SchedulerType.grid_dim(
                            batch_size, block_x, num_partitions.as_uint32()
                        ),
                        block_dim=(num_threads, 1, 1),
                        cluster_dim=cluster_dim,
                        shared_mem_bytes=smem_use,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                            UInt32(smem_use)
                        ),
                        attributes=pdl_launch_attributes(MHA_PDL_LEVEL),
                    )

                # --- ragged dispatch ---
                comptime if ragged:
                    with_valid_length[NonNullPointer[DType.uint32]](
                        {valid_length.as_immutable().as_unsafe_any_origin()}
                    )
                else:
                    with_valid_length[NullPointer[DType.uint32]]({})

            # --- kv_input_row_offsets dispatch ---
            if kv_input_row_offsets:
                with_kv_offsets[NonNullPointer[DType.uint32]](
                    {kv_input_row_offsets.value().ptr}
                )
            else:
                with_kv_offsets[NullPointer[DType.uint32]]({})

        # --- sink dispatch ---
        comptime if sink:
            with_sink[NonNullPointer[KVType.dtype]](
                {
                    rebind[UnsafePointer[Scalar[KVType.dtype], ImmutAnyOrigin]](
                        sink_weights.value().ptr
                    )
                }
            )
        else:
            with_sink[NullPointer[KVType.dtype]]({})

    # --- num_q dispatch ---
    # 1Q is only legal when single-CTA and qk_depth in [64, 256]; the
    # comptime gate prevents constructing fa4_config_1q (and its
    # supported() assert) on shapes 1Q can't run. Outside the gate
    # we unconditionally use 2Q.
    comptime can_use_1q: Bool = (
        not pair_cta and config.depth >= 64 and config.depth <= 256
    )
    comptime if can_use_1q:
        # Dynamic cluster dims (Stage B): compile the num_q==1 split-K kernel
        # ONCE at the portable cluster ceiling `P_MAX` (the 2-SM cap = 8) and
        # choose the partition count at LAUNCH via the runtime `cluster_dim`.
        # The count is AUTO-selected below from occupancy + KV length; there is
        # no compile-time `P` knob. `num_partitions` is an `OptionallyStaticInt`
        # so the 2Q config (and its in-kernel 1Q switch) carry a static `1` and
        # fold the split-K launch/grid/combine away, while the genuine 1Q split
        # path carries the runtime count via `DynamicInt`.
        comptime P_MAX = 4
        comptime fa4_config_1q = fa4_config_2q.with_num_q(1)
        comptime fa4_config_1q_splitk = fa4_config_1q.with_splitk(P_MAX)
        comptime assert (
            fa4_config_1q_splitk.supported()
        ), fa4_config_1q_splitk.description()

        # Heuristic: pick 1Q when (a) max_prompt_len <= 128 (so 2Q's
        # BM=256 would waste >= 50% of Q rows) or (b) the unclamped
        # 2Q grid only fills <= half the SMs, so halving BM doubles
        # the grid without oversubscribing.
        var max_prompt_len_u32: UInt32 = max_prompt_len_arg.as_uint32()
        var max_num_prompt_tiles_2q: UInt32 = ceildiv(
            max_prompt_len_u32, UInt32(fa4_config_2q.PairBM_eff())
        )
        comptime num_heads_sched_2q: UInt32 = UInt32(
            fa4_config_2q.num_kv_heads if fa4_config_2q.fuse_gqa else fa4_config_2q.num_q_heads
        )
        var raw_grid_2q: UInt32 = (
            max_num_prompt_tiles_2q * num_heads_sched_2q * batch_size
        )
        var sm_count: UInt32 = UInt32(
            ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        )
        var grid_threshold: UInt32 = UInt32(sm_count // 2)
        comptime bm_eff_1q = UInt32(fa4_config_1q_splitk.BM_eff())
        if max_prompt_len_u32 <= bm_eff_1q or raw_grid_2q <= grid_threshold:
            # Auto-select the split-K partition count.
            #   by_grid : how many extra CTAs we can add before oversubscribing
            #             (the 1Q grid widens x by this factor over the 2Q grid).
            #   by_cache: don't split a short KV into more partitions than there
            #             is K/V to go around (512 keys / partition, mirroring
            #             the decode split-K `//512` granularity); this avoids
            #             launching mostly-empty clusters.
            # Floor to a power of two in {1,2,4,8}: the B200-verified combine /
            # window envelope. (Non-pow2 P is correct by construction --
            # writer-combines-all and `splitk_window` handle any P -- but only
            # pow2 has been swept on hardware.)
            #
            # 128 x 12 or 256 x 12:
            # grid_2q = 12
            # grid_threshold = 74
            # 128: 74/12 = 6
            # 256: 74/12 = 6
            # 512: 74/24 = 3
            # better:
            # 128: 148/12 = 12
            # 256: 148/24 = 6
            # 512: 148/48 = 3
            #
            # (np-1)/np
            var max_num_prompt_tiles_1q: UInt32 = ceildiv(
                max_prompt_len_u32, UInt32(fa4_config_1q_splitk.PairBM_eff())
            )
            var raw_grid_1q: UInt32 = (
                max_num_prompt_tiles_1q * num_heads_sched_2q * batch_size
            )
            var by_grid: UInt32 = sm_count // max(raw_grid_1q, UInt32(1))
            var by_cache: UInt32 = max_cache_valid_length // UInt32(512)
            var np: UInt32 = min(min(by_grid, by_cache), UInt32(P_MAX))
            var num_partitions = 1 if np <= 1 else prev_power_of_two(np)
            # P==1 has nothing to combine: route to the 2Q config, whose
            # in-kernel 1Q switch (`switch_1q_config`) runs short / low-occupancy
            # sequences as a static single partition -- no cluster, no DSMEM, no
            # combine.
            if num_partitions >= 2:
                with_fa4_config[fa4_config_1q_splitk](
                    DynamicInt(Int(num_partitions))
                )
            else:
                with_fa4_config[fa4_config_1q](StaticInt[1]())
        else:
            with_fa4_config[fa4_config_2q](StaticInt[1]())
    else:
        with_fa4_config[fa4_config_2q](StaticInt[1]())
