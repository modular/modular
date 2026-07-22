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

"""
Provides the SM100 (Blackwell) flash-attention host-side dispatch entry point
that selects a 1Q or 2Q FA4 kernel configuration, builds the Q/K/V/O TMA
descriptors and tile scheduler, and enqueues the kernel onto the device.
"""

from std.collections import OptionalReg
from std.math import ceildiv
from std.sys import get_defined_int
from std.gpu.primitives.grid_controls import pdl_launch_attributes
from std.gpu.host import (
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
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import TransientScheduler
from nn.attention.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    StaticInt,
    _is_decoding,
)
from .attention_utils import (
    clusters_per_wave,
    kv_sub_tile_rows,
    kv_tma_fold_chunks,
    o_store_tma_blocks_per_op,
)
from .kernel import SM100MHA2Q

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
    """Dispatches the SM100 FA4 flash-attention kernel for a prefill workload.

    Selects between the 1Q split-K and 2Q FA4 configurations based on a
    occupancy and prompt-length heuristic, constructs the Q/K/V/O TMA tile
    descriptors and transient tile scheduler, threads optional ragged
    valid-length, KV-row-offset, and sink-attention arguments through to the
    compiled kernel, and enqueues the launch onto the supplied device context.

    Parameters:
        q_type: Element type of the query tensor (inferred).
        KVType: Key/value operand descriptor with dtype and page size
            (inferred).
        MaskType: Attention mask scheme applied to the Q@K' scores (inferred).
        output_type: Element type of the attention output buffer (inferred).
        MaxPromptLenType: Optionally-static type encoding the maximum prompt
            length (inferred).
        config: MHA configuration supplying dtype, head count, depth, and
            swizzle mode.
        group: Number of query heads per KV head (GQA group size).
        ragged: Whether to dispatch the variable-length valid-length path.
        sink: Whether to thread sink-attention weights into the kernel.
        _is_cache_length_accurate: Whether the supplied cache length is
            accurate, threaded to the compiled kernel.
        pair_cta: Whether to launch the pair-CTA configuration with a cluster
            size of 2.

    Args:
        output: Device buffer that receives the attention output rows.
        q_arg: Pointer to the query tensor data.
        k: Key operand descriptor.
        v: Value operand descriptor.
        num_rows_q: Number of query rows to attend over.
        mask: Attention mask applied to the Q@K' scores.
        valid_length: Per-row valid KV length pointer, used when `ragged` is set.
        max_prompt_len_arg: Maximum prompt length, optionally static.
        max_cache_valid_length_arg: Maximum valid KV cache length across the batch.
        scale: Scalar applied to the Q@K' product before softmax.
        kv_input_row_offsets: Optional per-row KV input offsets for ragged layouts.
        batch_size_arg: Number of sequences in the batch.
        ctx: Device context used to build TMA descriptors and enqueue the kernel.
        sink_weights: Optional sink-attention weights used when `sink` is set.
    """
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
        # `num_partitions` is ALWAYS static: the 2Q config (and its in-kernel 1Q
        # switch) carry a static 1, and each num_q==1 split-K config is compiled
        # once per static partition count `P` (cluster size `P`) — the dispatch
        # picks which `P` kernel to launch at runtime. So the cluster size is a
        # comptime constant that matches the kernel's static `nvvm.cluster_dim`.
        comptime assert Bool(
            NumPartitionsType.static_value
        ), "split-K num_partitions must be static (compiled once per P)"
        comptime swizzle_mode = fa4_config.swizzle_mode
        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # Q/K/V/S/P buffers governed by `swizzle_mode`). The softmax warp loads
        # O one-row-per-thread and writes it row-major, avoiding cross-thread
        # shuffles and swizzling while staying bank-conflict-free.
        comptime output_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE
        comptime BM = fa4_config.BM
        comptime fuse_gqa = fa4_config.fuse_gqa
        comptime num_threads = fa4_config.num_threads
        # `MMA_M // cta_group` drives q_tma_op and ragged_tma_store BM under a
        # unified expression: 128 for 2Q single-CTA (128 // 1), 2Q pair-CTA
        # (256 // 2), and 1Q single-CTA (128 // 1); 32 for the WS BM=32 config
        # (MMA_M=32 // 1).
        comptime BM_per_mma = fa4_config.MMA_M // fa4_config.cta_group()
        comptime assert BM == 32 or BM == 128 or BM == 256

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
        # The WS (MMA_M=32) combine likewise TMA-stores from WG0 via the
        # PER-BLOCK `fa4_tma_store_o_smem` (the B200-verified egress), so it too
        # needs the rank-3 store rather than the batched {0, half} box.
        comptime store_blocks_per_op = 0 if (
            fa4_config.splitk_partitions > 1 or fa4_config.use_ws
        ) else o_store_tma_blocks_per_op[
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
        # WS shared sub-tile ring: V depth-tile box (256x64), one TMA per
        # depth-tile. MUST match load_warp's `v_tma_op` BK + `v_fold_chunks`
        # exactly (single source of truth via `v_box_cols()`). Folds to the
        # full-depth V box for non-WS (which loads V whole).
        comptime v_sub_cols = fa4_config.v_box_cols()
        comptime v_fold_chunks = kv_tma_fold_chunks[
            KVType.dtype,
            fa4_config.swizzle_mode,
            BK=v_sub_cols,
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
            BK=v_sub_cols,
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
                    # Every config uses the static `kernel` entry; the cluster
                    # size is baked into its `nvvm.cluster_dim` metadata.
                    comptime kernel = KernelStruct.kernel

                    var cluster_dim: OptionalReg[Dim] = None
                    # Unifies pair-CTA (cluster_size==2) and num_q==1 split-K
                    # (cluster_size==P): both are the comptime `cluster_size()`,
                    # matching the static `nvvm.cluster_dim` metadata on the
                    # `kernel` entry (split-K is compiled once per static P).
                    comptime if fa4_config.cluster_size() > 1:
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
                    ctx.enqueue_function[kernel](
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
        # Static per-C split-K: the num_q==1 split-K kernel is compiled ONCE per
        # candidate partition count `P` (== cluster size) in SPLITK_CANDIDATES
        # (below); the dispatch picks which `P` to LAUNCH from occupancy + KV
        # length.
        # `num_partitions` is an `OptionallyStaticInt` carrying a static `P` (or
        # a static `1` for the 2Q config and its in-kernel 1Q switch), so the
        # cluster size / grid / combine are all comptime constants -- there is no
        # runtime cluster dimension.
        # Candidate split-K cluster sizes, scanned LARGEST-first; the
        # dispatch picks the largest that fits one wave (see the scan
        # below). Deliberately a COARSE ladder: 2 (fills all 148 SMs), 4
        # (132 SMs), 10 (110 SMs, for long-cache / high-partition shapes).
        # Each entry compiles a distinct static-P kernel, and the LARGE-P
        # kernels dominate compile time -- their cross-CTA combine / DSMEM
        # basic blocks are the biggest and most expensive to codegen (worst
        # at small page_size). Dropping 16, 8, 6 from the earlier
        # {16,10,8,6,4,2} set cut the worst-target compile ~5x (ps128_hs128
        # ~2240s -> ~480s) for <=~1.5% perf at the single transition it
        # changes (the P8->P4 pick around cache 4k-10k on a small grid;
        # long-cache shapes still get P10). P=10 exceeds the portable
        # cluster cap (8) and is non-portable, but the runtime sets
        # NON_PORTABLE_CLUSTER_SIZE_ALLOWED on every function load, so no
        # launch plumbing is needed (CUDADeviceContext::loadFunction).
        # Every candidate must be admitted by FA4Config.supported()
        # (asserted per-candidate in the scan) -- in particular, all
        # candidates are EVEN (the combine's SIMD-2 weight-normalize loop
        # requires it). To widen occupancy coverage again, re-add entries
        # (e.g. 6, 8, 16) at the cost of compile time; the scan / combine
        # math is P-general for any even P in {2, 4, 6, 8, 10, 16}.
        comptime fa4_config_1q = fa4_config_2q.with_num_q(1)
        # Split-K kernels reject UNKNOWN_MASK masks at compile time
        # (softmax_warp), so the candidate list stays empty for them.
        comptime splitk_mask_ok = MaskType.nonfull_sets[
            fa4_config_1q.PairBM_eff(), fa4_config_1q.BN
        ]()[0] != TileMaskStatus.UNKNOWN_MASK
        comptime SPLITK_CANDIDATES = (
            [10, 4, 2] if splitk_mask_ok else List[Int]()
        )

        # The GPC fragmentation model (`clusters_per_wave`) covers B200 (148 SMs)
        # and B300 (160 SMs); its LUT folds at comptime keyed on
        # `ctx.default_device_info.sm_count` (a comptime value). The helper's own
        # `else` branch is the single source of truth for an unsupported chip, so
        # no standalone assert is needed here.

        # 1Q-vs-2Q gate (unchanged): pick 1Q when (a) max_prompt_len fits one 1Q
        # tile (2Q's BM=256 would waste >= 50% of Q rows) or (b) the unclamped 2Q
        # grid only fills <= half the SMs, so halving BM doubles the grid without
        # oversubscribing.
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
        # SM count is the comptime compile-target value (same source the
        # `clusters_per_wave` wave-fit uses below via `default_device_info`),
        # not a runtime `get_attribute` query -- keeps the 1Q/2Q grid gate and
        # the split-K wave-fit reasoning off one consistent occupancy model.
        comptime sm_count: UInt32 = UInt32(ctx.default_device_info.sm_count)
        comptime grid_threshold: UInt32 = sm_count // 2
        # `BM_eff`/`PairBM_eff` are P-independent (split-K forces cta_group==1),
        # so the 1Q geometry uses `fa4_config_1q` directly.
        comptime bm_eff_1q = UInt32(fa4_config_1q.BM_eff())
        # Config-force override (bench / test only; default 0 = auto, so
        # production folds to the shape-driven selection below, byte-identical).
        # Config selection is otherwise purely shape-driven with no way to run WS
        # vs the path it replaces at a *fixed* shape, which is exactly what the WS
        # benchmark (Phase C3) and the reroute-pinned tests (C5) need. Read inside
        # this generic `def` so it elaborates with the consuming test/bench's
        # `-D FA4_FORCE_CONFIG=...` copts (same mechanism as softmax_warp's
        # `FA4_1Q_SPLITK_WRITER`).  1 = force WS (bypass the prompt-len/grid test,
        # still comptime-gated on `supported()`+`ws_mask_ok`); 2 = force the
        # baseline 1Q/split-K/2Q carve (skip the WS route entirely).
        comptime FA4_FORCE_CONFIG = get_defined_int["FA4_FORCE_CONFIG", 0]()
        # Warp-specialized BM=32 packed-TMEM datapath for very short prompts:
        # the BM=32 tile holds `group` q-heads x BM_eff seq positions under
        # fuse_gqa (BM_eff = 32 // group; 8 at group=4), and the 8-way intra-CTA
        # split-K
        # (2 softmax WGs x 4 packed-TMEM quarters) extracts the parallelism from
        # the KV reduction. Scoped to `depth in {64, 128}`: both
        # depths satisfy the shared sub-tile-ring invariant `256 // m_pack ==
        # BK0 == 64` (depth 128 -> num_qk_stages=2, depth 64 -> num_qk_stages=1;
        # BK0 = padded_qk_depth // num_qk_stages = 64 either way, P@V folds to
        # num_d_tiles=1 at depth 64). Sinks ARE supported: the intra-CTA fold is
        # confined to WG0 quarter 0 (softmax_warp.mojo `fold_sink and warp_idx
        # == 0`), so the 8-way split adds the sink mass exactly once -- the same
        # partition-0 discipline as split-K. `supported()` then prunes rope /
        # KV-scale / non-uniform-sub-tile shapes so dispatch degrades to the
        # 1Q / split-K path below. Routed BEFORE the `<= bm_eff_1q` split-K
        # carve so WS-eligible shapes (see the single-tile enablement) take WS
        # first; everything else falls through to the 1Q carve.
        comptime if config.depth == 128 or config.depth == 64:
            comptime fa4_config_ws = fa4_config_2q.with_bm(32)
            # Cross-CTA cluster split-K over the WS BM=32 config: each partition
            # count P groups P single-CTA WS kernels in a launch cluster that
            # partition the KV sequence and DSMEM-combine (a THIRD level, on top
            # of the 8-way intra-CTA split). Following #92167, each P compiles its
            # OWN static kernel (`StaticInt[P]`, `cluster_size() == P`) -- there is
            # no dynamic-cluster entry. Production auto-sizes P from the single-tile
            # scan below; `-D FA4_WS_SPLITK_FORCE=P` pins P for the split-K
            # correctness / bench targets. The WS route reuses the shared
            # `SPLITK_CANDIDATES` set (same `clusters_per_wave` wave-fit guard as
            # the 1Q carve): for P > m_pack (== 4) only `m_pack` partitions own a
            # depth band to write, but all P still partition the KV and are reduced
            # into the owning bands, so wider P raises SM utilization the same way
            # it does on the 1Q path (`fa4_ws_splitk_reduce_scatter_write`).
            comptime FA4_WS_SPLITK_FORCE = get_defined_int[
                "FA4_WS_SPLITK_FORCE", 0
            ]()
            # Pin the force knob to a supported WS split-K P (the even members of
            # `SPLITK_CANDIDATES`); 0/1 -- and any unsupported value -- => single-CTA WS.
            comptime ws_P_force = FA4_WS_SPLITK_FORCE if (
                FA4_WS_SPLITK_FORCE == 2
                or FA4_WS_SPLITK_FORCE == 4
                or FA4_WS_SPLITK_FORCE == 6
                or FA4_WS_SPLITK_FORCE == 8
                or FA4_WS_SPLITK_FORCE == 10
                or FA4_WS_SPLITK_FORCE == 16
            ) else 1
            # WS is validated only for masks whose visible range is statically
            # known and contiguous (`nonfull_sets[0] != UNKNOWN_MASK`: Null,
            # Causal, Chunked, SlidingWindow). Materialized/And/Or masks report
            # `{UNKNOWN_MASK}` and would take the WS softmax runtime-status path,
            # whose per-quarter `mask.status(...)` is issued over the 256-wide
            # tile window rather than the 64-wide packed-TMEM quarter -- correct
            # by superset but unverified on WS. Route them to the proven non-WS
            # 1Q / split-K / 2Q path instead. (Belt-and-suspenders today: these
            # masks are also blocked from this whole dispatch by the 1Q split-K
            # `UNKNOWN_MASK` comptime assert in `fa4_softmax`; this predicate
            # keeps them off the WS route if that block is ever lifted.)
            comptime ws_mask_ok = MaskType.nonfull_sets[
                fa4_config_ws.PairBM_eff(), fa4_config_ws.BN
            ]()[0] != TileMaskStatus.UNKNOWN_MASK
            # No cache-length gate. The WS 1Q shared-ring main loop
            # (`main_iters >= 1`, first exercised at T >= 4) is correct as of the
            # correction-SMEM sizing fix (2026-07-15: the region is sized by
            # softmax-thread count `2*WARPGROUP_SIZE`, not `BM`; see smem.mojo);
            # B200 matrix green for T in {1,2,3,5} across {Null,Causal,Chunked,
            # SlidingWindow} x depth{64,128}. Short prompt + long cache therefore
            # routes to single-CTA WS here. The perf guard that keeps a
            # huge-cache/short-prompt shape off single-CTA WS -- route WS only
            # while its finer BM=32 grid stays under full-SM occupancy -- is the
            # Phase C `ws_grid < sm_count` rule, landed with the WS benchmark.
            comptime if (
                fa4_config_ws.supported()
                and ws_mask_ok
                and FA4_FORCE_CONFIG != 2
            ):
                # Each WS split-K partition count P compiles its OWN static
                # single-CTA WS kernel (`StaticInt[P]`, `cluster_size() == P`),
                # mirroring the 1Q carve -- #92167 removed the dynamic-cluster
                # entry, so P is a comptime constant baked into `nvvm.cluster_dim`.
                comptime if FA4_FORCE_CONFIG == 1 or ws_P_force >= 2:
                    # Force override (bench C3 / pinned split-K tests): run WS
                    # regardless of prompt length / grid. P is the explicit pin
                    # `ws_P_force` (1 => single-CTA WS); each P is its own static
                    # kernel so forced runs stay deterministic. FA4_FORCE_CONFIG==1
                    # with no split-K pin runs single-CTA WS; a `ws_P_force >= 2`
                    # pin (with FORCE in {0, 1}) runs the static-P WS split-K.
                    comptime if ws_P_force >= 2:
                        comptime fa4_config_ws_splitk = fa4_config_ws.with_splitk(
                            ws_P_force
                        )
                        comptime assert (
                            fa4_config_ws_splitk.supported()
                        ), fa4_config_ws_splitk.description()
                        with_fa4_config[fa4_config_ws_splitk](
                            StaticInt[ws_P_force]()
                        )
                    else:
                        with_fa4_config[fa4_config_ws](StaticInt[1]())
                    return
                else:
                    # FORCE=0 production auto: route WS iff the whole prompt fits
                    # ONE WS tile -- `BM_eff >= max_prompt_len`, where fuse_gqa
                    # packs `group` q-heads into the BM=32 tile so one tile spans
                    # BM_eff = BM // group SEQ positions (8 for group=4), NOT 32.
                    # Beyond one tile the prompt shatters into ceildiv(seq, BM_eff)
                    # WS tiles (seq=32 => 4, seq=48 => 6) vs baseline's 1-2 BM=32
                    # tiles; at long cache those extra KV passes lose to baseline
                    # even when the grid is small, so an occupancy-only check
                    # over-routed the mid-seq/small-batch corner (measured
                    # seq=48/batch=1: 0.70-0.88x at cache>=8192). Single-tile WS
                    # never pays that shatter tax and still fills the SMs via the
                    # cross-CTA split-K candidate scan below.
                    var max_num_prompt_tiles_ws: UInt32 = ceildiv(
                        max_prompt_len_u32,
                        UInt32(fa4_config_ws.PairBM_eff()),
                    )
                    var raw_grid_ws: UInt32 = (
                        max_num_prompt_tiles_ws
                        * num_heads_sched_2q
                        * batch_size
                    )
                    if UInt32(fa4_config_ws.BM_eff()) >= max_prompt_len_u32:
                        # Production auto cross-CTA split-K (mirrors the 1Q carve
                        # below): pick the LARGEST WS P whose `raw_grid_ws` size-P
                        # clusters all fit ONE wave given GPC fragmentation
                        # (`clusters_per_wave`) AND that has enough KV per partition
                        # to amortize its cross-CTA combine. Scan descending; the
                        # first fit is the largest and returns. P==1 (nothing fit /
                        # short cache) => single-CTA WS.
                        comptime for C in SPLITK_CANDIDATES:
                            comptime ws_splitk_cfg = fa4_config_ws.with_splitk(
                                C
                            )
                            comptime assert (
                                ws_splitk_cfg.supported()
                            ), ws_splitk_cfg.description()
                            comptime fits_wave = UInt32(
                                clusters_per_wave[
                                    C, ctx.default_device_info.sm_count
                                ]()
                            )
                            # Min KV keys/partition to admit cluster size C. The WS
                            # BM=32 combine's cross-CTA DSMEM cost grows with C, so
                            # the LARGE clusters (C >= 10) only pay off with more KV
                            # per partition: require 1024 keys/partition vs 512 for
                            # C <= 8. Measured on B200 (g16_d128, batch=1): P8 beats
                            # P16 through cache=16384 and P16's cross-CTA combine only
                            # wins from ~24576, but the flat 512 floor (kept by the 1Q
                            # carve) let P16 grab the cache=8192 boundary at +3.8% vs
                            # the P8 optimum. This keeps 4096->P8, 2048->P4 while
                            # gating P16 to cache >= 16384.
                            comptime ws_min_kpp = 1024 if C >= 10 else 512
                            if (
                                UInt32(C)
                                <= max_cache_valid_length // UInt32(ws_min_kpp)
                                and raw_grid_ws <= fits_wave
                            ):
                                with_fa4_config[ws_splitk_cfg](StaticInt[C]())
                                return
                        with_fa4_config[fa4_config_ws](StaticInt[1]())
                        return
        if max_prompt_len_u32 <= bm_eff_1q or raw_grid_2q <= grid_threshold:
            var max_num_prompt_tiles_1q: UInt32 = ceildiv(
                max_prompt_len_u32, UInt32(fa4_config_1q.PairBM_eff())
            )
            # Number of split-K CLUSTERS the launch needs (one per work item);
            # each is a size-`P` cluster occupying `P` SMs within a single GPC.
            var raw_grid_1q: UInt32 = (
                max_num_prompt_tiles_1q * num_heads_sched_2q * batch_size
            )
            # Don't split a short KV into more partitions than there is K/V to go
            # around (512 keys / partition, mirroring decode split-K's `//512`);
            # avoids launching mostly-empty clusters.
            var by_cache: UInt32 = max_cache_valid_length // UInt32(512)
            # Pick the LARGEST P (<= by_cache) whose `raw_grid_1q` size-`P`
            # clusters ALL fit in ONE wave given GPC fragmentation: only
            # `clusters_per_wave[P]` size-`P` clusters fit per device. Scan
            # candidates descending (16, 10, 8, 6, 4, 2); the first that fits is
            # the largest, and the `return` keeps it to a single enqueue. (Only
            # pow2 P has been HW-swept; the combine / `splitk_window` math is
            # correct for any even P.)
            comptime for C in SPLITK_CANDIDATES:
                comptime splitk_cfg = fa4_config_1q.with_splitk(C)
                comptime assert splitk_cfg.supported(), splitk_cfg.description()
                comptime fits_wave = UInt32(
                    clusters_per_wave[C, ctx.default_device_info.sm_count]()
                )
                if UInt32(C) <= by_cache and raw_grid_1q <= fits_wave:
                    with_fa4_config[splitk_cfg](StaticInt[C]())
                    return
            # P==1 (nothing fit / nothing to combine): launch the 1Q
            # single-partition config -- no cluster, no DSMEM, no combine.
            with_fa4_config[fa4_config_1q](StaticInt[1]())
        else:
            with_fa4_config[fa4_config_2q](StaticInt[1]())
    else:
        with_fa4_config[fa4_config_2q](StaticInt[1]())
