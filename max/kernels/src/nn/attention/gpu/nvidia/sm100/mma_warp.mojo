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
"""MMA warp logic for FA4 (SM100 Flash Attention)."""

from std.math import align_up, ceildiv
from std.sys import size_of, get_defined_bool
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    ExplicitTMEMCrossP,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    blasst_vote_unanimous,
    elect,
    elect_mma_arrive,
    SM100TensorAccumulator,
    KConsumerPipeline,
    VConsumerPipeline,
    StagedPipeline,
    MBarType,
    splitk_window,
    splitk_partition_idx,
    splitk_num_partitions,
)
from nn.attention.mha_mask import MHAMask
from linalg.arch.sm100.mma import smem_descriptor
from .smem import SM100AttentionSMem


@always_inline
def fa4_mma[
    MaskType: MHAMask,
    //,
    config: FA4Config,
    *,
    page_size: Int,
    # Workspace (traditional/unfused) split-K: window the KV by a RUNTIME
    # partition count even at `config.splitk_partitions == 1`. Defaulted so
    # every non-workspace caller is byte-identical.
    workspace_split: Bool = False,
    # Effective cross-stage-P switch, computed once at the kernel level where
    # config, MaskType and the store shape are all visible. Defaults True so
    # the config-level gate alone decides for callers that do not thread it;
    # the MHA kernel passes its own narrower predicate.
    crossp_effective: Bool = True,
](
    smem: SM100AttentionSMem[config],
    tmem_addr: UInt32,
    seq_id: UInt32,
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
    ws_num_partitions: UInt32 = 1,
):
    """Executes the FA4 MMA warp loop for SM100 Flash Attention.

    Computes Q@K' into score TMEM (S0/S1) and P@V into output TMEM (O0/O1)
    across the KV tile sequence, coordinating with the load and softmax
    warps via barrier pipelines. Supports fused-KV and split-KV modes,
    single-Q and two-Q tile configurations, and partial-K contraction for
    paged sub-tiles (`page_size` zero disables paging; sub-BN enables
    partial-K).

    Args:
        smem: Shared memory allocator holding Q, K, V tiles and barriers.
        tmem_addr: Base TMEM address for this CTA's S/O accumulators.
        seq_id: Sequence index for the current query batch.
        score_row: Row offset of the query tile within the sequence.
        num_keys: Number of valid key columns to attend to.
        mask: Attention mask controlling tile iteration and masking bounds.
        ws_num_partitions: Runtime split-K partition count that windows the
            KV range when `workspace_split` is `True`; `1` disables windowing.
    """
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime num_q: Int = config.num_q
    comptime BM_mask: Int = config.PairBM_eff()
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime cta_group: Int = config.cta_group()

    # Mb: WS packed P@V (comptime-dead until Md/Layout-E elaborates a BM=32/64
    # config). Layout-G (MMA_M=32, m_pack=4) DEPTH-SCATTERS: the P@V op packs
    # `m_pack` key-quarters x `depth_tile` output cols into one MMA_N=256 tile
    # (F16 caps N at 256), so the full `padded_ov_depth` is covered by
    # `num_d_tiles` depth-tiled MMAs at C stride `depth_tile`, each writing a
    # DISJOINT output column range. Layout-E (MMA_M=64, m_pack=2) instead
    # REDUCTION-SPLITS: V is split along the KEY axis (not depth) into
    # `num_qk_stages` chunks of `pv_bk_chunk` keys each, packing the FULL
    # `m_pack*padded_ov_depth` per stack into one MMA_N tile; `num_d_tiles` is
    # repurposed as the reduction-chunk loop trip count and every chunk MMAs
    # into the SAME (un-offset) `o_tmem` (accumulating, not scattering) -- see
    # docs/plans/sm100-fa4-layout-e-mma64.md "the reduction-split geometry".
    # Every value folds to today's literal for non-WS (m_pack==1) -> the
    # non-WS issue stream is byte-identical. `depth_tile`/non-reduction-split
    # mirrors the consumer's derivation (softmax_warp.mojo:507-508).
    comptime is_reduction_split = config.use_ws and config.m_pack == 2
    comptime depth_tile = (
        256 // config.m_pack
    ) if config.use_ws else config.padded_ov_depth
    comptime num_d_tiles = (
        config.num_qk_stages if is_reduction_split else (
            config.padded_ov_depth // depth_tile
        ) if config.use_ws else 1
    )
    comptime pv_mma_n = (
        config.m_pack
        * config.padded_ov_depth if is_reduction_split else (
            config.m_pack * depth_tile
        ) if config.use_ws else config.padded_ov_depth
    )
    # Layout-E's reduction-chunk width: a partition's `BN // m_pack` owned
    # keys split `num_qk_stages` ways (=64 @depth128, matching Layout-G's own
    # `BN // m_pack` @ m_pack=4 -- a coincidence of today's depth-128 numbers,
    # not an identity that holds generally). This is exactly the V TMA box's
    # KEY-row count, so it reads from that one definition.
    comptime pv_bk_chunk = config.v_e_chunk_rows()
    comptime pv_bk = (
        pv_bk_chunk if is_reduction_split else (
            BN // config.m_pack
        ) if config.use_ws else BN
    )
    # Layout-E ONLY: P/S physically occupies `score_cols == BN//m_pack`
    # columns (one partition's FULL key range), but `UMMA1Type`'s BK ==
    # `pv_bk_chunk` covers only ONE reduction chunk's worth -- `stage_idx`
    # (p_stage) only offsets the A (P) operand WITHIN a `pv_bk_chunk`-wide
    # window at a manually-passed-in base (`attention_utils.mojo`'s
    # `a_tmem_offset = (k_offset*operand_size)//4`, relative to `a`, not to
    # any reduction-chunk notion). Without also advancing the BASE per `r`,
    # every reduction chunk would silently re-read chunk 0's P columns.
    # `s_tmem_chunk_words`: `pv_bk_chunk` KEYS worth of TMEM columns (4-byte
    # units, mirrors the accumulator's own `a_tmem_offset` conversion) = 32
    # @depth128 (64 keys * 2-byte bf16 // 4). Dead (unused) for Layout-G,
    # where `s_tmem` never advances per `v_stage` (a depth axis there).
    comptime s_tmem_chunk_words = (
        pv_bk_chunk * size_of[config.qkv_dtype]()
    ) // 4
    # Byte stride between adjacent V depth-tile regions in SMEM (one packed
    # `[pv_mma_n, pv_bk]` mn-major region). Only referenced in `_pv_full`'s
    # `v_stage > 0` branch; `_pv_full` is used only where `num_d_tiles == 1`
    # (non-WS), so this stays dead there. = 32768 @depth128.
    comptime v_region_bytes = pv_mma_n * pv_bk * size_of[config.qkv_dtype]()
    comptime if config.use_ws and not is_reduction_split:
        comptime assert (
            config.padded_ov_depth % depth_tile == 0
        ), "Mb: padded_ov_depth must be a multiple of depth_tile"

    var mbars = smem.misc_mbars()
    comptime mma_kind = (
        UMMAKind.KIND_F8F6F4 if config.qkv_dtype.is_float8() else UMMAKind.KIND_F16
    )

    # MMA types
    comptime UMMA0Type = SM100TensorAccumulator[
        config.qkv_dtype,
        accum_type,
        MMA_M=config.MMA_M,
        MMA_N=BN,
        BK=align_up(config.qk_depth, config.MMA_K),
        a_tmem=False,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        cta_group=cta_group,
        num_stages=num_qk_stages,
        mma_kind=mma_kind,
        b_page_dense=config.k_row_major(),
    ]
    comptime UMMA1Type = SM100TensorAccumulator[
        config.qkv_dtype,
        accum_type,
        MMA_M=config.MMA_M,
        MMA_N=pv_mma_n,
        BK=pv_bk,
        a_tmem=True,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        cta_group=cta_group,
        num_stages=num_pv_stages,
        mma_kind=mma_kind,
        b_page_dense=config.v_row_major(),
        # Layout-E's reduction-split P@V wants an EVEN 2-then-2 split of each
        # reduction chunk's own BK (so the P sub-stage chunks align with the
        # V reduction chunks); Layout-G keeps the default 3-then-1 (unchanged).
        allow_3_then_1_split=not is_reduction_split,
    ]

    # Runtime-k partial-page gate. Only the last KV tile can be partially
    # loaded (paged sub-tiles, page_size < BN), and skipping its unloaded V
    # tail in P@V avoids reading uninitialized SMEM (`0 * NaN = NaN`).
    # supported() guarantees page_size % MMA_K == 0 here, so the loaded
    # boundary is MMA_K-aligned and the cut is exact.
    comptime PARTIAL_K = page_size > 0 and page_size < BN

    # `tmem_addr` passed in by register (read once post-barrier in the kernel
    # prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var q_smem = smem.q_smem()

    var s0_tmem = tmem_addr + UInt32(config.TMEM_S0)
    var s1_tmem = tmem_addr + UInt32(config.TMEM_S1)
    var o0_tmem = tmem_addr + UInt32(config.TMEM_O0)
    var o1_tmem = tmem_addr + UInt32(config.TMEM_O1)

    # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
    var pipeline_s0 = mbars.producer_s0()
    var pipeline_s1 = mbars.producer_s1()
    # Keep consumer pointers for acquire operations (shared phase tracking)
    var consumer_s0 = pipeline_s0.consumer_mbar_base
    var consumer_s1 = pipeline_s1.consumer_mbar_base

    # O pipelines (producer side only; consumer wait is merged into S barriers)
    var pipeline_o0 = mbars.producer_o0()
    var pipeline_o1 = mbars.producer_o1()

    # Per-Q-tile element/byte counts.
    # 2Q: HalfBM * padded_qk_depth = 128 * d (one of two Q halves).
    # 1Q: BM * padded_qk_depth = 128 * d (the single full-BM Q tile).
    # Numerically identical because BM // num_q == 128 in both modes
    # (mirrors the load_warp.mojo Q-TMA invariance).
    comptime q0_size = (BM // num_q) * config.padded_qk_depth
    comptime q0_bytes = q0_size * size_of[config.qkv_dtype]()
    var q0 = smem_descriptor[
        BMN=config.BM // config.num_q,
        BK=config.BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](q_smem)
    var q1 = q0 + UInt32(q0_bytes)

    comptime q_sub_bytes = HalfBM * config.BK0 * size_of[config.qkv_dtype]()

    # WS Q depth-chunk stride: the full Q tile (q0_bytes) split into
    # `num_qk_stages` equal depth chunks. In the shared sub-tile ring the two
    # K depth-halves live in separate ring slots, so Q@K' is two
    # `mma[stage_idx=0]` and the second reads Q at `q0 + q_chunk_bytes`. Folds
    # to `q0_bytes` for num_qk_stages==1 (unused there). NOT `q_sub_bytes`
    # (=HalfBM*BK0*size, the 2Q row-half stride on the wrong axis).
    comptime q_chunk_bytes = q0_bytes // config.num_qk_stages

    var e = elect()

    # BLASST (arXiv 2512.12087): skip P@V (UMMA1) when all 4 of a WG's softmax
    # warps voted to skip; QK' (UMMA0) and the V load are never skipped. `and
    # not config.use_ws` since main's `_pv_ws` P@V path has no BLASST analog.
    comptime ENABLE_BLASST = get_defined_bool[
        "ENABLE_BLASST", False
    ]() and not config.use_ws
    var blasst_vote = smem.blasst_vote_smem()

    # Cross-stage P (BLASST enabler, 2Q + non-WS only): the MMA runs the
    # QK0-first schedule (see the gated block below) and reads each stage's P
    # from the OTHER stage's S columns (TMEM_P{wg}). Shared-KV path only (the
    # headline config); split-KV is rejected at compile time.
    comptime CrossP = crossp_effective and type_of(mbars).CrossP_enabled
    # Fires only for an explicit `-D FA4_TMEM_CROSS_P=true` on a config outside
    # the support matrix. The ON-by-default path resolves `crossp_on()` to
    # False on those configs instead, so it never reaches here.
    comptime assert not (ExplicitTMEMCrossP and not config.crossp_on()), (
        "FA4_TMEM_CROSS_P=true was requested on a config that does not"
        " support cross-stage P. Supported: 2Q, non-warp-specialized,"
        " single-CTA, shared-KV, MHA (no rope split), full pages. See"
        " FA4Config.crossp_supported"
    )

    # Called after consumer_s[*].wait(phase), which orders the vote write
    # (softmax) before this read. Peel writes no vote -> reads 0 -> never skips.
    @__parameter
    @always_inline
    def blasst_should_skip(wg: UInt32, phase: UInt32) -> Bool:
        return blasst_vote_unanimous(blasst_vote, wg, phase)

    @__parameter
    @always_inline
    def _commit(
        mbar: UnsafePointer[address_space=.SHARED, ...],
    ):
        """Arrive at mbar: multicast for pair-CTA, local elect for single."""
        comptime if config.pair_cta:
            if e != 0:
                # cta_mask = 0b11 = (1 << cta_group) - 1 for cta_group=2:
                # arrive on both CTAs' instance of the barrier.
                mma_arrive_multicast[cta_group](mbar, UInt16(0x3))
        else:
            elect_mma_arrive(mbar, e)

    # P@V contraction loop, factored out of the call sites below so the
    # wait + MMA body is written once. `_pv_full` is the original bulk path
    # (hot path, codegen unchanged); `_pv_partial` cuts the contraction at
    # the loaded-V boundary for a partially-loaded last KV tile (paged
    # sub-tiles). Each call site keeps its own `comptime if PARTIAL_K [and
    # num_q == 2]` gate and `valid_k_mmas` computation; only the duplicated
    # loop body lives here.
    @__parameter
    @always_inline
    def _pv_full(
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
        wg: UInt32 = 0,
    ):
        # BLASST (gated): the wait always runs (orders the vote read); only
        # the P@V UMMA is skipped on a unanimous vote, so O keeps its prior
        # value. Depth-tiling waits are hoisted so the vote can be read after
        # they retire.
        comptime if ENABLE_BLASST:
            comptime for p_stage in range(num_pv_stages):
                _ = consumer_s[p_stage].wait(wait_phase)
            if not blasst_should_skip(wg, wait_phase):
                comptime for v_stage in range(num_d_tiles):
                    comptime for p_stage in range(num_pv_stages):
                        comptime if v_stage == 0:
                            UMMA1Type.mma[stage_idx=p_stage](
                                s_tmem, v, o_tmem, elect=e, c_scale=c_scale
                            )
                        else:
                            UMMA1Type.mma[stage_idx=p_stage](
                                s_tmem,
                                v + UInt32(v_stage * v_region_bytes),
                                o_tmem + UInt32(v_stage * depth_tile),
                                elect=e,
                                c_scale=c_scale,
                            )
        else:
            # V-smem tile outer; P-tmem stage inner. V-smem is the binding
            # constraint, so each V region is fully contracted before the next.
            comptime for v_stage in range(num_d_tiles):
                comptime for p_stage in range(num_pv_stages):
                    comptime if v_stage == 0:
                        # First V pass: wait each P sub-stage exactly once (P
                        # stays live in TMEM across all V regions; preserves the
                        # 3/4+1/4 P-stage overlap). Non-WS folds to today's stream.
                        _ = consumer_s[p_stage].wait(wait_phase)
                        UMMA1Type.mma[stage_idx=p_stage](
                            s_tmem, v, o_tmem, elect=e, c_scale=c_scale
                        )
                    else:
                        UMMA1Type.mma[stage_idx=p_stage](
                            s_tmem,
                            v + UInt32(v_stage * v_region_bytes),
                            o_tmem + UInt32(v_stage * depth_tile),
                            elect=e,
                            c_scale=c_scale,
                        )

    @__parameter
    @always_inline
    def _pv_partial(
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
        valid_k_mmas: UInt32,
        wg: UInt32 = 0,
    ):
        # BLASST (gated): see `_pv_full` -- same skip, partial-K variant.
        comptime if ENABLE_BLASST:
            comptime for p_stage in range(num_pv_stages):
                _ = consumer_s[p_stage].wait(wait_phase)
            if not blasst_should_skip(wg, wait_phase):
                comptime for v_stage in range(num_d_tiles):
                    comptime for p_stage in range(num_pv_stages):
                        comptime if v_stage == 0:
                            UMMA1Type.mma_maybe_partial_k[stage_idx=p_stage](
                                s_tmem,
                                v,
                                o_tmem,
                                c_scale=c_scale,
                                elect=e,
                                valid_k_mmas=valid_k_mmas,
                            )
                        else:
                            UMMA1Type.mma_maybe_partial_k[stage_idx=p_stage](
                                s_tmem,
                                v + UInt32(v_stage * v_region_bytes),
                                o_tmem + UInt32(v_stage * depth_tile),
                                c_scale=c_scale,
                                elect=e,
                                valid_k_mmas=valid_k_mmas,
                            )
        else:
            # V-smem tile outer; P-tmem stage inner (see `_pv_full`).
            # `valid_k_mmas` is the KEY count, identical across V regions, so it
            # passes through unchanged for every v_stage.
            comptime for v_stage in range(num_d_tiles):
                comptime for p_stage in range(num_pv_stages):
                    comptime if v_stage == 0:
                        _ = consumer_s[p_stage].wait(wait_phase)
                        UMMA1Type.mma_maybe_partial_k[stage_idx=p_stage](
                            s_tmem,
                            v,
                            o_tmem,
                            c_scale=c_scale,
                            elect=e,
                            valid_k_mmas=valid_k_mmas,
                        )
                    else:
                        UMMA1Type.mma_maybe_partial_k[stage_idx=p_stage](
                            s_tmem,
                            v + UInt32(v_stage * v_region_bytes),
                            o_tmem + UInt32(v_stage * depth_tile),
                            c_scale=c_scale,
                            elect=e,
                            valid_k_mmas=valid_k_mmas,
                        )

    # Sliding-window / any non-zero `start_column` mask: the load and
    # softmax warps work the contraction in the `[start_column, num_keys)`
    # frame -- the producer iterates V tiles from `kv_row = start_column`
    # (load_warp.mojo) and softmax mirrors it (softmax_warp.mojo:1191). The
    # partial-K `valid_k_mmas` (`vkm`) below must use the SAME frame: the
    # last loaded tile sits at `start_column + (total_iters - 1) * BN`, so
    # the count of loaded MMA_K-blocks is measured against
    # `num_keys - start_column`, NOT `num_keys`. Omitting `start_column`
    # over-counts `vkm` by `start_column // MMA_K`, so P@V runs blocks over
    # V pages the producer never loaded (`0 * stale-NaN = NaN`). For causal
    # `start_column == 0`, so `v_eff_keys == num_keys` and every `vkm` site
    # below is bit-identical to before.
    #
    # 1Q split-K narrows the frame again: the `vkm` sites count tiles from
    # the partition's own window, so `v_eff_keys` is rebased by the window
    # start where split-K slices it below.
    var v_start_col: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_id, score_row
    )
    var v_eff_keys: UInt32 = num_keys - v_start_col

    comptime if config.use_shared_kv:
        # ---- Shared KV mode ----
        # In shared mode, K_nope and V alternate in a single StagedPipeline.
        # Stages: K0, V0, K1, V1, ...

        var kv_smem = smem.k_smem_base()  # same as v_smem_base in shared mode
        # Per-slot byte stride = ONE ring slot. Derived from the SMEM struct so
        # it can NEVER drift from the reservation/producer. WS: 32768 (one
        # 256x64 sub-tile); non-WS shared: 65536 (one full-depth tile). The old
        # `v_cols_per_cta()*BN*size` literal equalled the non-WS value but strode
        # 2x the WS reservation -> latent overrun; this reconciles it.
        comptime kv_stage_bytes = SM100AttentionSMem[config].k_stage_bytes
        comptime if config.use_ws:
            comptime assert (
                kv_stage_bytes
                == config.BN
                * (config.shared_kv_cols() // config.num_qk_stages)
                * size_of[config.qkv_dtype]()
            ), "WS kv_stage_bytes must equal one 256x64 sub-tile"
            # Uniform shared-ring sub-tile premise (mirrors the m_pack-selected
            # invariant already enforced at `attention.mojo`'s `supported()`):
            # Layout-G (m_pack==4) keeps its depth-scatter checks; Layout-E
            # (m_pack==2) asserts its reduction-split V sub-tile divides the
            # partition's keys evenly AND is byte-equal to one K/V ring slot.
            comptime if is_reduction_split:
                comptime assert (
                    config.BN // config.m_pack
                ) % config.num_qk_stages == 0, (
                    "WS Layout-E: (BN // m_pack) must divide evenly by"
                    " num_qk_stages"
                )
                comptime assert (
                    pv_mma_n * pv_bk_chunk * size_of[config.qkv_dtype]()
                    == kv_stage_bytes
                ), (
                    "WS Layout-E: reduction sub-tile bytes must equal one"
                    " K/V ring slot"
                )
            else:
                comptime assert (
                    num_d_tiles == config.num_qk_stages
                ), "WS uniform sub-tile: num_d_tiles == num_qk_stages"
                comptime assert (
                    256 // config.m_pack
                ) == config.BK0, "WS uniform sub-tile: depth_tile == BK0"

        # K descriptor: k_major for Q@K' (BK0=64 -> one depth-half sub-tile).
        var kv_desc_k = smem_descriptor[
            BMN=config.k_rows_per_cta(),
            BK=config.BK0,
            swizzle_mode=config.swizzle_mode,
            is_k_major=True,
            page_dense=config.k_row_major(),
        ](kv_smem)
        # V descriptor: mn_major for P@V. WS uses the packed [pv_mma_n, pv_bk]
        # box -- Layout-G's proven one-depth-tile box (256x64) or Layout-E's
        # reduction-chunk box (256x64, same numbers @depth128, different
        # meaning); non-WS keeps the full [v_cols_per_cta, BN] tile
        # (byte-identical).
        comptime v_desc_bmn = (
            pv_mma_n if config.use_ws else config.v_cols_per_cta()
        )
        comptime v_desc_bk = pv_bk if config.use_ws else config.BN
        var kv_desc_v = smem_descriptor[
            BMN=v_desc_bmn,
            BK=v_desc_bk,
            swizzle_mode=config.swizzle_mode,
            is_k_major=False,
            page_dense=config.v_row_major(),
        ](kv_smem)

        comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
        var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

        # We peel the first iteration, as we want to wait on q1.
        # 2Q: peel consumes 1 K_0 (shared); main loop decrements
        # once per iter. iter_count = total_iters - 1.
        # 1Q: peel consumes 2 K-tiles (K_e[0], K_o[0]) and 2 V-tiles
        # (V_e[0], V_o[0] held); main loop decrements once at top
        # (K_e consume) plus once inside a 1Q guard (K_o consume,
        # with a break-check between). iter_count = total_iters - 2.
        # 1Q at total_iters == 1 takes the T==1 fast path below after the
        # Q @ K_e[0] staged MMA; the iter_count underflow at T == 1
        # (1u32 - 2u32 wraps) is never read.
        var total_iters_runtime: UInt32 = mask.total_iters[
            BM_mask, BN, page_size
        ](seq_id, score_row, num_keys)
        # Split-K (1Q): slice the combined tile count to this partition's
        # window. mma needs only the per-partition COUNT -- load_warp emits
        # the matching cb-offset tiles that this warp consumes in order.
        # `total_iters == last_masked_set_end` for check_mask==False masks
        # (mha_mask.mojo:510-513/641-644/...), so all four warps derive the
        # same window.
        comptime if config.num_q == 1 and (
            config.splitk_partitions > 1 or workspace_split
        ):
            var _np = splitk_num_partitions[config](ws_num_partitions)
            var _w = splitk_window(
                total_iters_runtime,
                _np,
                splitk_partition_idx(_np),
            )
            total_iters_runtime = _w[1] - _w[0]
            # Empty partition (front-load trailing window, T < num_partitions;
            # or an M6 idle CTA): no tiles, so load_warp produces no K0 and the
            # peeled `consumer_wait()` below would hang. Skip all MMA work --
            # the empty partition's softmax stages a neutral identity, and the
            # kernel terminal `cluster_sync()` still runs after this return.
            if total_iters_runtime == 0:
                return
            # Rebase into this partition's tile window (see the `vkm` note).
            v_eff_keys -= _w[0] * UInt32(BN)
        # All-masked row (valid_length 0): total_iters == 0, so the
        # `- (3 - num_q)` below underflows and the MMA warp spins, hanging the
        # pipeline. Split-K is guarded above; guard the non-split path here.
        if total_iters_runtime == 0:
            return
        var iter_count: UInt32 = total_iters_runtime - UInt32(3 - num_q)

        # Release the KV slot at `release_idx`, advance to the next stage,
        # wait for it, and return its slot index. Bundles the
        # release/step/wait/capture idiom repeated across the 1Q path.
        # `consumer_mbar(idx)` with the current index is identical to the
        # no-arg `consumer_mbar()` (which forwards `state.index()`).
        @__parameter
        @always_inline
        def _advance_kv(release_idx: UInt32) -> UInt32:
            _commit(kv_pipeline.consumer_mbar(release_idx))
            kv_pipeline.state.step()
            kv_pipeline.consumer_wait()
            return kv_pipeline.state.index()

        # ===== Cross-stage P QK0-first JIT schedule + K-ahead KV ring =====
        # Two cursors over the shared K0,K1,V0,K2,V1,... ring: ACQUIRE walks
        # position order every step; RELEASE (`rel_slot`) lags by <=2 but fires
        # in the SAME producer-fill order, so per-slot mbar parity never skips.
        # Per-iter MMA order QK0(n)->PV0(n-1)->QK1(n)->PV1(n-1): QK0(n)
        # overwrites S0 (incl. P1's window S0[32:96]) before PV1(n-1) reads
        # P1(n-1) from there -- a WAR hazard made safe by the softmax JIT-P1
        # seed, which delays P1(n-1)'s store to land after QK0(n). 2Q + non-WS
        # only; aliased path below is byte-identical when off.
        comptime if CrossP:
            var p0_tmem = tmem_addr + UInt32(config.TMEM_P0)  # P0 in S1 window
            var p1_tmem = tmem_addr + UInt32(config.TMEM_P1)  # P1 in S0 window
            # MMA-side sfree consumers: QK{wg}(n) waits until softmax finished
            # reading S{wg}(n-1) before overwriting it. Peel QKs skip this (no
            # prior S). Persistent phase.
            var sfree0 = mbars.sfree_consumer(0)
            var sfree1 = mbars.sfree_consumer(1)

            # Private in-order release cursor (slot = pos % num_kv_stages). No
            # jump-back: releases fire in strict producer-fill order.
            var rel_slot: UInt32 = 0
            comptime last_stage = config.num_kv_stages - 1

            # ---- Peel: acquire K0 (pos0); QK0(0); QK1(0); release K0. ----
            kv_pipeline.consumer_wait()  # pos0 = K0
            var kc = (
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            )
            kv_pipeline.state.step()
            UMMA0Type.mma[stage_idx=0](q0, kc, s0_tmem, elect=e, c_scale=0)
            _commit(pipeline_s0.producer_mbar())
            var q1m = mbars.q1_wait_mbar()
            q1m[0].wait()
            UMMA0Type.mma[stage_idx=0](q1, kc, s1_tmem, elect=e, c_scale=0)
            _commit(pipeline_s1.producer_mbar())
            # release K0 (pos0)
            _commit(kv_pipeline.consumer_mbar(rel_slot))
            rel_slot = 0 if rel_slot == UInt32(last_stage) else rel_slot + 1

            var xphase: UInt32 = 0
            var xcscale: UInt32 = 0

            # ---- Main loop: iter n (n=1..NT-1) makes S(n), drains P(n-1). ----
            while iter_count != 0:
                iter_count -= 1

                # Acquire K_n (pos 2n-1).
                kv_pipeline.consumer_wait()
                var kn_idx = kv_pipeline.state.index()
                kv_pipeline.state.step()
                kc = kv_desc_k + UInt32(kv_stage_bytes) * kn_idx

                # QK0(n): acquire sfree0 (softmax read S0(n-1)); q0*K_n -> S0.
                sfree0.wait()
                sfree0.step()
                UMMA0Type.mma[stage_idx=0](q0, kc, s0_tmem, elect=e, c_scale=0)
                _commit(pipeline_s0.producer_mbar())

                # Acquire V_{n-1} (pos 2n).
                kv_pipeline.consumer_wait()
                var vn1_idx = kv_pipeline.state.index()
                kv_pipeline.state.step()
                var vn1 = kv_desc_v + UInt32(kv_stage_bytes) * vn1_idx

                # PV0(n-1): P0(n-1) * V_{n-1} -> O0 (cross p0_tmem in S1 window).
                _pv_full(p0_tmem, vn1, o0_tmem, consumer_s0, xphase, xcscale, 0)
                _commit(pipeline_o0.producer_mbar())

                # QK1(n): acquire sfree1; q1*K_n -> S1.
                sfree1.wait()
                sfree1.step()
                UMMA0Type.mma[stage_idx=0](q1, kc, s1_tmem, elect=e, c_scale=0)
                _commit(pipeline_s1.producer_mbar())

                # Release K_n (pos 2n-1) -- BEFORE the deferred PV1 (FI 1317).
                _commit(kv_pipeline.consumer_mbar(rel_slot))
                rel_slot = 0 if rel_slot == UInt32(last_stage) else rel_slot + 1

                # PV1(n-1): P1(n-1) * V_{n-1} -> O1 (cross p1_tmem in S0 window;
                # JIT-protected -- softmax stored P1(n-1) after QK0(n)).
                _pv_full(p1_tmem, vn1, o1_tmem, consumer_s1, xphase, xcscale, 1)
                _commit(pipeline_o1.producer_mbar())

                # Release V_{n-1} (pos 2n) -- after PV1 (FI 1333).
                _commit(kv_pipeline.consumer_mbar(rel_slot))
                rel_slot = 0 if rel_slot == UInt32(last_stage) else rel_slot + 1

                xcscale = 1
                xphase ^= 1

            # ---- Epilogue: acquire V_{NT-1} (pos 2NT-1); drain P0/P1. ----
            kv_pipeline.consumer_wait()
            var vl_idx = kv_pipeline.state.index()
            kv_pipeline.state.step()
            var vl = kv_desc_v + UInt32(kv_stage_bytes) * vl_idx
            comptime if PARTIAL_K:
                var vkm = ceildiv(
                    min(
                        v_eff_keys
                        - (total_iters_runtime - UInt32(1)) * UInt32(BN),
                        UInt32(BN),
                    ),
                    UInt32(UMMA1Type.MMA_K),
                )
                _pv_partial(
                    p0_tmem, vl, o0_tmem, consumer_s0, xphase, xcscale, vkm, 0
                )
                _commit(pipeline_o0.producer_mbar())
                _pv_partial(
                    p1_tmem, vl, o1_tmem, consumer_s1, xphase, xcscale, vkm, 1
                )
                _commit(pipeline_o1.producer_mbar())
            else:
                _pv_full(p0_tmem, vl, o0_tmem, consumer_s0, xphase, xcscale, 0)
                _commit(pipeline_o0.producer_mbar())
                _pv_full(p1_tmem, vl, o1_tmem, consumer_s1, xphase, xcscale, 1)
                _commit(pipeline_o1.producer_mbar())
            # release V_{NT-1} (pos 2NT-1)
            _commit(kv_pipeline.consumer_mbar(rel_slot))
            return

        # ---- WS sub-tile ring helpers (all fold to no-ops for non-WS) ----
        # `_qk_extra`: after the d=0 Q@K' mma at the first K sub-slot, do the
        # remaining num_qk_stages-1 depth-half sub-slots. Each is its OWN ring
        # slot (NOT contiguous), so we use two `mma[stage_idx=0]` with manual
        # K/Q bases and c_scale 0->1 -- NOT `mma[stage_idx=1]`, which adds an
        # internal +BK0 offset assuming one contiguous tile (would read OOB).
        # Empty (no-op) when num_qk_stages==1.
        @__parameter
        @always_inline
        def _qk_extra(q_base: MMASmemDescriptorPair, s_tmem: UInt32):
            comptime for d in range(1, config.num_qk_stages):
                var kd_idx = _advance_kv(kv_pipeline.state.index())
                UMMA0Type.mma[stage_idx=0](
                    q_base + UInt32(d * q_chunk_bytes),
                    kv_desc_k + UInt32(kv_stage_bytes) * kd_idx,
                    s_tmem,
                    elect=e,
                    c_scale=1,
                )

        # `_v_wait_rest`: wait the remaining num_d_tiles-1 V depth-tile sub-slots
        # (produced consecutively right after the first). Leaves the ring state
        # at the last of them. Empty when num_d_tiles==1.
        @__parameter
        @always_inline
        def _v_wait_rest():
            comptime for d in range(1, num_d_tiles):
                kv_pipeline.state.step()
                kv_pipeline.consumer_wait()

        # `_v_release_rest`: release the num_d_tiles-1 V sub-slots after v_idx0
        # (ring-adjacent). Empty when num_d_tiles==1.
        @__parameter
        @always_inline
        def _v_release_rest(v_idx0: UInt32):
            comptime for d in range(1, num_d_tiles):
                _commit(
                    kv_pipeline.consumer_mbar(
                        (v_idx0 + UInt32(d)) % UInt32(config.num_kv_stages)
                    )
                )

        # `_pv_ws`: P@V across num_d_tiles ring slots (WS only). `v_idx0` is
        # the first slot's ring index; the rest are ring-adjacent. V-slot
        # outer, P-stage inner (mirrors `_pv_full`); wait each P sub-stage
        # once on the first slot only (P stays live in TMEM across slots).
        #
        # Layout-G (m_pack==4, depth-scatter): `v_stage` selects a DISJOINT
        # output depth-tile (`o_tmem + v_stage*depth_tile`) from its own slot
        # base (NOT the `+d*v_region_bytes` intra-slot offset the non-WS path
        # uses for its contiguous V buffer); the caller's `c_scale` applies to
        # every v_stage unchanged (each writes independent columns).
        #
        # Layout-E (m_pack==2, reduction-accumulate): `v_stage` is instead a
        # reduction (key) chunk `r` -- every chunk MMAs into the SAME
        # (un-offset) `o_tmem`, so only `v_stage==0` uses the caller's
        # `c_scale`; later chunks always accumulate (scale=1), composing with
        # `UMMA1Type`'s own per-`p_stage` internal `stage_idx==0 -> c_scale
        # else 1` handling.
        @__parameter
        @always_inline
        def _pv_ws[
            partial: Bool
        ](
            s_tmem: UInt32,
            v_idx0: UInt32,
            o_tmem: UInt32,
            consumer_s: MBarType,
            wait_phase: UInt32,
            c_scale: UInt32,
            valid_k_mmas: UInt32,
        ):
            comptime for v_stage in range(num_d_tiles):
                var v_idx_d: UInt32
                comptime if v_stage == 0:
                    v_idx_d = v_idx0
                else:
                    v_idx_d = (v_idx0 + UInt32(v_stage)) % UInt32(
                        config.num_kv_stages
                    )
                var v_slot = kv_desc_v + UInt32(kv_stage_bytes) * v_idx_d
                # Per-`v_stage` O / P-column offsets, exactly one nonzero per
                # layout. Layout-G scatters O by depth-tile and never advances
                # the P columns. Layout-E accumulates into un-offset O but must
                # advance the P base by `v_stage` reduction chunks
                # (`s_tmem_chunk_words` TMEM columns each): `UMMA1Type.mma`'s
                # OWN internal offset only covers ONE `pv_bk_chunk`-wide window
                # relative to the base passed in, so without this advance every
                # chunk silently re-reads chunk 0's P columns against a
                # DIFFERENT V key range. `v_stage` is a comptime loop var, so
                # both offsets fold to constants.
                comptime o_col_offset = (
                    0 if is_reduction_split else v_stage * depth_tile
                )
                comptime s_col_offset = (
                    v_stage * s_tmem_chunk_words if is_reduction_split else 0
                )
                var stage_o_tmem = o_tmem + UInt32(o_col_offset)
                var stage_s_tmem = s_tmem + UInt32(s_col_offset)
                var stage_c_scale: UInt32 = c_scale
                comptime if is_reduction_split and v_stage != 0:
                    stage_c_scale = 1
                comptime for p_stage in range(num_pv_stages):
                    comptime if v_stage == 0:
                        _ = consumer_s[p_stage].wait(wait_phase)
                    comptime if partial:
                        UMMA1Type.mma_maybe_partial_k[stage_idx=p_stage](
                            stage_s_tmem,
                            v_slot,
                            stage_o_tmem,
                            c_scale=stage_c_scale,
                            elect=e,
                            valid_k_mmas=valid_k_mmas,
                        )
                    else:
                        UMMA1Type.mma[stage_idx=p_stage](
                            stage_s_tmem,
                            v_slot,
                            stage_o_tmem,
                            elect=e,
                            c_scale=stage_c_scale,
                        )

        # ---- Peeled iteration ----
        # Stage 0 = K0 (K_e[0]_d0)
        kv_pipeline.consumer_wait()
        var k0 = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
        UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
        _qk_extra(q0, s0_tmem)  # WS: K_e[0]_d1..; non-WS: no-op
        _commit(pipeline_s0.producer_mbar())

        # 1Q: release K_e[0] (last sub-slot); step to the next tile's first
        # sub-slot; wait. It holds K_o[0]_d0 for T >= 2 and V_e[0]_d0 for
        # T == 1 -- diverge on descriptor base only.
        comptime if num_q == 1:
            var slot1_offset = UInt32(kv_stage_bytes) * _advance_kv(
                kv_pipeline.state.index()
            )

            # T == 1 fast path: slot 1 holds V_e[0] (load_warp produced
            # K_e[0] + V_e[0] only, = 2*num_qk_stages sub-slots). Do
            # P_e @ V_e[0] -> o0 and return. Don't touch s1 / o1 -- softmax
            # WG1 takes its matching no-op path (softmax_warp.mojo:1254-1257).
            if total_iters_runtime == UInt32(1):
                var ve_idx0 = kv_pipeline.state.index()  # V_e[0]_d0
                _v_wait_rest()  # WS: wait V_e[0]_d1..
                comptime if PARTIAL_K:
                    var vkm = ceildiv(
                        min(v_eff_keys, UInt32(BN)), UInt32(UMMA1Type.MMA_K)
                    )
                    comptime if config.use_ws:
                        _pv_ws[partial=True](
                            s0_tmem, ve_idx0, o0_tmem, consumer_s0, 0, 0, vkm
                        )
                    else:
                        _pv_partial(
                            s0_tmem,
                            kv_desc_v + UInt32(kv_stage_bytes) * ve_idx0,
                            o0_tmem,
                            consumer_s0,
                            0,
                            0,
                            vkm,
                        )
                else:
                    comptime if config.use_ws:
                        _pv_ws[partial=False](
                            s0_tmem, ve_idx0, o0_tmem, consumer_s0, 0, 0, 0
                        )
                    else:
                        _pv_full(
                            s0_tmem,
                            kv_desc_v + UInt32(kv_stage_bytes) * ve_idx0,
                            o0_tmem,
                            consumer_s0,
                            0,
                            0,
                        )
                _commit(pipeline_o0.producer_mbar())
                _commit(kv_pipeline.consumer_mbar(ve_idx0))  # V_e[0]_d0
                _v_release_rest(ve_idx0)  # WS: release V_e[0]_d1..
                return

            k0 = kv_desc_k + slot1_offset  # K_o[0]_d0

        # Q_1 @ K_0 (2Q, q1 half, same K) / Q @ K_o[0] (1Q,
        # q0 + redefined k0), depth-split across sub-slots for WS.
        comptime if num_q == 2:
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].wait()
            UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)
            _qk_extra(q1, s1_tmem)  # no-op (2Q shared -> num_qk_stages==1)
        else:
            UMMA0Type.mma[stage_idx=0](q0, k0, s1_tmem, elect=e, c_scale=0)
            _qk_extra(q0, s1_tmem)  # WS: K_o[0]_d1..

        # Release K (K_0 in 2Q / K_o[0] last sub-slot in 1Q) and advance.
        _commit(kv_pipeline.consumer_mbar())
        kv_pipeline.state.step()
        _commit(pipeline_s1.producer_mbar())

        # Stage 1 = V_0 (2Q) / V_e[0] (1Q, single use; we will then
        # load V_o[0] and hold it for the first main-loop iter).
        kv_pipeline.consumer_wait()
        var v_prev_idx: UInt32 = kv_pipeline.state.index()  # V_e[0]_d0
        _v_wait_rest()  # WS: wait V_e[0]_d1..
        comptime if PARTIAL_K and num_q == 2:
            # 2Q peeled o0 contracts tile 0, which is the last (and only)
            # tile only when total_iters == 1; vkm self-clamps to full
            # (v_eff_keys >= BN) otherwise. 2Q shared is non-WS.
            var vkm = ceildiv(
                min(v_eff_keys, UInt32(BN)), UInt32(UMMA1Type.MMA_K)
            )
            _pv_partial(
                s0_tmem,
                kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                o0_tmem,
                consumer_s0,
                0,
                0,
                vkm,
            )
        else:
            comptime if config.use_ws:
                _pv_ws[partial=False](
                    s0_tmem, v_prev_idx, o0_tmem, consumer_s0, 0, 0, 0
                )
            else:
                _pv_full(
                    s0_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o0_tmem,
                    consumer_s0,
                    0,
                    0,
                )
        _commit(pipeline_o0.producer_mbar())
        var phase: UInt32 = 0

        var c_scale: UInt32 = 0

        # BLASST: epilogue P@V is WG1 by default; the 1Q odd-T tail re-points
        # s1 aliases to WG0, so flip this there too (DCE'd when off).
        var blasst_epi_wg: UInt32 = 1

        # 1Q: release V_e[0] (all sub-slots); load V_o[0] and hold its
        # first slot index in v_prev_idx for the first main-loop iter's
        # P_o @ V_o[0] MMA.
        comptime if num_q == 1:
            _v_release_rest(v_prev_idx)  # WS: release V_e[0]_d1..
            v_prev_idx = _advance_kv(v_prev_idx)  # -> V_o[0]_d0
            _v_wait_rest()  # WS: wait V_o[0]_d1..

        # ---- Main loop ----
        while iter_count != 0:
            iter_count -= 1

            # Advance past held V to get to next K
            kv_pipeline.state.step()

            # Kn (K_e[n]_d0, depth-split across sub-slots for WS)
            kv_pipeline.consumer_wait()
            var kn = (
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            )
            UMMA0Type.mma[stage_idx=0](q0, kn, s0_tmem, elect=e, c_scale=0)
            _qk_extra(q0, s0_tmem)  # WS: K_e[n]_d1..
            _commit(pipeline_s0.producer_mbar())

            # P1 @ V_{n-1} (held V_o[n-1]). Per-depth-tile ring slots for WS.
            comptime if config.use_ws:
                _pv_ws[partial=False](
                    s1_tmem, v_prev_idx, o1_tmem, consumer_s1, phase, c_scale, 0
                )
            else:
                _pv_full(
                    s1_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o1_tmem,
                    consumer_s1,
                    phase,
                    c_scale,
                    1,
                )
            _commit(pipeline_o1.producer_mbar())
            c_scale = 1
            _commit(kv_pipeline.consumer_mbar(v_prev_idx))  # V_{n-1}_d0
            _v_release_rest(v_prev_idx)  # WS: release V_{n-1}_d1..

            # 1Q: between K_e[n] and K_o[n] -- break-check for tail
            # iter when total K-tiles is odd, else consume K_o[n] by
            # releasing K_e[n] and reassigning kn = K_o[n].
            comptime if num_q == 1:
                if iter_count == 0:
                    # Tail iter (T odd): no K_o[k]. The remaining work
                    # -- P_e[k] @ V_e[k] -> o0_tmem -- has the same
                    # shape as the epilogue's P_1 @ V_last -> o1_tmem.
                    # Rebind o1-side aliases to o0-side resources and
                    # fall through; epilogue does the work unchanged.
                    # release K_e[k] (last sub-slot), wait V_e[k]_d0
                    v_prev_idx = _advance_kv(kv_pipeline.state.index())
                    _v_wait_rest()  # WS: wait V_e[k]_d1..
                    s1_tmem = s0_tmem
                    o1_tmem = o0_tmem
                    consumer_s1 = consumer_s0
                    pipeline_o1 = pipeline_o0
                    blasst_epi_wg = 0  # aliased to WG0 side
                    phase ^= 1  # advance from this iter's K@s1 phase
                    # to the V@o0 phase the s0 wait needs.
                    break
                iter_count -= 1
                # release K_e[n] (last sub-slot), wait K_o[n]_d0
                kn = kv_desc_k + UInt32(kv_stage_bytes) * _advance_kv(
                    kv_pipeline.state.index()
                )

            # Q_1 @ K_n (2Q, q1 + same kn) / Q @ K_o[n] (1Q,
            # q0 + redefined kn), depth-split across sub-slots for WS.
            comptime if num_q == 2:
                UMMA0Type.mma[stage_idx=0](q1, kn, s1_tmem, elect=e, c_scale=0)
                _qk_extra(q1, s1_tmem)  # no-op (2Q -> num_qk_stages==1)
            else:
                UMMA0Type.mma[stage_idx=0](q0, kn, s1_tmem, elect=e, c_scale=0)
                _qk_extra(q0, s1_tmem)  # WS: K_o[n]_d1..
            _commit(kv_pipeline.consumer_mbar())  # release K_n / K_o[n] last
            kv_pipeline.state.step()
            _commit(pipeline_s1.producer_mbar())
            phase ^= 1

            # Vn (2Q held for next iter) / V_e[n] (1Q single use,
            # then V_o[n] loaded and held).
            kv_pipeline.consumer_wait()
            v_prev_idx = kv_pipeline.state.index()  # V_e[n]_d0
            _v_wait_rest()  # WS: wait V_e[n]_d1..
            comptime if PARTIAL_K and num_q == 2:
                # 2Q: Vn is the last tile exactly when iter_count == 0
                # (the final main-loop iteration); otherwise full.
                var vkm = ceildiv(
                    min(
                        v_eff_keys
                        - (total_iters_runtime - UInt32(1)) * UInt32(BN),
                        UInt32(BN),
                    ),
                    UInt32(UMMA1Type.MMA_K),
                ) if iter_count == 0 else UInt32(UMMA1Type.num_k_mmas)
                _pv_partial(
                    s0_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o0_tmem,
                    consumer_s0,
                    phase,
                    1,
                    vkm,
                )
            else:
                comptime if config.use_ws:
                    _pv_ws[partial=False](
                        s0_tmem, v_prev_idx, o0_tmem, consumer_s0, phase, 1, 0
                    )
                else:
                    _pv_full(
                        s0_tmem,
                        kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                        o0_tmem,
                        consumer_s0,
                        phase,
                        1,
                    )
            _commit(pipeline_o0.producer_mbar())

            # 1Q: release V_e[n] (all sub-slots); load V_o[n] and hold
            # its first slot in v_prev_idx for the next iter / epilogue.
            comptime if num_q == 1:
                _v_release_rest(v_prev_idx)  # WS: release V_e[n]_d1..
                v_prev_idx = _advance_kv(v_prev_idx)  # -> V_o[n]_d0
                _v_wait_rest()  # WS: wait V_o[n]_d1..

        # ---- Epilogue ----
        comptime if PARTIAL_K:
            var vkm = ceildiv(
                min(
                    v_eff_keys - (total_iters_runtime - UInt32(1)) * UInt32(BN),
                    UInt32(BN),
                ),
                UInt32(UMMA1Type.MMA_K),
            )
            comptime if config.use_ws:
                _pv_ws[partial=True](
                    s1_tmem,
                    v_prev_idx,
                    o1_tmem,
                    consumer_s1,
                    phase,
                    c_scale,
                    vkm,
                )
            else:
                _pv_partial(
                    s1_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o1_tmem,
                    consumer_s1,
                    phase,
                    c_scale,
                    vkm,
                    blasst_epi_wg,
                )
        else:
            comptime if config.use_ws:
                _pv_ws[partial=False](
                    s1_tmem, v_prev_idx, o1_tmem, consumer_s1, phase, c_scale, 0
                )
            else:
                _pv_full(
                    s1_tmem,
                    kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx,
                    o1_tmem,
                    consumer_s1,
                    phase,
                    c_scale,
                    blasst_epi_wg,
                )
        _commit(pipeline_o1.producer_mbar())
        _commit(kv_pipeline.consumer_mbar(v_prev_idx))  # V_last_d0
        _v_release_rest(v_prev_idx)  # WS: release V_last_d1..

    else:
        # ---- Non-shared mode (original) ----

        var k_smem = rebind[SharedMemPointer[Scalar[config.qkv_dtype]]](
            smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[config.qkv_dtype]]](
            smem.v_smem_base()
        )
        comptime KPipeType = KConsumerPipeline[config.qkv_dtype, config]
        comptime VPipeType = VConsumerPipeline[config.qkv_dtype, config]
        var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
        var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

        # We peel the first iteration, as we want to wait on q1.
        # 2Q: peel consumes 1 K_0 (shared); main loop decrements
        # once per iter. iter_count = total_iters - 1.
        # 1Q: peel consumes 2 K-tiles (K_e[0], K_o[0]) and 2 V-tiles
        # (V_e[0], V_o[0] held); main loop decrements once at top
        # (K_e consume) plus once inside a 1Q guard (K_o consume,
        # with a break-check between). iter_count = total_iters - 2.
        # Unified: subtract (3 - num_q) -- 1 for 2Q, 2 for 1Q.
        # 1Q at total_iters == 1 takes an early-return fast path after the
        # Q @ K_e[0] staged MMA below, so the iter_count underflow at
        # T == 1 (1u32 - 2u32 wraps) is never read. Keep the raw value in
        # `total_iters_runtime` for the runtime branch.
        var total_iters_runtime: UInt32 = mask.total_iters[
            BM_mask, BN, page_size
        ](seq_id, score_row, num_keys)
        # Split-K (1Q): slice the combined tile count to this partition's
        # window. mma needs only the per-partition COUNT -- load_warp emits
        # the matching cb-offset tiles that this warp consumes in order.
        # `total_iters == last_masked_set_end` for check_mask==False masks
        # (mha_mask.mojo:510-513/641-644/...), so all four warps derive the
        # same window.
        comptime if config.num_q == 1 and (
            config.splitk_partitions > 1 or workspace_split
        ):
            var _np = splitk_num_partitions[config](ws_num_partitions)
            var _w = splitk_window(
                total_iters_runtime,
                _np,
                splitk_partition_idx(_np),
            )
            total_iters_runtime = _w[1] - _w[0]
            # Empty partition (front-load trailing window, T < num_partitions;
            # or an M6 idle CTA): no tiles, so load_warp produces no K0 and the
            # peeled `pipeline_k.wait_k` below would hang. Skip all MMA work --
            # the empty partition's softmax stages a neutral identity, and the
            # kernel terminal `cluster_sync()` still runs after this return.
            if total_iters_runtime == 0:
                return
            # Rebase into this partition's tile window (see the `vkm` note).
            v_eff_keys -= _w[0] * UInt32(BN)
        # All-masked row (valid_length 0): total_iters == 0, so the
        # `- (3 - num_q)` below underflows and the MMA warp spins, hanging the
        # pipeline. Split-K is guarded above; guard the non-split path here.
        if total_iters_runtime == 0:
            return
        var iter_count: UInt32 = total_iters_runtime - UInt32(3 - num_q)

        # Q_0 @ K_0' (2Q) / Q @ K_e[0]' (1Q), staged over num_qk_stages
        var k0 = pipeline_k.get_k()

        comptime for qk_stage in range(num_qk_stages):
            pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
            UMMA0Type.mma[stage_idx=qk_stage](
                q0, k0, s0_tmem, elect=e, c_scale=0
            )
            # 1Q: release K_e[0] stage (single use); step at last.
            comptime if num_q == 1:
                _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                comptime if qk_stage == num_qk_stages - 1:
                    pipeline_k.pipeline.state.step()
        _commit(pipeline_s0.producer_mbar())

        # 1Q T==1 fast path. Only one K-tile in the sequence (K_e[0]),
        # so K_o[0] is never produced by load_warp and Q @ K_o[0] would
        # hang on pipeline_k.wait_k. Do the single P_e @ V_e[0] -> o0
        # MMA and exit. Skip Q @ K_o[0] -> s1, V_o[0] hold, the main
        # loop, and the epilogue P @ V_held -> o1. No mbar on s1 / o1
        # is touched here; softmax_warp.mojo's WG1 takes the matching
        # no-op path so the s1/o1 producer-consumer balance is
        # preserved.
        comptime if num_q == 1:
            if total_iters_runtime == UInt32(1):
                var vlatest_t1 = pipeline_v.get_v()
                pipeline_v.wait_v()
                comptime if PARTIAL_K:
                    var vkm = ceildiv(
                        min(v_eff_keys, UInt32(BN)), UInt32(UMMA1Type.MMA_K)
                    )
                    _pv_partial(
                        s0_tmem, vlatest_t1, o0_tmem, consumer_s0, 0, 0, vkm
                    )
                else:
                    _pv_full(s0_tmem, vlatest_t1, o0_tmem, consumer_s0, 0, 0)
                _commit(pipeline_o0.producer_mbar())
                var ve_idx = pipeline_v.pipeline.state.index()
                pipeline_v.pipeline.consumer_release_at(ve_idx, e)
                return

        # 1Q: redefine k0 = K_o[0] for the s1 staged loop below.
        comptime if num_q == 1:
            k0 = pipeline_k.get_k()

        # Q_1 @ K_0' (2Q, q1 half, same k0) / Q @ K_o[0]' (1Q,
        # q0 + redefined k0), staged over num_qk_stages. Mojo's
        # `comptime if` introduces a new lexical scope, so q1_mbar
        # is declared per-iteration inside the 2Q branch
        # (mbars.q1_wait_mbar() is a const accessor; declaring it
        # each iter is free at codegen since comptime for unrolls).
        comptime for qk_stage in range(num_qk_stages):
            comptime if num_q == 2:
                var q1_mbar = mbars.q1_wait_mbar()
                q1_mbar[qk_stage].wait()  # wait on Q1
                UMMA0Type.mma[stage_idx=qk_stage](
                    q1, k0, s1_tmem, elect=e, c_scale=0
                )
            else:
                pipeline_k.wait_k[qk_stage=qk_stage]()
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, k0, s1_tmem, elect=e, c_scale=0
                )
            _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
            comptime if qk_stage == num_qk_stages - 1:
                pipeline_k.pipeline.state.step()
        _commit(pipeline_s1.producer_mbar())

        # V_0 (2Q held for first main iter) / V_e[0] (1Q single use,
        # then V_o[0] loaded and held).
        var vlatest = pipeline_v.get_v()  # [kv1]
        pipeline_v.wait_v()  # [kv1]

        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        comptime if PARTIAL_K and num_q == 2:
            # 2Q peeled o0 contracts tile 0; last (and only) tile only when
            # total_iters == 1 -- vkm self-clamps to full otherwise.
            var vkm = ceildiv(
                min(v_eff_keys, UInt32(BN)), UInt32(UMMA1Type.MMA_K)
            )
            _pv_partial(s0_tmem, vlatest, o0_tmem, consumer_s0, 0, 0, vkm)
        else:
            _pv_full(s0_tmem, vlatest, o0_tmem, consumer_s0, 0, 0)
        _commit(pipeline_o0.producer_mbar())
        var phase: UInt32 = 0

        var c_scale: UInt32 = 0

        # BLASST: see the fused-mode `blasst_epi_wg` above -- same 1Q tail flip.
        var blasst_epi_wg: UInt32 = 1

        # vo_prev_idx tracks the held V_o slot index in 1Q (needed
        # by the deferred consumer_release_at). Declared at outer
        # scope so it's visible across the 1Q comptime-if blocks
        # (peel, main-loop V_{n-1} release, main-loop V_o[n] hold).
        # Unused in 2Q (held V is at the current pipeline state).
        var vo_prev_idx: UInt32 = 0

        # 1Q: release V_e[0] (single use); advance to V_o[0]; load
        # and HOLD vlatest = V_o[0] in vo_prev_idx for the first
        # main-loop iter's P_o @ V_o[0] MMA. State is pre-advanced
        # past the held slot so subsequent get_v() returns V_e[1].
        comptime if num_q == 1:
            var ve_idx = pipeline_v.pipeline.state.index()
            pipeline_v.pipeline.consumer_release_at(ve_idx, e)
            pipeline_v.pipeline.state.step()
            vlatest = pipeline_v.get_v()  # V_o[0]
            pipeline_v.wait_v()
            vo_prev_idx = pipeline_v.pipeline.state.index()
            pipeline_v.pipeline.state.step()  # advance; do NOT release
        # wait order
        # s0.wait(1)              # Q0@K0'
        # s1.wait(1)              # Q1@K0'
        # s0.wait(0), o0.wait(1)  # P0@V0
        # s1.wait(0), o1.wait(1)  # P1@V0

        while iter_count != 0:
            iter_count -= 1
            # Q_0 @ K_n' (2Q) / Q @ K_e[n]' (1Q), staged over
            # num_qk_stages.
            var kn = pipeline_k.get_k()  # kv_{2n-1}->[kv_{2n}]

            comptime for qk_stage in range(num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()  # kv_{2n-1}->[kv_{2n}]
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
                # 1Q: release K_e[n] stage (single use); step at last.
                comptime if num_q == 1:
                    _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                    comptime if qk_stage == num_qk_stages - 1:
                        pipeline_k.pipeline.state.step()
            _commit(pipeline_s0.producer_mbar())

            # O_1 + P_1 @ V_{n-1} (2Q) / O_o + P_o @ V_o[n-1] (1Q).
            _pv_full(s1_tmem, vlatest, o1_tmem, consumer_s1, phase, c_scale, 1)
            _commit(pipeline_o1.producer_mbar())
            c_scale = 1
            # Release V_{n-1} (2Q at current state) / V_o[n-1] (1Q at
            # vo_prev_idx; state was pre-advanced when V_o was held).
            comptime if num_q == 2:
                _commit(pipeline_v.pipeline.consumer_mbar[0]())
                pipeline_v.pipeline.state.step()  # [kv_{2n-1}]
            else:
                pipeline_v.pipeline.consumer_release_at(vo_prev_idx, e)

            # 1Q: between K_e[n] and K_o[n] -- break-check for tail
            # iter when total K-tiles is odd, else load K_o[n] by
            # reassigning kn (K_e[n] was already released per-stage
            # in the Q@K_e[n] staged loop above).
            comptime if num_q == 1:
                if iter_count == 0:
                    # Tail iter (T odd). Same alias-swap pattern as
                    # shared-KV. K_e[k] was already released per
                    # qk_stage inside the Q@K_e[k] staged loop above,
                    # so no K release is needed here.
                    vlatest = pipeline_v.get_v()  # V_e[k]
                    pipeline_v.wait_v()
                    vo_prev_idx = pipeline_v.pipeline.state.index()
                    pipeline_v.pipeline.state.step()
                    s1_tmem = s0_tmem
                    o1_tmem = o0_tmem
                    consumer_s1 = consumer_s0
                    pipeline_o1 = pipeline_o0
                    blasst_epi_wg = 0  # aliased to WG0 side
                    phase ^= 1
                    break
                iter_count -= 1
                kn = pipeline_k.get_k()  # kn = K_o[n]

            # Q_1 @ K_n' (2Q, q1 + same kn) / Q @ K_o[n]' (1Q,
            # q0 + redefined kn), staged over num_qk_stages.
            comptime for qk_stage in range(num_qk_stages):
                comptime if num_q == 2:
                    UMMA0Type.mma[stage_idx=qk_stage](
                        q1, kn, s1_tmem, elect=e, c_scale=0
                    )
                else:
                    pipeline_k.wait_k[qk_stage=qk_stage]()
                    UMMA0Type.mma[stage_idx=qk_stage](
                        q0, kn, s1_tmem, elect=e, c_scale=0
                    )
                _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                comptime if qk_stage == num_qk_stages - 1:
                    pipeline_k.pipeline.state.step()  # [kv_{2n}]->kv_{2n+1}
            _commit(pipeline_s1.producer_mbar())
            phase ^= 1

            # O_0 + P_0 @ V_n (2Q) / O_e + P_e @ V_e[n] (1Q)
            vlatest = pipeline_v.get_v()  # [kv_{2n+1}]
            pipeline_v.wait_v()  # [kv_{2n+1}]

            comptime if PARTIAL_K and num_q == 2:
                # 2Q: Vn is the last tile exactly when iter_count == 0.
                var vkm = ceildiv(
                    min(
                        v_eff_keys
                        - (total_iters_runtime - UInt32(1)) * UInt32(BN),
                        UInt32(BN),
                    ),
                    UInt32(UMMA1Type.MMA_K),
                ) if iter_count == 0 else UInt32(UMMA1Type.num_k_mmas)
                _pv_partial(
                    s0_tmem, vlatest, o0_tmem, consumer_s0, phase, 1, vkm
                )
            else:
                _pv_full(s0_tmem, vlatest, o0_tmem, consumer_s0, phase, 1)
            _commit(pipeline_o0.producer_mbar())

            # 1Q: release V_e[n] (single use); advance to V_o[n];
            # redefine vlatest = V_o[n] and hold its slot index in
            # vo_prev_idx for the next iter / epilogue. State is
            # pre-advanced past the held slot.
            comptime if num_q == 1:
                var ve_idx = pipeline_v.pipeline.state.index()
                pipeline_v.pipeline.consumer_release_at(ve_idx, e)
                pipeline_v.pipeline.state.step()
                vlatest = pipeline_v.get_v()  # V_o[n]
                pipeline_v.wait_v()
                vo_prev_idx = pipeline_v.pipeline.state.index()
                pipeline_v.pipeline.state.step()  # advance; do NOT release

        comptime if PARTIAL_K:
            var vkm = ceildiv(
                min(
                    v_eff_keys - (total_iters_runtime - UInt32(1)) * UInt32(BN),
                    UInt32(BN),
                ),
                UInt32(UMMA1Type.MMA_K),
            )
            _pv_partial(
                s1_tmem,
                vlatest,
                o1_tmem,
                consumer_s1,
                phase,
                c_scale,
                vkm,
                blasst_epi_wg,
            )
        else:
            _pv_full(
                s1_tmem,
                vlatest,
                o1_tmem,
                consumer_s1,
                phase,
                c_scale,
                blasst_epi_wg,
            )
        _commit(pipeline_o1.producer_mbar())
