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
from std.sys import size_of
from std.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
    mma_arrive_multicast,
)
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    elect_mma_arrive,
    SM100TensorAccumulator,
    KConsumerPipeline,
    VConsumerPipeline,
    StagedPipeline,
    MBarType,
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
](
    smem: SM100AttentionSMem[config],
    tmem_addr: UInt32,
    seq_id: UInt32,
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
):
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime num_qo: Int = config.num_qo
    comptime BM_mask: Int = config.PairBM_eff()
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime cta_group: Int = config.cta_group()

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
        MMA_N=config.padded_ov_depth,
        BK=BN,
        a_tmem=True,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        cta_group=cta_group,
        num_stages=num_pv_stages,
        mma_kind=mma_kind,
        b_page_dense=config.v_row_major(),
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

    s0_tmem = tmem_addr + UInt32(config.TMEM_S0)
    s1_tmem = tmem_addr + UInt32(config.TMEM_S1)
    o0_tmem = tmem_addr + UInt32(config.TMEM_O0)
    o1_tmem = tmem_addr + UInt32(config.TMEM_O1)

    # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
    var pipeline_s0 = mbars.producer_s0()
    var pipeline_s1 = mbars.producer_s1()
    # Keep consumer pointers for acquire operations (shared phase tracking)
    consumer_s0 = pipeline_s0.consumer_mbar_base
    consumer_s1 = pipeline_s1.consumer_mbar_base

    # O pipelines (producer side only; consumer wait is merged into S barriers)
    var pipeline_o0 = mbars.producer_o0()
    var pipeline_o1 = mbars.producer_o1()

    # Per-Q-tile element/byte counts.
    # 2Q: HalfBM * padded_qk_depth = 128 * d (one of two Q halves).
    # 1Q: BM * padded_qk_depth = 128 * d (the single full-BM Q tile).
    # Numerically identical because BM // num_qo == 128 in both modes
    # (mirrors the load_warp.mojo Q-TMA invariance).
    comptime q0_size = (BM // num_qo) * config.padded_qk_depth
    comptime q0_bytes = q0_size * size_of[config.qkv_dtype]()
    q0 = smem_descriptor[
        BMN=config.BM // config.num_qo,
        BK=config.BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](q_smem)
    q1 = q0 + UInt32(q0_bytes)

    comptime q_sub_bytes = HalfBM * config.BK0 * size_of[config.qkv_dtype]()

    e = elect()

    @parameter
    @always_inline
    def _commit(
        mbar: UnsafePointer[address_space=AddressSpace.SHARED, ...],
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
    # num_qo == 2]` gate and `valid_k_mmas` computation; only the duplicated
    # loop body lives here.
    @parameter
    @always_inline
    def _pv_full(
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
    ):
        comptime for pv_stage in range(num_pv_stages):
            _ = consumer_s[pv_stage].wait(wait_phase)
            UMMA1Type.mma[stage_idx=pv_stage](
                s_tmem, v, o_tmem, elect=e, c_scale=c_scale
            )

    @parameter
    @always_inline
    def _pv_partial(
        s_tmem: UInt32,
        v: MMASmemDescriptorPair,
        o_tmem: UInt32,
        consumer_s: MBarType,
        wait_phase: UInt32,
        c_scale: UInt32,
        valid_k_mmas: UInt32,
    ):
        comptime for pv_stage in range(num_pv_stages):
            _ = consumer_s[pv_stage].wait(wait_phase)
            UMMA1Type.mma_maybe_partial_k[stage_idx=pv_stage](
                s_tmem,
                v,
                o_tmem,
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
    var v_start_col: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_id, score_row
    )
    var v_eff_keys: UInt32 = num_keys - v_start_col

    comptime if config.use_fused_kv:
        # ---- Fused KV mode ----
        # In fused mode, K_nope and V alternate in a single StagedPipeline.
        # Stages: K0, V0, K1, V1, ...

        var kv_smem = smem.k_smem_base()  # same as v_smem_base in fused mode
        comptime kv_stage_bytes = config.v_cols_per_cta() * BN * size_of[
            config.qkv_dtype
        ]()

        # K descriptor: k_major for Q@K'
        kv_desc_k = smem_descriptor[
            BMN=config.k_rows_per_cta(),
            BK=config.BK0,
            swizzle_mode=config.swizzle_mode,
            is_k_major=True,
            page_dense=config.k_row_major(),
        ](kv_smem)
        # V descriptor: mn_major for P@V
        kv_desc_v = smem_descriptor[
            BMN=config.v_cols_per_cta(),
            BK=config.BN,
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
        var iter_count: UInt32 = total_iters_runtime - UInt32(3 - num_qo)

        # Release the KV slot at `release_idx`, advance to the next stage,
        # wait for it, and return its slot index. Bundles the
        # release/step/wait/capture idiom repeated across the 1Q path.
        # `consumer_mbar(idx)` with the current index is identical to the
        # no-arg `consumer_mbar()` (which forwards `state.index()`).
        @parameter
        @always_inline
        def _advance_kv(release_idx: UInt32) -> UInt32:
            _commit(kv_pipeline.consumer_mbar(release_idx))
            kv_pipeline.state.step()
            kv_pipeline.consumer_wait()
            return kv_pipeline.state.index()

        # ---- Peeled iteration ----
        # Stage 0 = K0
        kv_pipeline.consumer_wait()
        k0 = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
        UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
        _commit(pipeline_s0.producer_mbar())

        # 1Q: release K_e[0]; step to slot 1; wait. Slot 1 holds
        # K_o[0] for T >= 2 and V_e[0] for T == 1 -- diverge on
        # descriptor base only.
        comptime if num_qo == 1:
            var slot1_offset = UInt32(kv_stage_bytes) * _advance_kv(
                kv_pipeline.state.index()
            )

            # T == 1 fast path: slot 1 holds V_e[0] (load_warp produced
            # K_e[0] + V_e[0] only). Do P_e @ V_e[0] -> o0 and return.
            # Don't touch s1 / o1 -- softmax WG1 takes its matching
            # no-op path at softmax_warp.mojo:1254-1257 (gated on
            # total_iters_combined == 1 && warp_group_idx == 1).
            if total_iters_runtime == UInt32(1):
                v0 = kv_desc_v + slot1_offset
                comptime if PARTIAL_K:
                    var vkm = ceildiv(
                        min(v_eff_keys, UInt32(BN)), UInt32(UMMA1Type.MMA_K)
                    )
                    _pv_partial(s0_tmem, v0, o0_tmem, consumer_s0, 0, 0, vkm)
                else:
                    _pv_full(s0_tmem, v0, o0_tmem, consumer_s0, 0, 0)
                _commit(pipeline_o0.producer_mbar())
                _commit(kv_pipeline.consumer_mbar())  # release V_e[0]
                return

            k0 = kv_desc_k + slot1_offset

        # Q_1 @ K_0 (2Q, q1 half, same K) / Q @ K_o[0] (1Q,
        # q0 + redefined k0)
        comptime if num_qo == 2:
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].wait()
            UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)
        else:
            UMMA0Type.mma[stage_idx=0](q0, k0, s1_tmem, elect=e, c_scale=0)

        # Release K (K_0 in 2Q / K_o[0] in 1Q) and advance.
        _commit(kv_pipeline.consumer_mbar())
        kv_pipeline.state.step()
        _commit(pipeline_s1.producer_mbar())

        # Stage 1 = V_0 (2Q) / V_e[0] (1Q, single use; we will then
        # load V_o[0] and hold it for the first main-loop iter).
        kv_pipeline.consumer_wait()
        var v_prev_idx: UInt32 = kv_pipeline.state.index()
        v0 = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
        comptime if PARTIAL_K and num_qo == 2:
            # 2Q peeled o0 contracts tile 0, which is the last (and only)
            # tile only when total_iters == 1; vkm self-clamps to full
            # (v_eff_keys >= BN) otherwise.
            var vkm = ceildiv(
                min(v_eff_keys, UInt32(BN)), UInt32(UMMA1Type.MMA_K)
            )
            _pv_partial(s0_tmem, v0, o0_tmem, consumer_s0, 0, 0, vkm)
        else:
            _pv_full(s0_tmem, v0, o0_tmem, consumer_s0, 0, 0)
        _commit(pipeline_o0.producer_mbar())
        var phase: UInt32 = 0

        var c_scale: UInt32 = 0

        # 1Q: release V_e[0] (single use); load V_o[0] and hold its
        # slot index in v_prev_idx for the first main-loop iter's
        # P_o @ V_o[0] MMA.
        comptime if num_qo == 1:
            v_prev_idx = _advance_kv(v_prev_idx)

        # ---- Main loop ----
        while iter_count != 0:
            iter_count -= 1

            # Advance past held V to get to next K
            kv_pipeline.state.step()

            # Kn
            kv_pipeline.consumer_wait()
            kn = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            UMMA0Type.mma[stage_idx=0](q0, kn, s0_tmem, elect=e, c_scale=0)
            _commit(pipeline_s0.producer_mbar())

            # P1 @ V_{n-1}
            v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                )
            _commit(pipeline_o1.producer_mbar())
            c_scale = 1
            _commit(kv_pipeline.consumer_mbar(v_prev_idx))  # release V_{n-1}

            # 1Q: between K_e[n] and K_o[n] -- break-check for tail
            # iter when total K-tiles is odd, else consume K_o[n] by
            # releasing K_e[n] and reassigning kn = K_o[n].
            comptime if num_qo == 1:
                if iter_count == 0:
                    # Tail iter (T odd): no K_o[k]. The remaining work
                    # -- P_e[k] @ V_e[k] -> o0_tmem -- has the same
                    # shape as the epilogue's P_1 @ V_last -> o1_tmem.
                    # Rebind o1-side aliases to o0-side resources and
                    # fall through; epilogue does the work unchanged.
                    # release K_e[k], wait V_e[k]
                    v_prev_idx = _advance_kv(kv_pipeline.state.index())
                    s1_tmem = s0_tmem
                    o1_tmem = o0_tmem
                    consumer_s1 = consumer_s0
                    pipeline_o1 = pipeline_o0
                    phase ^= 1  # advance from this iter's K@s1 phase
                    # to the V@o0 phase the s0 wait needs.
                    break
                iter_count -= 1
                # release K_e[n], wait K_o[n]
                kn = kv_desc_k + UInt32(kv_stage_bytes) * _advance_kv(
                    kv_pipeline.state.index()
                )

            # Q_1 @ K_n (2Q, q1 + same kn) / Q @ K_o[n] (1Q,
            # q0 + redefined kn)
            comptime if num_qo == 2:
                UMMA0Type.mma[stage_idx=0](q1, kn, s1_tmem, elect=e, c_scale=0)
            else:
                UMMA0Type.mma[stage_idx=0](q0, kn, s1_tmem, elect=e, c_scale=0)
            _commit(kv_pipeline.consumer_mbar())  # release K_n / K_o[n]
            kv_pipeline.state.step()
            _commit(pipeline_s1.producer_mbar())
            phase ^= 1

            # Vn (2Q held for next iter) / V_e[n] (1Q single use,
            # then V_o[n] loaded and held).
            kv_pipeline.consumer_wait()
            v_prev_idx = kv_pipeline.state.index()
            vn = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime if PARTIAL_K and num_qo == 2:
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
                _pv_partial(s0_tmem, vn, o0_tmem, consumer_s0, phase, 1, vkm)
            else:
                _pv_full(s0_tmem, vn, o0_tmem, consumer_s0, phase, 1)
            _commit(pipeline_o0.producer_mbar())

            # 1Q: release V_e[n] (single use); load V_o[n] and hold
            # its slot in v_prev_idx for the next iter / epilogue.
            comptime if num_qo == 1:
                v_prev_idx = _advance_kv(v_prev_idx)

        # ---- Epilogue ----
        v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
        comptime if PARTIAL_K:
            var vkm = ceildiv(
                min(
                    v_eff_keys - (total_iters_runtime - UInt32(1)) * UInt32(BN),
                    UInt32(BN),
                ),
                UInt32(UMMA1Type.MMA_K),
            )
            _pv_partial(
                s1_tmem, v_prev, o1_tmem, consumer_s1, phase, c_scale, vkm
            )
        else:
            _pv_full(s1_tmem, v_prev, o1_tmem, consumer_s1, phase, c_scale)
        _commit(pipeline_o1.producer_mbar())
        _commit(kv_pipeline.consumer_mbar(v_prev_idx))  # release V_last

    else:
        # ---- Split KV mode (original) ----

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
        # Unified: subtract (3 - num_qo) -- 1 for 2Q, 2 for 1Q.
        # 1Q at total_iters == 1 takes an early-return fast path after the
        # Q @ K_e[0] staged MMA below, so the iter_count underflow at
        # T == 1 (1u32 - 2u32 wraps) is never read. Keep the raw value in
        # `total_iters_runtime` for the runtime branch.
        var total_iters_runtime: UInt32 = mask.total_iters[
            BM_mask, BN, page_size
        ](seq_id, score_row, num_keys)
        var iter_count: UInt32 = total_iters_runtime - UInt32(3 - num_qo)

        # Q_0 @ K_0' (2Q) / Q @ K_e[0]' (1Q), staged over num_qk_stages
        k0 = pipeline_k.get_k()

        comptime for qk_stage in range(num_qk_stages):
            pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
            UMMA0Type.mma[stage_idx=qk_stage](
                q0, k0, s0_tmem, elect=e, c_scale=0
            )
            # 1Q: release K_e[0] stage (single use); step at last.
            comptime if num_qo == 1:
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
        comptime if num_qo == 1:
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
        comptime if num_qo == 1:
            k0 = pipeline_k.get_k()

        # Q_1 @ K_0' (2Q, q1 half, same k0) / Q @ K_o[0]' (1Q,
        # q0 + redefined k0), staged over num_qk_stages. Mojo's
        # `comptime if` introduces a new lexical scope, so q1_mbar
        # is declared per-iteration inside the 2Q branch
        # (mbars.q1_wait_mbar() is a const accessor; declaring it
        # each iter is free at codegen since comptime for unrolls).
        comptime for qk_stage in range(num_qk_stages):
            comptime if num_qo == 2:
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
        vlatest = pipeline_v.get_v()  # [kv1]
        pipeline_v.wait_v()  # [kv1]

        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        comptime if PARTIAL_K and num_qo == 2:
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
        comptime if num_qo == 1:
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
            kn = pipeline_k.get_k()  # kv_{2n-1}->[kv_{2n}]

            comptime for qk_stage in range(num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()  # kv_{2n-1}->[kv_{2n}]
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
                # 1Q: release K_e[n] stage (single use); step at last.
                comptime if num_qo == 1:
                    _commit(pipeline_k.pipeline.consumer_mbar[qk_stage]())
                    comptime if qk_stage == num_qk_stages - 1:
                        pipeline_k.pipeline.state.step()
            _commit(pipeline_s0.producer_mbar())

            # O_1 + P_1 @ V_{n-1} (2Q) / O_o + P_o @ V_o[n-1] (1Q)
            comptime for pv_stage in range(num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                )
            _commit(pipeline_o1.producer_mbar())
            c_scale = 1
            # Release V_{n-1} (2Q at current state) / V_o[n-1] (1Q at
            # vo_prev_idx; state was pre-advanced when V_o was held).
            comptime if num_qo == 2:
                _commit(pipeline_v.pipeline.consumer_mbar[0]())
                pipeline_v.pipeline.state.step()  # [kv_{2n-1}]
            else:
                pipeline_v.pipeline.consumer_release_at(vo_prev_idx, e)

            # 1Q: between K_e[n] and K_o[n] -- break-check for tail
            # iter when total K-tiles is odd, else load K_o[n] by
            # reassigning kn (K_e[n] was already released per-stage
            # in the Q@K_e[n] staged loop above).
            comptime if num_qo == 1:
                if iter_count == 0:
                    # Tail iter (T odd). Same alias-swap pattern as
                    # fused-KV. K_e[k] was already released per
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
                    phase ^= 1
                    break
                iter_count -= 1
                kn = pipeline_k.get_k()  # kn = K_o[n]

            # Q_1 @ K_n' (2Q, q1 + same kn) / Q @ K_o[n]' (1Q,
            # q0 + redefined kn), staged over num_qk_stages.
            comptime for qk_stage in range(num_qk_stages):
                comptime if num_qo == 2:
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

            comptime if PARTIAL_K and num_qo == 2:
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
            comptime if num_qo == 1:
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
                s1_tmem, vlatest, o1_tmem, consumer_s1, phase, c_scale, vkm
            )
        else:
            _pv_full(s1_tmem, vlatest, o1_tmem, consumer_s1, phase, c_scale)
        _commit(pipeline_o1.producer_mbar())
