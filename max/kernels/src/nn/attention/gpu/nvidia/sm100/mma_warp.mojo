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

from std.math import align_up
from std.sys import size_of
from nn.attention.gpu.nvidia.sm100.attention import FA4Config
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    elect,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    KConsumerPipeline,
    VConsumerPipeline,
    StagedPipeline,
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
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
):
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages

    var mbars = smem.misc_mbars()

    # MMA types
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        config.qkv_dtype,
        accum_type,
        MMA_M=HalfBM,
        MMA_N=BN,
        BK=align_up(config.qk_depth, config.MMA_K),
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        num_stages=num_qk_stages,
    ]
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        config.qkv_dtype,
        accum_type,
        MMA_M=HalfBM,
        MMA_N=config.padded_ov_depth,
        BK=BN,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        num_stages=num_pv_stages,
    ]

    var tmem_addr: UInt32 = smem.tmem_addr_ptr()[]
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

    comptime q0_size = HalfBM * config.padded_qk_depth
    comptime q0_bytes = q0_size * size_of[config.qkv_dtype]()
    q0 = smem_descriptor[
        BMN=config.BM // 2,
        BK=config.BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](q_smem)
    q1 = q0 + UInt32(q0_bytes)

    comptime q_sub_bytes = HalfBM * config.BK0 * size_of[config.qkv_dtype]()

    comptime if config.use_fused_kv:
        # ---- Fused KV mode ----
        # In fused mode, K_nope and V alternate in a single StagedPipeline.
        # Stages: K0, V0, K1, V1, ...

        var kv_smem = smem.k_smem_base()  # same as v_smem_base in fused mode
        comptime kv_stage_bytes = config.padded_ov_depth * BN * size_of[
            config.qkv_dtype
        ]()

        # K descriptor: k_major for Q@K'
        kv_desc_k = smem_descriptor[
            BMN=config.BN,
            BK=config.BK0,
            swizzle_mode=config.swizzle_mode,
            is_k_major=True,
        ](kv_smem)
        # V descriptor: mn_major for P@V
        kv_desc_v = smem_descriptor[
            BMN=config.padded_ov_depth,
            BK=config.BN,
            swizzle_mode=config.swizzle_mode,
            is_k_major=False,
        ](kv_smem)

        comptime KVPipeType = StagedPipeline[config.num_kv_stages, 1]
        var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[BM, BN, page_size](score_row, num_keys) - 1
        )

        e = elect()

        # ---- Peeled iteration ----
        # Stage 0 = K0
        kv_pipeline.consumer_wait()
        k0 = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
        UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
        pipeline_s0.commit_mma(e)

        # Q1 @ K0
        var q1_mbar = mbars.q1_wait_mbar()
        q1_mbar[0].wait()
        UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)
        kv_pipeline.consumer_release(e)  # release K0, step -> stage 1
        pipeline_s1.commit_mma(e)

        # Stage 1 = V0
        kv_pipeline.consumer_wait()
        var v_prev_idx: UInt32 = kv_pipeline.state.index()
        v0 = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
        comptime for pv_stage in range(num_pv_stages):
            _ = consumer_s0[pv_stage].wait(0)
            UMMA1Type.mma[stage_idx=pv_stage](
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

            # Kn
            kv_pipeline.consumer_wait()
            kn = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            UMMA0Type.mma[stage_idx=0](q0, kn, s0_tmem, elect=e, c_scale=0)
            pipeline_s0.commit_mma(e)

            # P1 @ V_{n-1}
            v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)
            c_scale = 1
            kv_pipeline.consumer_release_at(v_prev_idx, e)  # release V_{n-1}

            # Q1 @ Kn
            UMMA0Type.mma[stage_idx=0](q1, kn, s1_tmem, elect=e, c_scale=0)
            kv_pipeline.consumer_release(e)  # release Kn, step
            pipeline_s1.commit_mma(e)
            phase ^= 1

            # Vn
            kv_pipeline.consumer_wait()
            v_prev_idx = kv_pipeline.state.index()
            vn = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(num_pv_stages):
                _ = consumer_s0[pv_stage].wait(phase)
                UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, vn, o0_tmem, elect=e, c_scale=1
                )
            pipeline_o0.commit_mma(e)

        # ---- Epilogue ----
        v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
        comptime for pv_stage in range(num_pv_stages):
            _ = consumer_s1[pv_stage].wait(phase)
            UMMA1Type.mma[stage_idx=pv_stage](
                s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
            )
        pipeline_o1.commit_mma(e)
        kv_pipeline.consumer_release_at(v_prev_idx, e)  # release V_last

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

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[BM, BN, page_size](score_row, num_keys) - 1
        )

        # Q_0 @ K_0' (staged over num_qk_stages)
        k0 = pipeline_k.get_k()
        e = elect()

        comptime for qk_stage in range(num_qk_stages):
            pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
            UMMA0Type.mma[stage_idx=qk_stage](
                q0, k0, s0_tmem, elect=e, c_scale=0
            )
        pipeline_s0.commit_mma(e)

        # Q_1 @ K_0' (staged over num_qk_stages)
        var q1_mbar = mbars.q1_wait_mbar()

        comptime for qk_stage in range(num_qk_stages):
            q1_mbar[qk_stage].wait()  # wait on Q1
            UMMA0Type.mma[stage_idx=qk_stage](
                q1, k0, s1_tmem, elect=e, c_scale=0
            )
            pipeline_k.release_k[qk_stage=qk_stage](e)  # [kv0]->kv1
        pipeline_s1.commit_mma(e)

        vlatest = pipeline_v.get_v()  # [kv1]
        pipeline_v.wait_v()  # [kv1]

        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        comptime for pv_stage in range(num_pv_stages):
            _ = consumer_s0[pv_stage].wait(0)

            UMMA1Type.mma[stage_idx=pv_stage](
                s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0
            )
        pipeline_o0.commit_mma(e)
        var phase: UInt32 = 0

        var c_scale: UInt32 = 0
        # wait order
        # s0.wait(1)              # Q0@K0'
        # s1.wait(1)              # Q1@K0'
        # s0.wait(0), o0.wait(1)  # P0@V0
        # s1.wait(0), o1.wait(1)  # P1@V0

        while iter_count != 0:
            iter_count -= 1
            # Q_0 @ K_n' (staged over num_qk_stages)
            kn = pipeline_k.get_k()  # kv_{2n-1}->[kv_{2n}]

            comptime for qk_stage in range(num_qk_stages):
                pipeline_k.wait_k[qk_stage=qk_stage]()  # kv_{2n-1}->[kv_{2n}]
                UMMA0Type.mma[stage_idx=qk_stage](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
            pipeline_s0.commit_mma(e)

            # O_1 + P_1 @ V_{n-1}
            comptime for pv_stage in range(num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)
            c_scale = 1
            pipeline_v.release_v(e)  # [kv_{2n-1}]

            # Q_1 @ K_n' (staged over num_qk_stages)
            comptime for qk_stage in range(num_qk_stages):
                UMMA0Type.mma[stage_idx=qk_stage](
                    q1, kn, s1_tmem, elect=e, c_scale=0
                )
                pipeline_k.release_k[qk_stage=qk_stage](
                    e
                )  # [kv_{2n}]->kv_{2n+1}
            pipeline_s1.commit_mma(e)
            phase ^= 1

            # O_0 + P_0 @ V_n
            vlatest = pipeline_v.get_v()  # [kv_{2n+1}]
            pipeline_v.wait_v()  # [kv_{2n+1}]

            comptime for pv_stage in range(num_pv_stages):
                _ = consumer_s0[pv_stage].wait(phase)
                UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                )
            pipeline_o0.commit_mma(e)

        comptime for pv_stage in range(num_pv_stages):
            _ = consumer_s1[pv_stage].wait(phase)
            UMMA1Type.mma[stage_idx=pv_stage](
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
        pipeline_o1.commit_mma(e)
