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

from math import align_up
from sys import size_of
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout.tma_async import SharedMemBarrier
from nn.fa4_config import FA4Config, EnableForcedOrdering
from nn.sm100_attention_utils import (
    SharedMemPointer,
    FA4MiscMBars,
    elect,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    KConsumerPipeline,
    VConsumerPipeline,
)
from nn.mha_mask import MHAMask
from linalg.arch.sm100.mma import smem_descriptor


@always_inline
fn fa4_mma[
    qkv_type: DType,
    MaskType: MHAMask,
    config: FA4Config,
    page_size: Int,
](
    mbars: FA4MiscMBars[
        num_qk_stages = config.num_qk_stages,
        num_pv_stages = config.num_pv_stages,
        num_kv_stages = config.num_kv_stages,
        separate_kv=True,
        use_order_barriers=EnableForcedOrdering,
    ],
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

    # Offset calculations
    comptime q_offset: Int32 = 0
    comptime kv_offset: Int32 = q_offset + Int32(
        config.BM * config.padded_depth
    )
    comptime correction_offset: Int32 = (
        kv_offset
        + Int32(2 * config.num_kv_stages * config.padded_depth * config.BN)
    ) * Int32(size_of[qkv_type]()) // Int32(size_of[DType.float32]())
    comptime mbar_offset = (correction_offset + Int32(config.BM)) * Int32(
        size_of[DType.float32]()
    ) // Int32(size_of[SharedMemBarrier]())

    comptime MiscMBarsType = type_of(mbars)

    # MMA types
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        qkv_type,
        accum_type,
        MMA_M=HalfBM,
        MMA_N=BN,
        BK = align_up(config.depth, config.MMA_K),
        swizzle_a = config.swizzle_mode,
        swizzle_b = config.swizzle_mode,
        transpose_b=True,
        num_stages=num_qk_stages,
    ]
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        qkv_type,
        accum_type,
        MMA_M=HalfBM,
        MMA_N = config.padded_depth,
        BK=BN,
        swizzle_b = config.swizzle_mode,
        transpose_b=False,
        num_stages=num_pv_stages,
    ]

    var tmem_addr: UInt32 = (
        mbars.mbar_base + MiscMBarsType.num_mbars()
    ).bitcast[UInt32]()[]
    var q_smem: SharedMemPointer[Scalar[qkv_type]] = (
        mbars.mbar_base - mbar_offset
    ).bitcast[Scalar[qkv_type]]() + q_offset

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

    comptime q0_size = HalfBM * config.padded_depth
    comptime q0_bytes = q0_size * size_of[qkv_type]()
    q0 = smem_descriptor[
        BMN = config.BM // 2,
        BK = config.BK0,
        swizzle_mode = config.swizzle_mode,
        is_k_major=True,
    ](q_smem)
    q1 = q0 + UInt32(q0_bytes)
    kv_smem = q_smem + 2 * q0_size

    comptime q_sub_bytes = HalfBM * config.BK0 * size_of[qkv_type]()

    var k_smem = q_smem + config.BM * config.padded_depth
    var v_smem = (
        k_smem + (config.BN * config.padded_depth) * config.num_kv_stages
    )
    comptime KPipeType = KConsumerPipeline[qkv_type, config]
    comptime VPipeType = VConsumerPipeline[qkv_type, config]
    var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem}
    var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem}

    # We peel the first iteration, as we want to wait on q1
    var iter_count: UInt32 = (
        mask.total_iters[BM, BN, page_size](score_row, num_keys) - 1
    )

    # Q_0 @ K_0' (staged over num_qk_stages)
    k0 = pipeline_k.get_k()
    e = elect()

    @parameter
    for qk_stage in range(num_qk_stages):
        pipeline_k.wait_k[qk_stage=qk_stage]()  # [kv0]
        UMMA0Type.mma[stage_idx=qk_stage](q0, k0, s0_tmem, elect=e, c_scale=0)
    pipeline_s0.commit_mma(e)

    # Q_1 @ K_0' (staged over num_qk_stages)
    var q1_mbar = mbars.q1_wait_mbar()

    @parameter
    for qk_stage in range(num_qk_stages):
        q1_mbar[qk_stage].wait()  # wait on Q1
        UMMA0Type.mma[stage_idx=qk_stage](q1, k0, s1_tmem, elect=e, c_scale=0)
        pipeline_k.release_k[qk_stage=qk_stage](e)  # [kv0]->kv1
    pipeline_s1.commit_mma(e)

    vlatest = pipeline_v.get_v()  # [kv1]
    pipeline_v.wait_v()  # [kv1]

    # For the first V tile in the current KV stage buffer:
    # Use the SAME base pointer you used for K (no manual offset).
    @parameter
    for pv_stage in range(num_pv_stages):
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

        @parameter
        for qk_stage in range(num_qk_stages):
            pipeline_k.wait_k[qk_stage=qk_stage]()  # kv_{2n-1}->[kv_{2n}]
            UMMA0Type.mma[stage_idx=qk_stage](
                q0, kn, s0_tmem, elect=e, c_scale=0
            )
        pipeline_s0.commit_mma(e)

        # O_1 + P_1 @ V_{n-1}
        @parameter
        for pv_stage in range(num_pv_stages):
            _ = consumer_s1[pv_stage].wait(phase)
            UMMA1Type.mma[stage_idx=pv_stage](
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
        pipeline_o1.commit_mma(e)
        c_scale = 1
        pipeline_v.release_v(e)  # [kv_{2n-1}]

        # Q_1 @ K_n' (staged over num_qk_stages)
        @parameter
        for qk_stage in range(num_qk_stages):
            UMMA0Type.mma[stage_idx=qk_stage](
                q1, kn, s1_tmem, elect=e, c_scale=0
            )
            pipeline_k.release_k[qk_stage=qk_stage](e)  # [kv_{2n}]->kv_{2n+1}
        pipeline_s1.commit_mma(e)
        phase ^= 1

        # O_0 + P_0 @ V_n
        vlatest = pipeline_v.get_v()  # [kv_{2n+1}]
        pipeline_v.wait_v()  # [kv_{2n+1}]

        @parameter
        for pv_stage in range(num_pv_stages):
            _ = consumer_s0[pv_stage].wait(phase)
            UMMA1Type.mma[stage_idx=pv_stage](
                s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
            )
        pipeline_o0.commit_mma(e)

    @parameter
    for pv_stage in range(num_pv_stages):
        _ = consumer_s1[pv_stage].wait(phase)
        UMMA1Type.mma[stage_idx=pv_stage](
            s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
        )
    pipeline_o1.commit_mma(e)
