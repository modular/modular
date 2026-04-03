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
"""MMA warp logic for depth=512 pair-CTA SM100 attention.

Orchestrates Q@K' and P@V matrix multiplications using pair-CTA SS MMA
(cta_group=2). Both MMAs read both operands from SMEM, unlike FA4 which
uses TS MMA for P@V.

Q@K' produces S in TMEM (double-buffered S_even/S_odd across iterations).

P@V is split into two MMA groups:
  P@V_lo (MMA_N=ov_depth/2): produces O_lo in TMEM cols [0, ov_depth/2)
  P@V_hi (MMA_N=ov_depth/2): produces O_hi in TMEM cols [ov_depth/2, ov_depth)

The split enables pipelining: after V_lo completes, O_mma_lo fires and the
correction warp can start rescaling O_lo while V_hi sub-stages are still
running. V_lo and V_hi occupy separate KV pipeline slots, each holding a
[BK1, ov_depth/4] half-tile per CTA (with cta_group=2, each CTA contributes
ov_depth/4 columns to the B operand).

K is sub-staged into num_qk_stages=4 depth chunks (BK0 each). V is
sub-staged into num_pv_stages=2 BN chunks (BK1 each), with V_lo and V_hi
in separate slots (4 total V slots per iteration). Both use the fused KV
pipeline where K and V sub-tiles share buffer slots.

CTA role split (cta_group=2):
    Leader CTA (even rank): Owns all pipeline interactions — waits on KV
        producer barriers, issues MMA, releases KV consumer barriers with
        cta_group=2 commit (fences MMA read of both CTAs' SMEM, then
        signals both CTAs' consumer barriers), and commits S/O barriers.
    Peer CTA (odd rank): Returns immediately.
"""

from std.sys import size_of
from std.gpu.primitives.cluster import (
    block_rank_in_cluster,
    elect_one_sync_with_mask,
)
from std.gpu.compute.arch.mma_nvidia_sm100 import mma_arrive_multicast
from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulatorSS,
    StagedPipeline,
    elect,
)
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .smem import Depth512AttentionSMem
from std.utils.index import Index

# from std.gpu import thread_idx_int as thread_idx


@always_inline
def depth512_mma[
    MaskType: MHAMask,
    qkv_dtype: DType,
    config: Depth512SM100Config[qkv_dtype],
    page_size: Int,
](
    smem: Depth512AttentionSMem[config=config],
    score_row: UInt32,
    num_keys: UInt32,
    mask: MaskType,
):
    comptime accum_type = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime BK0 = config.BK0
    comptime BK1 = config.BK1
    comptime MMA_M = config.MMA_M
    comptime ov_depth = config.ov_depth
    comptime ov_half = ov_depth // 2
    comptime num_qk_stages = config.num_qk_stages
    comptime num_pv_stages = config.num_pv_stages
    comptime num_kv_stages = config.num_kv_stages
    comptime cta_group = config.cta_group
    comptime qkv_size = size_of[qkv_dtype]()

    # Full pair-CTA M dimension for mask computations.
    comptime PairBM = BM * 2

    comptime assert BK0 % config.MMA_K == 0, "BK0 must be a multiple of MMA_K"
    comptime assert BK1 % config.MMA_K == 0, "BK1 must be a multiple of MMA_K"

    # ---- MMA types -----------------------------------------------------------

    # Q@K' → S: SS MMA, cta_group=2
    comptime UMMA_QK = SM100TensorAccumulatorSS[
        qkv_dtype,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=BN,
        BK=BK0,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        cta_group=cta_group,
    ]

    # P@V_lo → O_lo: SS MMA, MMA_N=ov_depth/2, cta_group=2
    comptime UMMA_PV_lo = SM100TensorAccumulatorSS[
        qkv_dtype,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=ov_half,
        BK=BK1,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        cta_group=cta_group,
    ]

    # P@V_hi → O_hi: same type as lo, different TMEM target and V offset
    comptime UMMA_PV_hi = SM100TensorAccumulatorSS[
        qkv_dtype,
        accum_type,
        MMA_M=MMA_M,
        MMA_N=ov_half,
        BK=BK1,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        cta_group=cta_group,
    ]

    # ---- TMEM addresses ------------------------------------------------------

    var tmem_addr = smem.tmem_addr_ptr()[]
    o_lo_tmem = tmem_addr + UInt32(config.TMEM_O)
    o_hi_tmem = tmem_addr + UInt32(config.TMEM_O_hi)
    s_even_tmem = tmem_addr + UInt32(config.TMEM_S_even)
    s_odd_tmem = tmem_addr + UInt32(config.TMEM_S_odd)

    # ---- SMEM descriptors ----------------------------------------------------

    # Q: [BM, BK0] per depth sub-stage, k_major.
    q_desc = smem_descriptor[
        BMN=BM,
        BK=BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](smem.q_smem())

    # K: [BN//2, BK0] per CTA per pipeline slot, k_major.
    kv_desc_k = smem_descriptor[
        BMN=BN // 2,
        BK=BK0,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](smem.kv_smem_base())

    # P: descriptor with [BM, BN] strides (matching the full P buffer layout).
    p_desc = smem_descriptor[
        BMN=BM,
        BK=BN,
        swizzle_mode=config.swizzle_mode,
        is_k_major=True,
    ](smem.p_smem())

    # V: [BK1, ov_depth//4] per CTA per pipeline slot, mn_major.
    # V_lo and V_hi occupy separate pipeline slots. Each slot holds one
    # [BK1, ov_depth//4] half-tile with stride ov_depth//4. The MMA with
    # MMA_N=ov_depth/2 and cta_group=2 reads ov_depth/4 cols from each CTA.
    kv_desc_v = smem_descriptor[
        BMN=ov_depth // 4,
        BK=BK1,
        swizzle_mode=config.swizzle_mode,
        is_k_major=False,
    ](smem.kv_smem_base())

    # ---- Byte offsets for sub-staging ----------------------------------------

    comptime q_sub_bytes = BM * BK0 * qkv_size
    comptime kv_stage_bytes = (BN // 2) * BK0 * qkv_size
    # P is [BM, BN] k_major with swizzle.  Advancing by BK1 columns skips
    # BK1/sw_K swizzle blocks of BM*sw_K elements each = BM*BK1 elements.
    comptime p_sub_bytes = BM * BK1 * qkv_size

    # ---- Barrier / pipeline setup --------------------------------------------

    var mbars = Depth512MBars[num_kv_stages](smem.mbar_base())

    # S double-buffered pipelines (MMA is the producer of S in TMEM).
    var pipeline_s_even = mbars.producer_s_even()
    pipeline_s_even.state._phase = 1
    var pipeline_s_odd = mbars.producer_s_odd()
    pipeline_s_odd.state._phase = 1

    # O_lo pipeline: acquire waits PO_lo, commit arrives O_mma_lo.
    # Override phase to 0: first acquire must block until PO_lo fires.
    var pipeline_o_lo = mbars.producer_o_lo()
    pipeline_o_lo.state._phase = 0

    # O_hi pipeline: acquire waits PO_hi, commit arrives O_mma_hi.
    var pipeline_o_hi = mbars.producer_o_hi()
    pipeline_o_hi.state._phase = 0

    # KV consumer pipeline (load warp is the producer).
    comptime KVPipeType = StagedPipeline[num_kv_stages, 1]
    var kv_pipeline: KVPipeType = {mbars.get_kv_mbars()}

    # ---- Iteration bounds (must match load_warp exactly) ---------------------

    var kv_row: UInt32 = mask.start_column[PairBM, BN, page_size](score_row)
    var iter_count: UInt32 = (
        mask.last_masked_set_end[PairBM, BN, page_size](score_row, num_keys) - 1
    )

    comptime check_mask = (
        mask.nonfull_sets[PairBM, BN]()[0] == TileMaskStatus.UNKNOWN_MASK
    )

    # ---- CTA identity --------------------------------------------------------
    var is_leader = block_rank_in_cluster() % 2 == 0

    if not is_leader:
        return

    e = elect()

    # CTA mask for multicast arrive: signal both CTAs in the pair.
    comptime cta_mask = UInt16(0x3)
    var elect_one = elect_one_sync_with_mask()

    # ---- Helper: P@V_lo + P@V_hi with separate commits ----------------------

    @parameter
    @always_inline
    def pv_mma(*, is_first: Bool):
        """Execute P@V_lo → commit O_mma_lo, then P@V_hi → commit O_mma_hi.

        V_lo and V_hi occupy separate KV pipeline slots. Each loop
        consumes and releases its own slots independently.

        Args:
            is_first: True for the peeled iteration (c_scale=0 on stage 0).
        """
        # -- P@V_lo → O_lo (own pipeline slots) --
        pipeline_o_lo.acquire()
        comptime for pv_stage in range(num_pv_stages):
            kv_pipeline.consumer_wait()
            UMMA_PV_lo.mma(
                p_desc + UInt32(pv_stage * p_sub_bytes),
                kv_desc_v + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                o_lo_tmem,
                c_scale=UInt32(0) if is_first and pv_stage == 0 else UInt32(1),
                elect=e,
            )
            if elect_one:
                mma_arrive_multicast[cta_group](
                    kv_pipeline.consumer_mbar(), cta_mask
                )
            kv_pipeline.state.step()
        if elect_one:
            mma_arrive_multicast[cta_group](
                pipeline_o_lo.producer_mbar(), cta_mask
            )
        pipeline_o_lo.step()

        # -- P@V_hi → O_hi (own pipeline slots) --
        pipeline_o_hi.acquire()
        comptime for pv_stage in range(num_pv_stages):
            kv_pipeline.consumer_wait()
            UMMA_PV_hi.mma(
                p_desc + UInt32(pv_stage * p_sub_bytes),
                kv_desc_v + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                o_hi_tmem,
                c_scale=UInt32(0) if is_first and pv_stage == 0 else UInt32(1),
                elect=e,
            )
            if elect_one:
                mma_arrive_multicast[cta_group](
                    kv_pipeline.consumer_mbar(), cta_mask
                )
            kv_pipeline.state.step()
        if elect_one:
            mma_arrive_multicast[cta_group](
                pipeline_o_hi.producer_mbar(), cta_mask
            )
        pipeline_o_hi.step()

    # ---- Peeled first iteration ----------------------------------------------

    # Q@K' → S_even (4 K sub-stages)
    comptime for qk_stage in range(num_qk_stages):
        kv_pipeline.consumer_wait()
        UMMA_QK.mma(
            q_desc + UInt32(qk_stage * q_sub_bytes),
            kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
            s_even_tmem,
            c_scale=UInt32(0) if qk_stage == 0 else UInt32(1),
            elect=e,
        )
        if elect_one:
            mma_arrive_multicast[cta_group](
                kv_pipeline.consumer_mbar(), cta_mask
            )
        kv_pipeline.state.step()
    if elect_one:
        mma_arrive_multicast[cta_group](
            pipeline_s_even.producer_mbar(), cta_mask
        )
    pipeline_s_even.step()

    # P@V_lo → O_lo, then P@V_hi → O_hi
    pv_mma(is_first=True)

    # ---- Main loop (alternating S_even / S_odd) ------------------------------

    var s_cur_pipeline = pipeline_s_odd
    var s_cur_tmem = s_odd_tmem
    var s_nxt_pipeline = pipeline_s_even
    var s_nxt_tmem = s_even_tmem

    while iter_count != 0:
        iter_count -= 1
        kv_row += UInt32(BN)

        # Mask check: skip fully-masked tiles.
        comptime if check_mask:
            if (
                mask.status(
                    Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                    Index[dtype=DType.int32](PairBM, BN),
                )
                == TileMaskStatus.FULL_MASK
            ):
                continue

        # Q@K' → S (current buffer)
        s_cur_pipeline.acquire()
        comptime for qk_stage in range(num_qk_stages):
            kv_pipeline.consumer_wait()
            UMMA_QK.mma(
                q_desc + UInt32(qk_stage * q_sub_bytes),
                kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index(),
                s_cur_tmem,
                c_scale=UInt32(0) if qk_stage == 0 else UInt32(1),
                elect=e,
            )
            if elect_one:
                mma_arrive_multicast[cta_group](
                    kv_pipeline.consumer_mbar(), cta_mask
                )
            kv_pipeline.state.step()
        if elect_one:
            mma_arrive_multicast[cta_group](
                s_cur_pipeline.producer_mbar(), cta_mask
            )
        s_cur_pipeline.step()

        # P@V_lo → O_lo, then P@V_hi → O_hi
        pv_mma(is_first=False)

        # Swap S buffers for next iteration.
        var tmp_pipeline = s_cur_pipeline
        s_cur_pipeline = s_nxt_pipeline
        s_nxt_pipeline = tmp_pipeline
        var tmp_tmem = s_cur_tmem
        s_cur_tmem = s_nxt_tmem
        s_nxt_tmem = tmp_tmem
