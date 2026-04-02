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
"""Barrier infrastructure for depth=512 pair-CTA SM100 attention kernels.

Manages all mbarrier resources for the warp-specialized kernel. Each CTA has
12 warps (384 threads, 3 warp groups of 128) divided into 4 warp kinds:
softmax (warps 0-3, 128 threads), correction (warps 4-7, 128 threads),
MMA (warp 8), load (warp 9), and spare (warps 10-11).

O is split into two halves (O_lo, O_hi) produced by separate P@V MMA
instructions (MMA_N=ov_depth/2 each). Each half has its own commit barrier
(O_mma_lo, O_mma_hi) and ready barrier (PO_lo, PO_hi), enabling the
correction warp to rescale O_lo while the MMA is still producing O_hi.

Barrier layout (low to high index):
    [0]   PO_lo           (count-256: softmax 128 + correction 128)
    [1]   PO_hi           (count-128: correction 128 only)
    [2]   S_even consumer (count-128: softmax 4 warps)
    [3]   S_odd consumer  (count-128: softmax 4 warps)
    [4]   C producer      (count-128: softmax 4 warps)
    [5]   C consumer      (count-128: correction 4 warps)
    [6]   S_even producer (count-1: MMA mma_arrive)
    [7]   S_odd producer  (count-1: MMA mma_arrive)
    [8]   O_mma_lo        (count-1: MMA mma_arrive after V_lo stages)
    [9]   O_mma_hi        (count-1: MMA mma_arrive after V_hi stages)
    [10..] KV pipeline    (count-1: 2 per stage, producer+consumer pairs)

Synchronization flow per KV iteration (even S buffer):
    Load:       TMA K/V sub-tile → arrive KV[stage] producer
    MMA:        wait KV + S_even consumer → Q@K'→S → mma_arrive S_even producer
                wait KV + PO_lo → P@V_lo→O_lo → mma_arrive O_mma_lo
                wait PO_hi → P@V_hi→O_hi (reuse KV slots) → mma_arrive O_mma_hi
    Softmax:    wait S_even producer → load S TMEM→regs → arrive S_even consumer
                exp(S) → write P to SMEM → arrive PO_lo
                write correction → arrive C producer
    Correction: wait C producer → read correction
                wait O_mma_lo → rescale O_lo → arrive PO_lo
                wait O_mma_hi → rescale O_hi → arrive PO_hi
                → arrive C consumer
"""

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    MBarType,
    RolePipeline,
    ProducerPipeline,
    ConsumerPipeline,
)


struct Depth512MBars[
    num_kv_stages: Int,
](TrivialRegisterPassable):
    """Manages all mbarrier resources for depth=512 pair-CTA attention.

    Parameters:
        num_kv_stages: Number of fused KV pipeline buffer slots.
    """

    # ---- comptime barrier offsets ---------------------------------------------
    # Count-256 (index 0).
    comptime PO_lo_offset: Int = 0

    # Count-128 section (indices 1-5).
    comptime PO_hi_offset: Int = 1
    comptime S_even_consumer_offset: Int = 2
    comptime S_odd_consumer_offset: Int = 3
    comptime C_producer_offset: Int = 4
    comptime C_consumer_offset: Int = 5

    # Count-1 section (indices 6-9).
    comptime S_even_producer_offset: Int = 6
    comptime S_odd_producer_offset: Int = 7
    comptime O_mma_lo_offset: Int = 8
    comptime O_mma_hi_offset: Int = 9

    # KV pipeline barriers (count-1, 2 per stage).
    comptime KV_offset: Int = 10
    comptime KV_barriers: Int = 2 * Self.num_kv_stages

    # Totals.
    comptime num_high_count_barriers: Int = 6  # indices 0-5 are count >= 128
    comptime size: Int = 10 + Self.KV_barriers

    # ---- storage --------------------------------------------------------------
    var mbar_base: MBarType

    # ---- construction ---------------------------------------------------------

    @always_inline
    def __init__(out self, mbar_base: MBarType):
        """Wrap a base mbarrier pointer from Depth512AttentionSMem.mbar_base().
        """
        self.mbar_base = mbar_base

    # ---- initialization -------------------------------------------------------

    @staticmethod
    @always_inline
    def _init_count(lane_idx: Int32) -> Int32:
        """Return mbarrier thread count for the given barrier index.

        Index 0: count-256 (PO_lo: softmax 128 + correction 128).
        Index 1: count-128 (PO_hi: correction 128 only).
        Indices 2-5: count-128 (S consumers, C producer, C consumer).
        Indices 6+: count-1 (S producers, O mma lo/hi, KV pipeline).
        """
        if lane_idx == Int32(Self.PO_lo_offset):
            return 256
        if lane_idx < Int32(Self.num_high_count_barriers):
            return 128
        return 1

    @always_inline
    def init(self, *, lane_idx: Int32):
        """Initialize all barriers. One call per warp lane (lane_idx = tid % 32).

        With num_kv_stages <= 11, total size <= 32 = WARP_SIZE, so a single
        wave covers all barriers.
        """
        if lane_idx < Int32(Self.size):
            self.mbar_base[lane_idx].init(Self._init_count(lane_idx))

    # ---- S pipeline accessors (score MMA double-buffer) -----------------------
    # Even/odd alternation handled by caller; each buffer has 1 producer
    # barrier (count-1, MMA arrive) and 1 consumer barrier (count-128, softmax).

    comptime SPipelineProducer = RolePipeline[1, True, 1, 1, 2]
    comptime SPipelineConsumer = ConsumerPipeline[1]

    @always_inline
    def producer_s_even(self) -> Self.SPipelineProducer:
        """MMA warp: acquire waits on S_even consumer (buffer free),
        commit arrives at S_even producer (S written to TMEM)."""
        return {
            self.mbar_base + Self.S_even_producer_offset,
            self.mbar_base + Self.S_even_consumer_offset,
        }

    @always_inline
    def producer_s_odd(self) -> Self.SPipelineProducer:
        """MMA warp: acquire waits on S_odd consumer (buffer free),
        commit arrives at S_odd producer (S written to TMEM)."""
        return {
            self.mbar_base + Self.S_odd_producer_offset,
            self.mbar_base + Self.S_odd_consumer_offset,
        }

    @always_inline
    def consumer_s_even(self) -> Self.SPipelineConsumer:
        """Softmax warp: wait on S_even producer (S ready in TMEM),
        release arrives at S_even consumer (S loaded, buffer free)."""
        return {
            self.mbar_base + Self.S_even_producer_offset,
            self.mbar_base + Self.S_even_consumer_offset,
        }

    @always_inline
    def consumer_s_odd(self) -> Self.SPipelineConsumer:
        """Softmax warp: wait on S_odd producer (S ready in TMEM),
        release arrives at S_odd consumer (S loaded, buffer free)."""
        return {
            self.mbar_base + Self.S_odd_producer_offset,
            self.mbar_base + Self.S_odd_consumer_offset,
        }

    # ---- C pipeline accessors (correction factor handoff) ---------------------

    @always_inline
    def producer_c(self) -> ProducerPipeline[1]:
        """Softmax warp: acquire waits on C consumer (buffer free),
        commit arrives at C producer (correction written to SMEM)."""
        return {
            self.mbar_base + Self.C_producer_offset,
            self.mbar_base + Self.C_consumer_offset,
        }

    @always_inline
    def consumer_c(self) -> ConsumerPipeline[1]:
        """Correction warp: wait on C producer (correction ready),
        release arrives at C consumer (correction consumed)."""
        return {
            self.mbar_base + Self.C_producer_offset,
            self.mbar_base + Self.C_consumer_offset,
        }

    # ---- O pipeline accessors (split into lo/hi for pipelining) ---------------
    # O_mma_lo/hi (count-1): MMA arrives via mma_arrive after V_lo/V_hi.
    # PO_lo (count-256): softmax (128, P ready) + correction (128, O_lo rescaled).
    # PO_hi (count-128): correction (128, O_hi rescaled) only.

    @always_inline
    def producer_o_lo(self) -> RolePipeline[1, True, 1, 1, 2]:
        """MMA warp P@V_lo pipeline.
        Acquire: wait on PO_lo (P ready + previous O_lo rescaled).
        Commit: mma_arrive at O_mma_lo (V_lo accumulation done)."""
        return {
            self.mbar_base + Self.O_mma_lo_offset,
            self.mbar_base + Self.PO_lo_offset,
        }

    @always_inline
    def producer_o_hi(self) -> RolePipeline[1, True, 1, 1, 2]:
        """MMA warp P@V_hi pipeline.
        Acquire: wait on PO_hi (previous O_hi rescaled).
        Commit: mma_arrive at O_mma_hi (V_hi accumulation done)."""
        return {
            self.mbar_base + Self.O_mma_hi_offset,
            self.mbar_base + Self.PO_hi_offset,
        }

    @always_inline
    def consumer_o_lo(self) -> ConsumerPipeline[1]:
        """Correction warp O_lo pipeline.
        Wait: on O_mma_lo (V_lo accumulation done, O_lo safe to read).
        Release: arrive at PO_lo (O_lo rescaling done, 128 threads)."""
        return {
            self.mbar_base + Self.O_mma_lo_offset,
            self.mbar_base + Self.PO_lo_offset,
        }

    @always_inline
    def consumer_o_hi(self) -> ConsumerPipeline[1]:
        """Correction warp O_hi pipeline.
        Wait: on O_mma_hi (V_hi accumulation done, O_hi safe to read).
        Release: arrive at PO_hi (O_hi rescaling done, 128 threads)."""
        return {
            self.mbar_base + Self.O_mma_hi_offset,
            self.mbar_base + Self.PO_hi_offset,
        }

    # ---- PO raw accessors (prologue arrives) ----------------------------------

    @always_inline
    def po_lo_mbar(self) -> MBarType:
        """Raw pointer to PO_lo barrier.

        Softmax arrives here (128 threads) after writing P to SMEM.
        Correction also arrives (128 threads) via consumer_o_lo().release().
        MMA waits for all 256 arrives before P@V_lo.
        """
        return self.mbar_base + Self.PO_lo_offset

    @always_inline
    def po_hi_mbar(self) -> MBarType:
        """Raw pointer to PO_hi barrier.

        Correction arrives here (128 threads) via consumer_o_hi().release().
        MMA waits for all 128 arrives before P@V_hi.
        """
        return self.mbar_base + Self.PO_hi_offset

    # ---- KV pipeline barriers ------------------------------------------------

    @always_inline
    def get_kv_mbars(self) -> MBarType:
        """Base pointer for KV pipeline barriers.

        Layout: 2 × num_kv_stages barriers in {producer, consumer} pairs.
        Used by load warp (producer) and MMA warp (consumer).
        """
        return self.mbar_base + Self.KV_offset

    # ---- utility -------------------------------------------------------------

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        """Total number of mbarriers managed by this struct."""
        return UInt32(Self.size)
