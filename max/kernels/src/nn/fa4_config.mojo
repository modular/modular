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
"""FA4 (Flash Attention 4) configuration for SM100 (Blackwell) kernels."""

from math import ceildiv, align_up, align_down, gcd
from sys import size_of
from sys import env_get_bool
from bit import prev_power_of_two
from gpu.globals import WARP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200


comptime EnableForcedOrdering = env_get_bool[
    "FA4ForcedSoftmaxOrdering", False
]()
comptime EnableEarlyAdd = env_get_bool["FA4AddEarly", False]()


struct FA4Config(TrivialRegisterPassable):
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var depth: Int
    var padded_depth: Int  # align_up(depth, 64)
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var TMEM_C0: Int
    var TMEM_C1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_qk_stages: Int  # Stages for Q@K' (K loading pipelining)
    var num_pv_stages: Int  # Stages for P@V (P writing pipelining)
    var smem_used: Int
    var dtype_size: Int
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var split_m: Bool
    var swizzle_mode: TensorMapSwizzle

    comptime MMA_K = 16
    comptime sm100_smem_carveout = B200.shared_memory_per_multiprocessor - 1024
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    @always_inline
    fn num_qo(self) -> Int:
        return 2

    fn __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        dtype_size: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        is_mla: Bool = False,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.split_m = depth > 128 and not is_mla
        if self.split_m:
            self.BM = 128
            self.MMA_M = 64
        else:
            self.BM = 256
            self.MMA_M = 128
        self.dtype_size = dtype_size
        self.swizzle_mode = swizzle_mode
        swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)

        if self.split_m:
            self.BN = min(
                256, align_down(Self.sm100_tmem_cols - depth, Self.MMA_K)
            )
            # TODO : delete this as soon as we define spliting BN across the pages
            if page_size % self.BN != 0:
                self.BN = prev_power_of_two(self.BN)
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_O0 = Self.TMEM_S0 + self.BN
            self.TMEM_C0 = Self.TMEM_S0 + self.BN // 2

            self.TMEM_S1 = Self.TMEM_S0 + 16 << 16
            self.TMEM_P1 = self.TMEM_P0 + 16 << 16
            self.TMEM_O1 = self.TMEM_O0 + 16 << 16
            self.TMEM_C1 = self.TMEM_C0 + 16 << 16
            self.tmem_used = self.TMEM_O1 + depth
        else:
            # we use two q and o
            # determine BN via tmem:
            # 2*BN + 2*depth <= 512 -> BN + depth <= 256
            self.BN = min(
                256,
                align_down(
                    (Self.sm100_tmem_cols // 2 - self.padded_depth),
                    Self.MMA_K,
                ),
            )
            # TODO : delete this as soon as we define spliting BN across the pages
            if page_size % self.BN != 0:
                self.BN = prev_power_of_two(self.BN)
            self.TMEM_S1 = Self.TMEM_S0 + self.BN
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_P1 = self.TMEM_S1
            self.TMEM_C0 = self.TMEM_P0 + self.BN // 2
            self.TMEM_C1 = self.TMEM_P1 + self.BN // 2
            self.TMEM_O0 = self.TMEM_S1 + self.BN
            self.TMEM_O1 = self.TMEM_O0 + self.padded_depth
            self.tmem_used = self.TMEM_O1 + self.padded_depth

        # We have the following resources that need smem barriers:
        # KV: num_kv_stages
        # S: 2
        # C: 2
        # O: 2
        # softmax order: 2
        # q: 1, for Q1 synchronization
        # 4 for `o_pipeline` (2 consumer + 2 producer)
        # we need two per stage
        # Compute staging for Q@K' and P@V operations
        # num_qk_stages: Controls how K loading is pipelined for Q@K' MMA
        # num_pv_stages: Controls how P writing is pipelined for P@V MMA
        #
        # For Q@K': K can be loaded in stages, MMA starts after first stage arrives
        # For P@V: V must be complete, but P writing can be staged to unblock MMA sooner
        #
        # Divisibility constraints:
        # - num_qk_stages must divide padded_depth (for K column splitting)
        # - num_pv_stages must divide BN (for P column splitting)
        # - Both must respect MMA_K alignment (16 elements)
        #
        # Staging infrastructure:
        # - SM100TensorAccumulatorSS.mma and SM100TensorAccumulatorTS.mma support
        #   stage_idx parameter for processing in chunks when num_stages > 1
        # - KPipeline and VPipeline structs support separate K/V barrier management
        # - FA4MiscMBars is parameterized by num_pv_stages for S barriers
        # - load() loads K in num_qk_stages chunks with separate barriers per stage
        # - store_exp() writes P in num_pv_stages chunks with barriers per stage
        # - mma() loops over qk_stages for Q@K' and pv_stages for P@V
        #
        # Computed staging values:
        # - num_qk_stages: How many chunks to split K processing into for Q@K' MMA
        # - num_pv_stages: How many chunks to split P writing into for P@V MMA
        #
        if is_mla:
            self.num_qk_stages = 1
            self.num_pv_stages = 1
        else:
            # Q@K' staging is enabled: MMA processes K in num_qk_stages chunks,
            # allowing register pressure reduction and potential overlap.
            self.num_qk_stages = gcd(
                self.padded_depth // swizzle_elems,
                self.padded_depth // Self.MMA_K,
            )
            # P@V staging requires coordinated changes to store_exp and mma functions:
            # - store_exp must write P in stages and signal barriers per stage
            # - mma must wait for each P stage barrier before processing
            if self.BN % 32 != 0:
                self.num_pv_stages = 1
            elif self.BN % 3 == 0:
                self.num_pv_stages = 3
            else:
                self.num_pv_stages = 2

        var smem_use = 4
        # Compute misc_mbars fixed size (barriers that don't scale with num_kv_stages):
        # - S barriers: 2 * (1 + num_pv_stages) per warp group = 4 + 4*num_pv_stages
        # - C barriers: 4 (C0/C1 producer/consumer)
        # - Order barriers: 2 (only when EnableForcedOrdering)
        # - Q1Sync barriers: num_qk_stages
        # - O barriers: 4 (2 producer + 2 consumer)
        # Total fixed = 10 + order_barrier_count + 2*num_pv_stages + num_qk_stages
        comptime order_barrier_count: Int = 2 if EnableForcedOrdering else 0
        misc_mbars_fixed_size = (
            10
            + order_barrier_count
            + 2 * self.num_pv_stages
            + self.num_qk_stages
        )
        smem_use += misc_mbars_fixed_size * Self.mbar_size

        # BK0: K-dimension chunk size for Q@K' per stage
        self.BK0 = self.padded_depth // self.num_qk_stages
        # BK1: Full BN since V loading is not staged (V must be complete for P@V)
        self.BK1 = self.BN
        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        smem_use += self.BM * self.padded_depth * dtype_size
        # Barriers per KV stage (K/V barriers scale with num_kv_stages):
        # - K loading: 2 * num_qk_stages (producer + consumer per stage)
        # - V loading: 2 (single stage, producer + consumer) if separate_kv
        # Note: For MLA (separate_kv=False), V barriers are 0
        smem_per_kv = (
            2 * self.BN * self.padded_depth * dtype_size
            + 2 * Self.mbar_size * self.num_qk_stages  # K barriers
            + 2 * Self.mbar_size  # V barriers (MHA, 1 stage)
        )
        self.num_kv_stages = (
            Self.sm100_smem_carveout - smem_use
        ) // smem_per_kv
        # Example staging values (when implemented):
        # depth= 64: num_qk_stages=1, num_pv_stages=2
        # depth=128: num_qk_stages=2, num_pv_stages=2
        # depth=256: num_qk_stages=4, num_pv_stages=2
        # Currently both are 1 until staged operations are implemented.
        smem_use += self.num_kv_stages * smem_per_kv
        # Add space for correction smem when not using tmem for correction
        smem_use += (
            self.BM * Self.num_correction_cols * size_of[DType.float32]()
        )
        self.smem_used = smem_use

    fn supported(self) -> Bool:
        return (
            self.depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    fn correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols

    fn num_active_warps_per_group(self) -> Int:
        return 4

    fn num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()
