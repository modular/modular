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

from std.math import ceildiv, align_up, align_down, gcd
from std.sys import size_of
from std.sys import get_defined_bool
from std.bit import prev_power_of_two
from std.gpu.globals import WARP_SIZE
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host.info import B200


comptime EnableForcedOrdering = get_defined_bool[
    "FA4ForcedSoftmaxOrdering", False
]()
comptime EnableEarlyAdd = get_defined_bool["FA4AddEarly", False]()


struct FA4Config[
    qkv_dtype: DType,
    *,
    rope_dtype: DType = DType.invalid,
    scale_dtype: DType = DType.invalid,
](TrivialRegisterPassable):
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var qk_depth: Int
    var padded_qk_depth: Int  # align_up(qk_depth, swizzle_elems)
    var ov_depth: Int
    var padded_ov_depth: Int
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
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var split_m: Bool
    var swizzle_mode: TensorMapSwizzle
    var use_fused_kv: Bool

    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_dtype_size: Int = size_of[Self.rope_dtype]()
    comptime scale_dtype_size: Int = size_of[Self.scale_dtype]()

    comptime MMA_K: Int = 16 if Self.qkv_dtype.is_half_float() else 32
    comptime sm100_smem_carveout = B200.shared_memory_per_multiprocessor - 1024
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    @always_inline
    def num_qo(self) -> Int:
        return 2

    @always_inline
    def q_nope_bytes(self) -> Int:
        """Q nope region bytes: BM * padded_ov_depth * dtype_size."""
        return self.BM * self.padded_ov_depth * Self.qkv_dtype_size

    @always_inline
    def q_rope_bytes(self) -> Int:
        """Q rope region bytes. Uses rope_dtype_size when set, else dtype_size.
        """
        return self.BM * self.rope_depth() * Self.rope_dtype_size

    @always_inline
    def rope_depth(self) -> Int:
        """Depth of the rope part. Calculated as:
        padded_qk_depth - padded_ov_depth (0 for MHA where qk_depth == ov_depth).
        """
        return self.padded_qk_depth - self.padded_ov_depth

    @always_inline
    def num_rope_buffers(self) -> Int:
        """Number of separate rope smem buffers (fused mode only).

        In fused mode K tiles alternate with V tiles in the pipeline.
        At most ceildiv(num_kv_stages, 2) K tiles can be in-flight
        simultaneously, so we only need that many rope buffers.
        For MHA (rope_depth=0), no rope buffers are needed.
        """
        if self.use_fused_kv and self.rope_depth() > 0:
            return ceildiv(self.num_kv_stages, 2)
        return 0

    @always_inline
    def num_k_scale_bufs(self) -> Int:
        """Number of staged k_scale smem buffers.

        In fused mode, K tiles alternate with V tiles so at most
        ceildiv(num_kv_stages, 2) K tiles are in-flight simultaneously.
        In split mode, each KV stage has its own K buffer.
        Returns 0 when scale_dtype_size == 0 (no per-token scaling).
        """
        if self.scale_dtype_size == 0:
            return 0
        if self.use_fused_kv:
            return ceildiv(self.num_kv_stages, 2)
        return self.num_kv_stages

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        qk_depth: Int,
        ov_depth: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        is_mla: Bool,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.qk_depth = qk_depth
        self.split_m = qk_depth > 128 and not is_mla
        if self.split_m:
            self.BM = 128
            self.MMA_M = 64
        else:
            self.BM = 256
            self.MMA_M = 128
        self.swizzle_mode = swizzle_mode
        swizzle_elems = swizzle_mode.bytes() // Self.qkv_dtype_size
        self.ov_depth = ov_depth
        self.padded_qk_depth = align_up(qk_depth, swizzle_elems)
        self.padded_ov_depth = align_up(ov_depth, swizzle_elems)

        if self.split_m:
            self.BN = min(
                256, align_down(Self.sm100_tmem_cols - qk_depth, Self.MMA_K)
            )
            # TODO : delete this as soon as we define splitting BN across the pages
            if page_size % self.BN != 0:
                self.BN = prev_power_of_two(self.BN)
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_O0 = Self.TMEM_S0 + self.BN
            self.TMEM_C0 = Self.TMEM_S0 + self.BN // 2

            self.TMEM_S1 = Self.TMEM_S0 + 16 << 16
            self.TMEM_P1 = self.TMEM_P0 + 16 << 16
            self.TMEM_O1 = self.TMEM_O0 + 16 << 16
            self.TMEM_C1 = self.TMEM_C0 + 16 << 16
            self.tmem_used = self.TMEM_O1 + ov_depth
        else:
            # we use two q and o
            # determine BN via tmem:
            # 2*BN + 2*ov_depth <= 512 -> BN + ov_depth <= 256
            self.BN = min(
                256,
                align_down(
                    (Self.sm100_tmem_cols // 2 - self.padded_ov_depth),
                    Self.MMA_K,
                ),
            )
            # TODO : delete this as soon as we define splitting BN across the pages
            if page_size % self.BN != 0:
                self.BN = prev_power_of_two(self.BN)
            self.TMEM_S1 = Self.TMEM_S0 + self.BN
            self.TMEM_P0 = Self.TMEM_S0
            self.TMEM_P1 = self.TMEM_S1
            self.TMEM_C0 = self.TMEM_P0 + self.BN // 2
            self.TMEM_C1 = self.TMEM_P1 + self.BN // 2
            self.TMEM_O0 = self.TMEM_S1 + self.BN
            self.TMEM_O1 = self.TMEM_O0 + self.padded_ov_depth
            self.tmem_used = self.TMEM_O1 + self.padded_ov_depth

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
        else:
            # Q@K' staging is enabled: MMA processes K in num_qk_stages chunks,
            # allowing register pressure reduction and potential overlap.
            self.num_qk_stages = gcd(
                self.padded_qk_depth // swizzle_elems,
                self.padded_qk_depth // Self.MMA_K,
            )

        # P@V staging requires coordinated changes to store_exp and mma functions:
        # - store_exp must write P in stages and signal barriers per stage
        # - mma must wait for each P stage barrier before processing
        self.num_pv_stages = 2

        var smem_use = 4
        # Compute misc_mbars fixed size (barriers that don't scale with num_kv_stages):
        # - S consumers: 2 * num_pv_stages (num_pv_stages per warp group)
        # - S producers: 2 (1 per warp group)
        # - C barriers: 4 (C0/C1 producer/consumer)
        # - Order barriers: 2 (only when EnableForcedOrdering)
        # - Q1Sync barriers: num_qk_stages
        # - O producers: 2 (O consumers reuse S_consumer[0], not separate)
        # Total fixed = 8 + order_barrier_count + 2*num_pv_stages + num_qk_stages
        comptime order_barrier_count: Int = 2 if EnableForcedOrdering else 0
        misc_mbars_fixed_size = (
            8
            + order_barrier_count
            + 2 * self.num_pv_stages
            + self.num_qk_stages
        )
        smem_use += misc_mbars_fixed_size * Self.mbar_size

        rope_depth = self.padded_qk_depth - self.padded_ov_depth

        # smem use is (NOTE: smem uses padded depth):
        # BM*depth*dtype_size + num_kv_stages*(2*mbar_size + BN*depth*dtype_size) <= smem_remaining
        # num_kv_stages <= (smem_remaining - 2*BM*depth*dtype_size) // (2*mbar_size + BN*depth*dtype_size)
        # Q region: when rope_dtype_size > 0, Q nope and Q rope have different
        # dtype sizes (e.g. FP8 nope + BF16 rope for per-token-scale MLA).
        var qk_depth_bytes: Int
        comptime if Self.rope_dtype_size > 0:
            qk_depth_bytes = (
                self.padded_ov_depth * Self.qkv_dtype_size
                + rope_depth * Self.rope_dtype_size
            )
        else:
            qk_depth_bytes = self.padded_qk_depth * Self.qkv_dtype_size
        smem_use += self.BM * qk_depth_bytes
        # q_scale: always 1 buffer (per-token scale only; 0 when no scaling).
        smem_use += self.BM * Self.scale_dtype_size
        # Add space for correction smem when not using tmem for correction
        smem_use += (
            self.BM * Self.num_correction_cols * size_of[DType.float32]()
        )

        # We use one of two strategies:
        #  - split kv: more efficient/neater to track smem separately.
        #              nope and rope smem can be tracked together
        #  - fused kv: if the maximum number of `nope`s we can store is odd
        #              then splitting would require us to round down to
        #              an even number of stages. Fusing avoids this.
        # We divide bytes needed by `k` and `v` into shared and k-specific:
        bytes_per_kv = (
            self.BN * self.padded_ov_depth * Self.qkv_dtype_size
            + 2 * Self.mbar_size
        )  # KV barriers
        bytes_per_k = (
            self.BN * rope_depth * Self.rope_dtype_size
            + self.BN * Self.scale_dtype_size
        )  # k scale buffers

        # total k + v bytes is thus
        # fused_pipeline_stages * bytes_per_kv
        #   + ceildiv(fused_pipeline_stages,2) * bytes_per_k
        # If `fused_pipeline_stages` is even, we split the pipelines.

        remaining = Self.sm100_smem_carveout - smem_use
        # remaining >= fused_pipeline_stages * bytes_per_kv
        #   + ceildiv(fused_pipeline_stages,2) * bytes_per_k
        #   >= fused_pipeline_stages * bytes_per_kv
        #   +  (fused_pipeline_stages/2) * bytes_per_k
        #   = fused_pipeline_stages * (bytes_per_kv + bytes_per_k/2)
        fused_stages = remaining // (bytes_per_kv + bytes_per_k // 2)
        bytes_used = (
            fused_stages * bytes_per_kv + ceildiv(fused_stages, 2) * bytes_per_k
        )
        if bytes_used > remaining:
            fused_stages -= 1
            bytes_used = (
                fused_stages * bytes_per_kv
                + ceildiv(fused_stages, 2) * bytes_per_k
            )
        smem_use += bytes_used

        if fused_stages % 2 == 1:  # odd, fused
            self.use_fused_kv = True
            self.num_kv_stages = fused_stages
            self.num_qk_stages = 1
        else:
            self.use_fused_kv = False
            self.num_kv_stages = fused_stages // 2
            if is_mla:
                self.num_qk_stages = 1
            else:
                # we try to split num_qk_stages
                self.num_qk_stages = gcd(
                    self.padded_qk_depth // swizzle_elems,
                    self.padded_qk_depth // Self.MMA_K,
                )
                # we need an extra bytes
                barrier_bytes_per_stage = (
                    self.num_kv_stages * 2 * Self.mbar_size
                )
                total_smem_use = (
                    smem_use
                    + (self.num_qk_stages - 1) * barrier_bytes_per_stage
                )
                if total_smem_use < Self.sm100_smem_carveout:
                    smem_use = total_smem_use
                else:
                    self.num_qk_stages = 1

        # BK0: K-dimension chunk size for Q@K' per stage
        self.BK0 = self.padded_qk_depth // self.num_qk_stages
        # BK1: Full BN since V loading is not staged (V must be complete
        # for P@V)
        self.BK1 = self.BN
        self.smem_used = smem_use

    def supported(self) -> Bool:
        return (
            self.qk_depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    def description(self) -> String:
        return String(
            "qk_depth = ",
            self.qk_depth,
            "\nBN = ",
            self.BN,
            "\nnum_kv_stages = ",
            self.num_kv_stages,
            "\ntmem_used = ",
            self.tmem_used,
            "\nsmem_used = ",
            self.smem_used,
            "\nsm100_smem_carveout = ",
            Self.sm100_smem_carveout,
            "\nnope_dtype_size = ",
            Self.qkv_dtype_size,
            "\nrope_dtype_size = ",
            Self.rope_dtype_size,
            "\nscale_dtype_size = ",
            Self.scale_dtype_size,
            "\nuse_fused_kv = ",
            self.use_fused_kv,
        )

    def correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols

    def num_active_warps_per_group(self) -> Int:
        return 4

    def num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()
