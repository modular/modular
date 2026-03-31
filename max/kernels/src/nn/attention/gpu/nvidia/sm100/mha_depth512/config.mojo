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
"""Configuration for depth=512 pair-CTA SM100 (Blackwell) MHA kernels.

This config drives a fundamentally different kernel design from FA4:
two neighboring SMs cooperate via cta_group=2, cluster_shape=(2,1,1).
Each CTA processes BM=64 Q rows; the pair-CTA MMA instruction operates
on MMA_M=128 combined rows. P@V uses SS MMA (P in SMEM, not TMEM).

K and V are extensively sub-staged to fit in SMEM:
  - Q@K': K sub-tiled along depth into num_qk_stages=4 chunks (BK0=128)
  - P@V:  V sub-tiled along BN (reduction dim) into num_pv_stages=4 chunks
"""

from std.math import align_down
from std.sys import size_of
from std.bit import prev_power_of_two
from std.gpu.globals import WARP_SIZE
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host.info import B200


struct Depth512SM100Config[
    qkv_dtype: DType,
    *,
    rope_dtype: DType = DType.invalid,
    scale_dtype: DType = DType.invalid,
](TrivialRegisterPassable):
    # --- Type sizes ---
    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_dtype_size: Int = size_of[Self.rope_dtype]()
    comptime scale_dtype_size: Int = size_of[Self.scale_dtype]()

    # --- MMA geometry ---
    comptime MMA_K: Int = 16 if Self.qkv_dtype.is_half_float() else 32
    comptime MMA_M: Int = 128  # Pair-CTA MMA instruction M dimension
    comptime BM: Int = 64  # Per-CTA block tile M (each CTA's Q rows)

    # --- Pair-CTA constants ---
    comptime cta_group: Int = 2
    comptime num_threads: Int = 384  # 12 warps (3 warp groups of 128)

    # --- Sub-staging ---
    comptime num_qk_stages: Int = 4  # K sub-tiled along depth
    comptime num_pv_stages: Int = 2  # V sub-tiled along BN (reduction)

    # --- Hardware limits ---
    comptime sm100_smem_carveout: Int = (
        B200.shared_memory_per_multiprocessor - 1024
    )
    comptime sm100_tmem_cols: Int = 512
    comptime mbar_size: Int = size_of[DType.int64]()

    # --- Runtime fields ---
    var BN: Int
    var BK0: Int  # K sub-tile depth = qk_depth // num_qk_stages
    var BK1: Int  # V sub-tile BN = BN // num_pv_stages
    var qk_depth: Int
    var ov_depth: Int
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int

    # TMEM offsets (logical column coordinates)
    var TMEM_O: Int  # O region start (= O_lo start)
    var TMEM_O_hi: Int  # O_hi start (= ov_depth // 4, physical cols)
    var TMEM_S_even: Int  # S double-buffer even
    var TMEM_S_odd: Int  # S double-buffer odd
    var tmem_used: Int

    var num_kv_stages: Int  # Fused KV pipeline buffer slots
    var smem_used: Int
    var swizzle_mode: TensorMapSwizzle
    var p_buf_bytes: Int

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        qk_depth: Int,
        ov_depth: Int,
        swizzle_mode: TensorMapSwizzle,
        page_size: Int,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.qk_depth = qk_depth
        self.ov_depth = ov_depth
        self.swizzle_mode = swizzle_mode

        # BN from TMEM constraint: (ov_depth + 2*BN) / 2 <= 512
        # => BN <= 512 - ov_depth/2
        self.BN = min(
            256,
            align_down(Self.sm100_tmem_cols - ov_depth // 2, Self.MMA_K),
        )
        if page_size % self.BN != 0:
            self.BN = prev_power_of_two(self.BN)

        # BK sub-tile sizes
        self.BK0 = qk_depth // Self.num_qk_stages
        self.BK1 = self.BN // Self.num_pv_stages

        # TMEM layout in PHYSICAL columns (cta_group=2 halves logical→physical).
        # Each MMA with logical MMA_N produces MMA_N/2 physical TMEM columns.
        #
        # O_lo:   phys cols [0,            ov_depth/4)
        # O_hi:   phys cols [ov_depth/4,   ov_depth/2)
        # S_even: phys cols [ov_depth/2,   ov_depth/2 + BN/2)
        # S_odd:  phys cols [ov_depth/2 + BN/2, ov_depth/2 + BN)
        #
        # For ov_depth=512, BN=256:
        #   O: phys 0-255, S: phys 256-511 → exactly 512 physical cols.
        self.TMEM_O = 0
        self.TMEM_O_hi = ov_depth // 4
        self.TMEM_S_even = ov_depth // 2
        self.TMEM_S_odd = self.TMEM_S_even + self.BN // 2
        self.tmem_used = self.TMEM_S_odd + self.BN // 2

        # SMEM budget
        var smem_use = size_of[UInt32]()  # tmem_addr

        # Q: BM rows × full depth
        q_bytes = Self.BM * qk_depth * Self.qkv_dtype_size

        # P buffer: BM × BN (softmax writes P here for SS MMA P@V)
        self.p_buf_bytes = Self.BM * self.BN * Self.qkv_dtype_size

        # Correction: BM float32 elements
        correction_bytes = Self.BM * size_of[DType.float32]()

        # Fixed barriers:
        #   count-256: PO_lo (P ready + O_lo rescaled)
        #   count-128: PO_hi, S_even consumer, S_odd consumer, C producer, C consumer
        #   count-1:   S_even producer, S_odd producer, O_mma_lo, O_mma_hi
        misc_mbars_fixed = 10
        smem_use += q_bytes + self.p_buf_bytes + correction_bytes
        smem_use += misc_mbars_fixed * Self.mbar_size

        # KV pipeline: fused K/V sub-tiles share buffer slots.
        # K sub-tile: BN//2 × BK0 elements
        # V sub-tile: (BN/num_pv_stages) × ov_depth elements//2
        # These have equal element count when qk_depth == ov_depth
        # and num_qk_stages == num_pv_stages.
        kv_sub_tile_bytes = (self.BN // 2) * self.BK0 * Self.qkv_dtype_size
        kv_barrier_bytes = 2 * Self.mbar_size  # per buffer slot
        bytes_per_slot = kv_sub_tile_bytes + kv_barrier_bytes

        remaining = Self.sm100_smem_carveout - smem_use
        self.num_kv_stages = remaining // bytes_per_slot
        smem_use += self.num_kv_stages * bytes_per_slot

        self.smem_used = smem_use

    @always_inline
    def rope_depth(self) -> Int:
        return self.qk_depth - self.ov_depth

    @always_inline
    def num_qo(self) -> Int:
        return 1

    @always_inline
    def correction_smem_elements(self) -> Int:
        return Self.BM

    @always_inline
    def num_active_warps_per_group(self) -> Int:
        return 4

    @always_inline
    def num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()

    def supported(self) -> Bool:
        return (
            self.BN >= 32
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
            and self.num_kv_stages >= Self.num_qk_stages
        )

    def description(self) -> String:
        return String(
            "qk_depth = ",
            self.qk_depth,
            "\nov_depth = ",
            self.ov_depth,
            "\nBN = ",
            self.BN,
            "\nBK0 = ",
            self.BK0,
            "\nBK1 = ",
            self.BK1,
            "\nnum_kv_stages = ",
            self.num_kv_stages,
            "\ntmem_used = ",
            self.tmem_used,
            " (physical: ",
            self.tmem_used // 2,
            ")",
            "\nsmem_used = ",
            self.smem_used,
            "\nsm100_smem_carveout = ",
            Self.sm100_smem_carveout,
            "\nqkv_dtype_size = ",
            Self.qkv_dtype_size,
            "\np_buf_bytes = ",
            self.p_buf_bytes,
        )
