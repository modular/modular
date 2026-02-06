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

"""Configuration types for SM100 structured convolution kernels.

This module provides:
- Conv2dProblemShape: Defines the convolution problem geometry
- Conv2dConfig: Kernel configuration (tile sizes, pipeline depths, etc.)

The convolution is mapped to GEMM as implicit im2col:
- M = N_batch * H_out * W_out (output spatial)
- N = K_filters (output channels)
- K = C_in * R * S (input channels * filter spatial)
"""

from math import ceildiv

from gpu.host.info import B200
from gpu.host.nvidia.tma import TensorMapSwizzle
from sys import size_of
from utils.index import IndexList
from utils.numerics import get_accum_type


# =============================================================================
# Conv2dProblemShape - Convolution problem geometry
# =============================================================================


struct Conv2dProblemShape(Copyable, Movable):
    """Defines 2D convolution problem geometry.

    Layouts:
    - Activation: NHWC (batch, height, width, channels)
    - Filter: KRSC (output_channels, filter_h, filter_s, input_channels)
    - Output: NHWC (batch, out_height, out_width, output_channels)

    For Fprop with stride=1, no dilation, this maps to GEMM as:
    - M = N * H_out * W_out
    - N = K (output channels)
    - K = C * R * S (input channels * filter area)
    """

    # Input activation shape (NHWC)
    var batch: Int
    var in_height: Int
    var in_width: Int
    var in_channels: Int

    # Filter shape (KRSC) - K=out_channels, R=height, S=width, C=in_channels
    var out_channels: Int
    var filter_h: Int
    var filter_w: Int

    # Padding (symmetric for simplicity)
    var pad_h: Int
    var pad_w: Int

    # Stride (currently only stride=1 supported)
    var stride_h: Int
    var stride_w: Int

    # Dilation (currently only dilation=1 supported)
    var dilation_h: Int
    var dilation_w: Int

    # Groups (currently only groups=1 supported)
    var groups: Int

    @always_inline
    fn __init__(
        out self,
        batch: Int,
        in_height: Int,
        in_width: Int,
        in_channels: Int,
        out_channels: Int,
        filter_h: Int,
        filter_w: Int,
        pad_h: Int = 0,
        pad_w: Int = 0,
        stride_h: Int = 1,
        stride_w: Int = 1,
        dilation_h: Int = 1,
        dilation_w: Int = 1,
        groups: Int = 1,
    ):
        self.batch = batch
        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.dilation_h = dilation_h
        self.dilation_w = dilation_w
        self.groups = groups

    # ========== Derived Dimensions ==========

    @always_inline
    fn out_height(self) -> Int:
        """Compute output height."""
        var effective_filter_h = (self.filter_h - 1) * self.dilation_h + 1
        return (
            self.in_height + 2 * self.pad_h - effective_filter_h
        ) // self.stride_h + 1

    @always_inline
    fn out_width(self) -> Int:
        """Compute output width."""
        var effective_filter_w = (self.filter_w - 1) * self.dilation_w + 1
        return (
            self.in_width + 2 * self.pad_w - effective_filter_w
        ) // self.stride_w + 1

    # ========== GEMM Dimension Mapping ==========

    @always_inline
    fn gemm_m(self) -> Int:
        """GEMM M dimension = batch * output_height * output_width."""
        return self.batch * self.out_height() * self.out_width()

    @always_inline
    fn gemm_n(self) -> Int:
        """GEMM N dimension = output_channels."""
        return self.out_channels

    @always_inline
    fn gemm_k(self) -> Int:
        """GEMM K dimension = input_channels * filter_height * filter_width."""
        return self.in_channels * self.filter_h * self.filter_w

    # ========== Tile Count Helpers ==========

    @always_inline
    fn num_m_tiles(self, tile_m: Int) -> Int:
        """Number of tiles in M dimension."""
        return ceildiv(self.gemm_m(), tile_m)

    @always_inline
    fn num_n_tiles(self, tile_n: Int) -> Int:
        """Number of tiles in N dimension."""
        return ceildiv(self.gemm_n(), tile_n)

    @always_inline
    fn num_k_tiles(self, tile_k: Int) -> Int:
        """Number of tiles in K dimension."""
        return ceildiv(self.gemm_k(), tile_k)


# =============================================================================
# Conv2dConfig - Kernel configuration
# =============================================================================


@fieldwise_init
struct Conv2dConfig[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
](Copyable, Movable):
    """Configuration for SM100 Conv2D kernel.

    This mirrors MatmulConfig but with conv-specific semantics.

    Parameters:
        act_type: Activation (input) data type.
        filter_type: Filter (weight) data type.
        out_type: Output data type.
    """

    # ========== Tile Shape ==========
    # block_tile_shape maps to GEMM tile: (M, N, K)
    # - M: batch * output_spatial (multiple output pixels)
    # - N: output_channels (multiple filters)
    # - K: input_channels * filter_area (reduction dimension)
    var block_tile_shape: IndexList[3]

    # MMA instruction shape
    var mma_shape: IndexList[3]

    # Output tile shape for epilogue
    var output_tile_shape: IndexList[2]

    # ========== Pipeline Configuration ==========
    var num_pipeline_stages: Int
    var num_output_stages: Int
    var num_accum_pipeline_stages: Int
    var num_clc_pipeline_stages: Int
    var k_group_size: Int

    # ========== Cluster Configuration ==========
    var cluster_shape: IndexList[3]
    var cta_group: Int

    # ========== Memory Layout ==========
    var a_swizzle: TensorMapSwizzle
    var b_swizzle: TensorMapSwizzle
    var c_swizzle: TensorMapSwizzle

    # ========== Scheduling ==========
    var block_swizzle_size: Int

    # ========== Derived Types ==========

    @staticmethod
    @always_inline
    fn accum_type() -> DType:
        """Accumulator type derived from output type."""
        return get_accum_type[Self.out_type]()

    @staticmethod
    @always_inline
    fn default_bf16() -> Self:
        """Default configuration for BF16 conv2d (VAE-optimized).

        Uses 2-SM cluster mode (cta_group=2) with 128×128 block tiles, matching
        the standard SM100 matmul configuration pattern.

        For cta_group=2 with MMA_M=256, MMA_N=256:
        - block_tile_shape = mma_shape // cta_group = (128, 128, 64)
        - output_tile_shape = (128, 32) - each output tile is 128 rows × 32 cols
        - cluster_shape[0] = 2 (2 CTAs in M dimension)

        Pipeline stages are dynamically computed to maximize SMEM utilization.
        """
        var config = Self(
            # 128×128 block tiles with cta_group=2 (mma_shape = 2 * block_tile)
            block_tile_shape=IndexList[3](128, 128, 64),
            mma_shape=IndexList[3](256, 256, 16),
            # Output tile: (128, 32) with SWIZZLE_64B - matches matmul pattern
            output_tile_shape=IndexList[2](128, 32),
            # Pipeline depths - num_pipeline_stages will be computed below
            num_pipeline_stages=4,  # Placeholder, will be maximized
            num_output_stages=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            k_group_size=1,
            # 2-SM cluster mode
            cluster_shape=IndexList[3](2, 1, 1),
            cta_group=2,
            # Swizzle modes for bank-conflict-free access
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            # c_swizzle=SWIZZLE_64B matches output_tile_n=32 (32*2=64 bytes)
            c_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            # Block swizzle for L2 locality
            block_swizzle_size=1,
        )

        # Dynamically compute optimal pipeline stages based on SMEM budget
        config._maximize_pipeline_stages_by_default()

        return config^

    @staticmethod
    @always_inline
    fn default_bf16_1sm() -> Self:
        """Default configuration for BF16 conv2d using 1-SM mode.

        Uses 1-SM mode (cta_group=1) with 128×128 block tiles, matching
        the CUTLASS example configuration.

        For cta_group=1 with MMA_M=128, MMA_N=128, MMA_K=16:
        - block_tile_shape = (128, 128, 64) for tile sizes
        - mma_shape = (128, 128, 16) for MMA instruction shape
        - output_tile_shape = (128, 32) with c_swizzle=SWIZZLE_64B
        - cluster_shape = (1, 1, 1) (single CTA per cluster)

        Pipeline stages are dynamically computed to maximize SMEM utilization.
        """
        var config = Self(
            # 128×128 block tiles with cta_group=1
            block_tile_shape=IndexList[3](128, 128, 64),
            # MMA shape: M=128, N=128, K=16 (K is reduction per MMA instruction)
            mma_shape=IndexList[3](128, 128, 16),
            # Output tile: (128, 32) with SWIZZLE_64B
            output_tile_shape=IndexList[2](128, 32),
            # Pipeline depths - num_pipeline_stages will be computed below
            num_pipeline_stages=4,  # Placeholder, will be maximized
            num_output_stages=2,
            num_accum_pipeline_stages=2,
            num_clc_pipeline_stages=2,
            k_group_size=1,
            # 1-SM mode
            cluster_shape=IndexList[3](1, 1, 1),
            cta_group=1,
            # Swizzle modes for bank-conflict-free access
            a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
            # c_swizzle=SWIZZLE_64B matches output_tile_n=32 (32*2=64 bytes)
            c_swizzle=TensorMapSwizzle.SWIZZLE_64B,
            # Block swizzle for L2 locality
            block_swizzle_size=1,
        )

        # Dynamically compute optimal pipeline stages based on SMEM budget
        config._maximize_pipeline_stages_by_default()

        return config^

    @staticmethod
    @always_inline
    fn default_fp16() -> Self:
        """Default configuration for FP16 conv2d."""
        # Same as BF16 for now - inherits dynamic pipeline stages
        return Self.default_bf16()

    fn _maximize_pipeline_stages_by_default(mut self):
        """Dynamically compute optimal pipeline stages based on SMEM budget.

        This mirrors MatmulConfig._maximize_pipeline_stages_by_default() since
        the SMEM layout is identical (activation/filter tiles, output tiles,
        pipeline barriers, CLC storage).
        """
        # B200 has 228KB SMEM per SM; reserve 1KB for misc overhead
        comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

        # Fixed overhead: output tiles + tmem addr/mbar
        var c_smem_bytes = (
            self.output_tile_shape[0]
            * self.output_tile_shape[1]
            * self.num_output_stages
            * size_of[Self.out_type]()
        )
        # Add tmem addr (4 bytes) and tmem dealloc mbar (8 bytes)
        var output_smem_bytes = c_smem_bytes + 12

        # CLC pipeline: response 128B + clc mbar 16B + clc-load mbar 16B = 160B per stage
        var clc_smem_bytes = 160 * self.num_clc_pipeline_stages

        # MMA output pipeline barriers
        var mma_output_smem_bytes = self.num_accum_pipeline_stages * 16

        # Per-stage cost: activation tiles + filter tiles + barriers
        var act_smem_bytes_per_stage = (
            self.block_tile_shape[0]
            * self.block_tile_shape[2]
            * size_of[Self.act_type]()
        )
        var filter_smem_bytes_per_stage = (
            self.block_tile_shape[1]
            * self.block_tile_shape[2]
            * size_of[Self.filter_type]()
        )
        # Include 16 bytes for consumer and producer mbar per stage
        var input_smem_per_stage = (
            act_smem_bytes_per_stage + filter_smem_bytes_per_stage + 16
        )

        # Compute maximum pipeline stages
        self.num_pipeline_stages = (
            b200_smem
            - output_smem_bytes
            - clc_smem_bytes
            - mma_output_smem_bytes
        ) // input_smem_per_stage
