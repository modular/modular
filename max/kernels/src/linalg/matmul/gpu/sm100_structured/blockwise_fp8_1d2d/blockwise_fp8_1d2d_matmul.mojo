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
"""CPU entrypoint for grouped 1D-1D blockwise FP8 SM100 matmul.

This module provides the public API for launching the grouped 1D-1D blockwise
FP8 matmul kernel for Mixture of Experts (MoE) layers.

Usage:
    grouped_matmul_1d2d_blockwise_fp8[transpose_b=True, config=config](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        expert_scales,
        num_active_experts,
        ctx,
    )
"""

from collections import Optional
from math import ceildiv
from sys import size_of

from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import B200
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout as LegacyLayout, LayoutTensor
from ..structured_kernels.tile_types import create_tma_tile

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from ..structured_kernels.config import MatmulConfig
from ..structured_kernels.tile_types import GMEMTile, lt_to_tt, lt_to_tt_1d
from .blockwise_fp8_1d2d_smem import BlockwiseFP8_1D2DSmem
from .blockwise_fp8_1d2d_matmul_kernel import BlockwiseFP8_1D2DMatmulKernel


fn grouped_matmul_1d2d_blockwise_fp8[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    a_scales_type: DType,
    b_scales_type: DType,
    a_scales_layout: LegacyLayout,
    b_scales_layout: LegacyLayout,
    a_offsets_layout: LegacyLayout,
    expert_ids_layout: LegacyLayout,
    expert_scales_layout: LegacyLayout,
    transpose_b: Bool,
    //,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c_device: LayoutTensor[c_type, c_layout, ...],
    a_device: LayoutTensor[a_type, a_layout, ...],
    b_device: LayoutTensor[b_type, b_layout, ...],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, ...],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, ...],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, ...],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, ...],
    expert_scales: LayoutTensor[
        DType.float32, expert_scales_layout, MutAnyOrigin
    ],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launch grouped 1D-1D blockwise FP8 matmul kernel.

    This function sets up TMA descriptors and launches the kernel with the
    proper configuration for 1D-1D tensor layout with blockwise FP8 scaling.

    Args:
        c_device: Output tensor (total_tokens, N).
        a_device: Input A tensor (total_tokens, K).
        b_device: Weight tensor B (num_experts, N, K).
        a_scales: Scaling factors for A (K//128 x total_tokens), FP32.
        b_scales: Scaling factors for B (num_experts x N//128 x K//128), FP32.
        a_offsets: Per-expert offsets (num_active_experts + 1).
        expert_ids: Active expert IDs (num_active_experts).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        ctx: Device context.
    """
    constrained[transpose_b, "Only support transposed B"]()
    constrained[
        a_type == b_type and a_type == DType.float8_e4m3fn,
        "Only support float8_e4m3fn",
    ]()
    constrained[
        a_scales_type == b_scales_type,
        "a_scales_type and b_scales_type must match",
    ]()
    constrained[
        config.cta_group in (1, 2), "Only support cta_group == 1 or 2"
    ]()
    constrained[not config.AB_swapped, "Swapped AB is not supported"]()

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    constrained[BK == 128, "Only support BK = 128"]()

    comptime num_experts = b_layout.shape[0].value()
    comptime N = c_layout.shape[1].value()
    comptime K = a_layout.shape[1].value()
    comptime expert_n = N

    # Reshape B from (num_experts, N, K) to (num_experts * N, K)
    var b_2d = LayoutTensor[
        b_type,
        LegacyLayout.row_major(num_experts * N, K),
        b_device.origin,
        address_space = b_device.address_space,
    ](b_device.ptr)

    # Reshape b_scales from 3D (num_experts, N//128, K//128) to 2D
    comptime b_scales_expert = b_scales_layout.shape[0].value()
    comptime b_scales_n = b_scales_layout.shape[1].value()
    comptime b_scales_k = b_scales_layout.shape[2].value()
    var b_scales_2d = LayoutTensor[
        b_scales_type,
        LegacyLayout.row_major(b_scales_expert * b_scales_n, b_scales_k),
        b_scales.origin,
        address_space = b_scales.address_space,
    ](b_scales.ptr)

    # Shared memory size calculation
    comptime SmemType = BlockwiseFP8_1D2DSmem[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    # B200 SMEM limit
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Instantiate kernel type (computes TMA layouts from config)
    comptime BScalesTileType = GMEMTile[b_scales_type, b_scales_2d.layout]
    comptime CDeviceTileType = GMEMTile[c_type, c_layout]
    comptime KernelType = BlockwiseFP8_1D2DMatmulKernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        BScalesTileType.LayoutType,
        CDeviceTileType.LayoutType,
        transpose_b,
        config=config,
        static_N=expert_n,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
    ]
    comptime kernel = KernelType.run

    # Create TMA descriptors using kernel's derived legacy layouts
    var a_tma_op = create_tma_tile[
        KernelType.ATmaTile.tile_layout,
        KernelType.ATmaTile.desc_layout,
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
    ](ctx, a_device)

    var b_tma_op = create_tma_tile[
        KernelType.BTmaTile.tile_layout,
        KernelType.BTmaTile.desc_layout,
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode = config.b_swizzle,
    ](ctx, b_2d)

    var a_scales_tma_op = create_tma_tile[
        KernelType.AScalesTmaTile.tile_layout,
        KernelType.AScalesTmaTile.desc_layout,
        Index(1, BM),
    ](ctx, a_scales)

    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    # Thread configuration: 1 Load + 1 MMA + 4 Epilogue = 6 warps = 192 threads
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime epilogue_warps = 4

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        a_scales_tma_op,
        lt_to_tt(b_scales_2d),
        lt_to_tt_1d(a_offsets),
        lt_to_tt_1d(expert_ids),
        lt_to_tt_1d(expert_scales),
        lt_to_tt(c_device),
        num_active_experts,
        UInt32(K),
        grid_dim=grid_dim,
        block_dim=(32 * (load_warps + mma_warps + epilogue_warps)),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )


fn grouped_matmul_dynamic_scaled_fp8_1d2d[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    a_scales_type: DType,
    b_scales_type: DType,
    a_scales_layout: LegacyLayout,
    b_scales_layout: LegacyLayout,
    a_offsets_layout: LegacyLayout,
    expert_ids_layout: LegacyLayout,
    expert_scales_layout: LegacyLayout,
    //,
    transpose_b: Bool = True,
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[a_scales_type, a_scales_layout, MutAnyOrigin],
    b_scales: LayoutTensor[b_scales_type, b_scales_layout, MutAnyOrigin],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, MutAnyOrigin],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
    expert_scales: LayoutTensor[
        DType.float32, expert_scales_layout, MutAnyOrigin
    ],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Compatibility wrapper that matches the existing dispatch API.

    Creates the default config and calls the new structured kernel.
    """
    comptime umma_shape: IndexList[3] = Index(64, 64, 32)
    # A-scales: 1 x BM floats per pipeline stage
    comptime BM = umma_shape[0]  # cta_group=1
    comptime a_scales_smem_per_stage = BM * size_of[DType.float32]()

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(1, 1, 1),
        mma_shape=umma_shape,
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
        extra_smem_per_stage=a_scales_smem_per_stage,
    )

    grouped_matmul_1d2d_blockwise_fp8[
        transpose_b=transpose_b, config=matmul_config
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        expert_scales,
        num_active_experts,
        ctx,
    )
