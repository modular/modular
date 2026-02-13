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
"""CPU entrypoint for grouped 1D-1D block-scaled SM100 matmul.

This module provides the public API for launching the grouped 1D-1D matmul
kernel for Mixture of Experts (MoE) layers.

Usage:
    grouped_matmul_1d1d_nvfp4[transpose_b=True, config=config](
        c_tensor,  # Output: (total_tokens, N)
        a_tensor,  # Input A: (total_tokens, K)
        a_offsets,  # Per-expert offsets into A
        a_scale_offsets,  # Per-expert scale offsets
        b_tensor,  # Weights B: (num_experts, N, K)
        expert_ids,  # Active expert IDs
        a_scales,  # Scale factors for A
        b_scales,  # Scale factors for B
        expert_scales,  # Per-expert output scaling
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
from layout import Layout as LegacyLayout, LayoutTensor, RuntimeLayout
from ..structured_kernels.tile_types import create_tma_tile
from ..structured_kernels.tile_types import lt_to_tt, lt_to_tt_1d

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP8_SF_DTYPE,
)
from gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from ..structured_kernels.config import BlockScaledMatmulConfig
from .grouped_1d1d_matmul_kernel import Grouped1D1DMatmulKernel


fn grouped_matmul_1d1d_nvfp4[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    a_offsets_layout: LegacyLayout,
    a_scale_offsets_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    expert_ids_layout: LegacyLayout,
    sfa_dtype: DType,
    sfa_layout: LegacyLayout,
    sfb_dtype: DType,
    _sfb_layout: LegacyLayout,
    expert_scale_layout: LegacyLayout,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
](
    c_device: LayoutTensor[c_type, c_layout, ...],
    a_device: LayoutTensor[a_type, a_layout, ...],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, ...],
    a_scale_offsets: LayoutTensor[DType.uint32, a_scale_offsets_layout, ...],
    _b_device: LayoutTensor[b_type, b_layout, ...],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, ...],
    a_scales: LayoutTensor[sfa_dtype, sfa_layout, MutAnyOrigin],
    _b_scales: LayoutTensor[sfb_dtype, _sfb_layout, MutAnyOrigin],
    expert_scales: LayoutTensor[
        DType.float32, expert_scale_layout, MutAnyOrigin
    ],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Launch grouped 1D-1D block-scaled matmul kernel.

    This function sets up TMA descriptors and launches the kernel with the
    proper configuration for 1D-1D tensor layout.

    Args:
        c_device: Output tensor (total_tokens, N).
        a_device: Input A tensor (total_tokens, K).
        a_offsets: Per-expert offsets (num_active_experts + 1).
        a_scale_offsets: Per-expert scale offsets (num_active_experts).
        _b_device: Weight tensor B (num_experts, N, K).
        expert_ids: Active expert IDs (num_active_experts).
        a_scales: Scale factors for A (5D).
        _b_scales: Scale factors for B (6D).
        expert_scales: Per-expert output scaling (num_experts).
        num_active_experts: Number of active experts.
        ctx: Device context.
    """
    constrained[
        transpose_b,
        "Only support transposed B",
    ]()

    comptime assert (
        sfa_dtype == sfb_dtype
    ), "Only support same scales dtype for A and B"
    comptime assert sfa_dtype in (
        MXFP8_SF_DTYPE,
        NVFP4_SF_DTYPE,
    ), "Only support MXFP8_SF_DTYPE or NVFP4_SF_DTYPE for scales"

    constrained[not config.AB_swapped, "swap AB is not supported"]()

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    constrained[
        config.cta_group in (1, 2), "Only support cta_group == 1 or 2"
    ]()

    constrained[
        config.k_group_size == 1,
        "Only support k_group_size == 1",
    ]()

    constrained[
        config.num_split_k == 1,
        "Only support split_k == 1",
    ]()

    constrained[
        config.num_pipeline_stages % config.k_group_size == 0,
        "num_pipeline_stages must be a multiple of k_group_size",
    ]()

    constrained[
        a_scales.rank == 5,
        "a_scales must be 5D tensors",
    ]()
    constrained[
        _b_scales.rank == 6,
        "b_scales must be 6D tensors",
    ]()

    # Reshape b_scales from 6D to 5D
    comptime sfb_layout = LegacyLayout.row_major(
        _sfb_layout.shape[0].value() * _sfb_layout.shape[1].value(),
        _sfb_layout.shape[2].value(),
        _sfb_layout.shape[3].value(),
        _sfb_layout.shape[4].value(),
        _sfb_layout.shape[5].value(),
    )
    var b_scales = LayoutTensor[
        sfb_dtype,
        sfb_layout,
        _b_scales.origin,
        address_space = _b_scales.address_space,
    ](_b_scales.ptr)

    constrained[
        sfa_layout.shape[2].value()
        == sfb_layout.shape[2].value()
        == SF_ATOM_M[0],
        "",
    ]()
    constrained[
        sfa_layout.shape[3].value()
        == sfb_layout.shape[3].value()
        == SF_ATOM_M[1],
        "",
    ]()
    constrained[
        sfa_layout.shape[4].value() == sfb_layout.shape[4].value() == SF_ATOM_K,
        "",
    ]()

    @parameter
    if config.cta_group == 2:
        constrained[
            MMA_M == 256 and MMA_N in (128, 256),
            "Only support cta_group == 2 with MMA_M == 256",
        ]()
    else:
        constrained[
            MMA_M == 128 and MMA_N in (128, 256),
            (
                "Only support MMA_M == 128 and MMA_N in (128, 256) when"
                " cta_group == 1"
            ),
        ]()

    comptime cluster_shape = config.cluster_shape

    comptime num_experts = b_layout.shape[0].value()
    constrained[
        _sfb_layout.shape[0].value() == num_experts,
        "num_experts must be equal to _sfb_layout.shape[0]",
    ]()
    comptime N = c_layout.shape[1].value()
    comptime expert_n = N
    comptime K = a_layout.shape[1].value()
    comptime assert K % 16 == 0, (
        "Due to TMA limitations, K must be a multiple of 16 bytes"
        + " but got K = "
        + String(K)
    )

    # Reshape B from (num_experts, N, K) to (num_experts * N, K)
    var b_device = LayoutTensor[
        b_type,
        Layout.row_major(num_experts * N, K),
        _b_device.origin,
        address_space = _b_device.address_space,
    ](_b_device.ptr)

    var M = c_device.dim[0]()

    constrained[
        ceildiv(K, BK) % config.k_group_size == 0,
        "K iterations must be a multiple of k_group_size",
    ]()

    # Instantiate kernel -- c_device_layout derived from caller's TileTensor
    # so types match by construction in enqueue_function.
    comptime c_device_tt_layout = type_of(lt_to_tt(c_device)).LayoutType
    comptime matmul_kernel = Grouped1D1DMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        c_device_tt_layout,
        transpose_b,
        config=config,
        static_N=expert_n,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
    ]
    comptime KernelType = type_of(matmul_kernel)

    # Shared memory size calculation
    comptime SmemType = KernelType.SmemType
    comptime smem_size = size_of[SmemType]()

    # B200 SMEM limit
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Create TMA descriptors using kernel-derived layout types
    var a_tma_op = create_tma_tile[
        KernelType.ATmaTile.tile_layout,
        KernelType.ATmaTile.desc_layout,
        Index(BM // cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
    ](ctx, a_device)

    var b_tma_op = create_tma_tile[
        KernelType.BTmaTile.tile_layout,
        KernelType.BTmaTile.desc_layout,
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode = config.b_swizzle,
    ](ctx, b_device)

    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    constrained[
        (not config.AB_swapped) or config.c_swizzle.bytes() == 128,
        "Only support 128B swizzle mode when AB_swapped is True",
    ]()
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tma_tile[
        KernelType.CTmaTile.tile_layout,
        KernelType.CTmaTile.desc_layout,
        c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode = config.c_swizzle,
    ](ctx, c_device)

    # Reshape scale tensors to 4D for TMA
    comptime scales_4d_layout[layout: LegacyLayout] = LegacyLayout.row_major(
        layout.shape[0].value(),
        layout.shape[1].value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    comptime sfa_4d_layout = scales_4d_layout[sfa_layout]
    comptime sfb_4d_layout = scales_4d_layout[sfb_layout]

    var sfa_4d = LayoutTensor[sfa_dtype, sfa_4d_layout, MutAnyOrigin](
        a_scales.ptr,
        RuntimeLayout[sfa_4d_layout].row_major(
            IndexList[4](
                a_scales.dim(0),
                a_scales.dim(1),
                a_scales.dim(2),
                a_scales.dim(3) * a_scales.dim(4),
            ),
        ),
    )
    var sfb_4d = LayoutTensor[sfb_dtype, sfb_4d_layout, MutAnyOrigin](
        b_scales.ptr,
        RuntimeLayout[sfb_4d_layout].row_major(
            IndexList[4](
                b_scales.dim(0),
                b_scales.dim(1),
                b_scales.dim(2),
                b_scales.dim(3) * b_scales.dim(4),
            ),
        ),
    )

    comptime sfa_tma_tile_shape = Index(
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    var sfa_tma_op = create_tma_tile[
        KernelType.SFATmaTile.tile_layout,
        KernelType.SFATmaTile.desc_layout,
        sfa_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, sfa_4d)

    comptime sfb_tma_tile_shape = Index(
        MMA_N // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    var sfb_tma_op = create_tma_tile[
        KernelType.SFBTmaTile.tile_layout,
        KernelType.SFBTmaTile.desc_layout,
        sfb_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, sfb_4d)

    # Define kernel function
    comptime kernel = matmul_kernel.run

    var grid_dim = (
        B200.sm_count,
        1,
        1,
    )

    # Thread configuration: 3 warps for Load/MMA + 4 warps for Epilogue = 192 threads
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime epilogue_warps = 4

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        sfa_tma_op,
        sfb_tma_op,
        lt_to_tt_1d(a_offsets),
        lt_to_tt_1d(a_scale_offsets),
        lt_to_tt_1d(expert_ids),
        lt_to_tt_1d(expert_scales),
        lt_to_tt(
            c_device
        ),  # Types match: kernel param derived from same layout
        num_active_experts,
        UInt32(K),
        grid_dim=grid_dim,
        block_dim=(32 * (load_warps + mma_warps + epilogue_warps)),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )


fn grouped_matmul_dynamic_scaled_nvfp4[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    scales_type: DType,
    a_scales_layout: LegacyLayout,
    b_scales_layout: LegacyLayout,
    a_offsets_layout: LegacyLayout,
    a_scale_offsets_layout: LegacyLayout,
    expert_ids_layout: LegacyLayout,
    expert_scales_layout: LegacyLayout,
    //,
    transpose_b: Bool = True,
    target: StaticString = "cpu",
](
    c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    a_scales: LayoutTensor[scales_type, a_scales_layout, MutAnyOrigin],
    b_scales: LayoutTensor[scales_type, b_scales_layout, MutAnyOrigin],
    a_offsets: LayoutTensor[DType.uint32, a_offsets_layout, MutAnyOrigin],
    a_scale_offsets: LayoutTensor[
        DType.uint32, a_scale_offsets_layout, MutAnyOrigin
    ],
    expert_ids: LayoutTensor[DType.int32, expert_ids_layout, MutAnyOrigin],
    expert_scales: LayoutTensor[
        DType.float32, expert_scales_layout, MutAnyOrigin
    ],
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Performs grouped matrix multiplication with NVFP4 quantization.

    This is a compatibility wrapper that matches the old API from
    grouped_matmul_sm100_1d1d.mojo. It creates the default config and calls
    the new structured kernel implementation.

    Computes C = A @ B^T for multiple expert groups in a Mixture of Experts
    (MoE) layer. Inputs A and B are NVFP4 quantized (4-bit floating point),
    packed as uint8 (2 values per byte), with float8_e4m3fn scale factors.

    Parameters:
        c_type: Output tensor dtype.
        c_layout: Output tensor layout.
        a_type: Input A dtype (must be uint8 for packed NVFP4).
        a_layout: Input A layout.
        b_type: Input B dtype (must be uint8 for packed NVFP4).
        b_layout: Input B layout.
        scales_type: Scale factor dtype.
        a_scales_layout: A scales layout.
        b_scales_layout: B scales layout.
        a_offsets_layout: A offsets layout.
        a_scale_offsets_layout: A scale offsets layout.
        expert_ids_layout: Expert IDs layout.
        expert_scales_layout: Expert scales layout.
        transpose_b: Whether B is transposed (must be True).
        target: Target device (ignored, always runs on GPU).

    Args:
        c: Output tensor (total_tokens, N).
        a: Input A tensor (total_tokens, K).
        b: Weight tensor B (num_experts, N, K).
        a_scales: Scale factors for A.
        b_scales: Scale factors for B.
        a_offsets: Per-expert token offsets.
        a_scale_offsets: Per-expert scale offsets.
        expert_ids: Active expert IDs.
        expert_scales: Per-expert output scaling.
        num_active_experts: Number of active experts.
        ctx: Device context.
    """
    # Create the default config matching the old kernel
    comptime MMA_K = 32
    comptime bm = 128
    comptime bn = 128
    comptime mma_shape = Index(bm, bn, MMA_K)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_type, scales_type, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        cluster_shape=Index(1, 1, 1),
        mma_shape=mma_shape,
        block_swizzle_size=8,
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
        num_accum_pipeline_stages=2,
    )

    # Call the new structured kernel
    grouped_matmul_1d1d_nvfp4[transpose_b=transpose_b, config=matmul_config](
        c,
        a,
        a_offsets,
        a_scale_offsets,
        b,
        expert_ids,
        a_scales,
        b_scales,
        expert_scales,
        num_active_experts,
        ctx,
    )
