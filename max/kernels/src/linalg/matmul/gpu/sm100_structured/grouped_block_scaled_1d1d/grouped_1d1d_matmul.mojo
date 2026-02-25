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
        c_tensor,  # Output: TileTensor (total_tokens, N)
        a_tensor,  # Input A: TileTensor (total_tokens, K)
        a_offsets,  # Per-expert offsets: TileTensor 1D
        a_scale_offsets,  # Per-expert scale offsets: TileTensor 1D
        b_tensor,  # Weights B: TileTensor (num_experts, N, K)
        expert_ids,  # Active expert IDs: TileTensor 1D
        a_scales,  # Scale factors for A: TileTensor 5D
        b_scales,  # Scale factors for B: TileTensor 6D
        expert_scales,  # Per-expert output scaling: TileTensor 1D
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
from layout._layout import Layout as TileLayout, row_major
from layout._coord import Coord, Idx, RuntimeInt
from layout._tile_tensor import TileTensor
from ..structured_kernels.tile_types import create_tma_tile

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
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
](
    c_device: TileTensor,
    a_device: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    _b_device: TileTensor,
    expert_ids: TileTensor,
    a_scales: TileTensor,
    _b_scales: TileTensor,
    expert_scales: TileTensor,
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
    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        sfa_dtype == sfb_dtype
    ), "Only support same scales dtype for A and B"
    comptime assert sfa_dtype in (
        MXFP8_SF_DTYPE,
        NVFP4_SF_DTYPE,
    ), "Only support MXFP8_SF_DTYPE or NVFP4_SF_DTYPE for scales"

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert config.k_group_size == 1, "Only support k_group_size == 1"

    comptime assert config.num_split_k == 1, "Only support split_k == 1"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    # Extract static dimensions from TileTensor types.
    # B is (num_experts, N, K), C is (M_dynamic, N), A is (M_dynamic, K).
    comptime num_experts = type_of(_b_device).static_shape[0]
    comptime N = type_of(c_device).static_shape[1]
    comptime expert_n = N
    comptime K = type_of(a_device).static_shape[1]
    comptime assert K % 16 == 0, (
        "Due to TMA limitations, K must be a multiple of 16 bytes"
        + " but got K = "
        + String(K)
    )

    # Reshape B from (num_experts, N, K) to (num_experts * N, K)
    var b_device = _b_device.reshape(
        row_major(Coord(Idx[num_experts * N](), Idx[K]()))
    )

    comptime if config.cta_group == 2:
        comptime assert MMA_M == 256 and MMA_N in (
            128,
            256,
        ), "Only support cta_group == 2 with MMA_M == 256"
    else:
        comptime assert MMA_M == 128 and MMA_N in (128, 256), (
            "Only support MMA_M == 128 and MMA_N in (128, 256) when"
            " cta_group == 1"
        )

    comptime cluster_shape = config.cluster_shape

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    # Instantiate kernel -- c_device_layout derived from caller's TileTensor
    # so types match by construction in enqueue_function.
    comptime c_device_tt_layout = type_of(c_device).LayoutType
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

    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()

    comptime sfa_tma_tile_shape = Index(
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    comptime sfb_tma_tile_shape = Index(
        MMA_N // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )

    # Reshape scale tensors from 5D to 4D for TMA.
    # a_scales: (M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
    #        -> (M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K)
    var sfa_4d = a_scales.reshape(
        row_major(
            Coord(
                a_scales.layout.shape[0](),
                a_scales.layout.shape[1](),
                a_scales.layout.shape[2](),
                Idx[SF_ATOM_M[1] * SF_ATOM_K](),
            )
        )
    )

    # _b_scales is 6D; reshape directly to 4D:
    # (num_experts, N_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
    # -> (num_experts*N_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1]*SF_ATOM_K)
    var sfb_dim0 = (
        _b_scales.layout.shape[0]().value()
        * _b_scales.layout.shape[1]().value()
    )
    var sfb_4d = _b_scales.reshape(
        row_major(
            Coord(
                RuntimeInt[DType.int64](Scalar[DType.int64](sfb_dim0)),
                _b_scales.layout.shape[2](),
                _b_scales.layout.shape[3](),
                Idx[SF_ATOM_M[1] * SF_ATOM_K](),
            )
        )
    )

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

    # Re-wrap 1D TileTensors with GMEMLayout1D to match the kernel's
    # expected types. The caller's TileTensors may have a different symbolic
    # LayoutType (from _DimsToCoordLike) than the kernel's GMEMLayout1D.
    from memory import UnsafePointer as Ptr
    from ..structured_kernels.tile_types import GMEMLayout1D

    fn _to_1d[
        target_type: DType,
    ](t: TileTensor) -> TileTensor[target_type, GMEMLayout1D, MutAnyOrigin]:
        var shape = Coord(
            RuntimeInt[DType.int64](
                Scalar[DType.int64](t.layout.shape[0]().value())
            )
        )
        var stride = Coord(Idx[1]())
        return TileTensor[target_type, GMEMLayout1D, MutAnyOrigin](
            ptr=Ptr[Scalar[target_type], MutAnyOrigin](
                unsafe_from_address=Int(t.ptr)
            ),
            layout=GMEMLayout1D(shape, stride),
        )

    # When AB_swapped, swap A/B operands and their scale factors for TMA
    # and kernel launch. The @parameter if ensures compile-time branching.
    comptime if config.AB_swapped:
        var a_tma_op = create_tma_tile[
            KernelType.ATmaTile.tile_layout,
            KernelType.ATmaTile.desc_layout,
            Index(BM // cluster_shape[1], BK),
            swizzle_mode = config.a_swizzle,
        ](ctx, b_device)
        var b_tma_op = create_tma_tile[
            KernelType.BTmaTile.tile_layout,
            KernelType.BTmaTile.desc_layout,
            Index(
                BN // (cluster_shape[0] // config.cta_group), BK
            ) if transpose_b else Index(
                BK, BN // (cluster_shape[0] // config.cta_group)
            ),
            swizzle_mode = config.b_swizzle,
        ](ctx, a_device)
        var c_tma_op = create_tma_tile[
            KernelType.CTmaTile.tile_layout,
            KernelType.CTmaTile.desc_layout,
            Index(c_tma_tile_shape[0], c_tma_tile_shape_1),
            swizzle_mode = config.c_swizzle,
        ](ctx, c_device)
        var sfa_tma_op = create_tma_tile[
            KernelType.SFATmaTile.tile_layout,
            KernelType.SFATmaTile.desc_layout,
            sfa_tma_tile_shape,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx, sfb_4d)
        var sfb_tma_op = create_tma_tile[
            KernelType.SFBTmaTile.tile_layout,
            KernelType.SFBTmaTile.desc_layout,
            sfb_tma_tile_shape,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx, sfa_4d)
        ctx.enqueue_function[kernel, kernel](
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            _to_1d[DType.uint32](a_offsets),
            _to_1d[DType.uint32](a_scale_offsets),
            _to_1d[DType.int32](expert_ids),
            _to_1d[DType.float32](expert_scales),
            c_device,
            num_active_experts,
            UInt32(K),
            grid_dim=grid_dim,
            block_dim=(32 * (load_warps + mma_warps + epilogue_warps)),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
        )
    else:
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
        var c_tma_op = create_tma_tile[
            KernelType.CTmaTile.tile_layout,
            KernelType.CTmaTile.desc_layout,
            c_tma_tile_shape,
            swizzle_mode = config.c_swizzle,
        ](ctx, c_device)
        var sfa_tma_op = create_tma_tile[
            KernelType.SFATmaTile.tile_layout,
            KernelType.SFATmaTile.desc_layout,
            sfa_tma_tile_shape,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx, sfa_4d)
        var sfb_tma_op = create_tma_tile[
            KernelType.SFBTmaTile.tile_layout,
            KernelType.SFBTmaTile.desc_layout,
            sfb_tma_tile_shape,
            swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        ](ctx, sfb_4d)
        ctx.enqueue_function[kernel, kernel](
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            _to_1d[DType.uint32](a_offsets),
            _to_1d[DType.uint32](a_scale_offsets),
            _to_1d[DType.int32](expert_ids),
            _to_1d[DType.float32](expert_scales),
            c_device,
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
    transpose_b: Bool = True,
    target: StaticString = "cpu",
](
    c: TileTensor,
    a: TileTensor,
    b: TileTensor,
    a_scales: TileTensor,
    b_scales: TileTensor,
    a_offsets: TileTensor,
    a_scale_offsets: TileTensor,
    expert_ids: TileTensor,
    expert_scales: TileTensor,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Performs grouped matrix multiplication with NVFP4 quantization.

    This is a compatibility wrapper that creates the default config and calls
    the structured kernel implementation.

    Computes C = A @ B^T for multiple expert groups in a Mixture of Experts
    (MoE) layer. Inputs A and B are NVFP4 quantized (4-bit floating point),
    packed as uint8 (2 values per byte), with float8_e4m3fn scale factors.

    Parameters:
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
    # Extract dtypes from TileTensor arguments.
    comptime a_type = a.dtype
    comptime b_type = b.dtype
    comptime c_type = c.dtype
    comptime scales_type = a_scales.dtype

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
