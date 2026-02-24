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

"""CPU entry points for block-scaled SM100 matmul.

Creates TMA descriptors for A, B, C and scaling factors (SFA, SFB),
then launches the warp-specialized kernel.
"""

from math import align_up, ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import size_of

from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import (
    Layout as LegacyLayout,
    LayoutTensor,
    RuntimeLayout,
)
from layout._layout import RowMajorLayout, TensorLayout, row_major
from layout._coord import ComptimeInt, RuntimeInt, Coord, Idx
from layout._tile_tensor import TileTensor
from ..structured_kernels.tile_types import create_tma_tile

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ..structured_kernels.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.profiler import MatmulWarpSpecializationWorkSpaceManager
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)

# V3: Ported from working legacy kernel
from .block_scaled_matmul_kernel import BlackwellBlockScaledMatmulKernel

# Use structured SMEM struct for size calculation (matches V3 kernel's SmemType)
from .block_scaled_smem import BlockScaledSmem


# =============================================================================
# LayoutTensor helpers (kept for grouped_block_scaled_matmul.mojo which imports
# these). New code should use _to_batched_3d below.
# =============================================================================


@parameter
fn _reshape_to_3d[layout: LegacyLayout]() -> LegacyLayout:
    """Reshape 2D layout to 3D by prepending batch dimension of 1."""
    comptime rank = len(layout.shape)

    comptime if rank == 3:
        return materialize[layout]()
    else:
        return LegacyLayout.row_major(
            1,
            comptime (layout.shape[0].value()),
            comptime (layout.shape[1].value()),
        )


fn _convert_input_to_batched_tensor[
    dtype: DType,
    layout: LegacyLayout,
    reshape_layout: LegacyLayout = _reshape_to_3d[layout](),
](
    tensor: LayoutTensor[dtype, layout, ...],
) -> LayoutTensor[
    tensor.dtype,
    reshape_layout,
    tensor.origin,
    address_space = tensor.address_space,
]:
    """Convert 2D tensor to 3D batched tensor with batch=1."""
    return LayoutTensor[
        dtype,
        reshape_layout,
        tensor.origin,
        address_space = tensor.address_space,
    ](
        tensor.ptr,
        RuntimeLayout[reshape_layout].row_major(
            IndexList[3](
                1 if tensor.rank == 2 else tensor.dim(0),
                tensor.dim(0) if tensor.rank == 2 else tensor.dim(1),
                tensor.dim(1) if tensor.rank == 2 else tensor.dim(2),
            ),
        ),
    )


# =============================================================================
# TileTensor reshape helpers
# =============================================================================


comptime _Batched3DLayout[L: TensorLayout] = RowMajorLayout[
    ComptimeInt[1], L._shape_types[0], L._shape_types[1]
]
"""3D batched layout from a 2D layout: prepend batch=1, preserve shape types."""


fn _to_batched_3d(
    tensor: TileTensor[...],
) -> tensor.ViewType[_Batched3DLayout[type_of(tensor).LayoutType]]:
    """Reshape 2D TileTensor to 3D by prepending batch=1: (M, K) -> (1, M, K).

    The input must be rank 2. Shape types (static/dynamic) are preserved.
    """
    comptime L = type_of(tensor).LayoutType
    comptime assert L.rank == 2, "expected rank-2 TileTensor"
    return tensor.reshape(
        row_major(
            Coord(
                Idx[1](),
                tensor.layout.shape[0](),
                tensor.layout.shape[1](),
            )
        )
    )


comptime _Scales5DLayoutBatched[L: TensorLayout] = RowMajorLayout[
    L._shape_types[0],
    L._shape_types[1],
    L._shape_types[2],
    ComptimeInt[SF_ATOM_M[0]],
    ComptimeInt[SF_ATOM_M[1] * SF_ATOM_K],
]
"""5D scale factor layout for batched TMA. (B, sf_m, sf_k, atom0, atom1*atom_k).

Preserves the static/dynamic nature of B, sf_m, sf_k from the input layout.
"""

comptime _Scales5DLayoutNonBatched[L: TensorLayout] = RowMajorLayout[
    ComptimeInt[1],
    L._shape_types[0],
    L._shape_types[1],
    ComptimeInt[SF_ATOM_M[0]],
    ComptimeInt[SF_ATOM_M[1] * SF_ATOM_K],
]
"""5D scale factor layout for non-batched TMA. (1, sf_m, sf_k, atom0, atom1*atom_k).

Preserves the static/dynamic nature of sf_m and sf_k from the input layout.
"""


fn _to_scales_5d_batched(
    tensor: TileTensor[...],
) -> tensor.ViewType[_Scales5DLayoutBatched[type_of(tensor).LayoutType]]:
    """Reshape batched (rank 6) scale factors to 5D for TMA.

    Input: (B, sf_m, sf_k, atom0, atom1, atom_k)
    Output: (B, sf_m, sf_k, SF_ATOM_M[0], SF_ATOM_M[1]*SF_ATOM_K)
    """
    return tensor.reshape(
        row_major(
            Coord(
                tensor.layout.shape[0](),
                tensor.layout.shape[1](),
                tensor.layout.shape[2](),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1] * SF_ATOM_K](),
            )
        )
    )


fn _to_scales_5d_non_batched(
    tensor: TileTensor[...],
) -> tensor.ViewType[_Scales5DLayoutNonBatched[type_of(tensor).LayoutType]]:
    """Reshape non-batched (rank 5) scale factors to 5D for TMA.

    Prepends batch=1 and merges last two atom dims.
    Input: (sf_m, sf_k, atom0, atom1, atom_k)
    Output: (1, sf_m, sf_k, SF_ATOM_M[0], SF_ATOM_M[1]*SF_ATOM_K)
    """
    return tensor.reshape(
        row_major(
            Coord(
                Idx[1](),
                tensor.layout.shape[0](),
                tensor.layout.shape[1](),
                Idx[SF_ATOM_M[0]](),
                Idx[SF_ATOM_M[1] * SF_ATOM_K](),
            )
        )
    )


# =============================================================================
# TMA + Kernel Launch: operates on already-reshaped 3D/5D TileTensors
# =============================================================================


fn _create_tma_and_launch[
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[_, _, _, _, _, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    a_3d: TileTensor[...],
    b_3d: TileTensor[...],
    c_3d: TileTensor[...],
    sfa_5d: TileTensor[...],
    sfb_5d: TileTensor[...],
    ctx: DeviceContext,
    alpha: Float32,
) raises:
    """Create TMA descriptors from 3D/5D TileTensors and launch the kernel.

    This function accepts already-reshaped tensors (3D for A/B/C, 5D for
    scale factors) so that TMA descriptor creation and kernel launch live
    in the same scope -- avoiding scoping issues with @parameter if
    branches (TMA descriptors are scoped references).
    """
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime sfa_dtype = config.sfa_dtype
    comptime sfb_dtype = config.sfb_dtype
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]
    comptime cluster_shape = config.cluster_shape

    comptime max_profiled_tiles = (
        0 if max_profiled_tiles_per_SM
        is None else max_profiled_tiles_per_SM.value()
    )
    comptime enable_profiling = max_profiled_tiles > 0

    comptime matmul_kernel = BlackwellBlockScaledMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    var B = Int(c_3d.dim[0]())
    var M = Int(c_3d.dim[1]())
    var N = Int(c_3d.dim[2]())
    var M_maybe_swapped = Int(a_3d.dim[1]())
    var N_maybe_swapped = Int(b_3d.dim[1]())

    # A matrix TMA
    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    a_tma_op = create_tma_tile[
        matmul_kernel.ATmaTile.tile_layout,
        matmul_kernel.ATmaTile.desc_layout,
        a_tma_tile_shape,
        swizzle_mode = config.a_swizzle,
    ](ctx, a_3d)

    # B matrix TMA
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    ) if transpose_b else Index(
        1, BK, BN // (cluster_shape[0] // config.cta_group)
    )
    b_tma_op = create_tma_tile[
        matmul_kernel.BTmaTile.tile_layout,
        matmul_kernel.BTmaTile.desc_layout,
        b_tma_tile_shape,
        swizzle_mode = config.b_swizzle,
    ](ctx, b_3d)

    # C matrix TMA
    comptime c_tma_tile_shape_mma128 = Index(
        1, 64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(1, config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = Index(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    ) if (MMA_M == 256 or config.cta_group == 1) else c_tma_tile_shape_mma128
    comptime c_tma_tile_shape_final = c_tma_tile_shape if not config.AB_swapped else Index(
        1,
        c_tma_tile_shape[1],
        config.c_swizzle.bytes() // size_of[c_type](),
    )
    var c_tma_op = create_tma_tile[
        matmul_kernel.CTmaTile.tile_layout,
        matmul_kernel.CTmaTile.desc_layout,
        c_tma_tile_shape_final,
        swizzle_mode = config.c_swizzle,
    ](ctx, c_3d)

    # Scale factors TMA
    comptime sfa_tma_tile_shape = Index(
        1,
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfa_tma_op = create_tma_tile[
        matmul_kernel.SFATmaTile.tile_layout,
        matmul_kernel.SFATmaTile.desc_layout,
        sfa_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, sfa_5d)

    comptime sfb_tma_tile_shape = Index(
        1,
        MMA_N // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfb_tma_op = create_tma_tile[
        matmul_kernel.SFBTmaTile.tile_layout,
        matmul_kernel.SFBTmaTile.desc_layout,
        sfb_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
    ](ctx, sfb_5d)

    # Shared Memory
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime SmemType = BlockScaledSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()
    matmul_kernel.validate_config()
    comptime kernel = matmul_kernel.run

    # Grid and block dimensions
    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), cluster_shape[0]),
        align_up(ceildiv(N_maybe_swapped, MMA_N), cluster_shape[1]),
        B,
    )
    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4
    comptime K = type_of(a_3d).LayoutType.static_shape[2]
    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    # Profiling workspace
    var workspace: Span[UInt64, MutAnyOrigin]

    comptime if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = Span[UInt64, MutAnyOrigin](
            ptr=UnsafePointer[UInt64, origin=MutAnyOrigin](), length=0
        )

    # Launch
    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        sfa_tma_op,
        sfb_tma_op,
        alpha,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )

    comptime if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "block_scaled_profile")


# =============================================================================
# Main Entry Point: Block-Scaled Matmul
# =============================================================================


fn blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[_, _, _, _, _, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_tensor: TileTensor,
    a_tensor: TileTensor,
    b_tensor: TileTensor,
    a_scales_tensor: TileTensor,
    b_scales_tensor: TileTensor,
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    """Launch block-scaled FP8 matmul kernel on SM100.

    Computes C = scale(A) @ scale(B) where A and B are FP8 matrices with
    per-block scaling factors following MXFP8 conventions.

    When config.AB_swapped is True, internally swaps A and B operands
    (along with their scale factors) and transposes the output for better
    performance when M is small.

    Parameters:
        transpose_b: Whether B is transposed (must be True).
        config: Block-scaled matmul configuration.
        elementwise_compute_lambda_fn: Optional epilogue lambda.
        register_based_epilogue: Whether to use register-based epilogue.
        pdl_level: Programmatic dependent launch level.
        max_profiled_tiles_per_SM: Optional profiling tile count.

    Args:
        c_tensor: Output tensor (TileTensor).
        a_tensor: A matrix tensor (TileTensor).
        b_tensor: B matrix tensor (TileTensor).
        a_scales_tensor: A scaling factors (TileTensor).
        b_scales_tensor: B scaling factors (TileTensor).
        ctx: Device context for kernel launch.
        alpha: Tensor scale factor (scalar).

    Raises:
        If configuration constraints are violated.
    """

    comptime if config.AB_swapped:
        # When both A and B are K-major, C = A @ B'.
        # If we swap A and B: D = B @ A', and D' = (B @ A')' = A @ B' = C.
        # So swapping + transposing the output gives the same result.
        # The transpose is handled by transpose_c = config.AB_swapped in the
        # kernel.
        comptime new_config = config.swap_AB_type()
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b,
            config=new_config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            c_tensor,
            b_tensor,
            a_tensor,
            b_scales_tensor,
            a_scales_tensor,
            ctx,
            alpha,
        )
    else:
        _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b,
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            ctx,
            alpha,
        )


fn _blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[_, _, _, _, _, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_tensor: TileTensor,
    a_tensor: TileTensor,
    b_tensor: TileTensor,
    a_scales_tensor: TileTensor,
    b_scales_tensor: TileTensor,
    ctx: DeviceContext,
    alpha: Float32 = 1.0,
) raises:
    """Internal implementation for block-scaled FP8 matmul kernel launch.

    Creates TMA descriptors for A, B, C and scaling factors (SFA, SFB),
    then launches the warp-specialized kernel. Called by the public wrapper
    which handles AB swap dispatch.
    """
    # ===== Derive dtypes from config =====
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    comptime sfa_dtype = config.sfa_dtype
    comptime sfb_dtype = config.sfb_dtype

    # ===== Static Assertions =====
    comptime assert transpose_b, "Only support transposed B"

    comptime assert sfa_dtype == sfb_dtype, "sfa_dtype and sfb_dtype must match"

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    comptime assert config.k_group_size == 1, "Only support k_group_size == 1"

    comptime assert config.num_split_k == 1, "Only support split_k == 1"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime assert type_of(a_tensor).rank == type_of(b_tensor).rank == type_of(
        c_tensor
    ).rank and type_of(a_tensor).rank in (2, 3), (
        "a_tensor, b_tensor, and c_tensor must have the same rank and be 2D"
        " (non-batched) or 3D (batched) tensors"
    )

    comptime is_batched_matmul = type_of(a_tensor).rank == 3

    # ===== Reshape and create TMA descriptors =====
    # Non-batched: reshape 2D→3D and 5D→5D (prepend batch=1).
    # Batched: 3D pass-through and 6D→5D (merge atom dims).
    comptime if is_batched_matmul:
        _create_tma_and_launch[
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            a_tensor,
            b_tensor,
            c_tensor,
            _to_scales_5d_batched(a_scales_tensor),
            _to_scales_5d_batched(b_scales_tensor),
            ctx,
            alpha,
        )
    else:
        _create_tma_and_launch[
            config=config,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            pdl_level=pdl_level,
            max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
        ](
            _to_batched_3d(a_tensor),
            _to_batched_3d(b_tensor),
            _to_batched_3d(c_tensor),
            _to_scales_5d_non_batched(a_scales_tensor),
            _to_scales_5d_non_batched(b_scales_tensor),
            ctx,
            alpha,
        )
