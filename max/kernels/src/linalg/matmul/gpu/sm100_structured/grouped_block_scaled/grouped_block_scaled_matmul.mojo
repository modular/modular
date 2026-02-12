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
"""CPU entry points for grouped block-scaled SM100 matmul.

Supports multiple GEMM operations with variable problem sizes per group.
Uses TMATensorTileArray for per-block updatable TMA descriptors.

This module implements grouped block-scaled GEMM following the architecture
of NVIDIA CuTe DSL grouped_blockscaled_gemm.py:
1. Creates template TMA descriptors from the first group
2. Creates TMATensorTileArray with one tensormap per block
3. Launches GroupedBlockScaledMatmulKernel with per-group pointers

Usage:
    # Per-group pointers (device addresses)
    var a_ptrs = ... # (num_groups, 1) with uint64 addresses
    var b_ptrs = ... # (num_groups, 1)
    var c_ptrs = ... # (num_groups, 1)
    var sfa_ptrs = ... # (num_groups, 1)
    var sfb_ptrs = ... # (num_groups, 1)

    # Problem sizes per group
    var problem_sizes = ... # (num_groups, 4) with [M, N, K, L]

    grouped_block_scaled_matmul[...](
        a_ptrs, b_ptrs, c_ptrs, sfa_ptrs, sfb_ptrs,
        problem_sizes, num_groups, ctx
    )
"""

from collections import Optional
from math import align_up, ceildiv
from memory import UnsafePointer
from sys import size_of

from gpu.host import DeviceContext, FuncAttribute
from gpu.host.info import B200
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Layout as LegacyLayout,
    LayoutTensor,
    RuntimeLayout,
)
from layout.tma_async import TMATensorTileArray

from ..structured_kernels.tile_types import create_tma_tile

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from ..structured_kernels.config import BlockScaledMatmulConfig
from ..block_scaled.block_scaled_smem import BlockScaledSmem
from ..block_scaled.block_scaled_matmul import (
    _reshape_to_3d as working_reshape_to_3d,
    _convert_input_to_batched_tensor as working_convert_to_batched,
)
from .grouped_tile_scheduler import GroupedTileScheduler
from .grouped_block_scaled_matmul_kernel import (
    GroupedBlockScaledMatmulKernel,
    GroupedTensormapManager,
)
from .grouped_block_scaled_smem import GroupedBlockScaledSmem


# =============================================================================
# Helper: Reshape 2D layout to 3D (same as block_scaled_matmul.mojo)
# =============================================================================


@parameter
fn _reshape_to_3d[layout: LegacyLayout]() -> LegacyLayout:
    """Reshape 2D layout to 3D by prepending batch dimension of 1."""
    comptime rank = len(layout.shape)

    @parameter
    if rank == 3:
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


# Helper for scale factor tensors (5D -> 5D with batch)
# Matches the exact pattern from block_scaled_matmul.mojo
@parameter
fn _reshape_sf_to_5d[layout: LegacyLayout, is_batched: Bool]() -> LegacyLayout:
    """Reshape scale factor layout to 5D with batch dimension.

    For non-batched: (M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
                  -> (1, M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K)
    """

    @parameter
    if is_batched:
        # Input is 6D: (B, M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        return LegacyLayout.row_major(
            comptime (layout.shape[0].value()),
            comptime (layout.shape[1].value()),
            comptime (layout.shape[2].value()),
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        )
    else:
        # Input is 5D: (M_groups, K_groups, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K)
        return LegacyLayout.row_major(
            1,
            comptime (layout.shape[0].value()),
            comptime (layout.shape[1].value()),
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        )


# =============================================================================
# Helper: Compute total tiles across all groups
# =============================================================================


fn compute_total_tiles[
    tile_m: Int, tile_n: Int, max_groups: Int
](
    problem_sizes: LayoutTensor[
        DType.int32, LegacyLayout.row_major(max_groups, 4), MutAnyOrigin
    ],
    num_groups: Int,
) -> Int:
    """Compute total number of tiles across all groups.

    Args:
        problem_sizes: (num_groups, 4) tensor with [M, N, K, L] per group.
        num_groups: Number of GEMM problems.

    Returns:
        Total tile count.
    """
    var total = 0
    for g in range(num_groups):
        var m = Int(problem_sizes[g, 0])
        var n = Int(problem_sizes[g, 1])
        var m_tiles = ceildiv(m, tile_m)
        var n_tiles = ceildiv(n, tile_n)
        total += m_tiles * n_tiles
    return total


# =============================================================================
# Validation: Check constraints matching NVIDIA CuTe DSL
# =============================================================================


fn validate_grouped_gemm_constraints[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]():
    """Validate grouped GEMM configuration constraints.

    Constraints from NVIDIA CuTe DSL grouped_blockscaled_gemm.py:
    - MMA tiler M: 128 or 256
    - MMA tiler N: 128 or 256
    - Cluster M/N: Power of 2, <=4 per axis (for SF multicast)
    - Total cluster size: <=16
    - 16-byte alignment on contiguous dimensions
    """
    # MMA tiler constraints
    comptime assert config.mma_shape[0] in (
        128,
        256,
    ), "MMA tiler M must be 128 or 256"
    comptime assert config.mma_shape[1] in (
        128,
        256,
    ), "MMA tiler N must be 128 or 256"

    # Cluster constraints
    comptime assert (
        config.cluster_shape[0] <= 4
    ), "Cluster M must be <=4 for SF multicast"
    comptime assert (
        config.cluster_shape[1] <= 4
    ), "Cluster N must be <=4 for SF multicast"
    comptime assert (
        config.cluster_shape[0] * config.cluster_shape[1] <= 16
    ), "Total cluster size must be <=16"

    # Must be transposed B
    comptime assert transpose_b, "Only support transposed B"

    # SF dtype must match
    comptime assert sfa_dtype == sfb_dtype, "sfa_dtype and sfb_dtype must match"


# =============================================================================
# Main Entry Point: Grouped Block-Scaled Matmul
# =============================================================================


fn grouped_block_scaled_matmul[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    sfa_dtype: DType,
    sfa_layout: LegacyLayout,
    sfb_dtype: DType,
    sfb_layout: LegacyLayout,
    transpose_b: Bool,
    max_groups: Int,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
](
    # Per-group tensor pointers
    a_ptrs: LayoutTensor[
        DType.uint64, LegacyLayout.row_major(max_groups, 1), MutAnyOrigin
    ],
    b_ptrs: LayoutTensor[
        DType.uint64, LegacyLayout.row_major(max_groups, 1), MutAnyOrigin
    ],
    c_ptrs: LayoutTensor[
        DType.uint64, LegacyLayout.row_major(max_groups, 1), MutAnyOrigin
    ],
    sfa_ptrs: LayoutTensor[
        DType.uint64, LegacyLayout.row_major(max_groups, 1), MutAnyOrigin
    ],
    sfb_ptrs: LayoutTensor[
        DType.uint64, LegacyLayout.row_major(max_groups, 1), MutAnyOrigin
    ],
    # Per-group problem sizes: (max_groups, 4) with [M, N, K, L]
    problem_sizes: LayoutTensor[
        DType.int32, LegacyLayout.row_major(max_groups, 4), MutAnyOrigin
    ],
    # Number of active groups (runtime)
    num_groups: Int,
    # Total tiles across all groups (computed by caller on host)
    total_tiles: Int,
    # Template tensors from first group (for TMA descriptor creation)
    # These are 2D tensors (M, K) or (N, K) - will be converted to 3D internally
    a_template: LayoutTensor[a_type, a_layout, MutAnyOrigin],
    b_template: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    c_template: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    # Scale factor templates - 5D tensors
    sfa_template: LayoutTensor[sfa_dtype, sfa_layout, MutAnyOrigin],
    sfb_template: LayoutTensor[sfb_dtype, sfb_layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Launch grouped block-scaled FP8 matmul kernel on SM100.

    Computes C[g] = scale(A[g]) @ scale(B[g]) for g in range(num_groups),
    where each group can have different M, N, K dimensions.

    Parameters:
        c_type: Output element type.
        c_layout: Output tensor layout.
        a_type: A matrix element type (FP8).
        a_layout: A tensor layout.
        b_type: B matrix element type (FP8).
        b_layout: B tensor layout.
        sfa_dtype: A scaling factor type (F8-UE8M0).
        sfa_layout: A scaling factor tensor layout.
        sfb_dtype: B scaling factor type (F8-UE8M0).
        sfb_layout: B scaling factor tensor layout.
        transpose_b: Whether B is transposed (must be True).
        max_groups: Maximum number of groups (compile-time bound).
        config: Block-scaled matmul configuration.
        elementwise_compute_lambda_fn: Optional epilogue lambda for element-wise
            operations on output. Applied after matmul, before writing to global
            memory.
        register_based_epilogue: If True (default), apply epilogue in registers.
            If False, use SMEM-based epilogue path.

    Args:
        a_ptrs: Per-group A matrix pointers (max_groups, 1).
        b_ptrs: Per-group B matrix pointers (max_groups, 1).
        c_ptrs: Per-group C matrix pointers (max_groups, 1).
        sfa_ptrs: Per-group A scaling factor pointers (max_groups, 1).
        sfb_ptrs: Per-group B scaling factor pointers (max_groups, 1).
        problem_sizes: Per-group problem sizes (max_groups, 4) as [M, N, K, L].
        num_groups: Actual number of groups (runtime value <= max_groups).
        total_tiles: Total tiles across all groups (computed by caller).
        a_template: Template A tensor for TMA descriptor creation.
        b_template: Template B tensor for TMA descriptor creation.
        c_template: Template C tensor for TMA descriptor creation.
        sfa_template: Template SFA tensor for TMA descriptor creation.
        sfb_template: Template SFB tensor for TMA descriptor creation.
        ctx: Device context for kernel launch.

    Raises:
        If configuration constraints are violated.
    """
    # ===== Validate constraints =====
    validate_grouped_gemm_constraints[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b, config
    ]()

    # ===== Compute tile dimensions =====
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]
    comptime cluster_shape = config.cluster_shape
    comptime CLUSTER_SIZE = cluster_shape[0] * cluster_shape[1]

    # ===== Convert templates to batched tensors =====
    # Use working kernel's conversion functions to ensure compatibility
    comptime is_batched_matmul = a_template.rank == 3

    var a_tensor_batched = working_convert_to_batched(a_template)
    var b_tensor_batched = working_convert_to_batched(b_template)
    var c_tensor_batched = working_convert_to_batched(c_template)

    # Scale factor tensors: convert to 5D with batch dimension and merged last dims
    # Use exact same pattern as block_scaled_matmul.mojo's scales_5d_layout
    comptime scales_5d_layout[layout: LegacyLayout] = LegacyLayout.row_major(
        layout.shape[0].value() if is_batched_matmul else 1,
        layout.shape[1]
        .value() if is_batched_matmul else layout.shape[0]
        .value(),
        layout.shape[2]
        .value() if is_batched_matmul else layout.shape[1]
        .value(),
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    comptime sfa_5d_layout = scales_5d_layout[sfa_layout]
    comptime sfb_5d_layout = scales_5d_layout[sfb_layout]

    var sfa_5d_tensor = LayoutTensor[sfa_dtype, sfa_5d_layout, MutAnyOrigin](
        sfa_template.ptr,
        RuntimeLayout[sfa_5d_layout].row_major(
            IndexList[5](
                sfa_template.dim(0) if is_batched_matmul else 1,
                sfa_template.dim(1) if is_batched_matmul else sfa_template.dim(
                    0
                ),
                sfa_template.dim(2) if is_batched_matmul else sfa_template.dim(
                    1
                ),
                sfa_template.dim(3) if is_batched_matmul else sfa_template.dim(
                    2
                ),
                (
                    sfa_template.dim(4) * sfa_template.dim(5)
                ) if is_batched_matmul else (
                    sfa_template.dim(3) * sfa_template.dim(4)
                ),
            ),
        ),
    )
    var sfb_5d_tensor = LayoutTensor[sfb_dtype, sfb_5d_layout, MutAnyOrigin](
        sfb_template.ptr,
        RuntimeLayout[sfb_5d_layout].row_major(
            IndexList[5](
                sfb_template.dim(0) if is_batched_matmul else 1,
                sfb_template.dim(1) if is_batched_matmul else sfb_template.dim(
                    0
                ),
                sfb_template.dim(2) if is_batched_matmul else sfb_template.dim(
                    1
                ),
                sfb_template.dim(3) if is_batched_matmul else sfb_template.dim(
                    2
                ),
                (
                    sfb_template.dim(4) * sfb_template.dim(5)
                ) if is_batched_matmul else (
                    sfb_template.dim(3) * sfb_template.dim(4)
                ),
            ),
        ),
    )

    # ===== Instantiate Kernel First (TMA layouts computed from config) =====
    comptime matmul_kernel = GroupedBlockScaledMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
        max_groups=max_groups,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
    ]
    comptime KernelType = type_of(matmul_kernel)

    # ===== Create template TMA descriptors using kernel-derived layouts =====

    # A matrix TMA
    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    var a_tma_op = create_tma_tile[
        KernelType.ATmaTile.tile_layout,
        KernelType.ATmaTile.desc_layout,
        a_tma_tile_shape,
        swizzle_mode = config.a_swizzle,
    ](ctx, a_tensor_batched)

    # B matrix TMA
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    )
    var b_tma_op = create_tma_tile[
        KernelType.BTmaTile.tile_layout,
        KernelType.BTmaTile.desc_layout,
        b_tma_tile_shape,
        swizzle_mode = config.b_swizzle,
    ](ctx, b_tensor_batched)

    # C matrix TMA
    comptime c_tma_tile_shape = Index(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    )
    var c_tma_op = create_tma_tile[
        KernelType.CTmaTile.tile_layout,
        KernelType.CTmaTile.desc_layout,
        c_tma_tile_shape,
        swizzle_mode = config.c_swizzle,
    ](ctx, c_tensor_batched)

    # Scaling factors TMA - 5D tensors (using converted batched tensors)
    comptime sfa_tma_tile_shape = Index(
        1,
        BM // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfa_tma_op = create_tma_tile[
        KernelType.SFATmaTile.tile_layout,
        KernelType.SFATmaTile.desc_layout,
        sfa_tma_tile_shape,
    ](ctx, sfa_5d_tensor)

    comptime sfb_tma_tile_shape = Index(
        1,
        MMA_N // SF_MN_GROUP_SIZE,
        config.num_sf_k_tiles,
        SF_ATOM_M[0],
        SF_ATOM_M[1] * SF_ATOM_K,
    )
    var sfb_tma_op = create_tma_tile[
        KernelType.SFBTmaTile.tile_layout,
        KernelType.SFBTmaTile.desc_layout,
        sfb_tma_tile_shape,
    ](ctx, sfb_5d_tensor)

    # ===== Create TMATensorTileArray for per-block tensormaps =====
    # Each block gets its own tensormap copy that can be updated at runtime.
    # We allocate based on actual grid size at runtime,
    # but use a compile-time constant for the type parameter.
    # The type parameter is not used for bounds checking - actual indexing
    # is by pointer arithmetic: ptr + index * 128.

    # Allocate device memory for tensormap arrays (128 bytes per descriptor)
    # For 2SM, grid size = total_tiles * cluster_shape[0] (each cluster = 1 tile)
    comptime TMA_DESC_SIZE = 128
    var num_blocks = total_tiles * cluster_shape[0]  # Actual grid size
    var device_tensormaps_a = ctx.enqueue_create_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_b = ctx.enqueue_create_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_sfa = ctx.enqueue_create_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_sfb = ctx.enqueue_create_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var device_tensormaps_c = ctx.enqueue_create_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )

    # Create TMATensorTileArray instances
    # Note: The compile-time num_blocks parameter is just for type signature.
    # Actual array access uses pointer math, so we can safely use any valid index
    # up to the allocated size. Using CLUSTER_SIZE as type param for now.
    var tma_array_a = KernelType.TMATensorTileArrayA(device_tensormaps_a)
    var tma_array_b = KernelType.TMATensorTileArrayB(device_tensormaps_b)
    var tma_array_sfa = KernelType.TMATensorTileArraySFA(device_tensormaps_sfa)
    var tma_array_sfb = KernelType.TMATensorTileArraySFB(device_tensormaps_sfb)
    var tma_array_c = KernelType.TMATensorTileArrayC(device_tensormaps_c)

    # ===== Initialize per-block tensormaps from templates =====
    # Each block's tensormap slot needs to be initialized with the template descriptor.
    # The kernel will update these at runtime when groups change.
    # We copy the template descriptor to all block slots, then copy to device.

    # Create host buffers for tensormap initialization
    var host_buf_a = ctx.enqueue_create_host_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_b = ctx.enqueue_create_host_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_sfa = ctx.enqueue_create_host_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_sfb = ctx.enqueue_create_host_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )
    var host_buf_c = ctx.enqueue_create_host_buffer[DType.uint8](
        TMA_DESC_SIZE * num_blocks
    )

    # Copy template descriptor bytes to each block's slot
    for blk in range(num_blocks):
        for j in range(TMA_DESC_SIZE):
            host_buf_a.unsafe_ptr()[
                blk * TMA_DESC_SIZE + j
            ] = a_tma_op.descriptor.data[j]
            host_buf_b.unsafe_ptr()[
                blk * TMA_DESC_SIZE + j
            ] = b_tma_op.descriptor.data[j]
            host_buf_sfa.unsafe_ptr()[
                blk * TMA_DESC_SIZE + j
            ] = sfa_tma_op.descriptor.data[j]
            host_buf_sfb.unsafe_ptr()[
                blk * TMA_DESC_SIZE + j
            ] = sfb_tma_op.descriptor.data[j]
            host_buf_c.unsafe_ptr()[
                blk * TMA_DESC_SIZE + j
            ] = c_tma_op.descriptor.data[j]

    ctx.enqueue_copy(device_tensormaps_a, host_buf_a.unsafe_ptr())
    ctx.enqueue_copy(device_tensormaps_b, host_buf_b.unsafe_ptr())
    ctx.enqueue_copy(device_tensormaps_sfa, host_buf_sfa.unsafe_ptr())
    ctx.enqueue_copy(device_tensormaps_sfb, host_buf_sfb.unsafe_ptr())
    ctx.enqueue_copy(device_tensormaps_c, host_buf_c.unsafe_ptr())
    ctx.synchronize()

    # ===== Shared Memory Size =====
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    # Use GroupedBlockScaledSmem which includes SMEM for TMA descriptors
    comptime SmemType = GroupedBlockScaledSmem[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        transpose_b,
        config=config,
    ]
    comptime smem_size = size_of[SmemType]()

    # ===== Grid and Block Dimensions =====
    # For grouped GEMM, grid is based on total tiles
    # For 2SM (cta_group=2), each cluster handles 1 tile, so we need
    # total_tiles * cluster_shape[0] blocks = total_tiles clusters
    var grid_dim = (
        align_up(total_tiles * cluster_shape[0], cluster_shape[0]),
        1,
        1,
    )

    # Thread organization: 7 warps (224 threads)
    # 1 TMA + 1 MMA + 1 Scheduler + 4 Epilogue warps
    comptime num_threads = 32 * 7

    # ===== Create TileTensor wrappers for kernel args =====
    from layout._layout import row_major as new_row_major
    from memory import UnsafePointer as NewPtr

    var a_ptrs_tt = type_of(matmul_kernel).GroupPtrTile(
        ptr=NewPtr[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(a_ptrs.ptr)
        ),
        layout=new_row_major[max_groups, 1](),
    )
    var b_ptrs_tt = type_of(matmul_kernel).GroupPtrTile(
        ptr=NewPtr[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(b_ptrs.ptr)
        ),
        layout=new_row_major[max_groups, 1](),
    )
    var c_ptrs_tt = type_of(matmul_kernel).GroupPtrTile(
        ptr=NewPtr[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(c_ptrs.ptr)
        ),
        layout=new_row_major[max_groups, 1](),
    )
    var sfa_ptrs_tt = type_of(matmul_kernel).GroupPtrTile(
        ptr=NewPtr[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(sfa_ptrs.ptr)
        ),
        layout=new_row_major[max_groups, 1](),
    )
    var sfb_ptrs_tt = type_of(matmul_kernel).GroupPtrTile(
        ptr=NewPtr[Scalar[DType.uint64], MutAnyOrigin](
            unsafe_from_address=Int(sfb_ptrs.ptr)
        ),
        layout=new_row_major[max_groups, 1](),
    )
    var problem_sizes_tt = type_of(matmul_kernel).ProblemSizesTile(
        ptr=NewPtr[Scalar[DType.int32], MutAnyOrigin](
            unsafe_from_address=Int(problem_sizes.ptr)
        ),
        layout=new_row_major[max_groups, 4](),
    )

    # ===== Kernel Launch =====
    # Dispatch to run_2sm() for 2SM mode (cta_group=2), else run() for 1SM

    @parameter
    if config.cta_group == 2:
        # 2SM mode: use CLC-based run_2sm() for proper cluster synchronization
        ctx.enqueue_function[matmul_kernel.run_2sm, matmul_kernel.run_2sm](
            # Template TMA descriptors
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            # Per-block tensormap arrays
            tma_array_a,
            tma_array_b,
            tma_array_sfa,
            tma_array_sfb,
            tma_array_c,
            # Per-group pointer tensors (TileTensor)
            a_ptrs_tt,
            b_ptrs_tt,
            c_ptrs_tt,
            sfa_ptrs_tt,
            sfb_ptrs_tt,
            # Problem sizes and group count
            problem_sizes_tt,
            num_groups,
            grid_dim=grid_dim,
            block_dim=num_threads,
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
        )
    else:
        # 1SM mode: use linear iteration run()
        ctx.enqueue_function[matmul_kernel.run, matmul_kernel.run](
            # Template TMA descriptors
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            # Per-block tensormap arrays
            tma_array_a,
            tma_array_b,
            tma_array_sfa,
            tma_array_sfb,
            tma_array_c,
            # Per-group pointer tensors (TileTensor)
            a_ptrs_tt,
            b_ptrs_tt,
            c_ptrs_tt,
            sfa_ptrs_tt,
            sfb_ptrs_tt,
            # Problem sizes and group count
            problem_sizes_tt,
            num_groups,
            grid_dim=grid_dim,
            block_dim=num_threads,
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(b200_smem)
            ),
        )


# =============================================================================
# Helper: Calculate SMEM size for grouped kernel
# =============================================================================


fn grouped_smem_size[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]() -> Int:
    """Calculate shared memory size for grouped block-scaled kernel.

    Returns:
        SMEM size in bytes, including tensormap descriptor storage.
    """
    comptime SmemType = GroupedBlockScaledSmem[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b, config=config
    ]
    return size_of[SmemType]()
