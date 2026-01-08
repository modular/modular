# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections import OptionalReg
from math import align_up, ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import size_of

from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.primitives.grid_controls import pdl_launch_attributes, PDLLevel
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from layout.tma_async import create_tma_tile

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.profiler import MatmulWarpSpecializationWorkSpaceManager
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)

from .block_scaled_smem import BlockScaledSmem
from .block_scaled_matmul_kernels import BlackwellBlockScaledMatmulKernel


# =============================================================================
# Helper: Reshape tensors to batched format
# =============================================================================


@parameter
fn _reshape_to_3d[layout: Layout]() -> Layout:
    """Reshape 2D layout to 3D by prepending batch dimension of 1."""
    comptime rank = len(layout.shape)

    @parameter
    if rank == 3:
        return layout
    else:
        return Layout.row_major(
            1,
            comptime (layout.shape[0].value()),
            comptime (layout.shape[1].value()),
        )


fn _convert_input_to_batched_tensor[
    dtype: DType,
    layout: Layout,
    reshape_layout: Layout = _reshape_to_3d[layout](),
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
# Main Entry Point: Block-Scaled Matmul
# =============================================================================


fn blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    sfa_dtype: DType,
    sfa_layout: Layout,
    sfb_dtype: DType,
    sfb_layout: Layout,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: OptionalReg[UInt32] = None,
](
    c_tensor: LayoutTensor[c_type, c_layout, ...],
    a_tensor: LayoutTensor[a_type, a_layout, ...],
    b_tensor: LayoutTensor[b_type, b_layout, ...],
    a_scales_tensor: LayoutTensor[sfa_dtype, sfa_layout, MutAnyOrigin],
    b_scales_tensor: LayoutTensor[sfb_dtype, sfb_layout, MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    """Launch block-scaled FP8 matmul kernel on SM100.

    Computes C = scale(A) @ scale(B) where A and B are FP8 matrices with
    per-block scaling factors following MXFP8 conventions.

    Parameters:
        c_type: Output element type.
        c_layout: Output tensor layout.
        a_type: A matrix element type (FP8).
        a_layout: A matrix layout.
        b_type: B matrix element type (FP8).
        b_layout: B matrix layout.
        sfa_dtype: A scaling factor type (F8-UE8M0).
        sfa_layout: A scaling factor layout.
        sfb_dtype: B scaling factor type (F8-UE8M0).
        sfb_layout: B scaling factor layout.
        transpose_b: Whether B is transposed (must be True).
        config: Block-scaled matmul configuration.
        elementwise_compute_lambda_fn: Optional epilogue lambda.
        register_based_epilogue: Whether to use register-based epilogue.
        pdl_level: Programmatic dependent launch level.
        max_profiled_tiles_per_SM: Optional profiling tile count.

    Args:
        c_tensor: Output tensor.
        a_tensor: A matrix tensor.
        b_tensor: B matrix tensor.
        a_scales_tensor: A scaling factors.
        b_scales_tensor: B scaling factors.
        ctx: Device context for kernel launch.

    Raises:
        If configuration constraints are violated.
    """
    # ===== Static Assertions =====
    __comptime_assert transpose_b, "Only support transposed B"

    __comptime_assert (
        sfa_dtype == sfb_dtype == MXFP8_SF_DTYPE
    ), "Only support MXFP8_SF_DTYPE (F8-UE8M0) for scales"

    __comptime_assert not config.AB_swapped, "swap AB is not supported"

    __comptime_assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"

    __comptime_assert config.k_group_size == 1, "Only support k_group_size == 1"

    __comptime_assert config.num_split_k == 1, "Only support split_k == 1"

    __comptime_assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    __comptime_assert (
        a_tensor.rank == b_tensor.rank == c_tensor.rank
        and a_tensor.rank in (2, 3)
    ), (
        "a_tensor, b_tensor, and c_tensor must have the same rank and be 2D"
        " (non-batched) or 3D (batched) tensors"
    )

    # ===== Extract Dimensions =====
    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime cluster_shape = config.cluster_shape
    comptime is_batched_matmul = a_tensor.rank == 3

    # Convert to batched tensors if needed
    var a_tensor_batched = _convert_input_to_batched_tensor(a_tensor)
    var b_tensor_batched = _convert_input_to_batched_tensor(b_tensor)
    var c_tensor_batched = _convert_input_to_batched_tensor(c_tensor)

    var B = c_tensor_batched.dim[0]()
    var M = c_tensor_batched.dim[1]()
    var N = c_tensor_batched.dim[2]()
    var M_maybe_swapped = a_tensor_batched.dim[1]()
    var N_maybe_swapped = b_tensor_batched.dim[1]()

    __comptime_assert (
        a_tensor_batched.layout.shape[2].value()
        == b_tensor_batched.layout.shape[2].value()
    ), "A and B K dimension does not match"

    comptime K = a_tensor_batched.layout.shape[2].value()

    __comptime_assert (
        ceildiv(K, BK) % Int(config.k_group_size) == 0
    ), "K iterations must be a multiple of k_group_size"

    # ===== Create TMA Descriptors =====

    # A matrix TMA
    comptime a_tma_tile_shape = Index(1, BM // cluster_shape[1], BK)
    a_tma_op = create_tma_tile[
        a_tma_tile_shape,
        swizzle_mode = config.a_swizzle,
        __tile_layout = Layout.row_major(a_tma_tile_shape),
    ](ctx, a_tensor_batched)

    # B matrix TMA
    comptime b_tma_tile_shape = Index(
        1, BN // (cluster_shape[0] // config.cta_group), BK
    ) if transpose_b else Index(
        1, BK, BN // (cluster_shape[0] // config.cta_group)
    )
    b_tma_op = create_tma_tile[
        b_tma_tile_shape,
        swizzle_mode = config.b_swizzle,
        __tile_layout = Layout.row_major(b_tma_tile_shape),
    ](ctx, b_tensor_batched)

    # C matrix TMA
    comptime c_tma_tile_shape_mma128 = Index(
        1, 64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(1, config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = Index(
        1, config.output_tile_shape[0], config.output_tile_shape[1]
    ) if (MMA_M == 256 or config.cta_group == 1) else c_tma_tile_shape_mma128

    comptime c_tma_tile_shape_final = c_tma_tile_shape if not config.AB_swapped else Index(
        1, c_tma_tile_shape[0], config.c_swizzle.bytes() // size_of[c_type]()
    )
    var c_tma_op = create_tma_tile[
        c_tma_tile_shape_final,
        swizzle_mode = config.c_swizzle,
        __tile_layout = Layout.row_major(c_tma_tile_shape_final),
    ](ctx, c_tensor_batched)

    # Scaling factors TMA - 5D tensors
    comptime scales_5d_layout[layout: Layout] = Layout.row_major(
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
        a_scales_tensor.ptr,
        RuntimeLayout[sfa_5d_layout].row_major(
            IndexList[5](
                a_scales_tensor.dim(0) if is_batched_matmul else 1,
                a_scales_tensor.dim(
                    1
                ) if is_batched_matmul else a_scales_tensor.dim(0),
                a_scales_tensor.dim(
                    2
                ) if is_batched_matmul else a_scales_tensor.dim(1),
                a_scales_tensor.dim(
                    3
                ) if is_batched_matmul else a_scales_tensor.dim(2),
                (
                    a_scales_tensor.dim(4) * a_scales_tensor.dim(5)
                ) if is_batched_matmul else (
                    a_scales_tensor.dim(3) * a_scales_tensor.dim(4)
                ),
            ),
        ),
    )
    var sfb_5d_tensor = LayoutTensor[sfb_dtype, sfb_5d_layout, MutAnyOrigin](
        b_scales_tensor.ptr,
        RuntimeLayout[sfb_5d_layout].row_major(
            IndexList[5](
                b_scales_tensor.dim(0) if is_batched_matmul else 1,
                b_scales_tensor.dim(
                    1
                ) if is_batched_matmul else b_scales_tensor.dim(0),
                b_scales_tensor.dim(
                    2
                ) if is_batched_matmul else b_scales_tensor.dim(1),
                b_scales_tensor.dim(
                    3
                ) if is_batched_matmul else b_scales_tensor.dim(2),
                (
                    b_scales_tensor.dim(4) * b_scales_tensor.dim(5)
                ) if is_batched_matmul else (
                    b_scales_tensor.dim(3) * b_scales_tensor.dim(4)
                ),
            ),
        ),
    )

    comptime sfa_tma_tile_shape = Index(
        1, BM // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )
    var sfa_tma_op = create_tma_tile[
        sfa_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        __tile_layout = Layout.row_major(sfa_tma_tile_shape),
    ](ctx, sfa_5d_tensor)

    comptime sfb_tma_tile_shape = Index(
        1, MMA_N // SF_MN_GROUP_SIZE, 1, SF_ATOM_M[0], SF_ATOM_M[1] * SF_ATOM_K
    )
    var sfb_tma_op = create_tma_tile[
        sfb_tma_tile_shape,
        swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE,
        __tile_layout = Layout.row_major(sfb_tma_tile_shape),
    ](ctx, sfb_5d_tensor)

    # ===== Shared Memory Size =====
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

    # ===== Profiling Setup =====
    comptime max_profiled_tiles = (
        0 if max_profiled_tiles_per_SM
        is None else max_profiled_tiles_per_SM.value()
    )
    comptime enable_profiling = max_profiled_tiles > 0

    # ===== Instantiate Kernel =====
    # Note: The kernel struct is parameterized but run() not yet implemented
    # This serves as documentation for the launch interface
    comptime matmul_kernel = BlackwellBlockScaledMatmulKernel[
        a_type,
        b_type,
        c_type,
        sfa_dtype,
        sfb_dtype,
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        sfa_tma_op.layout,
        sfb_tma_op.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        sfa_tma_op.desc_layout,
        sfb_tma_op.desc_layout,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            config.cluster_shape[0],
            config.cluster_shape[1],
            config.cluster_shape[2],
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        pdl_level=pdl_level,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    # Validate kernel configuration
    matmul_kernel.validate_config()

    # Get the kernel entry point from the struct
    comptime kernel = matmul_kernel.run

    # ===== Grid and Block Dimensions =====
    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), Int(cluster_shape[0])),
        align_up(ceildiv(N_maybe_swapped, MMA_N), Int(cluster_shape[1])),
        B,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        ceildiv(grid_dim[0], cluster_shape[0]),
        ceildiv(grid_dim[1], cluster_shape[1]),
        1,
    )

    # Thread organization: 1 TMA + 1 MMA + 1 Scheduler + 4 Epilogue warps
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](M, N, K)

    # ===== Workspace for Profiling =====
    var workspace: Span[UInt64, MutAnyOrigin]

    @parameter
    if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = Span[UInt64, MutAnyOrigin](
            ptr=UnsafePointer[UInt64, origin=MutAnyOrigin](), length=0
        )

    # ===== Kernel Launch =====
    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        sfa_tma_op,
        sfb_tma_op,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(b200_smem),
        attributes=pdl_launch_attributes(pdl_level),
    )

    @parameter
    if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "block_scaled_profile")
