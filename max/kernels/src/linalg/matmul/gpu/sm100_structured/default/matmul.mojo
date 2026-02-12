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
"""SM100 Matmul CPU entry points - TMA setup and kernel launch wrappers.

This module contains the CPU-side code for SM100 matrix multiplication:
- TMA descriptor creation
- Kernel instantiation and launch via ctx.enqueue_function

All GPU code (kernel structs, runtime functions) is in matmul_kernels.mojo.
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
    UNKNOWN_VALUE,
    Layout as LegacyLayout,
    LayoutTensor,
    RuntimeLayout,
)
from ..structured_kernels.tile_types import create_tma_tile

from utils.index import Index
from utils.static_tuple import StaticTuple

from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ..structured_kernels.config import MatmulConfig
from ..structured_kernels.tile_scheduler_splitk import (
    get_required_locks_buffer_size_bytes,
    get_num_tiles,
)
from linalg.matmul.gpu.profiler import MatmulWarpSpecializationWorkSpaceManager

# Import kernel structs and GPU functions from matmul_kernels
from .matmul_kernels import (
    B200MatmulSmem,
    BlackwellMatmulSM100Kernel,
    BlackwellMatmulSM100FallbackKernel,
)


fn _blackwell_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, ...],
    a_device: LayoutTensor[a_type, a_layout, ...],
    b_device: LayoutTensor[b_type, b_layout, ...],
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

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

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    @parameter
    if config.cta_group == 2:
        comptime assert (
            MMA_M == 256 or MMA_M == 128
        ), "Only support cta_group == 2 with MMA_M == 128 or 256"
        comptime assert (MMA_M != 256) or (
            MMA_N % 16 == 0
        ), "MMA_N must be a multiple of 16 when MMA_M is 256"
        comptime assert (
            config.AB_swapped
            or MMA_M != 128
            or register_based_epilogue
            or elementwise_compute_lambda_fn is None
        ) or (MMA_N % 32 == 0), (
            "SM100 doesn't support shared memory based epilogue when MMA_M =="
            " 128 and MMA_N is not a multiple of 32"
        )
    else:
        comptime assert (
            MMA_M == 128 or MMA_M == 64
        ), "Only support MMA_M == 128 or 64 when cta_group == 1"

    comptime cluster_shape = config.cluster_shape

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var M_maybe_swapped = a_device.dim[0]()
    var N_maybe_swapped = b_device.dim[0]()
    comptime K = a_layout.shape[1].value()

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    # ctx.default_device_info.shared_memory_per_multiprocessor gives this magic number on B200
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime smem_size = size_of[SmemType]()

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    # Instantiate kernel first -- TMA layouts are computed from config
    comptime matmul_kernel = BlackwellMatmulSM100Kernel[
        a_type,
        b_type,
        c_type,
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

    # Create TMA descriptors using kernel-derived layout types
    comptime KernelType = type_of(matmul_kernel)

    a_tma_op = create_tma_tile[
        KernelType.ATmaTile.tile_layout,
        KernelType.ATmaTile.desc_layout,
        Index(BM // cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
    ](ctx, a_device)

    # fmt: off
    b_tma_op = create_tma_tile[
        KernelType.BTmaTile.tile_layout,
        KernelType.BTmaTile.desc_layout,
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode = config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    comptime assert (not config.AB_swapped) or config.c_swizzle.bytes() == 128, "Only support 128B swizzle mode when AB_swapped is True"
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tma_tile[
        KernelType.CTmaTile.tile_layout,
        KernelType.CTmaTile.desc_layout,
     c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode = config.c_swizzle,
    ](ctx, c_device)
    # fmt: on

    # Get the kernel entry point from the struct
    comptime kernel = matmul_kernel.run

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), cluster_shape[0]),
        align_up(ceildiv(N_maybe_swapped, MMA_N), cluster_shape[1]),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

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

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
        attributes=pdl_launch_attributes(pdl_level),
    )

    @parameter
    if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


fn blackwell_matmul_tma_umma_warp_specialized[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, ...],
    a_device: LayoutTensor[a_type, a_layout, ...],
    b_device: LayoutTensor[b_type, b_layout, ...],
    ctx: DeviceContext,
) raises:
    @parameter
    if config.AB_swapped:
        # Swap the a_type, b_type in signature
        # TODO: Do this without creating a new instance.
        comptime new_config = config.swap_AB_type()

        # When both A and B are K-major, then the matrix multiplication math is
        # C = A @ B'
        # If we swap A and B, we have
        # D = B @ A'
        # Note that D' = (B @ A')' = A'' @ B' = A @ B' which is the same as the
        # original math. Therefore, when we swap A and B, we need to transpose
        # the result for consistency and correctness.
        @parameter
        if config.num_split_k > 1:
            _blackwell_matmul_tma_umma_warp_specialized_split_k[
                c_type,
                c_layout,
                b_type,
                b_layout,
                a_type,
                a_layout,
                transpose_b,
                config=new_config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, b_device, a_device, ctx)
        else:
            _blackwell_matmul_tma_umma_warp_specialized[
                c_type,
                c_layout,
                b_type,
                b_layout,
                a_type,
                a_layout,
                transpose_b,
                config=new_config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, b_device, a_device, ctx)
    else:

        @parameter
        if config.num_split_k > 1:
            _blackwell_matmul_tma_umma_warp_specialized_split_k[
                c_type,
                c_layout,
                a_type,
                a_layout,
                b_type,
                b_layout,
                transpose_b,
                config=config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx)
        else:
            _blackwell_matmul_tma_umma_warp_specialized[
                c_type,
                c_layout,
                a_type,
                a_layout,
                b_type,
                b_layout,
                transpose_b,
                config=config,
                elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
                register_based_epilogue=register_based_epilogue,
                pdl_level=pdl_level,
                max_profiled_tiles_per_SM=max_profiled_tiles_per_SM,
            ](c_device, a_device, b_device, ctx)


fn _blackwell_matmul_tma_umma_warp_specialized_split_k[
    c_type: DType,
    c_layout: LegacyLayout,
    a_type: DType,
    a_layout: LegacyLayout,
    b_type: DType,
    b_layout: LegacyLayout,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    max_profiled_tiles_per_SM: Optional[UInt32] = None,
](
    c_device: LayoutTensor[c_type, c_layout, ...],
    a_device: LayoutTensor[a_type, a_layout, ...],
    b_device: LayoutTensor[b_type, b_layout, ...],
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

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

    @parameter
    if config.cta_group == 2:
        comptime assert (
            MMA_M == 256 or MMA_M == 128
        ), "Only support cta_group == 2 with MMA_M == 128 or 256"
        comptime assert (MMA_M != 256) or (
            MMA_N % 16 == 0
        ), "MMA_N must be a multiple of 16 when MMA_M is 256"

        # transpose_c => MMA_M == 256 is the same as (not transpose_c) or MMA_M == 256
        comptime assert (
            not config.AB_swapped
        ) or MMA_M == 256, "swapAB is only supported for MMA_M == 256"

    else:
        comptime assert (
            MMA_M == 128 or MMA_M == 64
        ), "Only support MMA_M == 128 or 64 when cta_group == 1"
        comptime assert (
            register_based_epilogue or elementwise_compute_lambda_fn is None
        ), "only register-based epilogue is supported for cta_group == 1"

    comptime cluster_shape = config.cluster_shape

    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var M_maybe_swapped = a_device.dim[0]()
    var N_maybe_swapped = b_device.dim[0]()
    comptime K = a_layout.shape[1].value()

    comptime assert (
        ceildiv(K, BK) % config.k_group_size == 0
    ), "K iterations must be a multiple of k_group_size"

    comptime assert (
        config.num_pipeline_stages % config.k_group_size == 0
    ), "num_pipeline_stages must be a multiple of k_group_size"

    comptime SmemType = B200MatmulSmem[
        a_type, b_type, c_type, transpose_b, config=config
    ]
    comptime smem_size = size_of[SmemType]()
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024

    comptime max_profiled_tiles = 0 if max_profiled_tiles_per_SM is None else max_profiled_tiles_per_SM.value()
    comptime enable_profiling = max_profiled_tiles > 0

    comptime reduction_layout = LegacyLayout.row_major(UNKNOWN_VALUE, BM, MMA_N)
    from ..structured_kernels.tile_types import lt_to_tt

    # Instantiate kernel first -- TMA layouts are computed from config
    comptime matmul_kernel = BlackwellMatmulSM100Kernel[
        a_type,
        b_type,
        c_type,
        transpose_b,
        config=config,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(config.cluster_shape[0]),
            Int32(config.cluster_shape[1]),
            Int32(config.cluster_shape[2]),
        ),
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        register_based_epilogue=register_based_epilogue,
        max_profiled_tiles_per_SM=max_profiled_tiles,
    ]

    # Create TMA descriptors using kernel-derived layout types
    comptime KernelType = type_of(matmul_kernel)

    a_tma_op = create_tma_tile[
        KernelType.ATmaTile.tile_layout,
        KernelType.ATmaTile.desc_layout,
        Index(BM // cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
    ](ctx, a_device)

    b_tma_op = create_tma_tile[
        KernelType.BTmaTile.tile_layout,
        KernelType.BTmaTile.desc_layout,
        Index(
            BN // (cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode = config.b_swizzle,
    ](ctx, b_device)

    # For MMA_M=128, output tile has 128 rows and each 64 rows belongs to one c tile.
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
    comptime c_tma_tile_shape_mma128 = Index(
        64, config.output_tile_shape[1]
    ) if not config.AB_swapped else Index(config.output_tile_shape[0], 64)
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    # c_swizzle is set to 32B mode when swapAB is enabled so we need to adjust
    # the tile shape with 128B swizzle mode, there should always be 64 elements
    # on the contiguous dim.
    comptime c_tma_tile_shape_1 = config.c_swizzle.bytes() // size_of[c_type]()
    var c_tma_op = create_tma_tile[
        KernelType.CTmaTile.tile_layout,
        KernelType.CTmaTile.desc_layout,
        c_tma_tile_shape if not config.AB_swapped else Index(
            c_tma_tile_shape[0], c_tma_tile_shape_1
        ),
        swizzle_mode = config.c_swizzle,
    ](ctx, c_device)

    # Get the split-K kernel entry point.
    # Reduction TileTensor layout: shape = (UNKNOWN, BM, MMA_N),
    # strides = (BM*MMA_N, MMA_N, 1) -- all strides are static.
    comptime ReductionTTLayout = type_of(lt_to_tt(reduction_tensor)).LayoutType
    comptime kernel = matmul_kernel.run_splitk[ReductionTTLayout]

    var grid_dim = (
        align_up(ceildiv(M_maybe_swapped, BM), cluster_shape[0]),
        align_up(ceildiv(N_maybe_swapped, MMA_N), cluster_shape[1]),
        config.num_split_k,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], cluster_shape[1])),
        1,
    )

    # TODO: integrate with existing enums
    comptime load_warps = 1
    comptime mma_warps = 1
    comptime scheduler_warps = 1
    comptime epilogue_warps = 4

    var mnk = StaticTuple[UInt32, 3](UInt32(M), UInt32(N), UInt32(K))

    var workspace: Span[UInt64, MutAnyOrigin]

    var output_tiles = get_num_tiles(
        Index(M, N, K),
        Index(BM, MMA_N, BK),
        Index(cluster_shape[0], cluster_shape[1]),
    )
    var num_output_tiles = output_tiles[0] * output_tiles[1]
    var lock_buffer_size_bytes = get_required_locks_buffer_size_bytes[
        config.accum_type
    ](
        Index(M, N, K),
        Index(BM, MMA_N, BK),
        Index(cluster_shape[0], cluster_shape[1]),
    )

    var locks_buffer = ctx.enqueue_create_buffer[DType.uint8](
        lock_buffer_size_bytes
    )
    var reduction_workspace = ctx.enqueue_create_buffer[config.accum_type](
        num_output_tiles * BM * MMA_N
    )

    var reduction_tensor = LayoutTensor[config.accum_type, reduction_layout](
        reduction_workspace,
        RuntimeLayout[reduction_layout].row_major(
            Index(num_output_tiles, BM, MMA_N)
        ),
    )

    ctx.enqueue_memset(locks_buffer, 0)

    @parameter
    if enable_profiling:
        workspace = MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].get_workspace(ctx)
    else:
        workspace = Span[UInt64, MutAnyOrigin]()

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        lt_to_tt(reduction_tensor),
        locks_buffer,
        cluster_dim,
        mnk,
        workspace,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps
        block_dim=(
            32 * (load_warps + mma_warps + scheduler_warps + epilogue_warps)
        ),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(b200_smem)
        ),
    )

    _ = reduction_workspace^
    _ = locks_buffer^

    @parameter
    if enable_profiling:
        ctx.synchronize()
        MatmulWarpSpecializationWorkSpaceManager[
            max_profiled_tiles
        ].dump_workspace_as_csv(ctx, workspace, "profile")


fn matmul_sm100_fallback[
    a_layout: LegacyLayout,
    b_layout: LegacyLayout,
    c_layout: LegacyLayout,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    *,
    transpose_b: Bool,
    umma_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[c_type, c_layout, ...],
    a: LayoutTensor[a_type, a_layout, ...],
    b: LayoutTensor[b_type, b_layout, ...],
    ctx: DeviceContext,
) raises:
    comptime assert transpose_b, "Only support transposed B"

    comptime assert a_type == b_type and a_type in (
        DType.bfloat16,
        DType.float8_e4m3fn,
    ), "Only support bfloat16 and float8_e4m3fn"

    comptime BM = block_tile_shape[0]
    comptime BN = block_tile_shape[1]
    comptime BK = block_tile_shape[2]

    comptime smem_use = (
        BM * size_of[a_type]() + BN * size_of[b_type]()
    ) * BK + 24

    comptime block_dim = 128

    # Instantiate fallback kernel first (TMA layouts computed from config)
    from ..structured_kernels.tile_types import GMEMTile, lt_to_tt

    comptime CDeviceTileType = GMEMTile[c_type, c_layout]
    comptime fallback_kernel = BlackwellMatmulSM100FallbackKernel[
        a_type,
        b_type,
        c_type,
        CDeviceTileType.LayoutType,
        block_tile_shape,
        umma_shape,
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        num_threads=block_dim,
        elementwise_lambda_fn=elementwise_lambda_fn,
    ]
    comptime FallbackKernelType = type_of(fallback_kernel)
    comptime kernel = fallback_kernel.run

    # Create TMA descriptors using kernel-derived layout types
    a_tma_op = create_tma_tile[
        FallbackKernelType.ATmaTile.tile_layout,
        FallbackKernelType.ATmaTile.desc_layout,
        Index(BM, BK),
        swizzle_mode=a_swizzle,
    ](ctx, a)
    b_tma_op = create_tma_tile[
        FallbackKernelType.BTmaTile.tile_layout,
        FallbackKernelType.BTmaTile.desc_layout,
        Index(BN, BK) if transpose_b else Index(BK, BN),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    var M = c.dim[0]()
    var N = c.dim[1]()
    var K = a.dim[1]()

    ctx.enqueue_function[kernel, kernel](
        a_tma_op,
        b_tma_op,
        lt_to_tt(c),
        UInt(ceildiv(K, BK)),
        grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
        block_dim=(block_dim),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )
