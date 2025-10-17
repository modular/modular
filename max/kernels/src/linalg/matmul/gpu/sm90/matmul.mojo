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
from collections import OptionalReg
from math import ceildiv
from sys import size_of

from buffer.buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.globals import WARPGROUP_SIZE
from gpu.grid_controls import pdl_launch_attributes
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import H100
from layout import Layout
from layout._ndbuffer_stub import from_ndbuffer_row_major
from layout.tma_async import create_tma_tile, create_tma_tile_template
from logger import Logger
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import MatmulConfig, get_hilbert_lut_with_cache
from ..tile_scheduler import MatmulSchedule, RasterOrder
from ..tile_scheduler_splitk import SplitKTileScheduler
from .matmul_kernels import HopperMatmulSM90Kernel, find_K_alignment_upto_16B


fn _is_valid_cluster_shape[
    cluster_shape: IndexList[3]
](grid_shape: IndexList[2], num_tiles_n: Int) -> Bool:
    if num_tiles_n % cluster_shape[0] != 0:
        return False

    @parameter
    for i in range(2):
        if (
            grid_shape[i] < cluster_shape[i]
            or grid_shape[i] % cluster_shape[i] != 0
        ):
            return False

    return True


fn _get_grid_shape[
    cluster_shape: IndexList[3] = Index(1, 1, 1)
](num_tiles_n: Int) -> IndexList[2]:
    # Hardcode values on purpose until we move this inside tile scheduler
    # in a more robust way.
    alias h100_num_SMs = H100.sm_count
    num_blocks_n = min(num_tiles_n, h100_num_SMs)
    adjusted_grid_shape = Index(
        num_blocks_n,
        h100_num_SMs // num_blocks_n,
    )

    # A Naive heuristic to select grid shape based on number of tile in N.
    if num_tiles_n % 8 == 0 or not _is_valid_cluster_shape[cluster_shape](
        adjusted_grid_shape, num_tiles_n
    ):
        return Index(8, 16)

    return adjusted_grid_shape


fn _is_valid_grid_shape[
    grid_shape: IndexList[2], cluster_shape: IndexList[3]
](num_tiles_n: Int) -> Bool:
    constrained[
        grid_shape[0] * grid_shape[1] <= H100.sm_count,
        "Total grid size exceed number of SMs in H100.",
    ]()

    if not _is_valid_cluster_shape[cluster_shape](grid_shape, num_tiles_n):
        return False

    if grid_shape[0] <= num_tiles_n:
        return num_tiles_n % grid_shape[0] == 0

    return grid_shape[0] % num_tiles_n == 0


fn warp_specialize_gemm_with_multicasting[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    hilbert_swizzle: Bool = False,
    splits: Int = 0,
    raster_order: RasterOrder = RasterOrder.AlongM,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    """Unified dispatcher for all matmul kernel variants."""
    if splits > 0:
        # Dispatch to split-k kernel
        warp_specialize_gemm_with_multicasting_splitk[
            c_type,
            c_shape,
            a_type,
            a_shape,
            b_type,
            b_shape,
            transpose_b=transpose_b,
            config=config,
            splits=splits,
            raster_order=raster_order,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ](c_device, a_device, b_device, ctx)
    else:
        # Dispatch to regular kernel
        _warp_specialize_gemm_with_multicasting_impl[
            c_type,
            c_shape,
            a_type,
            a_shape,
            b_type,
            b_shape,
            transpose_b=transpose_b,
            config=config,
            grid_shape=grid_shape,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            schedule=schedule,
            hilbert_swizzle=hilbert_swizzle,
        ](c_device, a_device, b_device, ctx)


fn _warp_specialize_gemm_with_multicasting_impl[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    grid_shape: OptionalReg[IndexList[2]] = None,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    schedule: MatmulSchedule = MatmulSchedule.NONE,
    hilbert_swizzle: Bool = False,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    alias N_static = c_shape.get[1]()
    alias K_static = a_shape.get[1]()
    var M = c_device.dim[0]()
    var N = c_device.dim[1]()
    var K = a_device.dim[1]()

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    constrained[
        (a_type == b_type is DType.float8_e4m3fn)
        or (a_type == b_type and a_type in (DType.bfloat16, DType.float32)),
        "Unsupported input dtype",
    ]()

    constrained[
        a_type != DType.float8_e4m3fn or BK == 128,
        "BK must be 128 for fp8 data type for numerical accuracy correctness",
    ]()

    constrained[
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None,
        "Either the epilogue lambda or the compute lambda can be used",
    ]()

    constrained[
        BM > 64 or (BM == 64 and config.num_consumer == 1),
        "Only support 1 consumer for BM=64",
    ]()

    alias k_align = find_K_alignment_upto_16B(K_static * size_of[a_type]())
    constrained[
        k_align in (4, 8, 16), "H100 matmul K dim must be multiple of 4B"
    ]()

    var logger = Logger()

    logger.info("Executing Warp Specialized Gemm with Multicasting")
    logger.info("block_tile_shape:", config.block_tile_shape)
    logger.info("cluster_shape:", config.cluster_shape)
    logger.info("mma_shape:", config.mma_shape)

    @parameter
    if schedule == MatmulSchedule.NONE:
        pass
    elif schedule == MatmulSchedule.DS_SCHEDULER:
        constrained[
            grid_shape is not None,
            "Grid shape must be provided for DS scheduler",
        ]()
        alias ds_grid_shape = grid_shape.value()
        constrained[
            ds_grid_shape[0] <= H100.sm_count and ds_grid_shape[1] == 1,
            "Deepseek scheduler only accepts grid shape with 1 column",
        ]()

    elif grid_shape:
        constrained[
            _is_valid_grid_shape[grid_shape.value(), config.cluster_shape](
                ceildiv(N_static, BN)
            ),
            String(
                "grid shape:",
                grid_shape.value(),
                "is not compatible with cluster shape:",
                config.cluster_shape,
                "and static N:",
                N_static,
                sep=" ",
            ),
        ]()

    alias grid_shape_adjusted = grid_shape.value() if grid_shape else _get_grid_shape[
        config.cluster_shape
    ](
        ceildiv(N_static, BN)
    )

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        Int(config.num_pipeline_stages),
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(),
        c_smem_layout.shape[1].value() // config.num_consumer,
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # make sure TMA_BN = 64 -> 128B swizzle, 32 -> 64B swizzle and etc.
    alias c_swizzle = TensorMapSwizzle(
        min(log2_floor(c_smem_tile[1] // 8), 3)
    ) if use_tma_store else TensorMapSwizzle.SWIZZLE_NONE

    var c_tma_op = create_tma_tile_template[
        c_type,
        2,
        c_smem_tile,
        swizzle_mode=c_swizzle,
        __desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1]),
    ]()

    @parameter
    if use_tma_store:
        c_tma_op = create_tma_tile[
            c_smem_tile,
            swizzle_mode=c_swizzle,
            __desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1]),
        ](ctx, c)

    var lut_ptr = ctx.enqueue_create_buffer[DType.uint32](0)

    @parameter
    if hilbert_swizzle:
        var grid_x = ceildiv(N, BN)
        var grid_y = ceildiv(M, BM)
        lut_ptr = get_hilbert_lut_with_cache(ctx, grid_x, grid_y)

    alias num_threads = WARPGROUP_SIZE * config.num_consumer + WARPGROUP_SIZE

    alias matmul_kernel[hilbert_swizzle: Bool = False] = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        a.layout,
        b.layout,
        c.layout,
        c_smem_layout,
        config.block_tile_shape,
        config.mma_shape,
        cluster_shape,
        Int(config.num_pipeline_stages),
        Int(num_threads),
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast = config.partitioned_multicast,
        use_tma_store=use_tma_store,
        promotion_frequency=1,
        pdl_level = config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        hilbert_swizzle=hilbert_swizzle,
    ]

    alias smem_size = matmul_kernel[].SMem.storage_size()

    constrained[
        smem_size <= H100.shared_memory_per_multiprocessor - 1024,
        "requested SMEM size exceeds 227KB limit.",
    ]()

    # TMA requires stride (K) multiple of 16B. If not satisfied,
    # we need to use cp.async.ca for 4B and 8B access, and ld for
    # 2B or smaller access.
    # Note that K * size_of[a_type]() decides the 2nd row's alignment
    # and Nvidia requires access alignment by access size.
    # Dispatch kernel using TMA load when the stride is multiple of 16B.
    @parameter
    if k_align == 16:
        var a_tma_op = create_tma_tile[
            Index(
                BM // CLUSTER_N, BK
            ) if config.partitioned_multicast else Index(BM, BK),
            swizzle_mode=a_swizzle,
        ](ctx, a)

        var b_tma_op = create_tma_tile[
            Index(
                BN // CLUSTER_M, BK
            ) if config.partitioned_multicast else Index(BN, BK),
            swizzle_mode=b_swizzle,
        ](ctx, b)

        @parameter
        if schedule != MatmulSchedule.NONE:
            alias kernel = matmul_kernel[].run_persistent[
                a_tma_op.layout,
                b_tma_op.layout,
                c_tma_op.layout,
                a_tma_op.desc_layout,
                b_tma_op.desc_layout,
                c_tma_op.desc_layout,
                grid_shape=grid_shape_adjusted,
                schedule=schedule,
            ]

            ctx.enqueue_function[kernel](
                a_tma_op,
                b_tma_op,
                c_tma_op,
                c,
                Index(M, N, K),
                grid_dim=(grid_shape_adjusted[0], grid_shape_adjusted[1]),
                block_dim=(num_threads),
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_size
                ),
                attributes=pdl_launch_attributes(config.pdl_level()),
            )
        else:
            alias kernel = matmul_kernel[hilbert_swizzle=hilbert_swizzle].run[
                a_tma_op.layout,
                b_tma_op.layout,
                c_tma_op.layout,
                a_tma_op.desc_layout,
                b_tma_op.desc_layout,
                c_tma_op.desc_layout,
            ]

            ctx.enqueue_function[kernel](
                a_tma_op,
                b_tma_op,
                c_tma_op,
                a,
                b,
                c,
                lut_ptr,
                grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
                block_dim=(num_threads),
                shared_mem_bytes=smem_size,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    smem_size
                ),
                attributes=pdl_launch_attributes(config.pdl_level()),
            )

    # Dispatch kernel using cp.async.ca when the stride is not multiple of 4B or 8B..
    else:
        alias kernel = matmul_kernel[].run_unaligned[
            c_tma_op.desc_layout,
            c_tma_op.layout,
        ]

        ctx.enqueue_function[kernel](
            c_tma_op,
            a,
            b,
            c,
            grid_dim=(ceildiv(N, BN), ceildiv(M, BM)),
            block_dim=(num_threads),
            shared_mem_bytes=smem_size,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                smem_size
            ),
            attributes=pdl_launch_attributes(config.pdl_level()),
        )


fn _get_c_smem_layout[
    block_tile_shape: IndexList[3],
    a_type: DType,
    b_type: DType,
    c_type: DType,
    num_pipeline_stages: Int,
]() -> Layout:
    alias BM = Int(block_tile_shape[0])
    alias BN = Int(block_tile_shape[1])
    alias BK = Int(block_tile_shape[2])

    alias WG_BM = BM
    alias MAX_WG_BN = 128

    alias available_smem_size = Int(
        H100.shared_memory_per_multiprocessor - 1024
    )
    alias pipeline_smem_size = Int(
        num_pipeline_stages
        * (
            BM * BK * size_of[a_type]()
            + BN * BK * size_of[b_type]()
            + (size_of[Int64]() * 2)
        )
    )

    alias available_c_smem_size = Int(available_smem_size - pipeline_smem_size)
    # We want the shared memory N to be at least 16 when using `stmatrix`
    # (c_type = bf16) because it would make TMA and masked copy from shared
    # memory to global memory easier.
    alias MIN_WG_BN = 16 if size_of[c_type]() == 2 else BN // 4

    @parameter
    if available_smem_size > (
        pipeline_smem_size + (WG_BM * MIN_WG_BN * size_of[c_type]())
    ):

        fn _get_max_wg_bn() capturing -> Int:
            var WG_BN = MAX_WG_BN
            while (
                available_c_smem_size < WG_BM * WG_BN * size_of[c_type]()
                or BN % WG_BN != 0
            ) and WG_BN > MIN_WG_BN:
                WG_BN //= 2
            return WG_BN

        alias max_wg_bn = _get_max_wg_bn()
        return Layout.row_major(WG_BM, max_wg_bn)
    else:
        constrained[
            False,
            "There is not enough SMEM to fit the pipeline yet alone the"
            " output tile!"
            + " available_smem_size: "
            + String(available_smem_size)
            + " pipeline_smem_size + WG_BM * MIN_WG_BN * size_of[c_type](): "
            + String(
                pipeline_smem_size + WG_BM * MIN_WG_BN * size_of[c_type]()
            ),
        ]()
        return Layout.row_major(0, 0)


fn warp_specialize_gemm_with_multicasting_splitk[
    c_type: DType,
    c_shape: DimList,
    a_type: DType,
    a_shape: DimList,
    b_type: DType,
    b_shape: DimList,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    splits: Int,
    raster_order: RasterOrder,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c_device: NDBuffer[c_type, 2, _, c_shape],
    a_device: NDBuffer[a_type, 2, _, a_shape],
    b_device: NDBuffer[b_type, 2, _, b_shape],
    ctx: DeviceContext,
) raises:
    var a = from_ndbuffer_row_major(a_device)
    var b = from_ndbuffer_row_major(b_device)
    var c = from_ndbuffer_row_major(c_device)

    var M = c.dim[0]()
    alias N = c_shape.get[1]()
    alias K = a_shape.get[1]()

    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    constrained[
        (a_type == b_type is DType.float8_e4m3fn)
        or (a_type == b_type and a_type in (DType.bfloat16, DType.float32)),
        "Unsupported input dtype",
    ]()

    constrained[
        a_type != DType.float8_e4m3fn or BK == 128,
        "BK must be 128 for fp8 data type for numerical accuracy correctness",
    ]()

    constrained[
        elementwise_lambda_fn is None or elementwise_compute_lambda_fn is None,
        "Either the epilogue lambda or the compute lambda can be used",
    ]()

    constrained[
        BM > 64 or (BM == 64 and config.num_consumer == 1),
        "Only support 1 consumer for BM=64",
    ]()

    var logger = Logger()

    logger.info("Executing Split-K Warp Specialized GEMM with Multicasting")
    logger.info("block_tile_shape:", config.block_tile_shape)
    logger.info("cluster_shape:", config.cluster_shape)
    logger.info("mma_shape:", config.mma_shape)

    alias cluster_shape = StaticTuple[Int32, 3](
        config.cluster_shape[0],
        config.cluster_shape[1],
        config.cluster_shape[2],
    )

    alias CLUSTER_N = UInt(cluster_shape[0])
    alias CLUSTER_M = UInt(cluster_shape[1])

    alias c_smem_layout = _get_c_smem_layout[
        config.block_tile_shape,
        a_type,
        b_type,
        c_type,
        Int(config.num_pipeline_stages),
    ]()
    alias c_smem_tile = Index(
        c_smem_layout.shape[0].value(),
        c_smem_layout.shape[1].value() // config.num_consumer,
    )

    alias a_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias b_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # make sure TMA_BN = 64 -> 128B swizzle, 32 -> 64B swizzle and etc.
    alias c_swizzle = TensorMapSwizzle(
        min(log2_floor(c_smem_tile[1] // 8), 3)
    ) if use_tma_store else TensorMapSwizzle.SWIZZLE_NONE

    a_tma_op = create_tma_tile[
        Index(BM // CLUSTER_N, BK) if config.partitioned_multicast else Index(
            BM, BK
        ),
        swizzle_mode=a_swizzle,
    ](ctx, a)
    b_tma_op = create_tma_tile[
        Index(BN // CLUSTER_M, BK) if config.partitioned_multicast else Index(
            BN, BK
        ),
        swizzle_mode=b_swizzle,
    ](ctx, b)

    c_tma_op = create_tma_tile[
        c_smem_tile,
        swizzle_mode=c_swizzle,
        __desc_layout = Layout.row_major(c_smem_tile[0], c_smem_tile[1]),
    ](ctx, c)

    alias scheduler = SplitKTileScheduler[
        Index(N, K),
        config.block_tile_shape,
        splits,
        config.num_consumer,
        config.num_pipeline_stages,
        Index(config.cluster_shape[1], config.cluster_shape[0]),
        raster_order,
    ]

    var launch_grid_shape = scheduler.get_grid_shape(
        config.cluster_shape,
        raster_order,
    )

    alias accum_type = DType.float32  # fix this

    var NUM_TILES = scheduler.get_num_tiles(
        Index(M, N, K),
        config.block_tile_shape,
        Index(config.cluster_shape[1], config.cluster_shape[0]),
    )

    var workspace_data = ctx.enqueue_create_buffer[accum_type](
        NUM_TILES * BM * BN
    )
    var reduction_workspace = NDBuffer[accum_type, 3](
        workspace_data.unsafe_ptr(),
        Index(NUM_TILES, BM, BN),
    )

    var locks_buffer_size_bytes = (
        scheduler.get_required_locks_buffer_size_bytes[
            accum_type, config.num_consumer
        ](
            Index(M, N, K),
            config.block_tile_shape,
            Index(CLUSTER_M, CLUSTER_N),
        )
    )

    var locks_ptr = ctx.enqueue_create_buffer[DType.uint8](
        locks_buffer_size_bytes
    )

    ctx.enqueue_memset(locks_ptr, 0)

    alias num_threads = config.num_consumer * 128 + 128

    alias matmul_kernel = HopperMatmulSM90Kernel[
        a_type,
        b_type,
        c_type,
        a.layout,
        b.layout,
        c.layout,
        c_smem_layout,
        config.block_tile_shape,
        config.mma_shape,
        cluster_shape,
        Int(config.num_pipeline_stages),
        Int(num_threads),
        transpose_b=True,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        c_swizzle=c_swizzle,
        partitioned_multicast = config.partitioned_multicast,
        use_tma_store=use_tma_store,
        promotion_frequency=1,
        pdl_level = config.pdl_level(),
        elementwise_lambda_fn=elementwise_lambda_fn,
        elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
    ]

    alias smem_size = matmul_kernel.SMem.storage_size()

    constrained[
        smem_size <= H100.shared_memory_per_multiprocessor - 1024,
        "requested SMEM size exceeds 227KB limit.",
    ]()

    alias kernel = matmul_kernel.run_splitk[
        a_tma_op.layout,
        b_tma_op.layout,
        c_tma_op.layout,
        a_tma_op.desc_layout,
        b_tma_op.desc_layout,
        c_tma_op.desc_layout,
        splits=splits,
        raster_order=raster_order,
    ]

    ctx.enqueue_function[kernel](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        c,
        reduction_workspace,
        locks_ptr,
        Index(M, N, K),
        grid_dim=(
            launch_grid_shape[0],
            launch_grid_shape[1],
            launch_grid_shape[2],
        ),
        block_dim=(num_threads),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_size),
        attributes=pdl_launch_attributes(config.pdl_level()),
    )

    _ = workspace_data^
    _ = locks_ptr^
