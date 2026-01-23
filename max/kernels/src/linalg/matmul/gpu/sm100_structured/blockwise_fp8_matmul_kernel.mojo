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

"""Blockwise FP8 SM100 matmul kernel - Structured kernel with register accumulation.

Unlike standard SM100 matmul which accumulates in TMEM, blockwise FP8 applies
scaling factors per-K-iteration in CUDA cores, accumulating in registers.

Architecture:
- Load warp: TMA loads A, B, and A-scales into SMEM
- MMA warp: Standard MMA operations (partial results to TMEM)
- Epilogue warp: Per-K TMEM read → scale → register accumulate → final output

Key differences from standard/block-scaled kernels:
- Uses MmaOpSM100_SS (not block-scaled MMA)
- A-scales loaded via TMA, B-scales from global memory
- BlockwiseFP8Accumulator for register-based K-loop accumulation
- BlockwiseFP8TileWriter for final register → SMEM → GMEM flow
"""

from collections import OptionalReg
from math import ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import size_of

from gpu import WARP_SIZE, barrier
from gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import block_id_in_cluster, block_idx, lane_id, thread_idx
from gpu import warp_id as get_warp_id
from gpu.memory import (
    AddressSpace,
    external_memory,
    fence_mbarrier_init,
)
from gpu.compute.arch.mma_nvidia_sm100 import *
from gpu.sync import (
    mbarrier_arrive,
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
)
from gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import PipelineState, SharedMemBarrier, TMATensorTile

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.utils import elementwise_compute_lambda_type
from .config import MatmulConfig
from .pipeline import ProducerConsumerPipeline

# Structured kernel imports
from .matmul_kernels import WarpRole, KernelContext
from .blockwise_fp8_smem import BlockwiseFP8Smem
from .tile_pipeline import (
    InputTilePipeline,
    InputProducerStage,
    InputConsumerStage,
    BlockwiseFP8TilePayload,
)
from .tile_scheduler import TileScheduler as StructuredTileScheduler
from .tile_loader import TileLoaderTMA, ScalesTileLoader
from .tmem import TmemAllocation, TmemTensor
from .barriers import TmemDeallocBarrier
from .warp_context import MmaWarpContext, EpilogueWarpContext
from .tile_pipeline import OutputTilePipeline

# Blockwise FP8 specific components
from .blockwise_fp8_accumulator import (
    BlockwiseFP8Accumulator,
    get_accumulator_layout,
    is_lower_fragment_required,
)
from .blockwise_fp8_output_writer import BlockwiseFP8TileWriter


# =============================================================================
# BlackwellBlockwiseFP8MatmulKernel - Structured blockwise FP8 matmul kernel
# =============================================================================


struct BlackwellBlockwiseFP8MatmulKernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    # Tensor layouts (from TMA descriptors)
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_scales_layout: Layout,
    b_scales_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    a_scales_desc_layout: Layout,
    # Configuration
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Cluster shape (for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
]:
    """Blockwise FP8 matmul kernel with register-based accumulation.

    This kernel implements per-K-iteration scaling in CUDA cores:
    1. Load warp: TMA loads A, B, A-scales to SMEM
    2. MMA warp: Standard MMA (partial to TMEM)
    3. Epilogue warp: TMEM read → scale → register accumulate → output
    """

    # ========== Derived Constants (from config) ==========

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime accum_type = DType.float32
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M = Int(Self.config.cluster_shape[0])
    comptime CLUSTER_N = Int(Self.config.cluster_shape[1])
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = Self.num_output_warps * WARP_SIZE

    comptime NUM_THREADS = (
        Self.SCHEDULER_THREADS
        + Self.TMA_LOAD_THREADS
        + Self.MMA_THREADS
        + Self.EPILOGUE_THREADS
    )

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = Self.num_pipeline_stages // Self.config.k_group_size
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages = Int(Self.config.num_output_stages)

    # TMEM configuration
    comptime NUM_TMEM_COLS = 512
    comptime stage_stride_cols = Self.NUM_TMEM_COLS // Self.config.num_accum_pipeline_stages

    # ========== Barrier Arrival Counts ==========

    comptime clc_producer_arv_count = 1
    comptime clc_consumer_arv_count = Self.SCHEDULER_THREADS + Self.CLUSTER_SIZE * (
        Self.TMA_LOAD_THREADS + Self.MMA_THREADS + Self.EPILOGUE_THREADS
    )
    comptime clc_throttle_producer_arv_count = Self.TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = Self.SCHEDULER_THREADS
    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = Self.cta_group * Self.EPILOGUE_THREADS

    # ========== Shared Memory Layout Types ==========

    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, swizzle_mode = Self.config.a_swizzle
    ]()

    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]()

    comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # A-scales layout: 1D row vector with BM elements
    comptime a_scales_smem_layout = Layout.row_major(1, Self.BM)

    # ========== TMA Load Size Constants ==========
    comptime a_expected_bytes = Self.a_smem_layout.size() * size_of[
        Self.a_type
    ]()
    comptime b_expected_bytes = Self.b_smem_layout.size() * size_of[
        Self.b_type
    ]()
    comptime a_scales_expected_bytes = Self.a_scales_smem_layout.size() * size_of[
        Self.a_scales_type
    ]()
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.a_scales_expected_bytes
    )

    # TMA descriptor layout sizes for peer CTA slicing
    comptime a_tma_load_size = Self.a_desc_layout.size()
    comptime b_tma_load_size = Self.b_desc_layout.size()
    comptime a_tma_rows = Self.a_desc_layout.shape[0].value()
    comptime b_tma_rows = Self.b_desc_layout.shape[0].value()

    # ========== Shared Memory Type ==========
    comptime SmemType = BlockwiseFP8Smem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        Self.transpose_b,
        config = Self.config,
    ]

    # ========== MMA Operation Type ==========
    # Standard MMA (not block-scaled) - scaling applied in CUDA cores
    comptime MmaOp = MmaOpSM100_SS[
        Self.c_type,
        Self.a_type,
        Self.b_type,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type = Self.accum_type,
        cta_group = Self.cta_group,
        cluster_shape = Self.config.cluster_shape,
        a_swizzle = Self.config.a_swizzle,
        b_swizzle = Self.config.b_swizzle,
        transpose_b = Self.transpose_b,
    ]

    # ========== Kernel Context Type ==========
    comptime Context = KernelContext[
        Self.num_clc_pipeline_stages,
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
    ]

    # TMEM allocation size
    comptime max_tmem_cols: UInt = 512

    # ========== Tile Scheduler Type ==========
    comptime Scheduler = StructuredTileScheduler[
        num_stages = Self.num_clc_pipeline_stages,
        cluster_shape = Index[dtype = DType.uint32](
            Self.config.cluster_shape[0],
            Self.config.cluster_shape[1],
            Self.config.cluster_shape[2],
        ),
        block_swizzle_size = Self.config.block_swizzle_size,
        rasterize_order = Self.config.raster_order,
    ]

    # ========== Tile Pipeline Type ==========
    comptime TilePayload = BlockwiseFP8TilePayload[
        Self.a_type,
        Self.b_type,
        Self.a_scales_type,
        Self.SmemType.a_smem_layout,
        Self.SmemType.b_smem_layout,
        Self.SmemType.a_scales_smem_layout,
        Self.SmemType.num_pipeline_stages,
    ]
    comptime InputTilePipeline = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Int(Self.config.k_group_size),
    ]

    # ========== Tile Loader Types ==========
    comptime ATileLoaderType = TileLoaderTMA[cta_group = Self.cta_group]
    comptime BTileLoaderType = TileLoaderTMA[cta_group = Self.cta_group]
    comptime AScalesLoaderType = ScalesTileLoader[cta_group = Self.cta_group]

    # ========== TMEM Types ==========
    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime TmemDealloc = TmemDeallocBarrier[Self.cta_group]

    # Layout-parameterized TMEM tensor for typed accumulator access
    comptime tmem_accum_layout = Layout.row_major(Self.MMA_M, Self.MMA_N)
    comptime AccumTensor = TmemTensor[
        Self.accum_type, Self.tmem_accum_layout, cta_group = Self.cta_group
    ]

    # ========== Output Pipeline Type ==========
    comptime OutputPipeline = OutputTilePipeline[
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]

    # ========== Warp Context Types ==========
    comptime MmaCtx = MmaWarpContext[
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # ========== Accumulator Type ==========
    comptime is_lower_required = is_lower_fragment_required[
        Self.cta_group, Self.config.block_tile_shape
    ]()

    comptime accum_layout = get_accumulator_layout[
        c_smem_layout = Self.c_smem_layout,
        block_tile_shape = Self.config.block_tile_shape,
        mma_shape = Self.config.mma_shape,
        cta_group = Self.cta_group,
    ]()

    comptime Accumulator = BlockwiseFP8Accumulator[
        Self.accum_type,
        Self.accum_layout,
        Self.is_lower_required,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        Self.CLUSTER_SIZE,
    ]

    # ========== Output Writer Type ==========
    comptime TileWriterType = BlockwiseFP8TileWriter[
        Self.c_type,
        Self.c_smem_layout,
        Self.accum_type,
        Self.accum_layout,
        block_tile_shape = Self.config.block_tile_shape,
        mma_shape = Self.config.mma_shape,
        is_lower_frag_required = Self.is_lower_required,
        cta_group = Self.cta_group,
        num_output_stages = Self.num_output_stages,
        num_output_warps = Self.num_output_warps,
        c_swizzle = Self.config.c_swizzle,
    ]

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    fn load_input_tiles[
        a_tma_origin: ImmutOrigin,
        b_tma_origin: ImmutOrigin,
        a_scales_tma_origin: ImmutOrigin,
        tiles_origin: MutOrigin,
        //,
    ](
        a_loader: TileLoaderTMA[
            a_tma_origin,
            Self.a_type,
            Self.a_layout,
            Self.a_desc_layout,
            cta_group = Self.cta_group,
        ],
        b_loader: TileLoaderTMA[
            b_tma_origin,
            Self.b_type,
            Self.b_layout,
            Self.b_desc_layout,
            cta_group = Self.cta_group,
        ],
        a_scales_loader: ScalesTileLoader[
            a_scales_tma_origin,
            Self.a_scales_type,
            Self.a_scales_layout,
            Self.a_scales_desc_layout,
            cta_group = Self.cta_group,
        ],
        tiles: InputProducerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Int(Self.config.k_group_size),
        ],
        peer_cta_coord: Tuple[UInt, UInt, UInt],
        work_tile_coord: Tuple[UInt, UInt],
        iter_idx: UInt,
        elect_one_cta: Bool,
    ):
        """Load A, B, and A-scales tiles using TMA.

        Args:
            a_loader: TileLoaderTMA for A matrix.
            b_loader: TileLoaderTMA for B matrix.
            a_scales_loader: ScalesTileLoader for A-scales.
            tiles: InputProducerStage context with encapsulated tile access.
            peer_cta_coord: Peer CTA coordinates for multicast.
            work_tile_coord: Current work tile M/N coordinates.
            iter_idx: K iteration index.
            elect_one_cta: Whether this is the elected CTA in the cluster.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        var a_gmem_m_coord = peer_m_rank * UInt(
            Self.a_tma_rows
        ) + work_tile_coord[0] * UInt(Self.BM)
        var b_gmem_n_coord = (
            peer_rank_m * UInt(Self.b_tma_rows)
            + peer_rank_n * UInt(Self.BN)
            + work_tile_coord[1] * UInt(Self.MMA_N)
        )

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()
            var stage = tiles.stage()

            # Get tiles
            var a_tile, b_tile, a_scales_tile = tiles.payload().get_tile[
                Int(Self.config.k_group_size)
            ](stage, 0)

            # Peer CTA slicing
            var a_peer_tile = type_of(a_tile)(
                a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size)
            )
            var b_peer_tile = type_of(b_tile)(
                b_tile.ptr + peer_rank_m * UInt(Self.b_tma_load_size)
            )

            # Load A and B using TileLoaders
            a_loader.load(
                a_peer_tile,
                barrier[0],
                iter_idx * UInt(Self.BK),
                a_gmem_m_coord,
            )
            b_loader.load(
                b_peer_tile,
                barrier[0],
                iter_idx * UInt(Self.BK),
                b_gmem_n_coord,
            )

            # Load A-scales using ScalesTileLoader
            a_scales_loader.load(
                a_scales_tile,
                barrier[0],
                Int(work_tile_coord[0]) * Self.BM,
                Int(iter_idx),
            )

    # ========== MMA Operation ==========

    @staticmethod
    @always_inline
    fn mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tiles: InputConsumerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Int(Self.config.k_group_size),
        ],
        mma_op: Self.MmaOp,
        accum_tensor: Self.AccumTensor,
    ):
        """Execute standard MMA operations (partial results to TMEM).

        For blockwise FP8, each K iteration writes a fresh partial to TMEM.
        The epilogue accumulates across K in registers, not TMEM.
        Therefore init_c is always True (unlike standard matmul).

        Args:
            tiles: Input consumer stage with A, B, A-scales tiles.
            mma_op: The MMA operator.
            accum_tensor: Typed TMEM tensor view for the accumulator stage.
        """
        if elect_one_sync():
            # Loop through k_group_size tiles (typically 1)
            for jj in range(Int(Self.config.k_group_size)):
                var a_tile, b_tile, _ = tiles.payload().get_tile[
                    Int(Self.config.k_group_size)
                ](tiles.stage(), jj)

                # Blockwise FP8: always init_c=True since epilogue accumulates
                # in registers, not TMEM. Use typed tensor's offset() for MMA.
                mma_op.mma(
                    a_tile,
                    b_tile,
                    accum_tensor.offset(),
                    init_c=True,
                )

            mma_op.commit(tiles.mbar())

    # ========== Compile-Time Validation ==========

    @staticmethod
    fn validate_config():
        """Validate configuration constraints at compile time."""
        constrained[Self.transpose_b, "Only support transposed B"]()
        constrained[
            Self.a_scales_type == Self.b_scales_type,
            "a_scales_type and b_scales_type must match",
        ]()
        constrained[
            Self.cta_group in (1, 2),
            "Only support cta_group == 1 or 2",
        ]()
        constrained[Self.BK == 128, "Only support BK = 128"]()

    # ========== Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(a_scales_tma_op, `nvvm.grid_constant`)
    fn run(
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        a_scales_tma_op: TMATensorTile[
            Self.a_scales_type, Self.a_scales_layout, Self.a_scales_desc_layout
        ],
        cluster_dim: StaticTuple[Int32, 3],
        num_iters: UInt,
        b_scales: LayoutTensor[
            Self.b_scales_type, Self.b_scales_layout, MutAnyOrigin
        ],
        problem_shape: StaticTuple[Int32, 3],
    ):
        """Kernel entry point for blockwise FP8 matmul."""
        Self.validate_config()

        # ===== Shared Memory Setup =====
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var a_scales_tiles = smem.a_scales_tiles()

        var input_barriers = smem.tma_mma_mbars()
        var accum_barriers = smem.accum_mbars()
        var clc_full = smem.clc_mbars_full()
        var clc_empty = smem.clc_mbars_empty()
        var clc_throttle = smem.clc_throttle_mbars()
        var clc_response_arr = smem.clc_response()
        var tmem_addr_arr = smem.tmem_addr()
        var tmem_addr_storage = tmem_addr_arr.ptr

        var tile_payload = Self.TilePayload(a_tiles, b_tiles, a_scales_tiles)
        var input_pipeline = Self.InputTilePipeline(
            input_barriers, tile_payload
        )

        var ctx = Self.Context(smem.tmem_addr().ptr)

        # ===== Barrier Initialization =====
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()
            a_scales_tma_op.prefetch_descriptor()

            # Include epilogue warps in consumer count (they also consume A-scales)
            Self.InputTilePipeline.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Self.config.cluster_shape[0] // Self.cta_group
                + Self.config.cluster_shape[1]
                - 1
                + Self.CLUSTER_SIZE * (Self.EPILOGUE_THREADS // 32),
            )

            ProducerConsumerPipeline[Self.config.num_accum_pipeline_stages](
                accum_barriers.ptr
            ).init_mbars(
                Self.accum_pipeline_producer_arv_count,
                Self.accum_pipeline_consumer_arv_count,
            )

            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Self.clc_throttle_producer_arv_count,
                Self.clc_throttle_consumer_arv_count,
            )

            smem.tmem_dealloc_mbar().ptr[].init(
                Self.EPILOGUE_THREADS * Self.cta_group
            )

            @parameter
            for i in range(Self.num_clc_pipeline_stages):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Self.clc_consumer_arv_count)

        fence_mbarrier_init()
        cluster_sync()

        var mma_op = Self.MmaOp()

        # Create structured scheduler
        var scheduler = Self.Scheduler(
            cluster_dim, clc_response_arr, clc_full, clc_empty, clc_throttle
        )

        var work_iter = scheduler.work_iterator()

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():
            var a_loader = Self.ATileLoaderType(
                Pointer(to=a_tma_op), ctx.a_multicast_mask
            )
            var b_loader = Self.BTileLoaderType(
                Pointer(to=b_tma_op), ctx.b_multicast_mask
            )
            var a_scales_loader = Self.AScalesLoaderType(
                Pointer(to=a_scales_tma_op)
            )

            with input_pipeline.producer() as producer:
                while work_iter.has_work():
                    with work_iter.next() as current:
                        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                        for i in range(num_iters):
                            with producer.acquire() as tiles:  # waits for consumer
                                Self.load_input_tiles(
                                    a_loader,
                                    b_loader,
                                    a_scales_loader,
                                    tiles,
                                    ctx.peer_cta_coord,
                                    current.coord(),
                                    i,
                                    ctx.elect_one_cta,
                                )

                producer.drain()  # wait for consumer before CTA exits

        # ===== SCHEDULER WARP =====
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:

            @parameter
            if Self.num_clc_pipeline_stages == 0:
                return

            var sched_iter = scheduler.scheduler_iterator()

            while sched_iter.has_work():
                with sched_iter.next():
                    sched_iter.signal_and_advance()

            sched_iter.drain()

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            var mma_ctx = Self.MmaCtx.create(
                smem.tmem_addr(),
                accum_barriers,
                smem.tmem_dealloc_mbar(),
                ctx.mma_complete_mask,
            )

            with mma_ctx:  # TMEM lifecycle
                while work_iter.has_work():
                    with work_iter.wait_and_advance():  # blocks on CLC
                        if ctx.elect_one_cta:
                            with input_pipeline.consumer() as consumer:
                                for i in range(num_iters):
                                    with mma_ctx.per_k_stage() as mma_stage:
                                        var accum = Self.AccumTensor(
                                            mma_stage.tmem.offset()
                                        )
                                        with consumer.acquire() as input_tiles:
                                            Self.mma(
                                                input_tiles,
                                                mma_op,
                                                accum,
                                            )

        # ===== EPILOGUE WARP =====
        if WarpRole.is_epilogue():
            Self.EpilogueCtx.Sync.wait()
            var epi_ctx = Self.EpilogueCtx.create(
                smem.tmem_addr(),
                accum_barriers,
                smem.tmem_dealloc_mbar(),
                ctx.mma_complete_mask,
            )

            with epi_ctx:
                while work_iter.has_work():
                    with work_iter.next() as current:
                        var accum = Self.Accumulator()

                        for k_iter in range(num_iters):
                            with epi_ctx.per_k_stage(
                                input_pipeline
                            ) as epi_stage:
                                accum.promote(
                                    b_scales,
                                    a_scales_tiles,
                                    epi_stage,
                                    work_tile_coord=current.coord(),
                                    k_iter=k_iter,
                                    problem_shape=problem_shape,
                                )

                        named_barrier[Self.num_output_warps * WARP_SIZE]()

                        Self.TileWriterType.write(
                            accum,
                            c_tiles,
                            c_tma_op,
                            c_coord=current.coord(),
                        )
