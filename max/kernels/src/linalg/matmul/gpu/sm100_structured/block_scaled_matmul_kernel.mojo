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

"""Block-scaled SM100 matmul kernel - Structured kernel using tile pipelines.

Uses patterns from matmul_kernels.mojo with typed SMEM accessors and
context manager-based pipeline synchronization for MXFP8 and NVFP4
block-scaled matrix multiplication.

Architecture:
- Uses Self.SmemType (BlockScaledSmem) with typed tile/barrier accessors
- Uses Self.InputTilePipeline (BlockScaledTilePipeline) for producer/consumer sync
- Load warp: with input_pipeline.producer() as stage -> Self.load_input_tiles()
- MMA warp: with input_pipeline.consumer() as stage -> Self.mma()
- Epilogue warp: Uses structured building blocks from tile_writer.mojo

Epilogue Building Blocks (from tile_writer.mojo):
- TmemArrayType / load_fragments() for TMEM load
- AccumBarrier.arrive() for barrier signaling
- TMEMToSMemWriter.write_fragments() for SMEM write
- tma_wait_pipelined() for TMA wait
- TMA store remains inline (3D batch coordinates)

Key structured patterns:
- Context manager pattern for pipeline synchronization
- ProducerStage/ConsumerStage encapsulate tiles and barriers
- stage.get_tiles(j) returns (a, b, sfa, sfb) tuple
- Automatic wait/step in context manager __enter__/__exit__
"""

from collections import OptionalReg
from math import ceildiv
from memory import LegacyUnsafePointer, Pointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import align_of, size_of, simd_width_of

from gpu import WARP_SIZE, barrier, warp_id
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
from gpu.primitives.grid_controls import (
    launch_dependent_grids,
    PDLLevel,
    wait_on_dependent_grids,
)
from gpu.sync import named_barrier, named_barrier_arrive, syncwarp
from gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor, RuntimeLayout
from layout.layout_tensor import LayoutTensorIter
from layout.int_tuple import IntTuple
from layout.swizzle import Swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_sf_layout_k_major,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_BlockScaled_SS
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
from .config import BlockScaledMatmulConfig
from linalg.matmul.gpu.profiler import MatmulProfileWarp
from .pipeline import ProducerConsumerPipeline

# Structured kernel imports
from .matmul_kernels import WarpRole, KernelContext
from .block_scaled_smem import BlockScaledSmem
from .tile_pipeline import (
    InputTilePipeline,
    InputProducerStage,
    InputConsumerStage,
    BlockScaledTilePayload,
)
from .tile_scheduler import TileScheduler as StructuredTileScheduler
from .tmem import TmemAllocation
from .tile_pipeline import OutputTilePipeline
from .barriers import TmemDeallocBarrier, WarpGroupBarrier
from .warp_context import MmaWarpContext, EpilogueWarpContext

# Block-scaled output writer for epilogue
from .block_scaled_output_writer import BlockScaledTileWriter


# =============================================================================
# BlackwellBlockScaledMatmulKernel - Structured block-scaled matmul kernel
# =============================================================================


struct BlackwellBlockScaledMatmulKernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    # Tensor layouts (from TMA descriptors)
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    sfa_layout: Layout,
    sfb_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    sfa_desc_layout: Layout,
    sfb_desc_layout: Layout,
    # Configuration
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    # Cluster shape (for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Optional features
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
]:
    """Block-scaled matmul kernel V3 - ported from working legacy kernel.

    This struct provides the structured interface while internally using
    the proven legacy kernel logic.
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

    comptime accum_type = DType.float32  # Hardcoded for block-scaled
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
    comptime SFA_NUM_COLS = Self.config.num_sf_k_tiles * (Self.BM // 32)
    comptime SFB_NUM_COLS = Self.config.num_sf_k_tiles * (Self.MMA_N // 32)
    comptime stage_stride_cols = Self.MMA_N

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

    # SF_K_GROUP_SIZE = SF_ATOM_K * vec_sf_size (from fp4_utils)
    comptime SF_K_GROUP_SIZE = SF_ATOM_K * Self.config.vec_sf_size

    comptime sfa_smem_layout = tile_sf_layout_k_major[
        Self.BM,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    comptime sfb_smem_layout = tile_sf_layout_k_major[
        Self.MMA_N,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    # ========== TMA Load Size Constants ==========
    # Expected bytes for TMA loads (used in expect_bytes)
    comptime a_expected_bytes = Self.a_smem_layout.size() * size_of[
        Self.a_type
    ]()
    comptime b_expected_bytes = Self.b_smem_layout.size() * size_of[
        Self.b_type
    ]()
    comptime sfa_expected_bytes = Self.sfa_smem_layout.size() * size_of[
        Self.sfa_dtype
    ]()
    comptime sfb_expected_bytes = Self.sfb_smem_layout.size() * size_of[
        Self.sfb_dtype
    ]()
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.sfa_expected_bytes
        + Self.sfb_expected_bytes
    ) * Int(Self.config.k_group_size)

    # TMA descriptor layout sizes for peer CTA slicing
    comptime a_tma_load_size = Self.a_desc_layout.size()
    comptime b_tma_load_size = Self.b_desc_layout.size()
    comptime a_tma_rows = Self.a_desc_layout.shape[1].value()
    comptime b_tma_rows = Self.b_desc_layout.shape[1].value()

    # ========== Shared Memory Type ==========
    # Uses BlockScaledSmem with typed accessors (same pattern as B200MatmulSmem)
    comptime SmemType = BlockScaledSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config = Self.config,
    ]

    # ========== MMA Operation Type ==========

    comptime MmaOp = MmaOpSM100_BlockScaled_SS[
        Self.c_type,
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.config.scaling_kind,
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
    # Encapsulates election variables, CTA coordinates, and multicast masks
    comptime Context = KernelContext[
        Self.num_clc_pipeline_stages,
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
    ]

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
    # Manages A, B, SFA, SFB tiles with producer-consumer synchronization
    # Uses generic TilePipeline with BlockScaledTilePayload for composition
    comptime TilePayload = BlockScaledTilePayload[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.SmemType.a_smem_layout,
        Self.SmemType.b_smem_layout,
        Self.SmemType.sfa_smem_layout,
        Self.SmemType.sfb_smem_layout,
        Self.SmemType.num_pipeline_stages,
    ]
    comptime InputTilePipeline = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Int(Self.config.k_group_size),
    ]

    # ========== TMEM and Output Pipeline Types ==========
    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime OutputPipeline = OutputTilePipeline[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]
    comptime TmemDealloc = TmemDeallocBarrier[Self.cta_group]

    # ========== Warp Context Types ==========
    # MMA-Epilogue handoff barrier (barrier_id=1)
    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.MMA_THREADS + Self.EPILOGUE_THREADS, 1
    ]

    # MMA warp context (TMEM + dealloc + OutputPipeline)
    comptime MmaCtx = MmaWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # Epilogue warp context
    comptime EpilogueCtx = EpilogueWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # ========== Block-Scaled Output Tile Writer ==========
    # Uses structured building blocks with 3D batch coordinates for TMA stores
    comptime TileWriterType = BlockScaledTileWriter[
        a_type = Self.a_type,
        accum_type = Self.accum_type,
        block_tile_shape = Self.config.block_tile_shape,
        mma_shape = Self.config.mma_shape,
        cta_group = Self.cta_group,
        num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages,
        c_swizzle = Self.config.c_swizzle,
        transpose_c = Self.config.AB_swapped,
        c_smem_layout = Self.SmemType.c_smem_layout,
        num_output_stages = Int(Self.config.num_output_stages),
        stage_stride_cols = Self.stage_stride_cols,
        num_output_warps = Self.num_output_warps,
    ]

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    fn load_input_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        sfa_tma_op: TMATensorTile[
            Self.sfa_dtype, Self.sfa_layout, Self.sfa_desc_layout
        ],
        sfb_tma_op: TMATensorTile[
            Self.sfb_dtype, Self.sfb_layout, Self.sfb_desc_layout
        ],
        tiles: InputProducerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Int(Self.config.k_group_size),
        ],
        peer_cta_coord: Tuple[UInt, UInt, UInt],
        work_tile_coord: Tuple[UInt, UInt, UInt],
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A, B, SFA, SFB tiles using TMA with InputProducerStage.

        This method uses the structured ProducerStage pattern from
        matmul_kernels.mojo, with tiles and barrier encapsulated in the stage.

        Args:
            a_tma_op: TMA descriptor for A matrix.
            b_tma_op: TMA descriptor for B matrix.
            sfa_tma_op: TMA descriptor for A scaling factors.
            sfb_tma_op: TMA descriptor for B scaling factors.
            tiles: ProducerStage context with encapsulated tile access.
            peer_cta_coord: (rank_n, rank_m, peer_m_rank) for peer CTA slicing.
            work_tile_coord: (m, n, k_start) coordinates of the work tile.
            a_multicast_mask: Multicast mask for A tiles.
            b_multicast_mask: Multicast mask for B tiles.
            iter_idx: K iteration index (base index for k_group).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # Global memory coordinates
        var a_gmem_m_coord = peer_m_rank * UInt(
            Self.a_tma_rows
        ) + work_tile_coord[0] * UInt(Self.BM)
        var b_gmem_n_coord = (
            peer_rank_m * UInt(Self.b_tma_rows)
            + peer_rank_n * UInt(Self.BN)
            + work_tile_coord[1] * UInt(Self.MMA_N)
        )
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            # Set expected bytes ONCE for all k_group tiles
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            # Get barrier for TMA multicast loads
            var barrier = tiles.barrier()

            for jj in range(Int(Self.config.k_group_size)):
                var j = UInt32(jj)

                # Get tiles at this pipeline stage using the payload accessor
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Int(Self.config.k_group_size)](
                        tiles.stage(), jj
                    )
                )

                # Peer CTA slice using pointer arithmetic
                var a_peer_tile = type_of(a_tile)(
                    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size)
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile.ptr + peer_rank_m * UInt(Self.b_tma_load_size)
                )

                var k_coord = UInt(iter_idx + j) * UInt(Self.BK)

                # Load A and B with multicast
                a_tma_op.async_multicast_load_3d[Self.cta_group](
                    a_peer_tile,
                    barrier[0],
                    (k_coord, UInt(a_gmem_m_coord), UInt(batch_coord)),
                    a_multicast_mask,
                )
                b_tma_op.async_multicast_load_3d[Self.cta_group](
                    b_peer_tile,
                    barrier[0],
                    (k_coord, UInt(b_gmem_n_coord), UInt(batch_coord)),
                    b_multicast_mask,
                )

                # Load SFA and SFB (no multicast, 5D addressing)
                sfa_tma_op.async_copy_5d[Self.cta_group](
                    sfa_tile,
                    barrier[0],
                    (
                        0,
                        0,
                        Int((iter_idx + j) * Self.config.num_sf_k_tiles),
                        Int(work_tile_coord[0]) * (Self.BM // SF_MN_GROUP_SIZE),
                        Int(batch_coord),
                    ),
                )
                sfb_tma_op.async_copy_5d[Self.cta_group](
                    sfb_tile,
                    barrier[0],
                    (
                        0,
                        0,
                        Int((iter_idx + j) * Self.config.num_sf_k_tiles),
                        Int(work_tile_coord[1])
                        * (Self.MMA_N // SF_MN_GROUP_SIZE),
                        Int(batch_coord),
                    ),
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
        tmem_addr: UInt32,
        sfa_tmem: UInt32,
        sfb_tmem: UInt32,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations using InputConsumerStage.

        This method uses the structured ConsumerStage pattern from
        matmul_kernels.mojo, with tiles and barrier encapsulated in the stage.

        Args:
            tiles: ConsumerStage context with encapsulated tile access.
            mma_op: Block-scaled MMA operation instance.
            tmem_addr: TMEM address for accumulators.
            sfa_tmem: TMEM base address for A scaling factors.
            sfb_tmem: TMEM base address for B scaling factors.
            iter_idx: K iteration index.
            k_start: Starting K iteration (for init_c determination).
        """
        if elect_one_sync():
            for jj in range(Int(Self.config.k_group_size)):
                var j = UInt32(jj)

                # Get tiles at this pipeline stage using the payload accessor
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Int(Self.config.k_group_size)](
                        tiles.stage(), jj
                    )
                )

                # Calculate tile index for TMEM offset calculation
                var tile_idx = (
                    Int(tiles.stage()) * Int(Self.config.k_group_size) + jj
                )

                # Calculate TMEM offsets for scaling factors
                var sfa_tmem_offset = sfa_tmem + UInt32(tile_idx) * UInt32(
                    Self.SFA_NUM_COLS
                )
                var sfb_tmem_offset = sfb_tmem + UInt32(tile_idx) * UInt32(
                    Self.SFB_NUM_COLS
                )

                var is_first_k = (iter_idx + j) == k_start

                mma_op.mma(
                    a_tile,
                    b_tile,
                    sfa_tile,
                    sfb_tile,
                    tmem_addr,
                    sfa_tmem_offset,
                    sfb_tmem_offset,
                    init_c=is_first_k,
                )

            mma_op.commit(tiles.mbar())

    # ========== Epilogue Entry Point ==========

    @staticmethod
    @always_inline
    fn epilogue(
        c_tiles: Self.SmemType.CTileArray,
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        mma_output_pipeline: ProducerConsumerPipeline[
            Self.config.num_accum_pipeline_stages
        ],
        tmem_addr: UInt32,
        work_tile_coord: Tuple[UInt32, UInt32, UInt32],
        elect_one_warp: Bool,
        M: UInt32,
        N: UInt32,
    ):
        """Execute epilogue to store accumulated results to global memory.

        Uses BlockScaledTileWriter which encapsulates:
        - TmemArrayType.load_fragments() for TMEM load
        - AccumBarrier.arrive() for barrier signaling
        - TMEMToSMemWriter.write_fragments() for SMEM write
        - 3D TMA store (M, N, Batch coordinates)
        - tma_wait_pipelined() for TMA wait

        Args:
            c_tiles: SMEM tile array for C output.
            c_tma_op: TMA descriptor for C matrix.
            mma_output_pipeline: Pipeline for MMA→epilogue sync.
            tmem_addr: Base TMEM address for accumulators.
            work_tile_coord: (m, n, k_start) coordinates.
            elect_one_warp: Whether this warp should execute (unused).
            M: Problem M dimension.
            N: Problem N dimension.
        """
        # Wait for MMA to finish and get the current stage
        var mma_output_stage = mma_output_pipeline.consumer_stage()
        mma_output_pipeline.wait_producer()

        # Compute TMEM offset for this stage
        var tmem_offset = (
            mma_output_stage * UInt32(Self.stage_stride_cols) + tmem_addr
        )

        # Create OutputStage from raw parts (unified abstraction with TileWriter)
        var output_stage = Self.TileWriterType.Stage.from_raw(
            mma_output_pipeline, mma_output_stage, tmem_offset
        )

        # Use BlockScaledTileWriter for structured epilogue
        var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))
        tile_writer.write(
            c_tiles,
            output_stage,
            work_tile_coord,
            (M, N),
        )

    # ========== Compile-Time Validation ==========

    @staticmethod
    fn validate_config():
        """Validate configuration constraints at compile time."""
        constrained[Self.transpose_b, "Only support transposed B"]()
        constrained[
            Self.sfa_dtype == Self.sfb_dtype,
            "sfa_dtype and sfb_dtype must match",
        ]()
        constrained[
            Self.cta_group in (1, 2),
            "Only support cta_group == 1 or 2",
        ]()
        constrained[
            Self.config.k_group_size == 1,
            "Only support k_group_size == 1 for block-scaled",
        ]()

    # ========== Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_op, `nvvm.grid_constant`)
    fn run(
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        sfa_tma_op: TMATensorTile[
            Self.sfa_dtype, Self.sfa_layout, Self.sfa_desc_layout
        ],
        sfb_tma_op: TMATensorTile[
            Self.sfb_dtype, Self.sfb_layout, Self.sfb_desc_layout
        ],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Kernel entry point - ported from legacy kernel."""
        Self.validate_config()

        # ===== Shared Memory Setup (structured pattern with typed accessors) =====
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Get typed tile arrays from SMEM accessors
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var sfa_tiles = smem.sfa_tiles()
        var sfb_tiles = smem.sfb_tiles()

        # Get typed barrier arrays from SMEM accessors
        var input_barriers = smem.tma_mma_mbars()
        var accum_barriers = smem.accum_mbars()
        var clc_full = smem.clc_mbars_full()
        var clc_empty = smem.clc_mbars_empty()
        var clc_throttle = smem.clc_throttle_mbars()
        var clc_response_arr = smem.clc_response()
        var tmem_addr_arr = smem.tmem_addr()

        # Extract pointer for TMEM address storage
        var tmem_addr_storage = tmem_addr_arr.ptr

        # Create pipelines
        # input_pipeline uses the structured context manager pattern with payload
        var tile_payload = Self.TilePayload(
            a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )
        var input_pipeline = Self.InputTilePipeline(
            input_barriers, tile_payload
        )

        # ===== Kernel Context =====
        # Encapsulates election variables, CTA coordinates, and multicast masks
        var ctx = Self.Context(tmem_addr_storage)

        # ===== Barrier Initialization =====
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()
            sfa_tma_op.prefetch_descriptor()
            sfb_tma_op.prefetch_descriptor()

            # Initialize input pipeline barriers
            Self.InputTilePipeline.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Self.config.cluster_shape[0] // Self.cta_group
                + Self.config.cluster_shape[1]
                - 1,
            )
            # Initialize output pipeline barriers (using static method)
            Self.OutputPipeline.init_barriers(
                accum_barriers.ptr,
                Self.accum_pipeline_producer_arv_count,
                Self.accum_pipeline_consumer_arv_count,
            )
            # Initialize throttle barriers via scheduler
            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Self.clc_throttle_producer_arv_count,
                Self.clc_throttle_consumer_arv_count,
            )

            # Initialize TMEM deallocation barrier
            smem.tmem_dealloc().ptr[].init(
                Self.EPILOGUE_THREADS * Self.cta_group
            )

            # Initialize CLC barriers
            @parameter
            for i in range(Self.num_clc_pipeline_stages):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Self.clc_consumer_arv_count)

        fence_mbarrier_init()
        cluster_sync()

        var mma_op = Self.MmaOp()

        # Create structured scheduler with typed barrier arrays
        var scheduler = Self.Scheduler(
            cluster_dim, clc_response_arr, clc_full, clc_empty, clc_throttle
        )

        # Per-warp work iterator - owns work_info, pipeline state, and throttle
        var work_iter = scheduler.work_iterator()

        # CTA coordinates and multicast masks come from context
        # ctx.rank_m, ctx.rank_n, ctx.peer_cta_coord
        # ctx.a_multicast_mask, ctx.b_multicast_mask, ctx.mma_complete_mask

        var num_iters: UInt32 = ceildiv(mnk[2], Self.BK)
        var tmem_addr: UInt32 = 0

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():
            with MatmulProfilerType[0](workspace, 0):

                @parameter
                if Self.pdl_level > PDLLevel.OFF:
                    wait_on_dependent_grids()

                while work_iter.has_work():
                    with work_iter.next() as current:
                        # CLC throttle prevents each CTA from going ahead
                        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                        # DO TMA LOAD for full K range
                        with input_pipeline.producer() as producer:
                            for i in range(
                                num_iters // UInt32(Self.config.k_group_size)
                            ):
                                with producer.acquire() as stage:
                                    Self.load_input_tiles(
                                        a_tma_op,
                                        b_tma_op,
                                        sfa_tma_op,
                                        sfb_tma_op,
                                        stage,
                                        ctx.peer_cta_coord,
                                        (
                                            UInt(current.m),
                                            UInt(current.n),
                                            UInt(current.k_start),
                                        ),
                                        ctx.a_multicast_mask,
                                        ctx.b_multicast_mask,
                                        i * UInt32(Self.config.k_group_size),
                                        ctx.elect_one_cta,
                                    )

                        # Ensure all TMA loads complete before advancing work
                        syncwarp()

                # Drain pipeline to prevent CTA exit while MMA is still consuming
                @parameter
                for i in range(Self.num_group_pipeline_stages):
                    input_pipeline.pipeline.wait_consumer()
                    input_pipeline.pipeline.producer_step()

        # ===== SCHEDULER WARP =====
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            # Implies each SM will only process initial work, there is no
            # more work to schedule.
            @parameter
            if Self.num_clc_pipeline_stages == 0:
                return

            # Scheduler warp uses its own iterator that manages both
            # producer and consumer state, plus throttle signaling
            var sched_iter = scheduler.scheduler_iterator()

            with MatmulProfilerType[1](workspace, 0):

                @parameter
                if Self.pdl_level > PDLLevel.OFF:
                    wait_on_dependent_grids()

                while sched_iter.has_work():
                    with sched_iter.next():
                        sched_iter.signal_and_advance()

                # Drain all pending CLC requests before kernel exit
                sched_iter.drain()

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            with MatmulProfilerType[2](workspace, 0):
                # Use structured TMEM allocation and warp context
                var tmem = Self.Tmem.allocate(smem.tmem_addr())
                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        accum_barriers, tmem, ctx.mma_complete_mask
                    ),
                    Self.TmemDealloc(smem.tmem_dealloc()),
                )

                # Compute SF TMEM offsets (block-scaled specific)
                var sfa_tmem = tmem.addr + UInt32(
                    Self.num_accum_pipeline_stages * Self.MMA_N
                )
                var sfb_tmem = sfa_tmem + UInt32(Self.SFA_NUM_COLS) * UInt32(
                    Self.num_pipeline_stages
                )

                with mma_ctx:
                    while work_iter.has_work():
                        # Prefetch next work BEFORE doing MMA (software pipelining)
                        with work_iter.next_prefetch():
                            if ctx.elect_one_cta:
                                with mma_ctx.output_pipeline.producer() as out_stage:
                                    # Get TMEM offset for this output stage
                                    var tmem_offset = UInt32(
                                        out_stage.tmem.offset()
                                    )

                                    with input_pipeline.consumer() as consumer:
                                        for i in range(
                                            num_iters
                                            // UInt32(Self.config.k_group_size)
                                        ):
                                            with consumer.acquire() as stage:
                                                Self.mma(
                                                    stage,
                                                    mma_op,
                                                    tmem_offset,
                                                    sfa_tmem,
                                                    sfb_tmem,
                                                    i
                                                    * UInt32(
                                                        Self.config.k_group_size
                                                    ),
                                                    0,
                                                )

                    @parameter
                    if Self.pdl_level > PDLLevel.OFF:
                        launch_dependent_grids()

        # ===== EPILOGUE WARPS =====
        if WarpRole.is_epilogue():
            # Wait for MMA to allocate TMEM before reading address
            Self.MmaEpilogueSync.wait()

            # Get TMEM address from shared memory
            var tmem = Self.Tmem.from_shared(smem.tmem_addr())

            # Create epilogue context with OutputPipeline
            # EpilogueCtx manages TMEM lifecycle and dealloc barrier signaling
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, ctx.mma_complete_mask
                ),
                Self.TmemDealloc(smem.tmem_dealloc()),
            )

            with epi_ctx:
                var tile_idx = 0

                while work_iter.has_work():
                    with work_iter.next() as current:
                        with MatmulProfilerType[3](workspace, tile_idx):
                            Self.epilogue(
                                c_tiles,
                                c_tma_op,
                                epi_ctx.output_pipeline.pipeline,
                                tmem.addr,
                                work_tile_coord=(
                                    current.m,
                                    current.n,
                                    current.k_start,
                                ),
                                elect_one_warp=ctx.elect_one_warp,
                                M=mnk[0],
                                N=mnk[1],
                            )
                            epi_ctx.output_pipeline.pipeline.consumer_step()

                    tile_idx += 1
