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

"""Block-scaled SM100 matmul kernel - Structured kernel using tile pipelines.

Uses patterns from matmul_kernels.mojo with typed SMEM accessors and
context manager-based pipeline synchronization for MXFP8 and NVFP4
block-scaled matrix multiplication.

Architecture:
- Uses Self.SmemType (BlockScaledSmem) with typed tile/barrier accessors
- Uses Self.InputTilePipeline (BlockScaledTilePipeline) for producer/consumer sync
- Load warp: with input_pipeline.producer() as stage -> Self.load_input_tiles()
- MMA warp: with input_pipeline.consumer() as stage -> Self.mma()
- Epilogue warp: Uses structured building blocks from epilogue_components.mojo

Epilogue Building Blocks (from epilogue_components.mojo):
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

from collections import Optional
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
from ..structured_kernels.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.profiler import MatmulProfileWarp

# Structured kernel imports
from ..structured_kernels.kernel_common import WarpRole, KernelContext
from .block_scaled_smem import BlockScaledSmem
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    InputProducerStage,
    InputConsumerStage,
    BlockScaledTilePayload,
)
from layout._layout import RowMajorLayout, _IntToComptimeInt
from ..structured_kernels.tile_types import (
    TMATile,
    TmaOpType,
    internal_k_major_128B,
    tma_desc_layout_3d,
    tma_desc_layout_5d,
)
from ..structured_kernels.tile_scheduler import (
    TileScheduler as StructuredTileScheduler,
)
from ..structured_kernels.tmem import TmemAllocation
from ..structured_kernels.tile_pipeline import OutputTilePipeline
from ..structured_kernels.barriers import TmemDeallocBarrier, WarpGroupBarrier
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)

# Block-scaled output writer for epilogue
from ..structured_kernels.output_writer import TileWriter


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
    # Configuration
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    # Cluster shape (for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Optional features
    elementwise_compute_lambda_fn: Optional[
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

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
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
    comptime num_output_stages: Int = Self.config.num_output_stages

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
    ) * Self.config.k_group_size

    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime a_tile_dim1 = Self.BM // Self.CLUSTER_N
    comptime b_tile_dim1 = Self.BN // (Self.CLUSTER_M // Self.cta_group)
    comptime a_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.a_type
    ]()
    comptime b_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.b_type
    ]()
    comptime c_swizzle_elems = Self.config.c_swizzle.bytes() // size_of[
        Self.c_type
    ]()

    # C tile shape depends on MMA shape, cta_group, and AB_swapped
    comptime c_tile_dim1 = (
        Self.OutputM if (Self.MMA_M == 256 or Self.cta_group == 1) else (
            Self.OutputM if Self.config.AB_swapped else 64
        )
    )

    # 3D tile layout types (batch=1, rows, cols)
    comptime ATileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.a_tile_dim1, Self.BK]
    ]
    comptime ADescLayout = tma_desc_layout_3d[
        Self.a_type, 1, Self.a_tile_dim1, Self.config.a_swizzle
    ]
    comptime BTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.b_tile_dim1, Self.BK]
    ]
    comptime BDescLayout = tma_desc_layout_3d[
        Self.b_type, 1, Self.b_tile_dim1, Self.config.b_swizzle
    ]
    # C tile shape: when AB_swapped, last dim is swizzle elems
    comptime c_tile_dim2 = Self.OutputN if not Self.config.AB_swapped else Self.c_swizzle_elems
    comptime CTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.c_tile_dim1, Self.c_tile_dim2]
    ]
    comptime CDescLayout = tma_desc_layout_3d[
        Self.c_type, 1, Self.c_tile_dim1, Self.config.c_swizzle
    ]

    # 5D scale factor layout types (SWIZZLE_NONE)
    comptime SFATileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            1,
            Self.BM // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFADescLayout = tma_desc_layout_5d[
        Self.sfa_dtype,
        1,
        Self.BM // SF_MN_GROUP_SIZE,
        Self.config.num_sf_k_tiles,
        SF_ATOM_M[0],
        TensorMapSwizzle.SWIZZLE_NONE,
    ]
    comptime SFBTileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            1,
            Self.MMA_N // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFBDescLayout = tma_desc_layout_5d[
        Self.sfb_dtype,
        1,
        Self.MMA_N // SF_MN_GROUP_SIZE,
        Self.config.num_sf_k_tiles,
        SF_ATOM_M[0],
        TensorMapSwizzle.SWIZZLE_NONE,
    ]

    # TMA operation types
    comptime ATmaTile = TMATile[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime ATmaOp = Self.ATmaTile.InnerType
    comptime BTmaTile = TMATile[Self.b_type, Self.BTileLayout, Self.BDescLayout]
    comptime BTmaOp = Self.BTmaTile.InnerType
    comptime CTmaTile = TMATile[Self.c_type, Self.CTileLayout, Self.CDescLayout]
    comptime CTmaOp = Self.CTmaTile.InnerType
    comptime SFATmaTile = TMATile[
        Self.sfa_dtype, Self.SFATileLayout, Self.SFADescLayout
    ]
    comptime SFATmaOp = Self.SFATmaTile.InnerType
    comptime SFBTmaTile = TMATile[
        Self.sfb_dtype, Self.SFBTileLayout, Self.SFBDescLayout
    ]
    comptime SFBTmaOp = Self.SFBTmaTile.InnerType

    # TMA load size constants (from desc layout dimensions)
    comptime a_tma_load_size = Self.a_tile_dim1 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim1 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim1
    comptime b_tma_rows = Self.b_tile_dim1

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
    # TileTensor-native payload - passed directly to TMA/MMA
    comptime TilePayload = BlockScaledTilePayload[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        # A tile dimensions (BM x BK)
        Self.SmemType.BM,
        Self.SmemType.BK,
        # B tile dimensions (BN x BK)
        Self.SmemType.BN,
        Self.SmemType.BK,
        # SFA tile dimensions
        Self.SmemType.SFA_DIM0,
        Self.SmemType.SFA_DIM1,
        # SFB tile dimensions
        Self.SmemType.SFB_DIM0,
        Self.SmemType.SFB_DIM1,
        Self.SmemType.num_pipeline_stages,
    ]
    comptime InputTilePipeline = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== Internal Swizzled Layouts ==========
    # These layouts encode the swizzle pattern used by TileTensor storage.
    # MMA extracts layout from TileTensor type parameters via coord_to_int_tuple.
    comptime a_internal_layout = internal_k_major_128B[
        Self.a_type, Self.BM, Self.BK
    ]
    comptime b_internal_layout = internal_k_major_128B[
        Self.b_type, Self.BN, Self.BK
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
    comptime TileWriterType = TileWriter[
        a_type = Self.a_type,
        accum_type = Self.accum_type,
        block_tile_shape = Self.config.block_tile_shape,
        mma_shape = Self.config.mma_shape,
        cta_group = Self.cta_group,
        num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages,
        c_swizzle = Self.config.c_swizzle,
        transpose_c = Self.config.AB_swapped,
        c_smem_dim0 = Self.SmemType.OutputM,
        c_smem_dim1 = Self.SmemType.OutputN,
        num_output_stages = Self.config.num_output_stages,
        stage_stride_cols = Self.stage_stride_cols,
        num_output_warps = Self.num_output_warps,
        batched=True,  # Block-scaled uses 3D batched coordinates
    ]

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    fn load_input_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        tiles: InputProducerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
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

            for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Self.config.k_group_size](
                        tiles.stage(), jj
                    )
                )

                # Peer CTA slice using TileTensor pattern (ptr + layout)
                var a_peer_tile = type_of(a_tile)(
                    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
                    a_tile.layout,
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile.ptr + peer_rank_m * UInt(Self.b_tma_load_size),
                    b_tile.layout,
                )

                var k_coord = UInt(iter_idx + j) * UInt(Self.BK)

                # Load A and B with multicast - TileTensor directly to TMA
                a_tma_op.async_multicast_load_3d[Self.cta_group](
                    a_peer_tile,
                    barrier[0],
                    (k_coord, a_gmem_m_coord, batch_coord),
                    a_multicast_mask,
                )
                b_tma_op.async_multicast_load_3d[Self.cta_group](
                    b_peer_tile,
                    barrier[0],
                    (k_coord, b_gmem_n_coord, batch_coord),
                    b_multicast_mask,
                )

                # Load SFA and SFB (no multicast, 5D addressing)
                # TMA 5D now has TileTensor overload - pass tiles directly
                sfa_tma_op.async_copy_5d[Self.cta_group](
                    sfa_tile,
                    barrier[0],
                    (
                        0,
                        0,
                        Int(
                            (iter_idx + j) * UInt32(Self.config.num_sf_k_tiles)
                        ),
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
                        Int(
                            (iter_idx + j) * UInt32(Self.config.num_sf_k_tiles)
                        ),
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
            Self.config.k_group_size,
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
            for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Self.config.k_group_size](
                        tiles.stage(), jj
                    )
                )

                # Calculate tile index for TMEM offset calculation
                var tile_idx = (
                    Int(tiles.stage()) * Self.config.k_group_size + jj
                )

                # Calculate TMEM offsets for scaling factors
                var sfa_tmem_offset = sfa_tmem + UInt32(tile_idx) * UInt32(
                    Self.SFA_NUM_COLS
                )
                var sfb_tmem_offset = sfb_tmem + UInt32(tile_idx) * UInt32(
                    Self.SFB_NUM_COLS
                )

                var is_first_k = (iter_idx + j) == k_start

                # Pass TileTensor directly to MMA - layout encoded in type
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
        c_tma_op: Self.CTmaOp,
        stage: Self.TileWriterType.Stage,
        work_tile_coord: Tuple[UInt32, UInt32, UInt32],
        M: UInt32,
        N: UInt32,
        alpha: Float32,
    ):
        """Execute epilogue to store accumulated results to global memory.

        Uses TileWriter which encapsulates:
        - TmemArrayType.load_fragments() for TMEM load
        - AccumBarrier.arrive() for barrier signaling
        - TMEMToSMemWriter.write_fragments() for SMEM write
        - 3D TMA store (M, N, Batch coordinates)
        - tma_wait_pipelined() for TMA wait

        Barrier synchronization (wait/step) is handled by caller via consumer() context.

        Args:
            c_tiles: SMEM tile array for C output.
            c_tma_op: TMA descriptor for C matrix.
            stage: OutputStage from consumer() context with pipeline, index, and TMEM.
            work_tile_coord: (m, n, k_start) coordinates.
            M: Problem M dimension.
            N: Problem N dimension.
            alpha: Tensor scale factor (scalar).
        """
        # Use TileWriter for structured epilogue
        var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))
        tile_writer.write_batched(
            c_tiles,
            stage,
            work_tile_coord,
            (M, N),
            alpha,
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
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c_tma_op: Self.CTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        alpha: Float32,
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
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var clc_full = smem.pipelines.clc_full()
        var clc_empty = smem.pipelines.clc_empty()
        var clc_throttle = smem.pipelines.clc_throttle()
        var clc_response_arr = smem.pipelines.clc_response()
        var tmem_addr_arr = smem.pipelines.tmem_addr()

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
                Int32(
                    Self.config.cluster_shape[0] // Self.cta_group
                    + Self.config.cluster_shape[1]
                    - 1
                ),
            )
            # Initialize output pipeline barriers (using static method)
            Self.OutputPipeline.init_barriers(
                accum_barriers.ptr,
                Self.accum_pipeline_producer_arv_count,
                Int32(Self.accum_pipeline_consumer_arv_count),
            )
            # Initialize throttle barriers via scheduler
            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Int32(Self.clc_throttle_producer_arv_count),
                Int32(Self.clc_throttle_consumer_arv_count),
            )

            # Initialize TMEM deallocation barrier
            smem.pipelines.tmem_dealloc().ptr[].init(
                Int32(Self.EPILOGUE_THREADS * Self.cta_group)
            )

            # Initialize CLC barriers
            @parameter
            for i in range(Self.num_clc_pipeline_stages):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Int32(Self.clc_consumer_arv_count))

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

        var num_iters: UInt32 = ceildiv(mnk[2], UInt32(Self.BK))
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

                with input_pipeline.producer() as producer:
                    while work_iter.has_work():
                        with work_iter.next() as current:
                            work_iter.throttle_signal(
                                ctx.is_first_cta_in_cluster
                            )

                            for i in range(
                                num_iters // UInt32(Self.config.k_group_size)
                            ):
                                with producer.acquire() as tiles:  # waits for consumer
                                    Self.load_input_tiles(
                                        a_tma_op,
                                        b_tma_op,
                                        sfa_tma_op,
                                        sfb_tma_op,
                                        tiles,
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

                            syncwarp()

                    producer.drain()  # wait for consumer before CTA exits

        # ===== SCHEDULER WARP =====
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:

            @parameter
            if Self.num_clc_pipeline_stages == 0:
                return

            var sched_iter = scheduler.scheduler_iterator()

            with MatmulProfilerType[1](workspace, 0):

                @parameter
                if Self.pdl_level > PDLLevel.OFF:
                    wait_on_dependent_grids()

                while sched_iter.has_work():
                    with sched_iter.next():
                        sched_iter.signal_and_advance()

                sched_iter.drain()

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            with MatmulProfilerType[2](workspace, 0):
                var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                var sfa_tmem = tmem.addr + UInt32(
                    Self.num_accum_pipeline_stages * Self.MMA_N
                )
                var sfb_tmem = sfa_tmem + UInt32(Self.SFA_NUM_COLS) * UInt32(
                    Self.num_pipeline_stages
                )

                with mma_ctx:  # TMEM lifecycle
                    while work_iter.has_work():
                        with work_iter.wait_and_advance():  # blocks on CLC
                            if ctx.elect_one_cta:
                                with mma_ctx.output_pipeline.producer() as output_stage:  # waits for epilogue
                                    var tmem_offset = UInt32(
                                        output_stage.tmem.offset()
                                    )

                                    with input_pipeline.consumer() as consumer:
                                        for i in range(
                                            num_iters
                                            // UInt32(Self.config.k_group_size)
                                        ):
                                            with consumer.acquire() as input_tiles:  # waits for TMA
                                                Self.mma(
                                                    input_tiles,
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
            Self.MmaEpilogueSync.wait()  # wait for MMA to publish TMEM addr

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with epi_ctx:  # signals TMEM dealloc on exit
                var tile_idx = 0

                while work_iter.has_work():
                    with work_iter.next() as current:
                        with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                            with epi_ctx.output_pipeline.consumer() as output_stage:  # waits for MMA
                                Self.epilogue(
                                    c_tiles,
                                    c_tma_op,
                                    output_stage,
                                    work_tile_coord=(
                                        current.m,
                                        current.n,
                                        current.k_start,
                                    ),
                                    M=mnk[0],
                                    N=mnk[1],
                                    alpha=alpha,
                                )

                    tile_idx += 1
