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

"""SM100 Default Matmul Kernel - Standard FP8/BF16 warp-specialized kernel.

This module contains the default SM100 matmul kernel implementation:
- B200MatmulSmem: Shared memory layout for the kernel
- BlackwellMatmulSM100Kernel: Main kernel struct with run() and run_splitk()
- BlackwellMatmulSM100FallbackKernel: Simple fallback kernel

Shared components (WarpRole, KernelContext) are in kernel_common.mojo.
Output pipeline (TileWriter, copy_accum_to_gmem) is in output_writer.mojo.
Low-level epilogue components (TMAStoreExecutor, etc.) are in epilogue_components.mojo.

The kernel implements a warp-specialized architecture:
- Scheduler warp: CLC-based tile scheduling
- TMA Load warp: Async memory transfers
- MMA warp: Tensor core operations with TMEM accumulators
- Epilogue warps: Output from TMEM to GMEM via TileWriter
"""

from math import ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import align_of, size_of

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
from gpu.sync import syncwarp
from gpu.compute.arch.tcgen05 import *
from layout import Layout, LayoutTensor, RuntimeLayout
from layout.int_tuple import IntTuple
from layout.swizzle import Swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import SharedMemBarrier, TMATensorTile

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from ..structured_kernels.config import MatmulConfig
from ..structured_kernels.pipeline import ProducerConsumerPipeline
from ..structured_kernels.tile_pipeline import (
    TilePipeline,
    InputTilePipeline,
    StandardTilePayload,
    InputProducerStage,
    InputConsumerStage,
    InputProducer,
    InputConsumer,
    StandardProducerStage,
    StandardConsumerStage,
    OutputTilePipeline,
)
from ..structured_kernels.barriers import TmemDeallocBarrier, WarpGroupBarrier
from ..structured_kernels.pipeline_storage import (
    InputPipelineStorage,
    OutputPipelineStorage,
    ClcPipelineStorage,
    TmemDeallocStorage,
    StandardTileStorage,
    OutputTileStorage,
    SmemPipelineBundle,
    SmemLayouts,
)
from ..structured_kernels.tmem import TmemAllocation, TmemTensor
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)
from ..structured_kernels.tile_loader import TileLoaderTMA
from ..structured_kernels.tile_scheduler import TileScheduler
from ..structured_kernels.tile_scheduler_splitk import (
    TileScheduler as TileSchedulerSplitK,
)
from ..structured_kernels.epilogue_components import EpilogueConfig
from linalg.structuring import (
    SMemPtr,
    SMemTile,
    SMemTileIter,
    SMemTileArray,
    SMemArray,
)
from linalg.matmul.gpu.profiler import MatmulProfileWarp

# Import shared kernel components from kernel_common
from ..structured_kernels.kernel_common import (
    WarpRole,
    KernelContext,
    consumer_main_loop,
)

# Import output pipeline from output_writer module
from ..structured_kernels.output_writer import TileWriter


# =============================================================================
# B200MatmulSmem - Shared memory layout for SM100 matmul
# =============================================================================


struct B200MatmulSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """Shared memory layout for B200 SM100 matrix multiplication kernel.

    This struct manages the shared memory allocation for:
    - Input tiles (A and B matrices) with multi-stage pipelining
    - Output tile (C matrix) for accumulation
    - Synchronization barriers for producer-consumer coordination
    - CLC (Cluster Launch Control) barriers and response storage
    - TMEM (Tensor Memory) address and deallocation barrier

    The memory is organized to support asynchronous TMA loads and efficient
    bank-conflict-free access patterns for tensor core operations.

    Type aliases are provided for tile types (ATile, BTile, CTile) to enable
    cleaner function signatures without verbose LayoutTensor declarations.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    # Pipeline stage counts
    comptime num_pipeline_stages: Int = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages: Int = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_output_stages: Int = Self.config.num_output_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages

    # ========== Layout Definitions ==========
    comptime Layouts = SmemLayouts[
        Self.a_type,
        Self.b_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.OutputM,
        Self.OutputN,
        Self.config.a_swizzle,
        Self.config.b_swizzle,
        Self.transpose_b,
    ]
    comptime a_smem_layout = Self.Layouts.a_smem_layout
    comptime b_smem_layout = Self.Layouts.b_smem_layout
    comptime c_smem_layout = Self.Layouts.c_smem_layout

    # ========== Tile Storage (Single Source of Truth) ==========
    # Input tiles: A and B matrices
    # Tiles are stored with row_major layout and converted to swizzled layouts
    # at TMA/MMA boundaries using LayoutTensor construction.
    comptime InputTiles = StandardTileStorage[
        Self.a_type,
        Self.b_type,
        # A tile dimensions (BM x BK)
        Self.BM,
        Self.BK,
        # B tile dimensions (BN x BK)
        Self.BN,
        Self.BK,
        Self.num_pipeline_stages,
    ]
    # Output tiles: C matrix (different stage count)
    comptime OutputTiles = OutputTileStorage[
        Self.c_type,
        Self.OutputM,
        Self.OutputN,
        Self.num_output_stages,
    ]

    # Re-export tile array types for external use
    # Re-export tile array types (all TileTensor-based now)
    comptime ATileArray = Self.InputTiles.ATileArray
    comptime BTileArray = Self.InputTiles.BTileArray
    comptime CTileArray = Self.OutputTiles.CTileArray  # TileTensor-based

    # ========== Tile Storage Fields ==========
    var input_tiles: Self.InputTiles
    var output_tiles: Self.OutputTiles

    # ========== Tile Accessors (Delegated) ==========
    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        return self.input_tiles.a_tiles()

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        return self.input_tiles.b_tiles()

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        return self.output_tiles.c_tiles_tt()

    # ========== Pipeline Storage (Composed Bundle) ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.num_group_pipeline_stages,
        Self.num_accum_pipeline_stages,
        Self.num_clc_pipeline_stages,
        StandardTilePayload[
            Self.a_type,
            Self.b_type,
            # A tile dimensions (BM x BK)
            Self.BM,
            Self.BK,
            # B tile dimensions (BN x BK)
            Self.BN,
            Self.BK,
            Self.num_pipeline_stages,
        ],
    ]
    var pipelines: Self.Pipelines

    # ========== Size Calculations ==========

    @staticmethod
    @always_inline
    fn ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.ATileArray.num_elements + Self.BTileArray.num_elements

    @staticmethod
    @always_inline
    fn c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.CTileArray.num_elements

    @staticmethod
    @always_inline
    fn total_tile_size() -> Int:
        """Total tile storage size (A+B+C) in elements."""
        return Self.ab_pipeline_size() + Self.c_output_size()


# ===----------------------------------------------------------------------=== #
# BlackwellMatmulSM100Kernel - Structured kernel for SM100 matrix multiplication
# ===----------------------------------------------------------------------=== #


struct BlackwellMatmulSM100Kernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    # Tensor layouts (from TMA descriptors)
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    c_desc_layout: Layout,
    # Configuration
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    # Cluster shape (must match config, needed for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Optional features
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
]:
    """Blackwell SM100 GEMM kernel with warp specialization.

    This struct unifies all parameters and derived types for the SM100
    matmul kernel, providing:
    - Compile-time parameter validation
    - Centralized derived type computation
    - Factory methods for kernel components
    - Multiple kernel entry points (standard, split-k)

    The SM100 kernel uses:
    - Tensor Memory (TMEM) for MMA accumulators
    - Cluster Launch Control (CLC) for dynamic tile scheduling
    - Warp specialization: Scheduler, TMA Load, MMA, Epilogue warps
    - Software pipelining for overlapping compute and memory operations
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

    comptime accum_type = Self.config.accum_type
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # MMA tile counts
    comptime num_m_mmas = Self.BM // (Self.MMA_M // Self.cta_group)
    comptime num_n_mmas = Self.BN // (Self.MMA_N // Self.cta_group)
    comptime num_k_mmas = Self.BK // Self.MMA_K

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    comptime SCHEDULER_THREADS = WARP_SIZE
    comptime TMA_LOAD_THREADS = WARP_SIZE
    comptime MMA_THREADS = WARP_SIZE
    comptime EPILOGUE_THREADS = Self.num_output_warps * WARP_SIZE

    # Total threads per block
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
    comptime stage_stride_cols = Self.NUM_TMEM_COLS // Self.num_accum_pipeline_stages

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

    comptime SmemType = B200MatmulSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.transpose_b,
        config = Self.config,
    ]

    # ========== MMA Operation Type ==========

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

    # ========== Tile Scheduler Type ==========

    comptime Scheduler = TileScheduler[
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
    # Uses generic TilePipeline with StandardTilePayload for composition

    comptime TilePayload = StandardTilePayload[
        Self.a_type,
        Self.b_type,
        # A tile dimensions (BM x BK)
        Self.BM,
        Self.BK,
        # B tile dimensions (BN x BK)
        Self.BN,
        Self.BK,
        Self.SmemType.num_pipeline_stages,
    ]
    comptime InputTilePipeline = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== LayoutTensor Types for Boundary Conversion ==========
    # Used for TMA/MMA ops - 128B alignment required for shared memory
    comptime ATileLT = LayoutTensor[
        Self.a_type,
        Self.SmemType.a_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]
    comptime BTileLT = LayoutTensor[
        Self.b_type,
        Self.SmemType.b_smem_layout,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    # ========== Tile Loader Types ==========
    # Loaders wrapping TMA operations. Orchestration is in kernel.
    # Origins inferred from constructor Pointer arguments.

    comptime ATileLoaderType = TileLoaderTMA[cta_group = Self.cta_group]
    comptime BTileLoaderType = TileLoaderTMA[cta_group = Self.cta_group]

    # Constants for TMA expected_bytes calculation
    comptime a_expected_bytes = Self.SmemType.a_smem_layout.size() * size_of[
        Self.a_type
    ]()
    comptime b_expected_bytes = Self.SmemType.b_smem_layout.size() * size_of[
        Self.b_type
    ]()
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes + Self.b_expected_bytes
    ) * Self.config.k_group_size

    # TMA descriptor layout sizes for peer CTA slicing
    comptime a_tma_load_size = Self.a_desc_layout.size()
    comptime b_tma_load_size = Self.b_desc_layout.size()
    comptime a_tma_rows = Self.a_desc_layout.shape[0].value()
    comptime b_tma_rows = Self.b_desc_layout.shape[0].value()

    # ========== Epilogue Configuration ==========
    # Note: stageN is typically c_smem_layout.shape[1] for non-transposed output

    comptime EpilogueConf = EpilogueConfig[
        Self.MMA_M,
        Self.MMA_N,
        Self.SmemType.c_smem_layout.shape[1].value(),  # stageN
        Self.cta_group,
        False,  # transpose_c (default)
    ]

    # ========== Tensor Memory Type ==========
    # TMEM allocation and typed accumulator tensor

    comptime Tmem = TmemAllocation[Self.cta_group]

    # Layout-parameterized TMEM tensor for type-safe accumulator access
    comptime accum_layout = Layout.row_major(Self.MMA_M, Self.MMA_N)
    comptime AccumTensor = TmemTensor[
        Self.accum_type, Self.accum_layout, cta_group = Self.cta_group
    ]

    # ========== Output Tile Pipeline Type ==========
    # Manages MMA→Epilogue pipeline for TMEM accumulator stages

    comptime OutputPipeline = OutputTilePipeline[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]

    # MMA-Epilogue handoff barrier (barrier_id=1)
    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.MMA_THREADS + Self.EPILOGUE_THREADS, 1
    ]

    # TMEM deallocation barrier for cluster synchronization
    comptime TmemDealloc = TmemDeallocBarrier[Self.cta_group]

    # MMA warp context (TMEM + dealloc + OutputPipeline)
    comptime MmaCtx = MmaWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # Epilogue warp context (works in run_splitk, issues in run with k_group>1)
    comptime EpilogueCtx = EpilogueWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    # ========== Output Tile Writer ==========
    # Instance-based TileWriter with explicit config parameters
    # tma_origin, c_type, c_layout, c_desc_layout inferred from constructor arg
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
        num_output_stages = Self.SmemType.num_output_stages,
        stage_stride_cols = Self.stage_stride_cols,
        num_output_warps = Self.num_output_warps,
        elementwise_compute_lambda_fn = Self.elementwise_compute_lambda_fn,
        register_based_epilogue = Self.register_based_epilogue,
    ]

    # ========== Kernel Context Type ==========
    # Type comptime for KernelContext with this kernel's parameters

    comptime Context = KernelContext[
        Self.num_clc_pipeline_stages,
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
    ]

    # ========== Compile-Time Validation ==========

    @staticmethod
    @always_inline
    fn validate_constraints():
        """Validate parameter constraints at compile time."""
        constrained[
            Self.c_type != DType.float32,
            "c_type cannot be float32",
        ]()
        constrained[
            Self.transpose_b,
            "Only support transposed B (K-major)",
        ]()
        constrained[
            Self.cta_group in (1, 2),
            "Only support cta_group == 1 or 2",
        ]()

        @parameter
        if Self.cta_group == 2:
            constrained[
                Self.MMA_M in (128, 256),
                "cta_group=2 requires MMA_M == 128 or 256",
            ]()
        else:
            constrained[
                Self.MMA_M in (64, 128),
                "cta_group=1 requires MMA_M == 64 or 128",
            ]()

    # ========== Static Helper Methods ==========

    @staticmethod
    @always_inline
    fn init_barriers(
        ctx: Self.Context,
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        clc_throttle: Self.SmemType.Pipelines.ClcThrottleBarriers,
        clc_full: Self.SmemType.Pipelines.ClcBarriers,
        clc_empty: Self.SmemType.Pipelines.ClcBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
    ):
        """Initialize barriers and prefetch TMA descriptors. Called by elect_one_warp && elect_one_thread.
        """
        if ctx.elect_one_warp and ctx.elect_one_thread:
            # Prefetch TMA descriptors
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()

            # Initialize pipeline barriers
            Self.InputTilePipeline.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Int32(
                    Self.config.cluster_shape[0] // Self.config.cta_group
                    + Self.config.cluster_shape[1]
                    - 1
                ),
            )
            Self.OutputPipeline.init_barriers(
                accum_barriers.ptr,
                Self.accum_pipeline_producer_arv_count,
                Int32(Self.accum_pipeline_consumer_arv_count),
            )
            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Int32(Self.clc_throttle_producer_arv_count),
                Int32(Self.clc_throttle_consumer_arv_count),
            )

            # Initialize TMEM deallocation barrier
            tmem_dealloc.ptr[].init(
                Int32(Self.EPILOGUE_THREADS * Self.config.cta_group)
            )

            # Initialize CLC barriers
            @parameter
            for i in range(Self.config.num_clc_pipeline_stages):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Int32(Self.clc_consumer_arv_count))

        fence_mbarrier_init()
        cluster_sync()

    @staticmethod
    @always_inline
    fn mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tmem_stage: Self.OutputPipeline.Stage.Tmem,
        tiles: InputConsumerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: MmaOpSM100_SS,
        elect_one_warp: Bool,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations for one pipeline stage.

        This is the core MMA function designed to be called within a consumer
        stage context:

            with consumer.acquire() as tiles:
                Self.mma(stage.tmem, tiles, mma_op, ...)

        Args:
            tmem_stage: TMEM stage for accumulators.
            tiles: InputConsumerStage context with encapsulated tile access.
            mma_op: The MMA operation instance.
            elect_one_warp: Whether this warp should execute.
            iter_idx: K iteration index.
            k_start: Starting K iteration (for init_c determination).
        """
        # Get typed accumulator tensor from TMEM stage
        var accum = tmem_stage.tensor[Self.accum_type, Self.accum_layout]()

        if elect_one_sync():

            @parameter
            for j in range(Self.config.k_group_size):
                # Get tiles using payload accessor - tiles have swizzled layout
                var a_tile, b_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)
                var is_first_k = (iter_idx + UInt32(j)) == k_start
                # Pass TileTensor directly to MMA - layout is encoded in type
                mma_op.mma(
                    a_tile,
                    b_tile,
                    UInt32(accum.offset()),
                    init_c=is_first_k,
                )
            mma_op.commit(tiles.mbar())

    @staticmethod
    @always_inline
    fn load_input_tiles[
        a_tma_origin: ImmutOrigin,
        b_tma_origin: ImmutOrigin,
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
        tiles: InputProducerStage[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        iter_idx: UInt32,
        work_m_coord: UInt,
        work_n_coord: UInt,
        peer_cta_coord: Tuple[UInt, UInt, UInt],
        elect_one_cta: Bool,
    ):
        """Load k_group_size A and B tiles using TMA.

        Orchestrates the tile loading operation including:
        - expect_bytes signaling
        - k-group iteration
        - Peer CTA slicing for 2-SM MMA

        Args:
            a_loader: TileLoaderTMA for A matrix.
            b_loader: TileLoaderTMA for B matrix.
            tiles: InputProducerStage context with encapsulated tile access.
            iter_idx: K iteration index (base index).
            work_m_coord: M coordinate of the output tile.
            work_n_coord: N coordinate of the output tile.
            peer_cta_coord: Peer CTA coordinates (rank_n, rank_m, peer_m_rank).
            elect_one_cta: True if this CTA should call expect_bytes.
        """
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # Global memory coordinates for A (M) and B (N)
        var a_gmem_m_coord = peer_m_rank * UInt(
            Self.a_tma_rows
        ) + work_m_coord * UInt(Self.BM)
        var b_gmem_n_coord = (
            peer_rank_m * UInt(Self.b_tma_rows)
            + peer_rank_n * UInt(Self.BN)
            + work_n_coord * UInt(Self.MMA_N)
        )

        if elect_one_sync():
            # Set expected bytes ONCE for all k_group tiles
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            # Get barrier for TMA multicast loads
            var barrier = tiles.barrier()

            @parameter
            for j in range(Self.config.k_group_size):
                # Get tiles using payload accessor
                var a_tile, b_tile = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), j)

                # Peer CTA slice using pointer arithmetic (not tile[]).
                # The tile[] method uses SMEM layout strides which can differ from
                # TMA descriptor layout. Pointer arithmetic with a_tma_load_size
                # preserves the original working behavior.
                var a_peer_tile = type_of(a_tile)(
                    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
                    a_tile.layout,
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile.ptr + peer_rank_m * UInt(Self.b_tma_load_size),
                    b_tile.layout,
                )

                var k_coord = UInt(iter_idx + UInt32(j)) * UInt(Self.BK)

                # TileTensor directly to loader (uses TileTensor TMA overload)
                a_loader.load(
                    a_peer_tile,
                    barrier[0],
                    k_coord,
                    a_gmem_m_coord,
                )
                b_loader.load(
                    b_peer_tile,
                    barrier[0],
                    k_coord,
                    b_gmem_n_coord,
                )

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run(
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Main kernel entry point for SM100 matrix multiplication."""
        Self.validate_constraints()

        # Access shared memory via bitcast
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Create input pipeline for TMA→MMA synchronization (with payload)
        var tile_payload = Self.TilePayload(smem.a_tiles(), smem.b_tiles())
        var input_pipeline = Self.InputTilePipeline(
            smem.pipelines.input_barriers(), tile_payload
        )

        # Create kernel context with election vars, CTA coords, and masks
        var ctx = Self.Context(smem.pipelines.tmem_addr())

        # Initialize all barriers (only elect_one_warp && elect_one_thread)
        Self.init_barriers(
            ctx,
            a_tma_op,
            b_tma_op,
            c_tma_op,
            smem.pipelines.input_barriers(),
            smem.pipelines.accum_barriers(),
            smem.pipelines.clc_throttle(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.tmem_dealloc(),
        )

        var mma_op = Self.MmaOp()

        # Scheduler owns CLC throttle pipeline internally
        var scheduler = Self.Scheduler(
            cluster_dim,
            smem.pipelines.clc_response(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.clc_throttle(),
        )

        # Per-warp work iterator - owns work_info, pipeline state, and throttle
        var work_iter = scheduler.work_iterator()

        # Create tile loaders for A and B matrices
        var a_loader = Self.ATileLoaderType(
            Pointer(to=a_tma_op), ctx.a_multicast_mask
        )
        var b_loader = Self.BTileLoaderType(
            Pointer(to=b_tma_op), ctx.b_multicast_mask
        )

        var num_iters: UInt32 = ceildiv(mnk[2], UInt32(Self.BK))

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

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
                                0, num_iters, Self.config.k_group_size
                            ):
                                with producer.acquire() as tiles:  # waits for consumer
                                    Self.load_input_tiles(
                                        a_loader,
                                        b_loader,
                                        tiles,
                                        UInt32(i),
                                        UInt(current.m),
                                        UInt(current.n),
                                        ctx.peer_cta_coord,
                                        ctx.elect_one_cta,
                                    )

                            syncwarp()

                    producer.drain()  # wait for consumer before CTA exits

        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            # Implies each SM will only process initial work, there is no
            # more work to schedule.
            @parameter
            if Self.config.num_clc_pipeline_stages == 0:
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

        if WarpRole.is_mma():
            with MatmulProfilerType[2](workspace, 0):
                var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        smem.pipelines.accum_barriers(),
                        tmem,
                        UInt16(ctx.mma_complete_mask),
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                with mma_ctx:  # TMEM lifecycle
                    while work_iter.has_work():
                        with work_iter.wait_and_advance():  # blocks on CLC
                            if ctx.elect_one_cta:
                                with mma_ctx.output_pipeline.producer() as output_stage:  # waits for epilogue
                                    with input_pipeline.consumer() as consumer:
                                        for i in range(
                                            0,
                                            num_iters,
                                            Self.config.k_group_size,
                                        ):
                                            with consumer.acquire() as input_tiles:  # waits for TMA
                                                Self.mma(
                                                    output_stage.tmem,
                                                    input_tiles,
                                                    mma_op,
                                                    ctx.elect_one_warp,
                                                    UInt32(i),
                                                    0,
                                                )

                    @parameter
                    if Self.pdl_level > PDLLevel.OFF:
                        launch_dependent_grids()

        if WarpRole.is_epilogue():
            Self.EpilogueCtx.Sync.wait()  # wait for MMA to publish TMEM addr

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    smem.pipelines.accum_barriers(),
                    tmem,
                    UInt16(ctx.mma_complete_mask),
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))

            with epi_ctx:  # signals TMEM dealloc on exit
                var tile_idx = 0

                while work_iter.has_work():
                    with work_iter.next() as current:
                        with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                            with epi_ctx.output_pipeline.consumer() as output_stage:  # waits for MMA
                                tile_writer.write(
                                    smem.c_tiles(),
                                    output_stage,
                                    (current.m, current.n),
                                    (mnk[0], mnk[1]),
                                    ctx.elect_one_warp,
                                )

                    tile_idx += 1

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_splitk[
        reduction_layout: Layout,
    ](
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c_tma_op: TMATensorTile[Self.c_type, Self.c_layout, Self.c_desc_layout],
        reduction_tensor: LayoutTensor[
            Self.config.accum_type, reduction_layout, MutAnyOrigin
        ],
        lock_ptr: UnsafePointer[UInt8],
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        workspace: Span[UInt64, MutAnyOrigin],
    ):
        """Split-K kernel entry point for better parallelism on small problems.

        Split-K divides the K dimension across multiple CTAs, with each CTA
        computing a partial result that is then reduced.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c_tma_op: TMA descriptor for matrix C.
            reduction_tensor: Workspace for partial results from each split.
            lock_ptr: Synchronization locks for reduction coordination.
            cluster_dim: Cluster dimensions.
            mnk: Problem dimensions (M, N, K).
            workspace: Workspace buffer for profiling/scheduling.
        """
        Self.validate_constraints()

        # Access shared memory via bitcast
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Create input pipeline for TMA→MMA synchronization (with payload)
        var tile_payload = Self.TilePayload(smem.a_tiles(), smem.b_tiles())
        var input_pipeline = Self.InputTilePipeline(
            smem.pipelines.input_barriers(), tile_payload
        )

        # Create kernel context with election vars, CTA coords, and masks
        var ctx = Self.Context(smem.pipelines.tmem_addr())

        # Initialize all barriers (only elect_one_warp && elect_one_thread)
        Self.init_barriers(
            ctx,
            a_tma_op,
            b_tma_op,
            c_tma_op,
            smem.pipelines.input_barriers(),
            smem.pipelines.accum_barriers(),
            smem.pipelines.clc_throttle(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.tmem_dealloc(),
        )

        var mma_op = MmaOpSM100_SS[
            Self.c_type,
            Self.a_type,
            Self.b_type,
            Self.config.block_tile_shape,
            Self.config.mma_shape,
            accum_type = Self.config.accum_type,
            cta_group = Self.config.cta_group,
            cluster_shape = Self.config.cluster_shape,
            a_swizzle = Self.config.a_swizzle,
            b_swizzle = Self.config.b_swizzle,
            transpose_b=True,
        ]()

        # Scheduler owns CLC throttle pipeline internally
        var scheduler = TileSchedulerSplitK[
            num_stages = Self.config.num_clc_pipeline_stages,
            reduction_tile_shape = Index(Self.BM, Self.MMA_N, Self.BK),
            cluster_shape = Index[dtype = DType.uint32](
                Self.config.cluster_shape[0],
                Self.config.cluster_shape[1],
                Self.config.cluster_shape[2],
            ),
            block_swizzle_size = Self.config.block_swizzle_size,
            rasterize_order = Self.config.raster_order,
            num_split_k = Self.config.num_split_k,
        ](
            cluster_dim,
            mnk,
            smem.pipelines.clc_response(),
            smem.pipelines.clc_full(),
            smem.pipelines.clc_empty(),
            smem.pipelines.clc_throttle(),
            lock_ptr,
        )

        # Per-warp work iterator - owns work_info, pipeline state, and throttle
        var work_iter = scheduler.work_iterator()

        # Create tile loaders for A and B matrices
        var a_loader = Self.ATileLoaderType(
            Pointer(to=a_tma_op), ctx.a_multicast_mask
        )
        var b_loader = Self.BTileLoaderType(
            Pointer(to=b_tma_op), ctx.b_multicast_mask
        )

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        if WarpRole.is_main_load():
            with MatmulProfilerType[0](workspace, 0):
                # Producer context: coordinates with MMA consumer via barriers
                with input_pipeline.producer() as producer:
                    while work_iter.has_work():
                        with work_iter.next() as current:
                            work_iter.throttle_signal(
                                ctx.is_first_cta_in_cluster
                            )

                            var k_start = current.k_start
                            var k_end = k_start + current.num_k_tiles
                            for i in range(
                                k_start, k_end, Self.config.k_group_size
                            ):
                                with producer.acquire() as tiles:  # waits for consumer
                                    Self.load_input_tiles(
                                        a_loader,
                                        b_loader,
                                        tiles,
                                        UInt32(i),
                                        UInt(current.m),
                                        UInt(current.n),
                                        ctx.peer_cta_coord,
                                        ctx.elect_one_cta,
                                    )

                            syncwarp()

                    producer.drain()  # wait for consumer before CTA exits

        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:

            @parameter
            if Self.config.num_clc_pipeline_stages == 0:
                return

            var sched_iter = scheduler.scheduler_iterator()

            with MatmulProfilerType[1](workspace, 0):
                while sched_iter.has_work():
                    with sched_iter.next():
                        sched_iter.signal_and_advance()

                sched_iter.drain()

        if WarpRole.is_mma():
            with MatmulProfilerType[2](workspace, 0):
                var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        smem.pipelines.accum_barriers(),
                        tmem,
                        UInt16(ctx.mma_complete_mask),
                    ),
                    Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
                )

                with mma_ctx:  # TMEM lifecycle
                    while work_iter.has_work():
                        with work_iter.wait_and_advance() as current:  # blocks on CLC
                            if ctx.elect_one_cta:
                                with mma_ctx.output_pipeline.producer() as output_stage:  # waits for epilogue
                                    var k_start = current.k_start
                                    var k_end = k_start + current.num_k_tiles
                                    with input_pipeline.consumer() as consumer:
                                        for i in range(
                                            k_start,
                                            k_end,
                                            Self.config.k_group_size,
                                        ):
                                            with consumer.acquire() as input_tiles:  # waits for TMA
                                                Self.mma(
                                                    output_stage.tmem,
                                                    input_tiles,
                                                    mma_op,
                                                    ctx.elect_one_warp,
                                                    UInt32(i),
                                                    k_start,
                                                )

        if WarpRole.is_epilogue():
            Self.EpilogueCtx.Sync.wait()  # wait for MMA to publish TMEM addr

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    smem.pipelines.accum_barriers(),
                    tmem,
                    UInt16(ctx.mma_complete_mask),
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))

            with epi_ctx:  # signals TMEM dealloc on exit
                var tile_idx = 0

                while work_iter.has_work():
                    with work_iter.next() as current:
                        with MatmulProfilerType[3](workspace, UInt32(tile_idx)):
                            with epi_ctx.output_pipeline.consumer() as output_stage:  # waits for MMA
                                tile_writer.write_splitk(
                                    smem.c_tiles(),
                                    output_stage,
                                    scheduler,
                                    reduction_tensor,
                                    current,
                                    (mnk[0], mnk[1]),
                                    ctx.elect_one_warp,
                                )

                    tile_idx += 1


# ============================================================================
# BlackwellMatmulSM100FallbackKernel - Simple non-warp-specialized kernel
# ============================================================================


struct BlackwellMatmulSM100FallbackKernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    a_desc_layout: Layout,
    b_desc_layout: Layout,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    transpose_b: Bool = True,
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1, 1, 1),
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    num_threads: Int = 128,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Simple fallback matmul kernel for SM100 (B200).

    This kernel is used when the warp-specialized kernel is not applicable,
    such as for small problem sizes or unsupported configurations.

    Unlike the main BlackwellMatmulSM100Kernel, this uses:
    - Single warp approach (no warp specialization)
    - Basic barrier synchronization (no CLC scheduling)
    - Direct LayoutTensor output (no TMA for C)
    - Simpler pipeline with single buffer
    """

    # ========== Derived Constants ==========
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime BK = Self.block_tile_shape[2]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]
    comptime MMA_K = Self.mma_shape[2]
    comptime num_m_mmas = Self.BM // Self.MMA_M
    comptime num_n_mmas = Self.BN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K

    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, swizzle_mode = Self.a_swizzle
    ]()
    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.b_swizzle
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.b_swizzle
    ]()

    comptime a_size = Self.a_smem_layout.size()
    comptime b_size = Self.b_smem_layout.size()

    # ========== Tile Type Aliases ==========
    comptime ATile = SMemTile[
        Self.a_type,
        Self.a_smem_layout,
        alignment=128,
    ]
    comptime BTile = SMemTile[
        Self.b_type,
        Self.b_smem_layout,
        alignment=128,
    ]

    comptime accum_type = get_accum_type[Self.a_type]()
    comptime c_frag_size = Self.MMA_M * Self.MMA_N // Self.num_threads
    comptime max_tmem_cols = 512

    # ========== Validation ==========
    @staticmethod
    @always_inline
    fn validate_constraints():
        """Validate compile-time constraints for this kernel configuration."""
        constrained[Self.num_threads == 128 or Self.num_threads == 256]()
        constrained[
            ((Self.a_size * size_of[Self.a_type]()) % 128) == 0,
            "preserve alignment",
        ]()
        constrained[
            ((Self.b_size * size_of[Self.b_type]()) % 16) == 0,
            "preserve alignment",
        ]()

    # ========== Kernel Entry Point ==========
    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    fn run(
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        c: LayoutTensor[Self.c_type, Self.c_layout, MutAnyOrigin],
        num_iters: UInt,
    ):
        """Run the fallback matmul kernel.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c: Output tensor C (LayoutTensor, not TMA).
            num_iters: Number of K-dimension iterations.
        """
        Self.validate_constraints()

        # Setup shared memory for A and B tiles
        var a_smem = rebind[SMemPtr[Scalar[Self.a_type]]](
            external_memory[
                Scalar[Self.a_type],
                address_space = AddressSpace.SHARED,
                alignment=128,
                name="tmem_test_dynamic_shared_memory",
            ]()
        )

        var b_smem = (a_smem + Self.a_size).bitcast[Scalar[Self.b_type]]()

        var a_smem_tile = Self.ATile(a_smem)
        var b_smem_tile = Self.BTile(b_smem)

        # Shared memory pointer to hold tensor memory address
        var ptr_tmem_addr = (b_smem + Self.b_size).bitcast[UInt32]()

        var c_frag = SIMD[Self.accum_type, Self.c_frag_size]()

        comptime a_expected_bytes = Self.a_size * size_of[Self.a_type]()
        comptime b_expected_bytes = Self.b_size * size_of[Self.b_type]()
        comptime expected_bytes = a_expected_bytes + b_expected_bytes

        var tma_mbar = (ptr_tmem_addr + 2).bitcast[SharedMemBarrier]()
        var mma_mbar = tma_mbar + 1

        if thread_idx.x == 0:
            tma_mbar[0].init()
            mma_mbar[0].init()

        var tma_phase: UInt32 = 0
        var mma_phase: UInt32 = 0

        var elect_one_warp = warp_id() == 0
        var elect_one_thread = thread_idx.x == 0
        var elect_one_cta = block_rank_in_cluster() % 2 == 0

        # Allocate tensor memory
        if elect_one_warp:
            tcgen05_alloc[1](ptr_tmem_addr, Self.max_tmem_cols)

        # Ensure all threads see initialized mbarrier and tensor memory allocation
        barrier()

        var tmem_addr = ptr_tmem_addr[0]

        # Create MmaOpSM100_SS instance
        var mma_op = MmaOpSM100_SS[
            Self.c_type,
            Self.a_type,
            Self.b_type,
            Self.block_tile_shape,
            Self.mma_shape,
            accum_type = Self.accum_type,
            cta_group=1,
            a_swizzle = Self.a_swizzle,
            b_swizzle = Self.b_swizzle,
            transpose_b = Self.transpose_b,
        ]()

        # Main loop over K dimension
        for i in range(num_iters):
            # Only one thread per CTA does the copy
            if elect_one_thread:
                tma_mbar[0].expect_bytes(Int32(expected_bytes))

                a_tma_op.async_copy(
                    a_smem_tile,
                    tma_mbar[0],
                    (Int(i) * Self.BK, Int(block_idx.y) * Self.BM),
                )
                b_tma_op.async_copy(
                    b_smem_tile,
                    tma_mbar[0],
                    (
                        Int(i) * Self.BK,
                        Int(block_idx.x) * Self.BN,
                    ) if Self.transpose_b else (
                        Int(block_idx.x) * Self.BN,
                        Int(i) * Self.BK,
                    ),
                )

            # Wait for the copy to finish
            tma_mbar[0].wait(tma_phase)
            tma_phase ^= 1

            # Perform MMA operation
            if elect_one_thread:
                mma_op.mma(
                    a_smem_tile,
                    b_smem_tile,
                    tmem_addr,
                    init_c=(i == 0),  # Initialize C on first iteration
                )
                mma_op.commit(mma_mbar)

            mma_mbar[0].wait(mma_phase)
            mma_phase ^= 1

        # Load accumulated result from tensor memory
        from ..structured_kernels.tmem import TmemAddress

        var tmem = TmemAddress(tmem_addr)
        c_frag = tmem.load_upper[
            Self.accum_type, Self.c_frag_size, 16, 256, Self.BN // 8
        ]()
        TmemAddress.wait_load()

        if elect_one_warp:
            tcgen05_release_allocation_lock[1]()
            tcgen05_dealloc[1](tmem_addr, Self.max_tmem_cols)

        # Write output to global memory
        comptime num_warps = Self.num_threads // WARP_SIZE
        var warp_id = get_warp_id()

        ctile, ctile_coords, _ = c.tile_with_offset[Self.BM, Self.BN](
            Int(block_idx.y), Int(block_idx.x)
        )
        comptime c_coord_type = type_of(ctile_coords)

        var M = c.dim[0]()
        comptime N = c.layout.shape[1].value()

        @parameter
        for m_mma in range(Self.num_m_mmas):

            @parameter
            for n_mma in range(Self.num_n_mmas):
                comptime mma_id = n_mma * Self.num_m_mmas + m_mma

                var c_gmem_warp_tile, _c_gmem_warp_tile_coords, _ = (
                    ctile.tile_with_offset[Self.MMA_M // num_warps, Self.MMA_N](
                        4 * m_mma + Int(warp_id), n_mma
                    )
                )
                var c_gmem_warp_tile_coords = ctile_coords + rebind[
                    c_coord_type
                ](_c_gmem_warp_tile_coords)

                var c_gmem_frag, _c_gmem_frag_coords, _ = (
                    c_gmem_warp_tile.vectorize[1, 2]().distribute_with_offset[
                        Layout.row_major(8, 4)
                    ](lane_id())
                )
                var new_c_gmem_frag_coords = rebind[c_coord_type](
                    _c_gmem_frag_coords
                )
                new_c_gmem_frag_coords[1] *= 2
                var c_gmem_frag_coords = (
                    c_gmem_warp_tile_coords + new_c_gmem_frag_coords
                )

                comptime num_vecs_m = c_gmem_frag.layout.shape[0].value()
                comptime num_vecs_n = c_gmem_frag.layout.shape[1].value()

                @parameter
                for n_vec in range(num_vecs_n):

                    @parameter
                    for m_vec in range(num_vecs_m):
                        comptime i_vec = n_vec * num_vecs_m + m_vec
                        comptime dst_idx = type_of(c_gmem_frag).layout(
                            IntTuple(m_vec, n_vec)
                        )
                        comptime dst_m_offset = dst_idx // N
                        comptime dst_n_offset = dst_idx % N
                        var m = UInt32(c_gmem_frag_coords[0] + dst_m_offset)
                        var n = UInt32(c_gmem_frag_coords[1] + dst_n_offset)

                        if m < UInt32(M) and n < UInt32(N):
                            var c_mn = SIMD[Self.accum_type, 2](
                                c_frag[2 * i_vec], c_frag[2 * i_vec + 1]
                            ).cast[Self.c_type]()

                            @parameter
                            if Self.elementwise_lambda_fn:
                                comptime alignment = align_of[
                                    SIMD[Self.c_type, 2]
                                ]()
                                comptime epilogue = (
                                    Self.elementwise_lambda_fn.value()
                                )
                                epilogue[alignment=alignment](
                                    (Int(m), Int(n)), c_mn
                                )
                            else:
                                c_gmem_frag[m_vec, n_vec] = rebind[
                                    c_gmem_frag.element_type
                                ](c_mn)
