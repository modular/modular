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

"""Block-scaled SM100 matmul kernel for MXFP8 matrix multiplication.

Warp-specialized architecture:
- Scheduler: CLC-based tile distribution
- TMA Load: Async loads for A, B, and their scaling factors (SFA, SFB)
- MMA: Block-scaled tensor core ops with TMEM accumulators
- Epilogue: TMEM → SMEM → GMEM output pipeline
"""

from collections import OptionalReg
from math import ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import align_of, size_of

from gpu import WARP_SIZE, barrier, warp_id
from gpu.cluster import (
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
from gpu.mma_sm100 import *
from gpu.primitives.grid_controls import (
    launch_dependent_grids,
    PDLLevel,
    wait_on_dependent_grids,
)
from gpu.sync import syncwarp
from gpu.tcgen05 import *
from layout import Layout, LayoutTensor, RuntimeLayout
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
    MXFP8_SF_VECTOR_SIZE,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.structuring import (
    SMemPtr,
    SMemTileType,
    SMemTileIter,
    SMemTileArrayType,
    SMemArrayType,
)
from linalg.matmul.gpu.profiler import MatmulProfileWarp

from .pipeline import ProducerConsumerPipeline
from .tile_pipeline import OutputTilePipeline
from .barriers import TmemDeallocBarrier, WarpGroupBarrier
from .tmem import TmemAllocation, TmemTensor, BlockScaledTmem
from .tile_scheduler import TileScheduler
from .tile_loader import TileLoaderTMA
from .tile_writer import EpilogueConfig
from .block_scaled_output_writer import BlockScaledTileWriter
from .warp_context import MmaWarpContext, EpilogueWarpContext
from .block_scaled_smem import (
    BlockScaledSmem,
    get_sfa_num_cols,
    get_sfb_num_cols,
)
from .block_scaled_tile_pipeline import (
    BlockScaledTilePipeline,
    BlockScaledConsumerStage,
    BlockScaledProducerStage,
)
from .block_scaled_tile_loader import ScalingFactorLoader, copy_sf_tmem

# Import WarpRole from the standard kernel
from .matmul_kernels import WarpRole


# =============================================================================
# BlockScaledKernelContext - Extended context with scaling factor info
# =============================================================================


struct BlockScaledKernelContext[
    num_clc_pipeline_stages: Int,
    cta_group: Int,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
    BM: Int,
    MMA_N: Int,
    num_pipeline_stages: Int,
](Copyable, Movable):
    """Per-CTA state: election flags, coordinates, multicast masks, TMEM offsets.
    """

    # ===== Election Variables =====
    var elect_one_warp: Bool
    var elect_one_thread: Bool
    var elect_one_cta: Bool
    var is_first_cta_in_cluster: Bool
    var warp_id: UInt32

    # ===== CTA Coordinates =====
    var rank_m: UInt
    var rank_n: UInt
    var peer_cta_coord: Tuple[UInt, UInt, UInt]

    # ===== Multicast Masks =====
    var a_multicast_mask: UInt16
    var b_multicast_mask: UInt16
    var mma_complete_mask: Int

    # ===== TMEM Pointers =====
    comptime TmemAddrArray = SMemArrayType[UInt32, 1]
    var ptr_tmem_addr: SMemPtr[UInt32]

    # ===== Scaling Factor TMEM Layout =====
    comptime SFA_NUM_COLS = Self.BM // 32
    comptime SFB_NUM_COLS = Self.MMA_N // 32

    @always_inline
    fn __init__(out self, tmem_addr_ptr: SMemPtr[UInt32]):
        """Initialize context; computes election flags and multicast masks."""
        self.warp_id = UInt32(get_warp_id())
        self.elect_one_warp = self.warp_id == 0
        self.elect_one_thread = elect_one_sync_with_mask()
        self.elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )
        self.is_first_cta_in_cluster = block_rank_in_cluster() == 0

        # CTA coordinates
        self.rank_m = block_id_in_cluster.x
        self.rank_n = block_id_in_cluster.y

        # Peer CTA coordinate: (peer_id, mma_coord_m, mma_coord_n)
        self.peer_cta_coord = (
            UInt(self.rank_m % UInt(Self.cta_group)),
            UInt(self.rank_m // UInt(Self.cta_group)),
            self.rank_n,
        )

        # Compute multicast masks
        self.a_multicast_mask = 0x0
        self.b_multicast_mask = 0x0

        @parameter
        for i in range(Self.CLUSTER_N):
            self.a_multicast_mask |= 1 << (i * Self.CLUSTER_M)

        @parameter
        for i in range(Self.CLUSTER_M // Self.cta_group):
            self.b_multicast_mask |= 1 << (i * Self.cta_group)

        self.a_multicast_mask <<= UInt16(self.rank_m)
        self.b_multicast_mask <<= UInt16(self.peer_cta_coord[0])
        self.b_multicast_mask <<= UInt16(self.rank_n * UInt(Self.CLUSTER_M))

        # MMA completion mask for barrier synchronization
        var self_mask = 1 << Int(block_rank_in_cluster())
        var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
        self.mma_complete_mask = self_mask | peer_mask

        self.ptr_tmem_addr = tmem_addr_ptr


# =============================================================================
# BlackwellBlockScaledMatmulKernel - Main kernel struct
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
    # Cluster shape (must match config, needed for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Optional features
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    max_profiled_tiles_per_SM: UInt32 = 0,
]:
    """SM100 block-scaled GEMM kernel for MXFP8 (FP8 with microscaling).

    Extends standard matmul with per-block scaling factors (SFA, SFB) that are
    loaded via TMA, copied to TMEM, and applied during MMA operations.
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

    comptime CLUSTER_M = Int(Self.config.cluster_shape[0])
    comptime CLUSTER_N = Int(Self.config.cluster_shape[1])
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Scaling Factor Configuration ==========

    comptime SFA_NUM_COLS = Self.BM // 32
    comptime SFB_NUM_COLS = Self.MMA_N // 32
    comptime NUM_TMEM_COLS = 512

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

    comptime num_pipeline_stages = Int(Self.config.num_pipeline_stages)
    comptime num_group_pipeline_stages = Self.num_pipeline_stages // Int(
        Self.config.k_group_size
    )
    comptime num_clc_pipeline_stages = Int(Self.config.num_clc_pipeline_stages)
    comptime num_accum_pipeline_stages = Int(
        Self.config.num_accum_pipeline_stages
    )
    comptime num_output_stages = Int(Self.config.num_output_stages)

    # TMEM stage stride
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

    comptime sfa_smem_layout = tile_sf_layout_k_major[
        Self.BM, Self.BK, MXFP8_SF_VECTOR_SIZE
    ]()

    comptime sfb_smem_layout = tile_sf_layout_k_major[
        Self.MMA_N, Self.BK, MXFP8_SF_VECTOR_SIZE
    ]()

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
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type = Self.accum_type,
        cta_group = Self.cta_group,
        cluster_shape = Self.config.cluster_shape,
        a_swizzle = Self.config.a_swizzle,
        b_swizzle = Self.config.b_swizzle,
        transpose_b = Self.transpose_b,
    ]

    # ========== TMA Loader Type Templates ==========
    # Note: Actual loader instances are created in the kernel with specific
    # TMA descriptor origins. These aliases document the expected types.

    # ATileLoader = TileLoaderTMA[origin, a_type, a_layout, a_desc_layout, ...]
    # BTileLoader = TileLoaderTMA[origin, b_type, b_layout, b_desc_layout, ...]
    # SFALoader = ScalingFactorLoader[origin, sfa_dtype, sfa_layout, ...]
    # SFBLoader = ScalingFactorLoader[origin, sfb_dtype, sfb_layout, ...]

    # ========== Kernel Context Type ==========

    comptime ContextType = BlockScaledKernelContext[
        Self.num_clc_pipeline_stages,
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
        Self.BM,
        Self.MMA_N,
        Self.num_pipeline_stages,
    ]

    # ========== Static Assertions ==========

    @staticmethod
    fn validate_config():
        """Validate configuration constraints at compile time."""
        constrained[Self.transpose_b, "Only support transposed B"]()
        constrained[
            Self.sfa_dtype == Self.sfb_dtype == MXFP8_SF_DTYPE,
            "Only support MXFP8_SF_DTYPE (F8-UE8M0) for scales",
        ]()
        constrained[
            Self.cta_group in (1, 2),
            "Only support cta_group == 1 or 2",
        ]()
        constrained[
            Self.config.k_group_size == 1,
            "Only support k_group_size == 1 for block-scaled",
        ]()
        constrained[
            Self.config.num_split_k == 1,
            "Only support split_k == 1 for block-scaled",
        ]()
        # Verify TMEM capacity for scaling factors
        comptime sf_tmem_usage = (
            Self.num_accum_pipeline_stages * Self.MMA_N
            + Self.SFA_NUM_COLS * Self.num_pipeline_stages
            + Self.SFB_NUM_COLS * Self.num_pipeline_stages
        )
        constrained[
            sf_tmem_usage <= Self.NUM_TMEM_COLS,
            "Scaling factor TMEM usage exceeds capacity",
        ]()

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

    # ========== Input Tile Pipeline Type ==========

    comptime InputTilePipeline = BlockScaledTilePipeline[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.a_smem_layout,
        Self.b_smem_layout,
        Self.sfa_smem_layout,
        Self.sfb_smem_layout,
        Self.num_pipeline_stages,
        Self.num_group_pipeline_stages,
        Int(Self.config.k_group_size),
    ]

    # ========== Output Tile Pipeline Type ==========

    comptime OutputPipeline = OutputTilePipeline[
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]

    # ========== TMEM Types ==========

    comptime Tmem = TmemAllocation[Self.cta_group]

    # Unified TMEM region for accumulators and scaling factors
    comptime TmemRegion = BlockScaledTmem[
        accum_dtype = Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = Self.MMA_N,
        num_accum_stages = Self.num_accum_pipeline_stages,
        sf_dtype = Self.sfa_dtype,
        BM = Self.BM,
        num_pipeline_stages = Self.num_pipeline_stages,
        cta_group = Self.cta_group,
        total_cols = Self.NUM_TMEM_COLS,
    ]

    # Typed tile accessors
    comptime AccumTensor = Self.TmemRegion.AccumTile
    comptime SFATensor = Self.TmemRegion.SFATile
    comptime SFBTensor = Self.TmemRegion.SFBTile

    # ========== Expected Bytes Calculation ==========

    @staticmethod
    @always_inline
    fn expected_bytes_per_k_group() -> Int:
        """Calculate expected bytes for TMA barrier per k-group iteration."""
        comptime a_bytes = Self.a_smem_layout.size() * size_of[Self.a_type]()
        comptime b_bytes = Self.b_smem_layout.size() * size_of[Self.b_type]()
        comptime sfa_bytes = Self.sfa_smem_layout.size() * size_of[
            Self.sfa_dtype
        ]()
        comptime sfb_bytes = Self.sfb_smem_layout.size() * size_of[
            Self.sfb_dtype
        ]()

        return (
            Self.cta_group
            * (a_bytes + b_bytes + sfa_bytes + sfb_bytes)
            * Int(Self.config.k_group_size)
        )

    # ========== Block-Scaled MMA Helper ==========

    @staticmethod
    @always_inline
    fn mma_block_scaled[
        tiles_origin: MutOrigin,
    ](
        accum: Self.AccumTensor,
        tmem_region: Self.TmemRegion,
        tiles: BlockScaledConsumerStage[
            tiles_origin,
            Self.a_type,
            Self.b_type,
            Self.sfa_dtype,
            Self.sfb_dtype,
            Self.a_smem_layout,
            Self.b_smem_layout,
            Self.sfa_smem_layout,
            Self.sfb_smem_layout,
            Self.num_pipeline_stages,
            Self.num_group_pipeline_stages,
            Int(Self.config.k_group_size),
        ],
        mma_op: Self.MmaOp,
        k_iter: UInt32,
        k_start: UInt32,
    ):
        """Copy scaling factors to TMEM and execute block-scaled MMA.

        Args:
            accum: Typed TMEM tensor for accumulators.
            tmem_region: TMEM region with typed accessors for scaling factors.
            tiles: Consumer stage with A, B, SFA, SFB tiles.
            mma_op: Block-scaled MMA operation instance.
            k_iter: Current K iteration index.
            k_start: Starting K iteration (for init_c).
        """
        var stage = tiles.stage()

        if elect_one_sync():

            @parameter
            for j in range(Int(Self.config.k_group_size)):
                # Compute linear index for this k-iteration
                var sf_idx = stage * UInt32(Self.config.k_group_size) + UInt32(
                    j
                )
                var a_tile, b_tile = tiles.get_tile(j)
                var sfa_smem, sfb_smem = tiles.get_sf_tile(j)

                # Get typed TMEM tiles for scaling factors
                var sfa_tmem = tmem_region.sfa(sf_idx)
                var sfb_tmem = tmem_region.sfb(sf_idx)

                # Copy scaling factors from SMEM to TMEM
                copy_sf_tmem[
                    Self.sfa_dtype,
                    Self.sfa_smem_layout,
                    Self.BM,
                    Self.cta_group,
                ](sfa_smem, sfa_tmem)
                copy_sf_tmem[
                    Self.sfb_dtype,
                    Self.sfb_smem_layout,
                    Self.MMA_N,
                    Self.cta_group,
                ](sfb_smem, sfb_tmem)

                mma_op.mma(
                    a_tile,
                    b_tile,
                    UInt32(accum.offset()),
                    UInt32(sfa_tmem.offset()),
                    UInt32(sfb_tmem.offset()),
                    init_c=((k_iter + UInt32(j)) == k_start),
                )

            mma_op.commit(tiles.mbar())

    # ========== Tile Writer Type ==========

    comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    comptime TileWriterType = BlockScaledTileWriter[
        config = Self.config,
        c_smem_layout = Self.c_smem_layout,
        num_output_stages = Self.num_output_stages,
        stage_stride_cols = UInt(Self.stage_stride_cols),
        num_output_warps = Self.num_output_warps,
        max_tmem_cols = UInt(Self.NUM_TMEM_COLS),
        elementwise_compute_lambda_fn = Self.elementwise_compute_lambda_fn,
        register_based_epilogue = Self.register_based_epilogue,
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

    comptime TmemDeallocBarrier = TmemDeallocBarrier[Self.cta_group]

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
        """Kernel entry point. Dispatches to warp-specialized roles."""
        Self.validate_config()

        # Access shared memory via bitcast
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Create tile arrays from SMEM storage
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var sfa_tiles = smem.sfa_tiles()
        var sfb_tiles = smem.sfb_tiles()
        var c_tiles = smem.c_tiles()

        # Create barrier arrays
        var input_barriers = smem.tma_mma_mbars()
        var accum_barriers = smem.accum_mbars()
        var clc_full = smem.clc_mbars_full()
        var clc_empty = smem.clc_mbars_empty()
        var clc_throttle = smem.clc_throttle_mbars()
        var clc_response = smem.clc_response()
        var tmem_dealloc = smem.tmem_dealloc_mbar()
        var tmem_addr_arr = smem.tmem_addr()

        # Create input pipeline
        var input_pipeline = Self.InputTilePipeline(
            input_barriers, a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )

        # Create kernel context
        var ctx = Self.ContextType(tmem_addr_arr.ptr)

        var mma_op = Self.MmaOp()

        # Initialize barriers (elect_one_warp && elect_one_thread only)
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()
            sfa_tma_op.prefetch_descriptor()
            sfb_tma_op.prefetch_descriptor()

            Self.InputTilePipeline.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Self.config.cluster_shape[0] // Self.config.cta_group
                + Self.config.cluster_shape[1]
                - 1,
            )
            Self.OutputPipeline.init_barriers(
                accum_barriers.ptr,
                Self.accum_pipeline_producer_arv_count,
                Self.accum_pipeline_consumer_arv_count,
            )
            Self.Scheduler.init_throttle_barriers(
                clc_throttle.ptr,
                Self.clc_throttle_producer_arv_count,
                Self.clc_throttle_consumer_arv_count,
            )

            tmem_dealloc.ptr[].init(
                Self.EPILOGUE_THREADS * Self.config.cta_group
            )

            @parameter
            for i in range(Self.config.num_clc_pipeline_stages):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Self.clc_consumer_arv_count)

        fence_mbarrier_init()
        cluster_sync()

        # Create scheduler
        var scheduler = Self.Scheduler(
            cluster_dim,
            clc_response,
            clc_full,
            clc_empty,
            clc_throttle,
        )

        var work_iter = scheduler.work_iterator()

        # Create tile loaders
        var a_loader = TileLoaderTMA[cta_group = Self.cta_group](
            Pointer(to=a_tma_op), ctx.a_multicast_mask
        )
        var b_loader = TileLoaderTMA[cta_group = Self.cta_group](
            Pointer(to=b_tma_op), ctx.b_multicast_mask
        )
        var sfa_loader = ScalingFactorLoader[
            cta_group = Self.cta_group,
            BM_or_MMA_N = Self.BM,
        ](Pointer(to=sfa_tma_op))
        var sfb_loader = ScalingFactorLoader[
            cta_group = Self.cta_group,
            BM_or_MMA_N = Self.MMA_N,
        ](Pointer(to=sfb_tma_op))

        var num_iters: UInt32 = ceildiv(mnk[2], Self.BK)

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        # TMA descriptor layout sizes for peer CTA slicing
        comptime a_tma_rows = Self.a_desc_layout.shape[0].value()
        comptime b_tma_rows = Self.b_desc_layout.shape[0].value()

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():

            @parameter
            @always_inline
            fn load_tiles[
                tiles_origin: MutOrigin
            ](
                tiles: BlockScaledProducerStage[
                    tiles_origin,
                    Self.a_type,
                    Self.b_type,
                    Self.sfa_dtype,
                    Self.sfb_dtype,
                    Self.a_smem_layout,
                    Self.b_smem_layout,
                    Self.sfa_smem_layout,
                    Self.sfb_smem_layout,
                    Self.num_pipeline_stages,
                    Self.num_group_pipeline_stages,
                    Int(Self.config.k_group_size),
                ],
                tile_m: UInt32,
                tile_n: UInt32,
                k_iter: Int,
            ):
                """Load A, B, SFA, SFB tiles for one k-group. Captures loaders/ctx.
                """
                var mbar = tiles.barrier()

                @parameter
                for j in range(Int(Self.config.k_group_size)):
                    var a, b = tiles.get_tile(j)
                    var sfa, sfb = tiles.get_sf_tile(j)
                    var k = UInt(k_iter + j)

                    # GMEM coordinates
                    var a_m = ctx.peer_cta_coord[2] * UInt(a_tma_rows) + UInt(
                        tile_m
                    ) * UInt(Self.BM)
                    var b_n = (
                        ctx.peer_cta_coord[1] * UInt(b_tma_rows)
                        + ctx.peer_cta_coord[0] * UInt(Self.BN)
                        + UInt(tile_n) * UInt(Self.MMA_N)
                    )

                    # Load A, B with peer CTA slicing
                    a_loader.load(
                        a.tile[a_tma_rows, Self.BK](
                            Int(ctx.peer_cta_coord[2]), 0
                        ),
                        mbar[0],
                        k * UInt(Self.BK),
                        a_m,
                    )
                    b_loader.load(
                        b.tile[b_tma_rows, Self.BK](
                            Int(ctx.peer_cta_coord[1]), 0
                        ),
                        mbar[0],
                        k * UInt(Self.BK),
                        b_n,
                    )

                    # Load scaling factors
                    sfa_loader.load(sfa, mbar[0], k, UInt(tile_m), UInt(0))
                    sfb_loader.load(sfb, mbar[0], k, UInt(tile_n), UInt(0))

            with MatmulProfilerType[0](workspace, 0):

                @parameter
                if Self.pdl_level > PDLLevel.OFF:
                    wait_on_dependent_grids()

                while work_iter.has_work():
                    with work_iter.next() as current:
                        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                        with input_pipeline.producer() as producer:
                            for i in range(
                                0, num_iters, Self.config.k_group_size
                            ):
                                with producer.acquire() as tiles:
                                    if ctx.elect_one_cta:
                                        tiles.expect_bytes(
                                            Self.expected_bytes_per_k_group()
                                        )

                                    if elect_one_sync():
                                        load_tiles(
                                            tiles, current.m, current.n, i
                                        )

                        syncwarp()

                with input_pipeline.producer() as producer:
                    producer.drain()

        # ===== SCHEDULER WARP =====
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:

            @parameter
            if Self.config.num_clc_pipeline_stages == 0:
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
                var tmem = Self.Tmem.allocate(tmem_addr_arr)

                # Create unified TMEM region for accumulators and scaling factors
                var tmem_region = Self.TmemRegion(tmem)

                var mma_ctx = Self.MmaCtx(
                    tmem,
                    Self.OutputPipeline(
                        accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                    ),
                    Self.TmemDeallocBarrier(tmem_dealloc),
                )

                with mma_ctx:
                    while work_iter.has_work():
                        with work_iter.next_prefetch():
                            if ctx.elect_one_cta:
                                with mma_ctx.output_pipeline.producer() as stage:
                                    with input_pipeline.consumer() as consumer:
                                        for i in range(
                                            0,
                                            num_iters,
                                            Self.config.k_group_size,
                                        ):
                                            with consumer.acquire() as tiles:
                                                # Get typed accumulator tensor
                                                var accum = tmem_region.accum(
                                                    stage.index
                                                )

                                                Self.mma_block_scaled(
                                                    accum,
                                                    tmem_region,
                                                    tiles,
                                                    mma_op,
                                                    i,
                                                    0,
                                                )

                    @parameter
                    if Self.pdl_level > PDLLevel.OFF:
                        launch_dependent_grids()

        # ===== EPILOGUE WARPS =====
        if WarpRole.is_epilogue():
            Self.EpilogueCtx.Sync.wait()

            var tmem = Self.Tmem.from_shared(tmem_addr_arr)
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDeallocBarrier(tmem_dealloc),
            )

            var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))

            with epi_ctx:
                var tile_idx = 0

                while work_iter.has_work():
                    with work_iter.next() as current:
                        with MatmulProfilerType[3](workspace, tile_idx):
                            with epi_ctx.output_pipeline.consumer() as stage:
                                tile_writer.write(
                                    c_tiles,
                                    stage,
                                    (current.m, current.n),
                                    (mnk[0], mnk[1]),
                                    ctx.elect_one_warp,
                                )

                    tile_idx += 1
