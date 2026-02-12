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
"""Grouped 1D-1D block-scaled SM100 matmul kernel.

This kernel implements grouped GEMM for Mixture of Experts (MoE) layers using
the 1D-1D tensor layout with offset-based addressing.

Key characteristics:
- 3-warp specialization (Load, MMA, Epilogue) - no scheduler warp
- Grid-constant TMA descriptors (no runtime tensormap updates)
- Offset-based addressing via a_offsets for contiguous token buffers
- Per-expert output scaling via expert_scales tensor

Architecture:
- TMA warp: Loads A, B, SFA, SFB tiles using grid-constant TMAs
- MMA warp: Executes block-scaled matrix multiply
- Epilogue warps: Stores results with expert_scale applied

This is a port of grouped_matmul_sm100_1d1d.mojo to the structured kernels
architecture.
"""

from collections import Optional
from math import ceildiv
from memory import UnsafePointer, Pointer
from sys import size_of

from gpu import WARP_SIZE, block_idx, grid_dim, lane_id, thread_idx, warp_id
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.sync import named_barrier, syncwarp
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout as LegacyLayout, LayoutTensor
from layout._layout import TensorLayout
from layout._tile_tensor import TileTensor
from ..structured_kernels.tile_types import (
    GMEMLayout1D,
    GMEMTile,
    TMATile,
    TmaOpType,
    lt_to_tt,
    lt_to_tt_1d,
    static_row_major,
    tma_desc_layout_3d,
    tma_desc_layout_4d,
)
from layout._coord import CoordLike, ComptimeInt, RuntimeInt
from layout._layout import RowMajorLayout, _IntToComptimeInt
from layout.tma_async import SharedMemBarrier, TMATensorTile
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_sf_layout_k_major,
)

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_BlockScaled_SS
from linalg.fp4_utils import SF_MN_GROUP_SIZE, SF_ATOM_M, SF_ATOM_K
from linalg.utils import elementwise_compute_lambda_type

from ..structured_kernels.config import BlockScaledMatmulConfig
from ..structured_kernels.kernel_common import KernelContext
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    InputProducerStage,
    InputConsumerStage,
    OutputTilePipeline,
    BlockScaledTilePayload,
)
from ..structured_kernels.tmem import BlockScaledTmem, TmemAllocation
from ..structured_kernels.barriers import TmemDeallocBarrier, WarpGroupBarrier
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)

from .grouped_1d1d_smem import Grouped1D1DSmem
from .grouped_1d1d_tile_scheduler import (
    GroupedWorkIterator1D1D,
    GroupedWorkContext1D1D,
)
from ..structured_kernels.output_writer import TileWriter


# =============================================================================
# Warp Role for 3-warp specialization (no scheduler)
# =============================================================================


struct WarpRole1D1D(TrivialRegisterPassable):
    """Warp role for 1D-1D kernel with 3-warp specialization.

    Thread layout (192 threads total) - matches original kernel:
    - Warps 0-3 (threads 0-127): Epilogue (4 warps)
    - Warp 4 (threads 128-159): TMA Load
    - Warp 5 (threads 160-191): MMA

    This layout matches the original grouped_matmul_sm100_1d1d.mojo kernel
    which uses WarpRole[has_scheduler=False]. The epilogue warps being at
    0-3 is important because TMAStoreCoords uses `warp_id == 0` for election.

    No scheduler warp - work distribution uses linear grid traversal.
    """

    comptime EPILOGUE_WARP_START = 0
    comptime LOAD_WARP_START = 128
    comptime MMA_WARP_START = 160

    comptime NUM_EPILOGUE_THREADS = 128  # 4 warps
    comptime NUM_LOAD_THREADS = 32
    comptime NUM_MMA_THREADS = 32

    comptime TOTAL_THREADS = 192

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        """Returns True if current thread is in an epilogue warp (warps 0-3)."""
        return thread_idx.x < Self.LOAD_WARP_START

    @staticmethod
    @always_inline
    fn is_load() -> Bool:
        """Returns True if current thread is in the TMA load warp (warp 4)."""
        return (
            thread_idx.x >= Self.LOAD_WARP_START
            and thread_idx.x < Self.MMA_WARP_START
        )

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        """Returns True if current thread is in the MMA warp (warp 5)."""
        return thread_idx.x >= Self.MMA_WARP_START


# =============================================================================
# Grouped1D1DMatmulKernel - Main kernel struct
# =============================================================================


struct Grouped1D1DMatmulKernel[
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
    # Static N dimension (expert output size)
    static_N: Int,
    # Cluster shape
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Epilogue fusion
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
]:
    """Grouped 1D-1D block-scaled matmul kernel.

    Uses 3-warp specialization (Load, MMA, Epilogue) with grid-constant TMAs.
    Work distribution via GroupedWorkIterator1D1D using offset-based addressing.
    """

    # ========== Derived Constants ==========

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

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    comptime NUM_THREADS = WarpRole1D1D.TOTAL_THREADS

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # TMEM configuration
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = Self.config.num_sf_k_tiles * (Self.BM // 32)
    comptime SFB_NUM_COLS = Self.config.num_sf_k_tiles * (Self.MMA_N // 32)
    comptime stage_stride_cols = Self.MMA_N

    # ========== Barrier Arrival Counts ==========

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = (
        Self.cta_group * WarpRole1D1D.NUM_EPILOGUE_THREADS
    )

    # ========== Shared Memory Type ==========

    comptime SmemType = Grouped1D1DSmem[
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

    # ========== Tile Pipeline Types ==========
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

    comptime InputTilePipelineType = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== TMEM and Output Pipeline Types ==========

    comptime Tmem = TmemAllocation[Self.cta_group]

    comptime TmemRegion = BlockScaledTmem[
        Self.accum_type,
        Self.MMA_M,
        Self.MMA_N,
        Self.num_accum_pipeline_stages,
        Self.sfa_dtype,
        Self.BM,
        Self.num_pipeline_stages,
        cta_group = Self.cta_group,
        num_sf_k_tiles = Self.config.num_sf_k_tiles,
    ]

    comptime OutputPipeline = OutputTilePipeline[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    ]

    comptime TmemDealloc = TmemDeallocBarrier[Self.cta_group]

    # ========== Warp Context Types ==========

    comptime MmaEpilogueSync = WarpGroupBarrier[
        WarpRole1D1D.NUM_MMA_THREADS + WarpRole1D1D.NUM_EPILOGUE_THREADS, 1
    ]

    comptime MmaCtx = MmaWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        WarpRole1D1D.NUM_MMA_THREADS,
        WarpRole1D1D.NUM_EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        WarpRole1D1D.NUM_MMA_THREADS,
        WarpRole1D1D.NUM_EPILOGUE_THREADS,
    ]

    # ========== Tile Writer Type ==========

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
        batched=False,  # 1D-1D uses 2D coordinates with bounds checking
    ]

    # ========== Work Iterator Type ==========

    comptime WorkIterator = GroupedWorkIterator1D1D[
        static_N = Self.static_N,
        tile_shape = Self.config.block_tile_shape,
        cluster = Self.config.cluster_shape,
        cta_group = Self.cta_group,
    ]

    # ========== TMA Load Size Constants ==========

    comptime a_expected_bytes = Self.SmemType.a_smem_layout.size() * size_of[
        Self.a_type
    ]()
    comptime b_expected_bytes = Self.SmemType.b_smem_layout.size() * size_of[
        Self.b_type
    ]()
    comptime sfa_expected_bytes = Self.SmemType.sfa_smem_layout.size() * size_of[
        Self.sfa_dtype
    ]()
    comptime sfb_expected_bytes = Self.SmemType.sfb_smem_layout.size() * size_of[
        Self.sfb_dtype
    ]()

    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.sfa_expected_bytes
        + Self.sfb_expected_bytes
    ) * Self.config.k_group_size

    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime a_tile_dim0 = Self.BM // Self.CLUSTER_N
    comptime b_tile_dim0 = Self.BN // (Self.CLUSTER_M // Self.cta_group)
    comptime a_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.a_type
    ]()
    comptime b_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.b_type
    ]()
    comptime c_swizzle_elems = Self.config.c_swizzle.bytes() // size_of[
        Self.c_type
    ]()

    # C tile shape -- same logic as default/block_scaled kernels
    comptime c_tile_dim0 = Self.OutputM if (
        Self.MMA_M == 256 or Self.cta_group == 1 or Self.config.AB_swapped
    ) else 64
    comptime c_tile_dim1 = Self.c_swizzle_elems if (
        Self.config.AB_swapped
    ) else Self.OutputN

    # A, B, C: 2D TMA layouts
    comptime ATileLayout = static_row_major[Self.a_tile_dim0, Self.BK]
    comptime ADescLayout = static_row_major[
        Self.a_tile_dim0, Self.a_swizzle_elems
    ]
    comptime BTileLayout = static_row_major[Self.b_tile_dim0, Self.BK]
    comptime BDescLayout = static_row_major[
        Self.b_tile_dim0, Self.b_swizzle_elems
    ]
    comptime CTileLayout = static_row_major[Self.c_tile_dim0, Self.c_tile_dim1]
    comptime CDescLayout = static_row_major[
        Self.c_tile_dim0, Self.c_swizzle_elems
    ]

    # SFA, SFB: 4D TMA layouts (no batch dim, unlike block_scaled's 5D)
    comptime SFATileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            Self.BM // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFADescLayout = tma_desc_layout_4d[
        Self.sfa_dtype,
        Self.BM // SF_MN_GROUP_SIZE,
        Self.config.num_sf_k_tiles,
        SF_ATOM_M[0],
        TensorMapSwizzle.SWIZZLE_NONE,
    ]
    comptime SFBTileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            Self.MMA_N // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            SF_ATOM_M[0],
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFBDescLayout = tma_desc_layout_4d[
        Self.sfb_dtype,
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

    # 1D data TileTensor types (offsets, expert IDs, scales)
    comptime OffsetsTile = TileTensor[DType.uint32, GMEMLayout1D, MutAnyOrigin]
    comptime AScaleOffsetsTile = TileTensor[
        DType.uint32, GMEMLayout1D, MutAnyOrigin
    ]
    comptime ExpertIdsTile = TileTensor[DType.int32, GMEMLayout1D, MutAnyOrigin]
    comptime ExpertScalesTile = TileTensor[
        DType.float32, GMEMLayout1D, MutAnyOrigin
    ]

    # C device layout: (M_dynamic, N_static) row-major, computed from static_N.
    comptime CDeviceLayout = RowMajorLayout[
        RuntimeInt[DType.int64], ComptimeInt[Self.static_N]
    ]

    # C device tensor type (for bounds-checked stores)
    comptime CDeviceTile = TileTensor[
        Self.c_type, Self.CDeviceLayout, MutAnyOrigin
    ]

    # TMA load size constants (from desc layout dimensions)
    comptime a_tma_load_size = Self.a_tile_dim0 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim0 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim0
    comptime b_tma_rows = Self.b_tile_dim0

    # ========== Validation ==========

    @staticmethod
    fn validate_config():
        """Compile-time validation of kernel configuration."""
        constrained[
            Self.a_type == Self.b_type,
            "A and B types must match for block-scaled GEMM",
        ]()
        constrained[
            Self.sfa_dtype == Self.sfb_dtype,
            "SFA and SFB types must match",
        ]()
        constrained[
            Self.cta_group in (1, 2),
            "Only support cta_group == 1 or 2",
        ]()
        constrained[Self.transpose_b, "Only support transposed B"]()

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
        # Grid-constant TMA descriptors
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c_tma_op: Self.CTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        # Offset tensors for 1D-1D addressing (TileTensor)
        a_offsets: Self.OffsetsTile,
        a_scale_offsets: Self.AScaleOffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
        # C tensor for bounds-checked stores (TileTensor)
        c_device: Self.CDeviceTile,
        # Number of active experts
        num_active_experts: Int,
        # K dimension for iteration
        K: UInt32,
    ):
        """Grouped 1D-1D block-scaled GEMM kernel entry point.

        Uses grid-constant TMAs with offset-based addressing for 1D-1D layout.
        """
        Self.validate_config()

        # ===== Shared Memory Setup =====
        ref smem = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Get typed tile arrays from SMEM
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var sfa_tiles = smem.sfa_tiles()
        var sfb_tiles = smem.sfb_tiles()

        # Get typed barrier arrays
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var tmem_addr_storage = smem.pipelines.tmem_addr().ptr

        # Create input pipeline with tile payload
        var tile_payload = Self.TilePayload(
            a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )
        var input_pipeline = Self.InputTilePipelineType(
            input_barriers, tile_payload
        )

        # ===== Warp/Thread Election =====
        var elect_one_warp = thread_idx.x // UInt(WARP_SIZE) == 0
        var elect_one_thread = elect_one_sync_with_mask()
        var elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )

        # Peer CTA coordinates for multicast
        var peer_rank_n = UInt(block_rank_in_cluster() % UInt32(Self.CLUSTER_N))
        var peer_rank_m = UInt(
            block_rank_in_cluster()
            // UInt32(Self.CLUSTER_N)
            % UInt32(Self.CLUSTER_M)
        )
        var peer_m_rank = peer_rank_m % UInt(Self.cta_group)
        var peer_cta_coord = (peer_rank_n, peer_rank_m, peer_m_rank)

        # Multicast masks
        var a_multicast_mask = UInt16((1 << Self.CLUSTER_M) - 1)
        var b_multicast_mask = UInt16((1 << Self.CLUSTER_N) - 1)
        var mma_complete_mask = UInt16((1 << Self.cta_group) - 1)

        # K iteration count
        var num_k_iters = Int(ceildiv(Int(K), Self.BK))

        # ===== Barrier Initialization =====
        if elect_one_warp and elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()
            sfa_tma_op.prefetch_descriptor()
            sfb_tma_op.prefetch_descriptor()

            # Initialize input pipeline barriers
            Self.InputTilePipelineType.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Int32(
                    Self.config.cluster_shape[0] // Self.cta_group
                    + Self.config.cluster_shape[1]
                    - 1
                ),
            )

            # Initialize output pipeline barriers
            Self.OutputPipeline.init_barriers(
                accum_barriers.ptr,
                Self.accum_pipeline_producer_arv_count,
                Int32(Self.accum_pipeline_consumer_arv_count),
            )

            # Initialize TMEM deallocation barrier
            smem.pipelines.tmem_dealloc().ptr[].init(
                Int32(WarpRole1D1D.NUM_EPILOGUE_THREADS * Self.cta_group)
            )

        fence_mbarrier_init()
        cluster_sync()

        var mma_op = Self.MmaOp()

        # ===== Work Iterator Setup =====
        var work_iter = Self.WorkIterator(
            num_active_experts,
            a_offsets,
            expert_ids,
            expert_scales,
        )

        # ===== TMA LOAD WARP =====
        if WarpRole1D1D.is_load():
            with input_pipeline.producer() as producer:
                while True:
                    var ctx = work_iter.next()
                    if ctx.info.is_done():
                        break
                    if not ctx.info.is_valid():
                        continue

                    var next_ready = True
                    if num_k_iters > 0:
                        next_ready = producer.try_acquire()

                    for k_tile in range(num_k_iters):
                        with producer.acquire_if_needed(next_ready) as tiles:
                            Self.load_input_tiles(
                                a_tma_op,
                                b_tma_op,
                                sfa_tma_op,
                                sfb_tma_op,
                                tiles,
                                peer_cta_coord,
                                ctx,
                                a_scale_offsets,
                                UInt32(k_tile),
                                elect_one_cta,
                            )
                        next_ready = True
                        if k_tile + 1 < num_k_iters:
                            next_ready = producer.try_acquire()

                    syncwarp()

                producer.drain()

        # ===== MMA WARP =====
        if WarpRole1D1D.is_mma():
            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(accum_barriers, tmem, mma_complete_mask),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            with mma_ctx:
                while True:
                    var ctx = work_iter.next()
                    if ctx.info.is_done():
                        break
                    if not ctx.info.is_valid():
                        continue

                    if elect_one_cta:
                        with mma_ctx.output_pipeline.producer() as output_stage:
                            var tmem_offset = UInt32(output_stage.tmem.offset())

                            with input_pipeline.consumer() as consumer:
                                var next_ready = True
                                if num_k_iters > 0:
                                    next_ready = consumer.try_acquire()

                                for k_tile in range(num_k_iters):
                                    with consumer.acquire_if_needed(
                                        next_ready
                                    ) as input_tiles:
                                        Self.mma(
                                            input_tiles,
                                            mma_op,
                                            tmem_offset,
                                            tmem_region,
                                            UInt32(k_tile),
                                            0,
                                        )
                                    next_ready = True
                                    if k_tile + 1 < num_k_iters:
                                        next_ready = consumer.try_acquire()

        # ===== EPILOGUE WARPS =====
        if WarpRole1D1D.is_epilogue():
            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(accum_barriers, tmem, mma_complete_mask),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with epi_ctx:
                while True:
                    var ctx = work_iter.next()
                    if ctx.info.is_done():
                        break
                    if not ctx.info.is_valid():
                        continue

                    with epi_ctx.output_pipeline.consumer() as output_stage:
                        Self.epilogue(
                            c_tiles,
                            c_tma_op,
                            c_device,
                            output_stage,
                            ctx,
                        )

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
        work_ctx: GroupedWorkContext1D1D,
        a_scale_offsets: Self.AScaleOffsetsTile,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A, B, SFA, SFB tiles using TMA."""
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # M coordinate in contiguous token space
        var m_coord = work_ctx.m()
        var n_coord = work_ctx.n()
        var expert_id = work_ctx.expert_id()
        var group_idx = work_ctx.group_idx()

        var a_gmem_m_coord = peer_m_rank * UInt(Self.a_tma_rows) + UInt(m_coord)
        var b_gmem_n_coord = (
            peer_rank_m * UInt(Self.b_tma_rows)
            + peer_rank_n * UInt(Self.BN)
            + UInt(n_coord)
            + UInt(expert_id) * UInt(Self.static_N)
        )

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            @parameter
            for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor
                var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)

                # Peer CTA slice using TileTensor pattern (ptr + layout)
                var a_peer_tt = type_of(a_tt)(
                    a_tt.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
                    a_tt.layout,
                )
                var b_peer_tt = type_of(b_tt)(
                    b_tt.ptr + peer_rank_m * UInt(Self.b_tma_load_size),
                    b_tt.layout,
                )

                var k_coord = UInt(iter_idx + j) * UInt(Self.BK)

                # TileTensor directly to TMA (uses TileTensor overload)
                a_tma_op.async_multicast_load[Self.cta_group](
                    a_peer_tt,
                    barrier[0],
                    (k_coord, a_gmem_m_coord),
                    UInt16((1 << Self.CLUSTER_M) - 1),
                )
                b_tma_op.async_multicast_load[Self.cta_group](
                    b_peer_tt,
                    barrier[0],
                    (k_coord, b_gmem_n_coord),
                    UInt16((1 << Self.CLUSTER_N) - 1),
                )

                # Scale factor load with offset
                # TMA 4D now has TileTensor overload - pass tiles directly
                var a_scale_offset = rebind[Scalar[DType.uint32]](
                    a_scale_offsets[Int(group_idx)]
                )
                var sfa_m_coord = Int(m_coord) // SF_MN_GROUP_SIZE + Int(
                    a_scale_offset
                )
                sfa_tma_op.async_copy_4d[Self.cta_group](
                    sfa_tt,
                    barrier[0],
                    (
                        0,
                        0,
                        Int(
                            (iter_idx + j) * UInt32(Self.config.num_sf_k_tiles)
                        ),
                        sfa_m_coord,
                    ),
                )

                var sfb_n_coord = (
                    Int(n_coord) + Int(expert_id) * Self.static_N
                ) // SF_MN_GROUP_SIZE
                sfb_tma_op.async_copy_4d[Self.cta_group](
                    sfb_tt,
                    barrier[0],
                    (
                        0,
                        0,
                        Int(
                            (iter_idx + j) * UInt32(Self.config.num_sf_k_tiles)
                        ),
                        sfb_n_coord,
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
        tmem_region: Self.TmemRegion,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations."""
        if elect_one_sync():

            @parameter
            for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor
                var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)

                var tile_idx = (
                    Int(tiles.stage()) * Self.config.k_group_size + jj
                )

                var sfa_tmem_offset = UInt32(tmem_region.sfa(tile_idx).col_addr)
                var sfb_tmem_offset = UInt32(tmem_region.sfb(tile_idx).col_addr)

                var is_first_k = (iter_idx + j) == k_start

                # MMA has TileTensor overload - pass tiles directly
                # (layout is extracted from TileTensor type parameters)
                mma_op.mma(
                    a_tt,
                    b_tt,
                    sfa_tt,
                    sfb_tt,
                    tmem_addr,
                    sfa_tmem_offset,
                    sfb_tmem_offset,
                    init_c=is_first_k,
                )

            mma_op.commit(tiles.mbar())

    # ========== Epilogue ==========

    @staticmethod
    @always_inline
    fn epilogue(
        c_tiles: Self.SmemType.CTileArray,
        c_tma_op: Self.CTmaOp,
        c_device: Self.CDeviceTile,
        stage: Self.TileWriterType.Stage,
        work_ctx: GroupedWorkContext1D1D,
    ):
        """Execute epilogue to store accumulated results with expert_scale."""
        var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))

        # For 1D-1D, pass absolute coordinates directly (not tile indices)
        # to handle unaligned expert offsets correctly
        var m_abs = work_ctx.m()  # Absolute M in contiguous token space
        var n_abs = work_ctx.n()  # Absolute N in output space

        # Get problem dimensions
        # M is the end offset for current expert (used for bounds checking)
        var M = work_ctx.m_end
        var N = UInt32(Self.static_N)

        # Convert TileTensor to LayoutTensor at output_writer boundary
        var c_lt = c_device.to_layout_tensor()

        tile_writer.write_absolute_with_bounds_check[type_of(c_lt).layout](
            c_tiles,
            stage,
            m_abs,
            n_abs,
            M,  # m_end for bounds checking
            work_ctx.expert_scale,
            c_lt,
        )
