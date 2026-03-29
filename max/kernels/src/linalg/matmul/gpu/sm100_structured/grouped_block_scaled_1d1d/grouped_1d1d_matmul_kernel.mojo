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
- Warp specialization (Load, MMA, Epilogue, optional SFB Load)
- Grid-constant TMA descriptors (no runtime tensormap updates)
- Offset-based addressing via a_offsets for contiguous token buffers
- Per-expert output scaling via expert_scales tensor

Architecture (MMA_N >= 64, 192 threads):
- TMA warp: Loads A, B, SFA, SFB tiles using grid-constant TMAs
- MMA warp: Executes block-scaled matrix multiply (SFB via tcgen05_cp)
- Epilogue warps: Stores results with expert_scale applied

Architecture (MMA_N < 64, 352 threads):
- TMA warp: Loads A, B, SFA tiles using grid-constant TMAs
- MMA warp: Executes block-scaled matrix multiply
- Epilogue warps: Stores results with expert_scale applied
- SFB TMA Load warp: Loads SFB from GMEM to SMEM via TMA (reduced tile)
- SFB TMEM Load warps: Reads SFB from SMEM, writes to TMEM via tcgen05_st

This is a port of grouped_matmul_sm100_1d1d.mojo to the structured kernels
architecture.
"""

from std.collections import Optional
from std.math import align_up, ceildiv
from std.memory import Pointer, bitcast
from std.math.uutils import ufloordiv, umod
from std.sys import align_of, size_of

from std.gpu import (
    WARP_SIZE,
    block_id_in_cluster,
    lane_id_uint as lane_id,
    thread_idx_int as thread_idx,
)
from std.gpu.memory import (
    AddressSpace,
    external_memory,
    fence_mbarrier_init,
)
from std.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from std.gpu.sync import syncwarp
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_fence_before,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout.tma_async import PipelineState
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Coord,
    Idx,
    RowMajorLayout,
    RuntimeInt,
    TensorLayout,
    TileTensor,
    row_major,
)
from structured_kernels.tile_types import (
    GMEMLayout1D,
    TmaOpType,
    static_row_major,
    tma_desc_layout_4d,
)
from layout.tile_layout import Layout as TileLayout, _IntToComptimeInt

from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_BlockScaled_SS
from linalg.fp4_utils import SF_MN_GROUP_SIZE, SF_ATOM_M, SF_ATOM_K
from linalg.utils import elementwise_compute_lambda_type

from ..structured_kernels.config import (
    BlockScaledMatmulConfig,
    OutputPipelineConfig,
)
from structured_kernels.kernel_common import (
    WarpRole1D1D,
    compute_tma_tile_dims,
    compute_accum_barrier_counts,
    compute_input_consumer_count,
    init_core_barriers,
)
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    ProducerTiles,
    ConsumerTiles,
    OutputTilePipeline,
    BlockScaledTilePayload,
)
from structured_kernels.pipeline import ProducerConsumerPipeline
from ..structured_kernels.tmem import (
    BlockScaledTmem,
    TmemAllocation,
    TmemDeallocBarrier,
)
from structured_kernels.barriers import WarpGroupBarrier
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
# Grouped1D1DMatmulKernel - Main kernel struct
# =============================================================================


struct Grouped1D1DMatmulKernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    # C device layout (TensorLayout from caller's TileTensor)
    c_device_layout: TensorLayout,
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

    # SFB N dimension aligned up to SF_MN_GROUP_SIZE (e.g. 64 → 128).
    # Used for TMA descriptors and SMEM layout (both need the full SF group).
    comptime SFB_N_ALIGNED = align_up(Self.MMA_N, SF_MN_GROUP_SIZE)

    # TMEM configuration — stride matches MMA output width for scaled kernels.
    # SFB TMEM width must be SFB_N_ALIGNED (not MMA_N) because
    # _copy_sf_to_tmem_tt writes SF_MN_GROUP_SIZE//32 = 4 columns per
    # SF group regardless of MMA_N.  Matches the sm100/block_scaled kernel.
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = Self.config.num_sf_k_tiles * (Self.BM // 32)
    comptime SFB_NUM_COLS = Self.config.num_sf_k_tiles * (
        Self.SFB_N_ALIGNED // 32
    )
    comptime stage_stride_cols = Self.MMA_N

    # Output pipeline config (bundles accum stages, stride, and cta_group)
    comptime opc = OutputPipelineConfig(
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    )

    # ========== Barrier Arrival Counts ==========

    comptime _accum_barrier_counts = compute_accum_barrier_counts[
        WarpRole1D1D.NUM_EPILOGUE_THREADS, Self.cta_group
    ]()
    comptime accum_pipeline_producer_arv_count = Self._accum_barrier_counts[0]
    comptime accum_pipeline_consumer_arv_count = Self._accum_barrier_counts[1]

    # ========== Shared Memory Type ==========

    comptime SmemType = Grouped1D1DSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config=Self.config,
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
        accum_type=Self.accum_type,
        cta_group=Self.cta_group,
        cluster_shape=Self.config.cluster_shape,
        a_swizzle=Self.config.a_swizzle,
        b_swizzle=Self.config.b_swizzle,
        transpose_b=Self.transpose_b,
    ]

    # ========== Tile Pipeline Types ==========
    # TileTensor-native payload - passed directly to TMA/MMA

    comptime TilePayload = BlockScaledTilePayload[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        IndexList[2](
            Self.SmemType.Core.BM, Self.SmemType.Core.BK
        ),  # A tile shape
        IndexList[2](
            Self.SmemType.Core.BN, Self.SmemType.Core.BK
        ),  # B tile shape
        IndexList[2](
            Self.SmemType.Core.SFA_DIM0, Self.SmemType.Core.SFA_DIM1
        ),  # SFA shape
        IndexList[2](
            Self.SmemType.Core.SFB_DIM0, Self.SmemType.Core.SFB_DIM1
        ),  # SFB shape
        Self.SmemType.Core.num_pipeline_stages,
    ]

    comptime InputTilePipelineType = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.Core.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== TMEM and Output Pipeline Types ==========

    comptime Tmem = TmemAllocation[Self.opc.cta_group]

    comptime TmemRegion = BlockScaledTmem[
        Self.accum_type,
        Self.MMA_M,
        Self.MMA_N,
        Self.num_accum_pipeline_stages,
        Self.sfa_dtype,
        Self.BM,
        Self.num_pipeline_stages,
        cta_group=Self.cta_group,
        num_sf_k_tiles=Self.config.num_sf_k_tiles,
        SFB_N=Self.SFB_N_ALIGNED,
    ]

    comptime OutputPipeline = OutputTilePipeline[Self.opc]

    comptime TmemDealloc = TmemDeallocBarrier[Self.opc.cta_group]

    # ========== Warp Context Types ==========

    comptime MmaEpilogueSync = WarpGroupBarrier[
        WarpRole1D1D.NUM_MMA_THREADS + WarpRole1D1D.NUM_EPILOGUE_THREADS, 1
    ]

    # Barrier for MMA+SFB sync (MMA_N < 64 only, barrier_id=2)
    # SFB load warps wait for MMA to allocate TMEM before reading the address.
    comptime MmaSfbSync = WarpGroupBarrier[
        WarpRole1D1D.NUM_MMA_THREADS + WarpRole1D1D.NUM_SFB_LOAD_THREADS, 2
    ]

    comptime MmaCtx = MmaWarpContext[
        Self.opc,
        WarpRole1D1D.NUM_MMA_THREADS,
        WarpRole1D1D.NUM_EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.opc,
        WarpRole1D1D.NUM_MMA_THREADS,
        WarpRole1D1D.NUM_EPILOGUE_THREADS,
    ]

    # ========== Tile Writer Type ==========

    comptime TileWriterType = TileWriter[
        a_type=Self.a_type,
        accum_type=Self.accum_type,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        opc=Self.opc,
        c_swizzle=Self.config.c_swizzle,
        transpose_c=Self.config.AB_swapped,
        c_smem_dim0=Self.SmemType.Core.OutputM,
        c_smem_dim1=Self.SmemType.Core.OutputN,
        num_output_stages=Self.config.num_output_stages,
        num_output_warps=Self.num_output_warps,
        batched=False,  # 1D-1D uses 2D coordinates with bounds checking
        problem_n=Self.static_N,
    ]

    # ========== Work Iterator Type ==========

    comptime WorkIterator = GroupedWorkIterator1D1D[
        static_N=Self.static_N,
        tile_shape=Self.config.block_tile_shape,
        cluster=Self.config.cluster_shape,
        cta_group=Self.cta_group,
        AB_swapped=Self.config.AB_swapped,
    ]

    # ========== TMA Load Size Constants ==========

    comptime a_expected_bytes = Self.BM * Self.BK * size_of[Self.a_type]()
    comptime b_expected_bytes = Self.BN * Self.BK * size_of[Self.b_type]()
    comptime sfa_expected_bytes = Self.SmemType.Core.sfa_smem_layout.size() * size_of[
        Self.sfa_dtype
    ]()
    comptime sfb_expected_bytes = Self.SmemType.Core.sfb_smem_layout.size() * size_of[
        Self.sfb_dtype
    ]()

    # For MMA_N < 64, SFB is loaded by the dedicated SfbTMALoad warp
    # on a separate pipeline, so exclude it from the input pipeline bytes.
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.sfa_expected_bytes
        + (Self.sfb_expected_bytes if Self.MMA_N >= 64 else 0)
    ) * Self.config.k_group_size

    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime _tma_tile_dims = compute_tma_tile_dims[
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.OutputM,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
        Self.cta_group,
        AB_swapped=Self.config.AB_swapped,
    ]()
    comptime a_tile_dim0 = Self._tma_tile_dims[0]
    comptime b_tile_dim0 = Self._tma_tile_dims[1]
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
    comptime c_tile_dim0 = Self._tma_tile_dims[2]
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
    # When c_swizzle is SWIZZLE_NONE (MMA_N=8), c_swizzle_elems is 0.
    # The TMA descriptor dim1 must equal the tile dim1 in that case.
    comptime c_desc_dim1 = Self.c_tile_dim1 if Self.c_swizzle_elems == 0 else Self.c_swizzle_elems
    comptime CDescLayout = static_row_major[Self.c_tile_dim0, Self.c_desc_dim1]

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
    # SFB TMA tile: for MMA_N < 64 load 1 k-atom at a time with only
    # MMA_N rows; for MMA_N >= 64 unchanged (full atom, all k-atoms).
    comptime SFB_TMA_ROWS = Self.MMA_N if Self.MMA_N < SF_ATOM_M[
        0
    ] else SF_ATOM_M[0]
    comptime SFB_TMA_K_ATOMS = 1 if Self.MMA_N < 64 else Self.config.num_sf_k_tiles
    comptime SFBTileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            Self.SFB_N_ALIGNED // SF_MN_GROUP_SIZE,
            Self.SFB_TMA_K_ATOMS,
            Self.SFB_TMA_ROWS,
            SF_ATOM_M[1] * SF_ATOM_K,
        ]
    ]
    comptime SFBDescLayout = tma_desc_layout_4d[
        Self.sfb_dtype,
        Self.SFB_N_ALIGNED // SF_MN_GROUP_SIZE,
        Self.SFB_TMA_K_ATOMS,
        Self.SFB_TMA_ROWS,
        TensorMapSwizzle.SWIZZLE_NONE,
    ]

    # TMA operation types
    comptime ATmaOp = TmaOpType[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime BTmaOp = TmaOpType[Self.b_type, Self.BTileLayout, Self.BDescLayout]
    comptime CTmaOp = TmaOpType[Self.c_type, Self.CTileLayout, Self.CDescLayout]
    comptime SFATmaOp = TmaOpType[
        Self.sfa_dtype, Self.SFATileLayout, Self.SFADescLayout
    ]
    comptime SFBTmaOp = TmaOpType[
        Self.sfb_dtype, Self.SFBTileLayout, Self.SFBDescLayout
    ]

    # 1D data TileTensor types (offsets, expert IDs, scales)
    comptime OffsetsTile = TileTensor[DType.uint32, GMEMLayout1D, MutAnyOrigin]
    comptime AScaleOffsetsTile = TileTensor[
        DType.uint32, GMEMLayout1D, MutAnyOrigin
    ]
    comptime ExpertIdsTile = TileTensor[DType.int32, GMEMLayout1D, MutAnyOrigin]
    comptime ExpertScalesTile = TileTensor[
        DType.float32, GMEMLayout1D, MutAnyOrigin
    ]

    # C device tensor type (for bounds-checked stores)
    comptime CDeviceTile = TileTensor[
        Self.c_type, Self.c_device_layout, MutAnyOrigin
    ]

    # TMA load size constants (from desc layout dimensions)
    comptime a_tma_load_size = Self.a_tile_dim0 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim0 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim0
    comptime b_tma_rows = Self.b_tile_dim0

    # ========== Validation ==========

    @staticmethod
    def validate_config():
        """Compile-time validation of kernel configuration."""
        comptime assert (
            Self.a_type == Self.b_type
        ), "A and B types must match for block-scaled GEMM"
        comptime assert (
            Self.sfa_dtype == Self.sfb_dtype
        ), "SFA and SFB types must match"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "Only support cta_group == 1 or 2"
        comptime assert Self.transpose_b, "Only support transposed B"
        comptime if Self.MMA_N < 64:
            comptime assert (
                Self.cta_group == 1
            ), "MMA_N < 64 cooperative SFB loading requires cta_group=1"

    # ========== Static Helper Methods ==========

    @staticmethod
    @always_inline
    def init_barriers(
        elect_one_warp: Bool,
        elect_one_thread: Bool,
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c_tma_op: Self.CTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
    ):
        """Initialize barriers and prefetch TMA descriptors."""
        if elect_one_warp and elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            c_tma_op.prefetch_descriptor()
            sfa_tma_op.prefetch_descriptor()
            sfb_tma_op.prefetch_descriptor()

            init_core_barriers[
                Self.num_group_pipeline_stages,
                Self.num_accum_pipeline_stages,
            ](
                input_barriers.ptr,
                Int32(
                    compute_input_consumer_count[
                        Self.CLUSTER_M, Self.CLUSTER_N, Self.cta_group
                    ]()
                ),
                accum_barriers.ptr,
                Int32(Self.accum_pipeline_producer_arv_count),
                Int32(Self.accum_pipeline_consumer_arv_count),
                tmem_dealloc.ptr,
                Int32(WarpRole1D1D.NUM_EPILOGUE_THREADS * Self.cta_group),
            )

        fence_mbarrier_init()
        cluster_sync()

    # ========== Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_op, `nvvm.grid_constant`)
    def run(
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
            address_space=AddressSpace.SHARED,
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
        var elect_one_warp = ufloordiv(thread_idx.x, WARP_SIZE) == 0
        var elect_one_thread = elect_one_sync_with_mask()
        var elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )

        # CTA coordinates in cluster (matches KernelContext pattern)
        var rank_m = Int(block_id_in_cluster.x)
        var rank_n = Int(block_id_in_cluster.y)

        # Peer CTA coordinates: (peer_id, mma_coord_m, mma_coord_n)
        # Following KernelContext convention:
        #   [0] = rank_m % cta_group  (peer ID within CTA group)
        #   [1] = rank_m // cta_group (MMA coordinate in M)
        #   [2] = rank_n              (MMA coordinate in N)
        var peer_cta_coord = (
            umod(rank_m, Self.cta_group),
            ufloordiv(rank_m, Self.cta_group),
            rank_n,
        )

        # Per-CTA multicast masks (following KernelContext)
        var a_multicast_mask = UInt16(0)

        comptime for i in range(Self.CLUSTER_N):
            a_multicast_mask |= UInt16(1 << (i * Self.CLUSTER_M))
        a_multicast_mask <<= UInt16(rank_m)

        var b_multicast_mask = UInt16(0)

        comptime for i in range(Self.CLUSTER_M // Self.cta_group):
            b_multicast_mask |= UInt16(1 << (i * Self.cta_group))
        b_multicast_mask <<= UInt16(umod(rank_m, Self.cta_group))
        b_multicast_mask <<= UInt16(rank_n * Self.CLUSTER_M)

        var mma_complete_mask = UInt16((1 << Self.cta_group) - 1)

        # K iteration count
        var num_k_iters = ceildiv(Int(K), Self.BK)

        # ===== Barrier Initialization =====
        Self.init_barriers(
            elect_one_warp,
            elect_one_thread,
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            input_barriers,
            accum_barriers,
            smem.pipelines.tmem_dealloc(),
        )

        # Init SFB load barriers (MMA_N < 64 only).
        # Separate from init_barriers to avoid changing its interface.
        comptime if Self.MMA_N < 64:
            if elect_one_warp and elect_one_thread:
                var sfb_mbars_ptr = smem.sfb_load_mbars_ptr()
                comptime for i in range(Self.num_group_pipeline_stages):
                    sfb_mbars_ptr[i].init(
                        Int32(WarpRole1D1D.NUM_SFB_LOAD_THREADS)
                    )

                # Init SFB TMA pipeline barriers (SfbTMALoad ↔ MMA).
                # Producer count = 1 (SfbTMALoad expect_bytes + TMA auto-arrive).
                # Consumer count = 1 (MMA arrive after consuming TMEM data).
                ProducerConsumerPipeline[Self.num_group_pipeline_stages](
                    smem.sfb_tma_mbars_ptr()
                ).init_mbars(Int32(1), Int32(1))

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
        # For cta_group=2: BOTH CTAs run the production loop to keep
        # pipeline state in sync.  UMMA multicast arrives on both
        # CTAs' EMPTY barriers, so both must advance through stages
        # to match.  Inside load_input_tiles, elect_one_cta gates
        # expect_bytes and the cta_group parameter on TMA ops ensures
        # only the leader CTA issues loads.
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
                                a_multicast_mask,
                                b_multicast_mask,
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
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, mma_complete_mask
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            # Signal SFB load warps that TMEM is allocated
            comptime if Self.MMA_N < 64:
                Self.MmaSfbSync.arrive()

            # SFB pipeline state: tracks sfb_load_mbar phase for MMA_N<64.
            # For MMA_N>=64, SFB is loaded via tcgen05_cp inside mma_op.mma().
            var sfb_mbars = smem.sfb_load_mbars_ptr()
            var sfb_pipe_state = PipelineState[Self.num_group_pipeline_stages]()

            # SFB TMA pipeline: MMA is the consumer that signals the
            # SfbTMALoad warp when SMEM is free for reuse.
            var sfb_tma_pipeline = ProducerConsumerPipeline[
                Self.num_group_pipeline_stages
            ](smem.sfb_tma_mbars_ptr())

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

                            var sfb_tmem_adj = Self._compute_sfb_tmem_adj(
                                ctx.m(), ctx.n(), ctx.m_start()
                            )

                            with input_pipeline.consumer() as consumer:
                                var next_ready = True
                                if num_k_iters > 0:
                                    next_ready = consumer.try_acquire()

                                for k_tile in range(num_k_iters):
                                    with consumer.acquire_if_needed(
                                        next_ready
                                    ) as input_tiles:
                                        # Wait for SFB load warps to
                                        # finish writing SFB to TMEM.
                                        comptime if Self.MMA_N < 64:
                                            sfb_mbars[
                                                sfb_pipe_state.index()
                                            ].wait(sfb_pipe_state.phase())

                                        Self.mma(
                                            input_tiles,
                                            mma_op,
                                            tmem_offset,
                                            tmem_region,
                                            UInt32(k_tile),
                                            0,
                                            sfb_tmem_adj,
                                        )

                                        comptime if Self.MMA_N < 64:
                                            sfb_pipe_state.step()

                                            # Signal SfbTMALoad that SMEM
                                            # slot is consumed and can be
                                            # reused.
                                            if elect_one_sync():
                                                _ = sfb_tma_pipeline.consumer_mbar(
                                                    sfb_tma_pipeline.consumer_stage()
                                                )[
                                                    0
                                                ].arrive()
                                            sfb_tma_pipeline.consumer_step()
                                    next_ready = True
                                    if k_tile + 1 < num_k_iters:
                                        next_ready = consumer.try_acquire()

        # ===== EPILOGUE WARPS =====
        if WarpRole1D1D.is_epilogue():
            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, mma_complete_mask
                ),
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

        # ===== SFB TMA LOAD WARP (MMA_N < 64 only) =====
        # Dedicated warp that loads SFB scale factors from GMEM to SMEM
        # via TMA, issuing one copy per k-atom with reduced tile height
        # (MMA_N rows instead of full SF_ATOM_M[0]=32 rows).
        comptime if Self.MMA_N < 64:
            if WarpRole1D1D.is_sfb_tma_load():
                var sfb_tma_pipeline = ProducerConsumerPipeline[
                    Self.num_group_pipeline_stages
                ](smem.sfb_tma_mbars_ptr())

                # Bytes per k-group iteration: all k-atoms × reduced tile.
                comptime ROW_STRIDE = SF_ATOM_M[1] * SF_ATOM_K
                comptime sfb_single_copy_bytes = (
                    (Self.SFB_N_ALIGNED // SF_MN_GROUP_SIZE)
                    * Self.SFB_TMA_ROWS
                    * ROW_STRIDE
                    * size_of[Self.sfb_dtype]()
                )
                comptime sfb_tma_expected_bytes = (
                    Self.config.num_sf_k_tiles
                    * sfb_single_copy_bytes
                    * Self.config.k_group_size
                )

                # Layout mapping (k_atom, row) → flat SMEM offset within atom.
                comptime sfb_atom_layout = TileLayout(
                    Coord(
                        Idx[Self.config.num_sf_k_tiles](),
                        Idx[SF_ATOM_M[0]](),
                    ),
                    Coord(
                        Idx[SF_ATOM_M[0] * ROW_STRIDE](),
                        Idx[ROW_STRIDE](),
                    ),
                )

                while True:
                    var ctx = work_iter.next()
                    if ctx.info.is_done():
                        break
                    if not ctx.info.is_valid():
                        continue

                    # Hoist loop-invariant SF coords outside k_tile loop.
                    # These depend only on the work context, not k_tile.
                    var row_in_atom: Int = 0
                    var sfb_n_coord: Int = 0
                    if elect_one_sync():
                        comptime if Self.config.AB_swapped:
                            row_in_atom = (
                                Int(ctx.m()) - Int(ctx.m_start())
                            ) % SF_ATOM_M[0]
                        else:
                            row_in_atom = Int(ctx.n()) % SF_ATOM_M[0]

                        var a_scale_offset = rebind[Scalar[DType.uint32]](
                            a_scale_offsets[Int(ctx.group_idx())]
                        )
                        _, sfb_n_coord = Self._get_sf_coords(
                            ctx.m(),
                            ctx.n(),
                            ctx.expert_id(),
                            a_scale_offset,
                            ctx.m_start(),
                        )

                    for k_tile in range(num_k_iters):
                        sfb_tma_pipeline.wait_consumer()
                        var stage = sfb_tma_pipeline.producer_stage()
                        var sfb_tma_mbar = sfb_tma_pipeline.producer_mbar(stage)

                        if elect_one_sync():
                            if elect_one_cta:
                                sfb_tma_mbar[0].expect_bytes(
                                    Int32(sfb_tma_expected_bytes)
                                )

                            comptime for kg in range(Self.config.k_group_size):
                                var offset = stage * UInt32(
                                    Self.config.k_group_size
                                ) + UInt32(kg)
                                var sfb_smem_tile = sfb_tiles[offset]

                                comptime for k_atom in range(
                                    Self.config.num_sf_k_tiles
                                ):
                                    var smem_offset = Int(
                                        sfb_atom_layout(
                                            Coord(
                                                Idx[k_atom](),
                                                RuntimeInt(
                                                    Scalar[DType.int64](
                                                        row_in_atom
                                                    )
                                                ),
                                            )
                                        )
                                    )

                                    var atom_dst = TileTensor(
                                        sfb_smem_tile.ptr + smem_offset,
                                        row_major[
                                            Self.SFB_TMA_ROWS, ROW_STRIDE
                                        ](),
                                    )

                                    var k_tile_base = (
                                        Int(
                                            UInt32(k_tile)
                                            * UInt32(Self.config.k_group_size)
                                            + UInt32(kg)
                                        )
                                        * Self.config.num_sf_k_tiles
                                    )

                                    sfb_tma_op.async_copy_4d[Self.cta_group](
                                        atom_dst,
                                        sfb_tma_mbar[0],
                                        (
                                            0,
                                            row_in_atom,
                                            k_tile_base + k_atom,
                                            sfb_n_coord,
                                        ),
                                    )

                        sfb_tma_pipeline.producer_step()

                # Drain: prevent exit while SfbTMEMLoad is still working.
                comptime for i in range(Self.num_group_pipeline_stages):
                    sfb_tma_pipeline.wait_consumer()
                    sfb_tma_pipeline.producer_step()

        # ===== SFB TMEM LOAD WARPS (MMA_N < 64 only) =====
        # Dedicated warps that read SFB scale factors from SMEM and
        # write them to TMEM via tcgen05_st.  Wait on the sfb_tma_pipeline
        # for TMA loads to complete before reading SMEM.
        comptime if Self.MMA_N < 64:
            if WarpRole1D1D.is_sfb_load():
                # Wait for MMA warp to allocate TMEM
                Self.MmaSfbSync.wait()

                var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
                var tmem_region = Self.TmemRegion(tmem)

                # Track the sfb_tma_pipeline (SfbTMALoad→SfbTMEMLoad).
                var sfb_input_pipeline = ProducerConsumerPipeline[
                    Self.num_group_pipeline_stages
                ](smem.sfb_tma_mbars_ptr())
                var sfb_pipe_state = PipelineState[
                    Self.num_group_pipeline_stages
                ]()
                var sfb_mbars = smem.sfb_load_mbars_ptr()

                while True:
                    var ctx = work_iter.next()
                    if ctx.info.is_done():
                        break
                    if not ctx.info.is_valid():
                        continue

                    for k_tile in range(num_k_iters):
                        var stage = sfb_input_pipeline.consumer_stage()
                        sfb_input_pipeline.wait_producer()

                        Self._sfb_load_to_tmem(
                            sfb_tiles,
                            tmem_region,
                            stage,
                            UInt32(k_tile),
                            ctx,
                        )

                        _ = sfb_mbars[sfb_pipe_state.index()].arrive()
                        sfb_input_pipeline.consumer_step()
                        sfb_pipe_state.step()

    # ========== SFB Load to TMEM (MMA_N < 64) ==========

    @staticmethod
    @always_inline
    def _sfb_load_to_tmem(
        sfb_tiles: Self.SmemType.Core.SFBTileArray,
        tmem_region: Self.TmemRegion,
        stage: UInt32,
        k_tile: UInt32,
        work_ctx: GroupedWorkContext1D1D,
    ):
        """Load SFB scale factors from SMEM to TMEM via tcgen05_st.

        Matches the SFB load pattern from block_scaled_matmul_small_bn.mojo.
        Each of the 4 SFB load warps (128 threads) covers 32 datapaths via
        tcgen05_st[datapaths=32].  Only lanes 0..MMA_N-1 read valid data;
        others write zero (harmless — UMMA only reads dp 0..MMA_N-1).
        """
        comptime k_group_size = Self.config.k_group_size

        comptime for kg in range(k_group_size):
            var offset = stage * UInt32(k_group_size) + UInt32(kg)

            # SFB SMEM tile at this pipeline offset
            var sfb_smem_tile = sfb_tiles[offset]
            var sfb_tmem_offset = UInt32(tmem_region.sfb(Int(offset)).col_addr)

            comptime SFB_TILE_BYTES = SF_ATOM_M[0] * SF_ATOM_M[
                1
            ] * SF_ATOM_K * size_of[Self.sfb_dtype]()

            comptime for sf_idx in range(Self.config.num_sf_k_tiles):
                var sfb_scales = SIMD[Self.sfb_dtype, SF_ATOM_K]()
                if lane_id() < UInt(Self.MMA_N):
                    # Compute N-position within SF group.
                    # work_ctx.n() is in element space (not tile index),
                    # so no multiplication by MMA_N needed.
                    var outer: UInt
                    comptime if Self.config.AB_swapped:
                        # AB_swapped: N position is derived from M
                        var m_in_group = UInt(work_ctx.m()) - UInt(
                            work_ctx.m_start()
                        )
                        outer = m_in_group % UInt(SF_MN_GROUP_SIZE) + lane_id()
                    else:
                        outer = (
                            UInt(work_ctx.n()) % UInt(SF_MN_GROUP_SIZE)
                            + lane_id()
                        )

                    var scales_offset = (
                        UInt(sf_idx) * UInt(SFB_TILE_BYTES)
                        + (outer % UInt(SF_ATOM_M[0]))
                        * (UInt(SF_ATOM_M[1]) * UInt(SF_ATOM_K))
                        + (outer / UInt(SF_ATOM_M[0])) * UInt(SF_ATOM_K)
                    )
                    sfb_scales = sfb_smem_tile.ptr.load[
                        width=SF_ATOM_K,
                        alignment=align_of[SIMD[Self.sfb_dtype, SF_ATOM_K]](),
                    ](scales_offset)

                syncwarp()

                var _sfb_st = InlineArray[Scalar[DType.uint32], 1](
                    uninitialized=True
                )
                _sfb_st[0] = bitcast[DType.uint32, 1](sfb_scales)[0]
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](
                    sfb_tmem_offset + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    _sfb_st,
                )
                tcgen05_store_wait()
                tcgen05_fence_before()

    @staticmethod
    @always_inline
    def _get_sf_coords(
        m_coord: UInt32,
        n_coord: UInt32,
        expert_id: Int32,
        a_scale_offset: Scalar[DType.uint32],
        m_start: UInt32,
    ) -> Tuple[Int, Int]:
        """Return (sfa_m_coord, sfb_n_coord), swapped when AB_swapped."""
        var expert_sf_coord = (
            Int(expert_id) * ceildiv(Self.static_N, SF_MN_GROUP_SIZE)
            + Int(n_coord) // SF_MN_GROUP_SIZE
        )

        # Use expert-relative position for the SF group index to avoid
        # incorrect floor-division when MMA_N < SF_MN_GROUP_SIZE and the
        # expert starts at a non-SF_MN_GROUP_SIZE-aligned token offset.
        # (m_coord - m_start) // G gives the correct per-expert group;
        # m_start // G + a_scale_offset restores the absolute SF row.
        var token_sf_coord = (
            Int(m_coord - m_start) // SF_MN_GROUP_SIZE
            + Int(m_start) // SF_MN_GROUP_SIZE
            + Int(a_scale_offset)
        )

        comptime if Self.config.AB_swapped:
            return (expert_sf_coord, token_sf_coord)
        else:
            return (token_sf_coord, expert_sf_coord)

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    def load_input_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_ctx: GroupedWorkContext1D1D,
        a_scale_offsets: Self.AScaleOffsetsTile,
        iter_idx: UInt32,
        elect_one_cta: Bool,
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
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

        var a_gmem_m_coord: Int
        var b_gmem_n_coord: Int

        comptime if Self.config.AB_swapped:
            # A loads weights (b_device): per-CTA weight row offset.
            # peer_rank_n differentiates CTAs in the weight dimension
            # (0 for CTA0, 1 for CTA1 in a 2SM cluster).
            # Each CTA provides BM weight rows to the UMMA.
            a_gmem_m_coord = (
                peer_rank_n * Self.BM
                + Int(n_coord)
                + Int(expert_id) * Self.static_N
            )
            # B loads tokens (a_device): per-CTA token offset.
            # The UMMA combines B rows from both CTAs, so each CTA
            # loads a different portion of the token range.
            b_gmem_n_coord = (
                peer_rank_m * Self.b_tma_rows
                + peer_rank_n * Self.BN
                + Int(m_coord)
            )
        else:
            # Normal: A loads tokens, B loads weights
            a_gmem_m_coord = peer_m_rank * Self.a_tma_rows + Int(m_coord)
            b_gmem_n_coord = (
                peer_rank_m * Self.b_tma_rows
                + peer_rank_n * Self.BN
                + Int(n_coord)
                + Int(expert_id) * Self.static_N
            )

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            comptime for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor
                var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)

                # Peer CTA slice using TileTensor pattern (ptr + layout)
                var a_peer_tt = type_of(a_tt)(
                    a_tt.ptr + peer_m_rank * Self.a_tma_load_size,
                    a_tt.layout,
                )
                var b_peer_tt = type_of(b_tt)(
                    b_tt.ptr + peer_rank_m * Self.b_tma_load_size,
                    b_tt.layout,
                )

                var k_coord = Int(iter_idx + j) * Self.BK

                # TileTensor directly to TMA (uses TileTensor overload)
                a_tma_op.async_multicast_load[Self.cta_group](
                    a_peer_tt,
                    barrier[0],
                    (k_coord, a_gmem_m_coord),
                    a_multicast_mask,
                )
                b_tma_op.async_multicast_load[Self.cta_group](
                    b_peer_tt,
                    barrier[0],
                    (k_coord, b_gmem_n_coord),
                    b_multicast_mask,
                )

                # Scale factor load with offset
                # TMA 4D now has TileTensor overload - pass tiles directly
                var a_scale_offset = rebind[Scalar[DType.uint32]](
                    a_scale_offsets[Int(group_idx)]
                )

                var sfa_m_coord: Int
                var sfb_n_coord: Int
                # For AB_swapped 2SM, each CTA needs scale factors
                # for its own weight rows. Add per-CTA offset (BM)
                # to the weight coordinate used for SFA lookup.
                var n_coord_sf = n_coord
                comptime if Self.config.AB_swapped:
                    n_coord_sf = UInt32(Int(n_coord) + peer_rank_n * Self.BM)

                sfa_m_coord, sfb_n_coord = Self._get_sf_coords(
                    m_coord,
                    n_coord_sf,
                    expert_id,
                    a_scale_offset,
                    work_ctx.m_start(),
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

                # For MMA_N < 64, SFB is loaded by the SfbTMALoad warp.
                comptime if Self.MMA_N >= 64:
                    sfb_tma_op.async_copy_4d[Self.cta_group](
                        sfb_tt,
                        barrier[0],
                        (
                            0,
                            0,
                            Int(
                                (iter_idx + j)
                                * UInt32(Self.config.num_sf_k_tiles)
                            ),
                            sfb_n_coord,
                        ),
                    )

    # ========== MMA Operation ==========

    @staticmethod
    @always_inline
    def _compute_sfb_tmem_adj(
        m_coord: UInt32, n_coord: UInt32, m_start: UInt32
    ) -> UInt32:
        """Compute SFB TMEM column adjustment for MMA_N < SF_MN_GROUP_SIZE.

        When MMA_N reads exactly 2 TMEM columns (MMA_N=64 or 192), one SF
        group (128 elements = 4 TMEM columns) spans two adjacent tiles.
        The adjustment selects the correct half.

        For MMA_N < 64, each SF atom covers 32 N positions in one TMEM
        column.  The adj selects which column (atom) within the 128-N SF
        group to read: adj = n_in_sf_group // SF_ATOM_M[0].

        Divides by MMA_N (not BN): in 2SM mode BN = MMA_N/2, but both CTAs
        must supply the same adj because the paired UMMA distributes SF data
        internally.
        """
        comptime if Self.MMA_N < 64:
            # SFB is loaded externally to TMEM via dedicated SFB load
            # warps using tcgen05_st.  Data is placed at dp 0..MMA_N-1
            # of the base TMEM column, so no adjustment is needed.
            return UInt32(0)
        elif Self.MMA_N % SF_MN_GROUP_SIZE != 0:
            var effective_n: Int
            comptime if Self.config.AB_swapped:
                effective_n = Int(m_coord) - Int(m_start)
            else:
                effective_n = Int(n_coord)
            return UInt32(effective_n // Self.MMA_N % 2) * 2
        else:
            return UInt32(0)

    @staticmethod
    @always_inline
    def mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tiles: ConsumerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: Self.MmaOp,
        tmem_addr: UInt32,
        tmem_region: Self.TmemRegion,
        iter_idx: UInt32,
        k_start: UInt32,
        sfb_tmem_adj: UInt32,
    ):
        """Execute MMA operations.

        For MMA_N >= 64: SFB is loaded to TMEM via tcgen05_cp inside
        mma_op.mma().
        For MMA_N < 64: SFB is pre-loaded by dedicated SFB load warps
        via tcgen05_st. The MMA warp waits on sfb_load_mbars before
        entering this function.
        """
        if elect_one_sync():
            comptime for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)
                var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)
                var tile_idx = (
                    Int(tiles.stage()) * Self.config.k_group_size + jj
                )
                var sfa_tmem_offset = UInt32(tmem_region.sfa(tile_idx).col_addr)
                var sfb_tmem_offset = UInt32(tmem_region.sfb(tile_idx).col_addr)
                var is_first_k = (iter_idx + j) == k_start
                mma_op.mma(
                    a_tt,
                    b_tt,
                    sfa_tt,
                    sfb_tt,
                    tmem_addr,
                    sfa_tmem_offset,
                    sfb_tmem_offset,
                    init_c=is_first_k,
                    sfb_tmem_adj=sfb_tmem_adj,
                )
            mma_op.commit(tiles.mbar())

    # ========== Epilogue ==========

    @staticmethod
    @always_inline
    def epilogue(
        c_tiles: Self.SmemType.Core.CTileArray,
        c_tma_op: Self.CTmaOp,
        c_device: Self.CDeviceTile,
        stage: Self.TileWriterType.Stage,
        work_ctx: GroupedWorkContext1D1D,
    ):
        """Execute epilogue to store accumulated results with expert_scale."""
        var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))

        # For 1D-1D, pass absolute coordinates directly (not tile indices)
        # to handle unaligned expert offsets correctly.
        # m_abs = token offset, n_abs = weight offset, m_end = token boundary.
        # When transpose_c (AB_swapped), the writer handles the coordinate
        # swap internally.

        # For AB_swapped 2SM, each CTA computes different weight rows.
        # Add per-CTA offset (BM) to the N (weight) coordinate so each CTA
        # writes its portion: CTA0 writes n..n+BM-1, CTA1 writes n+BM..n+2*BM-1.
        var n_abs = work_ctx.n()
        comptime if Self.config.AB_swapped and Self.cta_group > 1:
            var rank_m = Int(block_id_in_cluster.x)
            var cta_n_offset = umod(rank_m, Self.cta_group) * Self.BM
            n_abs = UInt32(Int(n_abs) + cta_n_offset)

        tile_writer.write_absolute_with_bounds_check[Self.c_device_layout](
            c_tiles,
            stage,
            work_ctx.m(),  # Absolute M in contiguous token space
            n_abs,  # Absolute N in output space (per-CTA for 2SM)
            work_ctx.m_end,  # Token dim end for bounds checking
            work_ctx.expert_scale,
            c_device,
        )
