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

"""SM100 Matmul Kernel Structs - GPU kernel entry points and helpers.

This module contains the GPU kernel structs for SM100 matmul:
- WarpRole: Warp specialization roles (MMA, Load, Scheduler, Epilogue)
- KernelContext: Common kernel state (election vars, CTA coords, masks)
- B200MatmulSmem: Shared memory layout for the kernel
- BlackwellMatmulSM100Kernel: Main kernel struct with run() and run_splitk()
- BlackwellMatmulSM100FallbackKernel: Simple fallback kernel
- consumer_main_loop: MMA consumer loop (for external callers)

Output pipeline functions (copy_accum_to_gmem, multi_stage_store_C) are in
matmul_output.mojo.

The kernel implements a warp-specialized architecture:
- Scheduler warp: CLC-based tile scheduling
- TMA Load warp: Async memory transfers
- MMA warp: Tensor core operations with TMEM accumulators
- Epilogue warps: Output from TMEM to GMEM (see matmul_output.mojo)
"""

from collections import OptionalReg
from math import ceildiv
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, *_, **_]
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
from gpu.sync import (
    named_barrier,
    named_barrier_arrive,
    syncwarp,
    umma_arrive_leader_cta,
    mbarrier_arrive,
)
from gpu.tcgen05 import *
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

from ....arch.sm100 import MmaOpSM100_SS
from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ..sm100.config import MatmulConfig
from .pipeline import ProducerConsumerPipeline
from .tile_pipeline import (
    TilePipeline,
    ProducerStage,
    ConsumerStage,
    OutputTilePipeline,
    OutputStage,
)
from .tile_loader import TileLoaderTMA
from .tile_scheduler import TileScheduler
from .tile_scheduler_splitk import (
    TileScheduler as TileSchedulerSplitK,
    WorkInfo as WorkInfoSplitK,
)
from .tile_writer import (
    EpilogueConfig,
    store_fragment_to_smem,
)
from ....structuring import (
    NVIDIASharedMemoryManager as SharedMemoryManager,
    SMemPtr,
    SMemTileType,
    SMemTileIter,
    SMemTileArrayType,
)
from ..profiler import MatmulProfileWarp

# Import output pipeline functions from matmul_output module
from .matmul_output import (
    accum_arrive,
    copy_accum_to_gmem,
    multi_stage_store_C,
    multi_stage_store_C_split_k,
)


# =============================================================================
# WarpRole - Warp specialization roles
# =============================================================================


@fieldwise_init
@register_passable("trivial")
struct WarpRole(ImplicitlyCopyable, Movable):
    """Warp role identifiers for SM100 warp-specialized kernel."""

    var _role: Int32

    comptime Mma = Self(6)
    comptime MainLoad = Self(5)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    fn __eq__(self, other: UInt) -> Bool:
        return self._role == other

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    fn __ge__(self, other: UInt) -> Bool:
        return self._role >= other

    @staticmethod
    @always_inline
    fn is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()

    @staticmethod
    @always_inline
    fn is_scheduler() -> Bool:
        return Self.Scheduler == get_warp_id()


# =============================================================================
# KernelContext - Common state for kernel entry points
# =============================================================================


struct KernelContext[
    num_clc_pipeline_stages: Int,
    cta_group: Int,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](Copyable, Movable):
    """Shared kernel state: election vars, CTA coords, multicast masks, pipeline states.
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

    # Note: Pipeline states (producer and consumer) are now managed by
    # SchedulerWorkIterator and WorkIterator respectively.

    # ===== TMEM Pointer =====
    var ptr_tmem_addr: SMemPtr[UInt32]

    @always_inline
    fn __init__(out self, ptr_tmem_addr: SMemPtr[UInt32]):
        """Initialize context from TMEM pointer; computes all derived state."""
        # Election variables
        self.warp_id = get_warp_id()
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

        self.a_multicast_mask <<= self.rank_m
        self.b_multicast_mask <<= self.peer_cta_coord[0]
        self.b_multicast_mask <<= self.rank_n * UInt(Self.CLUSTER_M)

        # MMA completion mask for barrier synchronization
        var self_mask = 1 << Int(block_rank_in_cluster())
        var peer_mask = 1 << Int(block_rank_in_cluster() + 1)
        self.mma_complete_mask = self_mask | peer_mask

        # TMEM pointer
        self.ptr_tmem_addr = ptr_tmem_addr


# =============================================================================
# consumer_main_loop - MMA consumer loop (external API)
# =============================================================================


# NOTE: Used by warp_specialized_blockwise_fp8.mojo
@always_inline
fn consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = Index(1, 1, 1),
    k_group_size: UInt = 1,
](
    tmem_addr: UInt32,
    a_smem_iter: SMemTileIter[a_type, a_smem_layout],
    b_smem_iter: SMemTileIter[b_type, b_smem_layout],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ],
    elect_one_warp: Bool,
    iter_idx: UInt32,
    k_start: UInt32,
):
    """Consume tiles from shared memory and execute MMA operations.

    This is the public API for external callers using SMemTileIter.
    """
    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    if elect_one_sync():
        for j in range(k_group_size):
            var a_smem_tile = a_smem_iter.next(stage * k_group_size + j)[]
            var b_smem_tile = b_smem_iter.next(stage * k_group_size + j)[]
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                tmem_addr,
                init_c=((iter_idx + j) == k_start),
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))


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
    comptime num_pipeline_stages = Int(Self.config.num_pipeline_stages)
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Int(Self.config.k_group_size)
    )
    comptime num_output_stages = Int(Self.config.num_output_stages)
    comptime num_accum_pipeline_stages = Int(
        Self.config.num_accum_pipeline_stages
    )
    comptime num_clc_pipeline_stages = Int(Self.config.num_clc_pipeline_stages)

    # ========== Layout Definitions ==========
    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, swizzle_mode = Self.config.a_swizzle
    ]()

    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]()

    comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # ========== Tile Type Aliases ==========
    # Using structuring module types for clean function signatures
    comptime SMM = SharedMemoryManager[]
    comptime ATile = Self.SMM.Tile[Self.a_type, Self.a_smem_layout]
    comptime BTile = Self.SMM.Tile[Self.b_type, Self.b_smem_layout]
    comptime CTile = Self.SMM.Tile[Self.c_type, Self.c_smem_layout]

    # ========== Tile Array Type Aliases ==========
    # SMemTileArrayType for indexing into pipeline stages (uses [index] syntax)
    comptime ATileArray = SMemTileArrayType[
        Self.a_type,
        Self.a_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime BTileArray = SMemTileArrayType[
        Self.b_type,
        Self.b_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime CTileArray = SMemTileArrayType[
        Self.c_type,
        Self.c_smem_layout,
        Self.num_output_stages,
        alignment=128,
    ]

    # ========== Storage Fields ==========
    # Tile storage sized to match tile array requirements
    var a_tiles_storage: Self.ATileArray.StorageType
    var b_tiles_storage: Self.BTileArray.StorageType
    var c_tiles_storage: Self.CTileArray.StorageType

    @always_inline
    fn a_tiles(ref [AddressSpace.SHARED]self) -> Self.ATileArray:
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref [AddressSpace.SHARED]self) -> Self.BTileArray:
        return Self.BTileArray(self.b_tiles_storage)

    @always_inline
    fn c_tiles(ref [AddressSpace.SHARED]self) -> Self.CTileArray:
        return Self.CTileArray(self.c_tiles_storage)

    # Input pipeline barriers (TMA→MMA synchronization)
    var input_barriers: InlineArray[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]

    # Accumulator pipeline barriers (MMA→Epilogue synchronization)
    var accum_barriers: InlineArray[
        SharedMemBarrier, Self.num_accum_pipeline_stages * 2
    ]

    # CLC (Cluster Launch Control) barriers and response
    var clc_full: InlineArray[SharedMemBarrier, Self.num_clc_pipeline_stages]
    var clc_empty: InlineArray[SharedMemBarrier, Self.num_clc_pipeline_stages]
    var clc_throttle: InlineArray[
        SharedMemBarrier, Self.num_clc_pipeline_stages * 2
    ]
    var clc_response: InlineArray[UInt128, Self.num_clc_pipeline_stages]

    # TMEM (Tensor Memory) management
    var tmem_dealloc: InlineArray[SharedMemBarrier, 1]
    var tmem_addr: InlineArray[UInt32, 1]

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
    elementwise_compute_lambda_fn: OptionalReg[
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

    comptime CLUSTER_M = Int(Self.config.cluster_shape[0])
    comptime CLUSTER_N = Int(Self.config.cluster_shape[1])
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

    comptime num_pipeline_stages = Int(Self.config.num_pipeline_stages)
    comptime num_group_pipeline_stages = Self.num_pipeline_stages // Int(
        Self.config.k_group_size
    )
    comptime num_clc_pipeline_stages = Int(Self.config.num_clc_pipeline_stages)
    comptime num_accum_pipeline_stages = Int(
        Self.config.num_accum_pipeline_stages
    )
    comptime num_output_stages = Int(Self.config.num_output_stages)

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

    comptime InputTilePipeline = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.SmemType.a_smem_layout,
        Self.SmemType.b_smem_layout,
        Self.SmemType.num_pipeline_stages,
        Self.SmemType.num_group_pipeline_stages,
        Int(Self.config.k_group_size),
    ]

    # ========== Tile Loader Type ==========
    # Partial comptime binding explicit params; Origins inferred from constructor args.
    # See tile_loader.mojo for the partial type binding pattern explanation.

    comptime TileLoaderTMA = TileLoaderTMA[
        Self.SmemType.a_smem_layout,
        Self.SmemType.b_smem_layout,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.MMA_N,
        Self.cta_group,
        Int(Self.config.k_group_size),
        Self.SmemType.num_pipeline_stages,
        Self.SmemType.num_group_pipeline_stages,
    ]

    # ========== Epilogue Configuration ==========
    # Note: stageN is typically c_smem_layout.shape[1] for non-transposed output

    comptime EpilogueConf = EpilogueConfig[
        Self.MMA_M,
        Self.MMA_N,
        Self.SmemType.c_smem_layout.shape[1].value(),  # stageN
        Self.cta_group,
        False,  # transpose_c (default)
    ]

    # ========== Output Tile Pipeline Type ==========
    # Manages MMA→Epilogue pipeline for TMEM accumulator stages

    comptime OutputPipeline = OutputTilePipeline[
        Int(Self.config.num_accum_pipeline_stages),
        Self.stage_stride_cols,
        Self.cta_group,
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
            Self.c_type is not DType.float32,
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
        tma_mma_mbars_ptr: SMemPtr[SharedMemBarrier],
        accum_mbars_ptr: SMemPtr[SharedMemBarrier],
        clc_throttle_ptr: SMemPtr[SharedMemBarrier],
        clc_full_mbar: SMemPtr[SharedMemBarrier],
        clc_empty_mbar: SMemPtr[SharedMemBarrier],
        tmem_dealloc_mbar: SMemPtr[SharedMemBarrier],
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
                tma_mma_mbars_ptr,
                Int32(1),
                Self.config.cluster_shape[0] // Self.config.cta_group
                + Self.config.cluster_shape[1]
                - 1,
            )
            Self.OutputPipeline.init_barriers(
                accum_mbars_ptr,
                Self.accum_pipeline_producer_arv_count,
                Self.accum_pipeline_consumer_arv_count,
            )
            Self.Scheduler.init_throttle_barriers(
                clc_throttle_ptr,
                Self.clc_throttle_producer_arv_count,
                Self.clc_throttle_consumer_arv_count,
            )

            # Initialize TMEM deallocation barrier
            tmem_dealloc_mbar[].init(
                Self.EPILOGUE_THREADS * Self.config.cta_group
            )

            # Initialize CLC barriers
            @parameter
            for i in range(Self.config.num_clc_pipeline_stages):
                clc_full_mbar[i].init(Self.clc_producer_arv_count)
                clc_empty_mbar[i].init(Self.clc_consumer_arv_count)

        fence_mbarrier_init()
        cluster_sync()

    @staticmethod
    @always_inline
    fn _load_AB_tiles[
        tiles_origin: MutOrigin,
        //,
    ](
        a_tma_op: TMATensorTile[Self.a_type, Self.a_layout, Self.a_desc_layout],
        b_tma_op: TMATensorTile[Self.b_type, Self.b_layout, Self.b_desc_layout],
        tiles: ProducerStage[
            tiles_origin,
            Self.a_type,
            Self.b_type,
            Self.SmemType.a_smem_layout,
            Self.SmemType.b_smem_layout,
            Self.SmemType.num_pipeline_stages,
            Self.SmemType.num_group_pipeline_stages,
            Int(Self.config.k_group_size),
        ],
        peer_cta_coord: Tuple[UInt, UInt, UInt],
        work_tile_coord: Tuple[UInt, UInt],
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A/B tiles via TMA multicast within producer stage context."""
        comptime a_expected_bytes = (
            Self.SmemType.a_smem_layout.size() * size_of[Self.a_type]()
        )
        comptime b_expected_bytes = (
            Self.SmemType.b_smem_layout.size() * size_of[Self.b_type]()
        )
        # Leader CTAs expect SMEM from itself and their peers
        comptime expected_bytes = Self.config.cta_group * (
            a_expected_bytes + b_expected_bytes
        ) * Int(Self.config.k_group_size)

        comptime a_tma_load_size = Self.a_desc_layout.size()
        comptime b_tma_load_size = Self.b_desc_layout.size()
        comptime a_tma_rows = Self.a_desc_layout.shape[0].value()
        comptime b_tma_rows = Self.b_desc_layout.shape[0].value()

        # Global memory row coordinates for TMA loads
        var a_gmem_m_coord = peer_cta_coord[2] * UInt(
            a_tma_rows
        ) + work_tile_coord[0] * UInt(Self.BM)
        var b_gmem_n_coord = (
            peer_cta_coord[1] * UInt(b_tma_rows)
            + peer_cta_coord[0] * UInt(Self.BN)
            + work_tile_coord[1] * UInt(Self.config.mma_shape[1])
        )

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(expected_bytes)

            var barrier = tiles.barrier()

            for j in range(Int(Self.config.k_group_size)):
                var a_tile, b_tile = tiles.get_tile(j)

                # Offset to peer CTA's portion within the tile
                var a_peer_slice = type_of(a_tile)(
                    a_tile.ptr + peer_cta_coord[2] * UInt(a_tma_load_size)
                )
                var b_peer_slice = type_of(b_tile)(
                    b_tile.ptr + peer_cta_coord[1] * UInt(b_tma_load_size)
                )

                a_tma_op.async_multicast_load[Self.config.cta_group](
                    a_peer_slice,
                    barrier[0],
                    (UInt(iter_idx + j) * UInt(Self.BK), UInt(a_gmem_m_coord)),
                    a_multicast_mask,
                )

                b_tma_op.async_multicast_load[Self.config.cta_group](
                    b_peer_slice,
                    barrier[0],
                    (UInt(iter_idx + j) * UInt(Self.BK), UInt(b_gmem_n_coord)),
                    b_multicast_mask,
                )

    @staticmethod
    @always_inline
    fn mma(
        tmem_addr: UInt32,
        tiles: ConsumerStage,
        mma_op: MmaOpSM100_SS,
        elect_one_warp: Bool,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations for one pipeline stage.

        This is the core MMA function designed to be called within a consumer
        stage context:

            with consumer.acquire() as tiles:
                Self.mma(tmem_addr, tiles, mma_op, ...)

        Args:
            tmem_addr: Tensor memory address for accumulators.
            tiles: ConsumerStage context with encapsulated tile access.
            mma_op: The MMA operation instance.
            elect_one_warp: Whether this warp should execute.
            iter_idx: K iteration index.
            k_start: Starting K iteration (for init_c determination).
        """
        if elect_one_sync():
            for j in range(Int(Self.config.k_group_size)):
                var a_tile, b_tile = tiles.get_tile(j)
                var is_first_k = (iter_idx + j) == k_start
                mma_op.mma(a_tile, b_tile, tmem_addr, init_c=is_first_k)
            mma_op.commit(tiles.mbar())

    # ========== Kernel Entry Points ==========
    # Note: These delegate to the existing kernel functions for now.
    # As refactoring progresses, the kernel body will move into these methods.

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

        # Create input pipeline for TMA→MMA synchronization
        var input_pipeline = Self.InputTilePipeline(
            smem.input_barriers.unsafe_ptr(), smem.a_tiles(), smem.b_tiles()
        )

        comptime accum_type = get_accum_type[Self.a_type]()
        comptime max_tmem_cols = 512

        # Create kernel context with election vars, CTA coords, and masks
        var ctx = Self.Context(smem.tmem_addr.unsafe_ptr())

        # Initialize all barriers (only elect_one_warp && elect_one_thread)
        Self.init_barriers(
            ctx,
            a_tma_op,
            b_tma_op,
            c_tma_op,
            smem.input_barriers.unsafe_ptr(),
            smem.accum_barriers.unsafe_ptr(),
            smem.clc_throttle.unsafe_ptr(),
            smem.clc_full.unsafe_ptr(),
            smem.clc_empty.unsafe_ptr(),
            smem.tmem_dealloc.unsafe_ptr(),
        )

        var mma_op = Self.MmaOp()

        # Scheduler owns CLC throttle pipeline internally
        var scheduler = Self.Scheduler(
            cluster_dim,
            smem.clc_response.unsafe_ptr(),
            smem.clc_full.unsafe_ptr(),
            smem.clc_empty.unsafe_ptr(),
            smem.clc_throttle.unsafe_ptr(),
        )

        # Per-warp work iterator - owns work_info, pipeline state, and throttle
        var work_iter = scheduler.work_iterator()

        # Create tile loader for TMA operations
        var tile_loader = Self.TileLoaderTMA(
            Pointer(to=a_tma_op),
            Pointer(to=b_tma_op),
            ctx.a_multicast_mask,
            ctx.b_multicast_mask,
            ctx.peer_cta_coord,
        )

        var num_iters: UInt32 = ceildiv(mnk[2], Self.BK)

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        if WarpRole.is_main_load():
            with MatmulProfilerType[0](workspace, 0):

                @parameter
                if Self.pdl_level > PDLLevel.OFF:
                    wait_on_dependent_grids()

                while work_iter.has_work():
                    with work_iter.next() as current:
                        # CLC throttle prevents each CTA from going ahead
                        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                        # Set work tile coordinates for this iteration
                        tile_loader.set_work_tile(
                            UInt(current.m), UInt(current.n)
                        )

                        # DO TMA LOAD for full K range: [0, num_iters)
                        # Pattern: for i in range(k_start, k_end, k_group_size)
                        with input_pipeline.producer() as producer:
                            for i in range(
                                0, num_iters, Self.config.k_group_size
                            ):
                                with producer.acquire() as tiles:
                                    tile_loader.load_tiles(
                                        tiles, i, ctx.elect_one_cta
                                    )

                        # Ensure all TMA loads complete before advancing work
                        syncwarp()

                # Prevent CTA from exiting while peer CTA is still working on MMA
                with input_pipeline.producer() as producer:
                    producer.drain()

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
                tcgen05_alloc[Self.config.cta_group](
                    ctx.ptr_tmem_addr, max_tmem_cols
                )
                syncwarp()
                # non blocking, arrives and proceeds
                named_barrier_arrive[Self.MMA_THREADS + Self.EPILOGUE_THREADS](
                    1
                )

                var tmem_addr = ctx.ptr_tmem_addr[0]

                # Create output pipeline for MMA→Epilogue synchronization
                var output_pipeline = Self.OutputPipeline(
                    smem.accum_barriers.unsafe_ptr(),
                    tmem_addr,
                    ctx.mma_complete_mask,
                )

                while work_iter.has_work():
                    # Prefetch next work BEFORE doing MMA (software pipelining)
                    with work_iter.next_prefetch():
                        # DO MMA for full K range: [0, num_iters), init at k=0
                        if ctx.elect_one_cta:
                            with output_pipeline.producer() as stage:
                                with input_pipeline.consumer() as consumer:
                                    for i in range(
                                        0, num_iters, Self.config.k_group_size
                                    ):
                                        with consumer.acquire() as tiles:
                                            Self.mma(
                                                stage.tmem_offset,
                                                tiles,
                                                mma_op,
                                                ctx.elect_one_warp,
                                                i,
                                                0,  # init_k=0 for regular matmul
                                            )

                @parameter
                if Self.pdl_level > PDLLevel.OFF:
                    launch_dependent_grids()

                tcgen05_release_allocation_lock[Self.config.cta_group]()

                # wait for epilogue to finish
                smem.tmem_dealloc.unsafe_ptr()[].wait()

                tcgen05_dealloc[Self.config.cta_group](tmem_addr, max_tmem_cols)

        if WarpRole.is_epilogue():
            named_barrier[Self.MMA_THREADS + Self.EPILOGUE_THREADS](1)
            var tmem_addr = ctx.ptr_tmem_addr[0]

            # Create output pipeline for MMA→Epilogue synchronization
            var output_pipeline = Self.OutputPipeline(
                smem.accum_barriers.unsafe_ptr(),
                tmem_addr,
                ctx.mma_complete_mask,
            )

            var tile_idx = 0

            while work_iter.has_work():
                with work_iter.next() as current:
                    with MatmulProfilerType[3](workspace, tile_idx):
                        # WAIT FOR MMA TO FINISH AND STORE RESULT
                        with output_pipeline.consumer() as stage:
                            multi_stage_store_C[
                                Self.c_type,
                                Self.SmemType.c_smem_layout,
                                Self.c_layout,
                                Self.c_desc_layout,
                                Self.SmemType.num_accum_pipeline_stages,
                                Self.SmemType.num_output_stages,
                                input_type = Self.a_type,
                                accum_type=accum_type,
                                block_tile_shape = Self.config.block_tile_shape,
                                mma_shape = Self.config.mma_shape,
                                stage_stride_cols = UInt(
                                    Self.stage_stride_cols
                                ),
                                c_swizzle = Self.config.c_swizzle,
                                cta_group = Self.config.cta_group,
                                num_output_warps = Self.num_output_warps,
                                max_tmem_cols=max_tmem_cols,
                                elementwise_compute_lambda_fn = Self.elementwise_compute_lambda_fn,
                                register_based_epilogue = Self.register_based_epilogue,
                                transpose_c = Self.config.AB_swapped,
                            ](
                                smem.c_tiles(),
                                c_tma_op,
                                stage,  # Self-contained OutputStage
                                work_tile_coord=(current.m, current.n),
                                elect_one_warp=ctx.elect_one_warp,
                                M=mnk[0],
                                N=mnk[1],
                            )

                tile_idx += 1

            @parameter
            if Self.config.cta_group == 2:
                _ = smem.tmem_dealloc.unsafe_ptr()[].arrive_cluster(
                    block_rank_in_cluster() ^ 1
                )
            _ = smem.tmem_dealloc.unsafe_ptr()[].arrive()

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

        # Create input pipeline for TMA→MMA synchronization
        var input_pipeline = Self.InputTilePipeline(
            smem.input_barriers.unsafe_ptr(), smem.a_tiles(), smem.b_tiles()
        )

        comptime max_tmem_cols = 512

        # Create kernel context with election vars, CTA coords, and masks
        var ctx = Self.Context(smem.tmem_addr.unsafe_ptr())

        # Initialize all barriers (only elect_one_warp && elect_one_thread)
        Self.init_barriers(
            ctx,
            a_tma_op,
            b_tma_op,
            c_tma_op,
            smem.input_barriers.unsafe_ptr(),
            smem.accum_barriers.unsafe_ptr(),
            smem.clc_throttle.unsafe_ptr(),
            smem.clc_full.unsafe_ptr(),
            smem.clc_empty.unsafe_ptr(),
            smem.tmem_dealloc.unsafe_ptr(),
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
            num_stages = Int(Self.config.num_clc_pipeline_stages),
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
            smem.clc_response.unsafe_ptr(),
            smem.clc_full.unsafe_ptr(),
            smem.clc_empty.unsafe_ptr(),
            smem.clc_throttle.unsafe_ptr(),
            lock_ptr,
        )

        # Per-warp work iterator - owns work_info, pipeline state, and throttle
        var work_iter = scheduler.work_iterator()

        # Create tile loader for TMA operations
        var tile_loader = Self.TileLoaderTMA(
            Pointer(to=a_tma_op),
            Pointer(to=b_tma_op),
            ctx.a_multicast_mask,
            ctx.b_multicast_mask,
            ctx.peer_cta_coord,
        )

        comptime MatmulProfilerType[warp_role: UInt32] = MatmulProfileWarp[
            warp_role, Self.max_profiled_tiles_per_SM
        ]

        if WarpRole.is_main_load():
            with MatmulProfilerType[0](workspace, 0):
                while work_iter.has_work():
                    with work_iter.next() as current:
                        # CLC throttle prevents each CTA from going ahead
                        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                        # Set work tile coordinates for this iteration
                        tile_loader.set_work_tile(
                            UInt(current.m), UInt(current.n)
                        )

                        # DO TMA LOAD for K slice: [k_start, k_start + num_k_tiles)
                        var k_start = current.k_start
                        var k_end = k_start + current.num_k_tiles
                        with input_pipeline.producer() as producer:
                            for i in range(
                                k_start, k_end, Self.config.k_group_size
                            ):
                                with producer.acquire() as tiles:
                                    tile_loader.load_tiles(
                                        tiles, i, ctx.elect_one_cta
                                    )

                        # Ensure all TMA loads complete before advancing work
                        syncwarp()

                # Prevent CTA from exiting while peer CTA is still working on MMA
                with input_pipeline.producer() as producer:
                    producer.drain()

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
                while sched_iter.has_work():
                    with sched_iter.next():
                        sched_iter.signal_and_advance()

                # Drain all pending CLC requests before kernel exit
                sched_iter.drain()

        if WarpRole.is_mma():
            with MatmulProfilerType[2](workspace, 0):
                tcgen05_alloc[Self.config.cta_group](
                    ctx.ptr_tmem_addr, max_tmem_cols
                )
                syncwarp()
                # non blocking, arrives and proceeds
                named_barrier_arrive[Self.MMA_THREADS + Self.EPILOGUE_THREADS](
                    1
                )

                var tmem_addr = ctx.ptr_tmem_addr[0]

                # Create output pipeline for MMA→Epilogue synchronization
                var output_pipeline = Self.OutputPipeline(
                    smem.accum_barriers.unsafe_ptr(),
                    tmem_addr,
                    ctx.mma_complete_mask,
                )

                while work_iter.has_work():
                    # Prefetch next work BEFORE doing MMA (software pipelining)
                    with work_iter.next_prefetch() as current:
                        # DO MMA for K slice: [k_start, k_start + num_k_tiles)
                        # Init accumulator at k_start (first K of this split)
                        if ctx.elect_one_cta:
                            with output_pipeline.producer() as stage:
                                var k_start = current.k_start
                                var k_end = k_start + current.num_k_tiles
                                with input_pipeline.consumer() as consumer:
                                    for i in range(
                                        k_start, k_end, Self.config.k_group_size
                                    ):
                                        with consumer.acquire() as tiles:
                                            Self.mma(
                                                stage.tmem_offset,
                                                tiles,
                                                mma_op,
                                                ctx.elect_one_warp,
                                                i,
                                                k_start,  # init at k_start for split-K
                                            )

                tcgen05_release_allocation_lock[Self.config.cta_group]()

                # wait for epilogue to finish
                smem.tmem_dealloc.unsafe_ptr()[].wait()

                tcgen05_dealloc[Self.config.cta_group](tmem_addr, max_tmem_cols)

        if WarpRole.is_epilogue():
            named_barrier[Self.MMA_THREADS + Self.EPILOGUE_THREADS](1)
            var tmem_addr = ctx.ptr_tmem_addr[0]

            # Create output pipeline for MMA→Epilogue synchronization
            var output_pipeline = Self.OutputPipeline(
                smem.accum_barriers.unsafe_ptr(),
                tmem_addr,
                ctx.mma_complete_mask,
            )

            var tile_idx = 0

            while work_iter.has_work():
                with work_iter.next() as current:
                    with MatmulProfilerType[3](workspace, tile_idx):
                        # WAIT FOR MMA TO FINISH AND STORE RESULT
                        with output_pipeline.consumer() as stage:
                            multi_stage_store_C_split_k[
                                input_type = Self.a_type,
                                accum_type = Self.config.accum_type,
                                block_tile_shape = Self.config.block_tile_shape,
                                mma_shape = Self.config.mma_shape,
                                stage_stride_cols = UInt(
                                    Self.stage_stride_cols
                                ),
                                c_swizzle = Self.config.c_swizzle,
                                cta_group = Self.config.cta_group,
                                num_output_warps = Self.num_output_warps,
                                max_tmem_cols=max_tmem_cols,
                                elementwise_compute_lambda_fn = Self.elementwise_compute_lambda_fn,
                                register_based_epilogue = Self.register_based_epilogue,
                                transpose_c = Self.config.AB_swapped,
                            ](
                                scheduler,
                                reduction_tensor,
                                smem.c_tiles(),
                                c_tma_op,
                                stage,  # Self-contained OutputStage
                                work_info=current,
                                elect_one_warp=ctx.elect_one_warp,
                                M=mnk[0],
                                N=mnk[1],
                            )

                tile_idx += 1

            @parameter
            if Self.config.cta_group == 2:
                _ = smem.tmem_dealloc.unsafe_ptr()[].arrive_cluster(
                    block_rank_in_cluster() ^ 1
                )
            _ = smem.tmem_dealloc.unsafe_ptr()[].arrive()


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
    num_threads: UInt = 128,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
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
    comptime ATile = SMemTileType[
        Self.a_type,
        Self.a_smem_layout,
        alignment=128,
    ]
    comptime BTile = SMemTileType[
        Self.b_type,
        Self.b_smem_layout,
        alignment=128,
    ]

    comptime accum_type = get_accum_type[Self.a_type]()
    comptime c_frag_size = Self.MMA_M * Self.MMA_N // Int(Self.num_threads)
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
                tma_mbar[0].expect_bytes(expected_bytes)

                a_tma_op.async_copy(
                    a_smem_tile,
                    tma_mbar[0],
                    (UInt(i) * UInt(Self.BK), block_idx.y * UInt(Self.BM)),
                )
                b_tma_op.async_copy(
                    b_smem_tile,
                    tma_mbar[0],
                    (
                        UInt(i) * UInt(Self.BK),
                        block_idx.x * UInt(Self.BN),
                    ) if Self.transpose_b else (
                        block_idx.x * UInt(Self.BN),
                        UInt(i) * UInt(Self.BK),
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
        c_frag = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat = Self.BN // 8,
            dtype = Self.accum_type,
            pack=False,
            width = Self.c_frag_size,
        ](tmem_addr)

        tcgen05_load_wait()

        if elect_one_warp:
            tcgen05_release_allocation_lock[1]()
            tcgen05_dealloc[1](tmem_addr, Self.max_tmem_cols)

        # Write output to global memory
        comptime num_warps = Self.num_threads // UInt(WARP_SIZE)
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
                    ctile.tile_with_offset[
                        Self.MMA_M // Int(num_warps), Self.MMA_N
                    ](4 * m_mma + Int(warp_id), n_mma)
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

                        if m < M and n < N:
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


comptime RLayout32Bits[layout: Layout] = RuntimeLayout[
    layout, element_type = DType.uint32, linear_idx_type = DType.uint32
]


@always_inline
fn f32_frag_to_smem[
    swizzle_mode: TensorMapSwizzle,
    stageN: UInt,
](vec: SIMD, dst: SMemTileType):
    # TODO: apply swizzle. Somehow swizzle+distribute results in wrong values.
    # comptime swizzle = make_swizzle[DType.float64, swizzle_mode]() # hack
    # var dst_frag = dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4), swizzle=swizzle](lane_id())
    var dst_frag = dst.vectorize[1, 2]().distribute[Layout.row_major(8, 4)](
        lane_id()
    )
    constrained[
        2 * dst_frag.layout.size() == vec.size,
        "2*dst_frag.layout.size() must be equal to vec.size",
    ]()

    @parameter
    for i in range(dst_frag.layout.shape[0].value()):

        @parameter
        for j in range(dst_frag.layout.shape[1].value()):
            comptime i_vec = i + j * dst_frag.layout.shape[0].value()
            val = SIMD[dst.dtype, 2](
                rebind[Scalar[dst.dtype]](vec[2 * i_vec]),
                rebind[Scalar[dst.dtype]](vec[2 * i_vec + 1]),
            )
            dst_frag[i, j] = rebind[dst_frag.element_type](val)


@always_inline
fn stsm_helper[
    swizzle: Swizzle,
    stageN: UInt,
    transpose_c: Bool = False,
    swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](vec: SIMD, dst: SMemTileType, warp_offset: UInt32 = 0):
    """Store a fragment to shared memory using st.matrix.

    Delegates to store_fragment_to_smem for non-float32 types,
    and to f32_frag_to_smem for float32.
    """

    @parameter
    if size_of[dst.dtype]() == 4:
        constrained[not transpose_c, "transpose_c must be False"]()
        return f32_frag_to_smem[swizzle_mode, stageN](vec, dst)

    # Validate layout constraint
    constrained[
        dst.layout.stride[1].value() == 1,
        "stride1 must be 1. Got: "
        + String(dst.layout.stride[1].value())
        + " for layout: "
        + String(dst.layout),
    ]()

    # Delegate to tile_writer component
    store_fragment_to_smem[swizzle, Int(stageN), transpose_c, swizzle_mode](
        vec, dst, warp_offset
    )
