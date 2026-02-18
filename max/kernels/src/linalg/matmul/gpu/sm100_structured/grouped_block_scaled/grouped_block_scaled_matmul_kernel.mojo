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
"""Grouped block-scaled SM100 matmul kernel for multiple GEMM problems.

This kernel extends the block_scaled_matmul_kernel to support grouped GEMM
with variable problem sizes per group. It uses:

1. GroupedTileScheduler: For linear tile iteration across groups
2. TMATensorTileArray: For per-block updatable TMA descriptors
3. Dynamic tensormap updates: When transitioning between groups

Architecture (aligned with NVIDIA CuTe DSL grouped_blockscaled_gemm.py):
- TMA warp: Initializes A/B/SFA/SFB tensormaps, handles group transitions
- MMA warp: Consumes input tiles, performs block-scaled MMA
- Epilogue warps: Initializes C tensormap, handles C group transitions
- Named barrier synchronization between warps for tensormap init

Key differences from block_scaled_matmul_kernel.mojo:
1. TMA descriptors are per-block (TMATensorTileArray) not grid constants
2. SMEM tensormap buffers for dynamic updates (5 x 128 bytes)
3. GroupedWorkInfo provides group_idx, k_tile_count, group_changed
4. When group_changed=True, tensormaps are updated before loading tiles
5. K-loop uses per-group k_tile_count instead of global K dimension
"""

from collections import Optional
from math import ceildiv
from memory import stack_allocation, UnsafePointer, Pointer
from sys import size_of

from gpu import WARP_SIZE, block_idx, grid_dim, lane_id, thread_idx, warp_id
from gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from gpu.primitives.cluster import cluster_sync, elect_one_sync
from gpu.sync import named_barrier, named_barrier_arrive, syncwarp
from gpu.host.nvidia.tma import TMADescriptor, TensorMapSwizzle
from sys import inlined_assembly
from layout import Layout as LegacyLayout, LayoutTensor
from layout._layout import TensorLayout
from layout._tile_tensor import TileTensor
from layout._layout import RowMajorLayout, _IntToComptimeInt
from layout._coord import ComptimeInt
from ..structured_kernels.tile_types import (
    TMATile,
    TmaOpType,
    tma_desc_layout_3d,
    tma_desc_layout_5d,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    TMATensorTileArray,
)
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
from linalg.structuring import SMemPtr
from ..structured_kernels.config import BlockScaledMatmulConfig
from ..structured_kernels.tile_types import internal_k_major_128B
from ..structured_kernels.kernel_common import WarpRole, KernelContext
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
from ..structured_kernels.output_writer import TileWriter
from ..structured_kernels.tile_scheduler import (
    TileScheduler as WorkingTileScheduler,
)
from ..block_scaled.block_scaled_smem import BlockScaledSmem
from .grouped_block_scaled_smem import GroupedBlockScaledSmem
from .grouped_tile_scheduler import (
    GroupedTileScheduler,
    GroupedWorkInfo,
    GroupedWorkIterator,
    GroupedCLCWorkIterator,
    GroupedCLCSchedulerIterator,
)


comptime _GroupPtrLayout[max_groups: Int] = RowMajorLayout[
    ComptimeInt[max_groups], ComptimeInt[1]
]
comptime _GroupPtrTile[max_groups: Int] = TileTensor[
    DType.uint64, _GroupPtrLayout[max_groups], MutAnyOrigin
]
comptime _ProblemSizesLayout[max_groups: Int] = RowMajorLayout[
    ComptimeInt[max_groups], ComptimeInt[4]
]
comptime _ProblemSizesTile[max_groups: Int] = TileTensor[
    DType.int32, _ProblemSizesLayout[max_groups], MutAnyOrigin
]


# =============================================================================
# Constants
# =============================================================================

# TMA descriptor size in bytes (128 bytes per descriptor)
comptime TMA_DESCRIPTOR_SIZE = 128

# Number of tensormaps for grouped GEMM (A, B, SFA, SFB, C)
comptime NUM_TENSORMAPS = 5


# =============================================================================
# GroupedTensormapSmem - SMEM storage for tensormap descriptors
# =============================================================================


@fieldwise_init
struct GroupedTensormapSmem(TrivialRegisterPassable):
    """Shared memory pointers for tensormap descriptors.

    Points to 5 TMA descriptors (128 bytes each) in SMEM for dynamic updates:
    - A, B, SFA, SFB for input loading
    - C for output storing

    These pointers should come from the main SMEM struct (GroupedBlockScaledSmem)
    to ensure all warps access the same SMEM locations.
    """

    var desc_a: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var desc_b: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var desc_sfa: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var desc_sfb: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]
    var desc_c: UnsafePointer[
        TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
    ]

    @staticmethod
    @always_inline
    fn from_smem(
        ptr_a: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        ptr_b: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        ptr_sfa: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        ptr_sfb: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
        ptr_c: UnsafePointer[
            TMADescriptor, MutAnyOrigin, address_space = AddressSpace.SHARED
        ],
    ) -> Self:
        """Create tensormap pointers from explicit SMEM pointers.

        Args:
            ptr_a: Pointer to A tensormap in SMEM.
            ptr_b: Pointer to B tensormap in SMEM.
            ptr_sfa: Pointer to SFA tensormap in SMEM.
            ptr_sfb: Pointer to SFB tensormap in SMEM.
            ptr_c: Pointer to C tensormap in SMEM.

        Returns:
            GroupedTensormapSmem with the provided pointers.
        """
        return Self(ptr_a, ptr_b, ptr_sfa, ptr_sfb, ptr_c)


# =============================================================================
# GroupedTensormapManager - Manages tensormap updates for grouped GEMM
# =============================================================================


@fieldwise_init
struct GroupedTensormapManager(TrivialRegisterPassable):
    """Manages tensormap SMEM state and updates for grouped GEMM.

    Handles the 4-step CuTe DSL update pattern:
    1. tensormap_fence_acquire() - Acquire fence on block's GMEM tensormap
    2. replace_tensormap_global_address_in_shared_mem() - Update SMEM descriptor
    3. tensormap_cp_fence_release() - Copy SMEM -> block's GMEM tensormap
    4. syncwarp() - Sync before using updated tensormap

    TMA descriptor arrays are passed by reference (as UnsafePointer from
    TMATensorTileArray[blk]) to methods rather than stored by value. This
    ensures PTX tensormap operations receive valid GMEM addresses with correct
    address space semantics.

    The manager stores only SMEM descriptor pointers, which are shared across
    all warps within a CTA.
    """

    # SMEM descriptors for in-place tensormap updates
    var smem: GroupedTensormapSmem

    @always_inline
    fn init_ab_tensormaps[
        a_dtype: DType,
        a_layout: Layout,
        a_desc: Layout,
        b_dtype: DType,
        b_layout: Layout,
        b_desc: Layout,
        sfa_dtype: DType,
        sfa_layout: Layout,
        sfa_desc: Layout,
        sfb_dtype: DType,
        sfb_layout: Layout,
        sfb_desc: Layout,
    ](
        self,
        template_a: TMATensorTile[a_dtype, a_layout, a_desc],
        template_b: TMATensorTile[b_dtype, b_layout, b_desc],
        template_sfa: TMATensorTile[sfa_dtype, sfa_layout, sfa_desc],
        template_sfb: TMATensorTile[sfb_dtype, sfb_layout, sfb_desc],
    ):
        """Initialize A/B/SFA/SFB tensormaps in SMEM from grid-constant templates.

        Called by MMA warp (lane 0). Copies template descriptors to SMEM.
        Templates must be kernel parameters with nvvm.grid_constant metadata.
        """
        if lane_id() == 0:
            template_a.smem_tensormap_init(self.smem.desc_a)
            template_b.smem_tensormap_init(self.smem.desc_b)
            template_sfa.smem_tensormap_init(self.smem.desc_sfa)
            template_sfb.smem_tensormap_init(self.smem.desc_sfb)

    @always_inline
    fn init_c_tensormap[
        c_dtype: DType,
        c_layout: Layout,
        c_desc: Layout,
    ](self, template_c: TMATensorTile[c_dtype, c_layout, c_desc],):
        """Initialize C tensormap in SMEM from grid-constant template.

        Called by epilogue warp (lane 0). Copies template descriptor to SMEM.
        """
        if lane_id() == 0:
            template_c.smem_tensormap_init(self.smem.desc_c)

    @always_inline
    fn update_ab_for_group[
        a_dtype: DType,
        a_layout: Layout,
        a_desc: Layout,
        b_dtype: DType,
        b_layout: Layout,
        b_desc: Layout,
        sfa_dtype: DType,
        sfa_layout: Layout,
        sfa_desc: Layout,
        sfb_dtype: DType,
        sfb_layout: Layout,
        sfb_desc: Layout,
        max_groups: Int,
    ](
        self,
        group_idx: UInt32,
        group_a_ptrs: _GroupPtrTile[max_groups],
        group_b_ptrs: _GroupPtrTile[max_groups],
        group_sfa_ptrs: _GroupPtrTile[max_groups],
        group_sfb_ptrs: _GroupPtrTile[max_groups],
        tma_a: UnsafePointer[
            TMATensorTile[a_dtype, a_layout, a_desc], MutAnyOrigin
        ],
        tma_b: UnsafePointer[
            TMATensorTile[b_dtype, b_layout, b_desc], MutAnyOrigin
        ],
        tma_sfa: UnsafePointer[
            TMATensorTile[sfa_dtype, sfa_layout, sfa_desc], MutAnyOrigin
        ],
        tma_sfb: UnsafePointer[
            TMATensorTile[sfb_dtype, sfb_layout, sfb_desc], MutAnyOrigin
        ],
    ):
        """Update A/B/SFA/SFB tensormaps for the specified group.

        Called when group_changed=True in TMA load warp.
        TMA pointers must be from TMATensorTileArray[block_idx.x] (GMEM).
        """
        # Step 1: Acquire fences on GMEM tensormaps
        tma_a[].tensormap_fence_acquire()
        tma_b[].tensormap_fence_acquire()
        tma_sfa[].tensormap_fence_acquire()
        tma_sfb[].tensormap_fence_acquire()

        # Step 2: Update SMEM descriptors (lane 0 only)
        if lane_id() == 0:
            var g = Int(group_idx)

            var a_ptr = UnsafePointer[mut=True, Scalar[a_dtype], MutAnyOrigin](
                unsafe_from_address=Int(group_a_ptrs[g, 0])
            )
            var b_ptr = UnsafePointer[mut=True, Scalar[b_dtype], MutAnyOrigin](
                unsafe_from_address=Int(group_b_ptrs[g, 0])
            )
            var sfa_ptr = UnsafePointer[
                mut=True, Scalar[sfa_dtype], MutAnyOrigin
            ](unsafe_from_address=Int(group_sfa_ptrs[g, 0]))
            var sfb_ptr = UnsafePointer[
                mut=True, Scalar[sfb_dtype], MutAnyOrigin
            ](unsafe_from_address=Int(group_sfb_ptrs[g, 0]))

            tma_a[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_a, a_ptr
            )
            tma_b[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_b, b_ptr
            )
            tma_sfa[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_sfa, sfa_ptr
            )
            tma_sfb[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_sfb, sfb_ptr
            )

        syncwarp()

        # Step 3: Fence release copies SMEM -> GMEM
        tma_a[].tensormap_cp_fence_release(self.smem.desc_a)
        tma_b[].tensormap_cp_fence_release(self.smem.desc_b)
        tma_sfa[].tensormap_cp_fence_release(self.smem.desc_sfa)
        tma_sfb[].tensormap_cp_fence_release(self.smem.desc_sfb)

        # Step 4: Sync within warp
        syncwarp()

    @always_inline
    fn update_c_for_group[
        c_dtype: DType,
        c_layout: Layout,
        c_desc: Layout,
        max_groups: Int,
    ](
        self,
        group_idx: UInt32,
        group_c_ptrs: _GroupPtrTile[max_groups],
        tma_c: UnsafePointer[
            TMATensorTile[c_dtype, c_layout, c_desc], MutAnyOrigin
        ],
    ):
        """Update C tensormap for the specified group.

        Called when group_changed=True in epilogue warp.
        TMA pointer must be from TMATensorTileArray[block_idx.x] (GMEM).
        """
        # Step 1: Acquire fence
        tma_c[].tensormap_fence_acquire()

        # Step 2: Update SMEM descriptor (lane 0 only)
        if lane_id() == 0:
            var g = Int(group_idx)
            var c_ptr = UnsafePointer[mut=True, Scalar[c_dtype], MutAnyOrigin](
                unsafe_from_address=Int(group_c_ptrs[g, 0])
            )

            tma_c[].replace_tensormap_global_address_in_shared_mem(
                self.smem.desc_c, c_ptr
            )

        syncwarp()

        # Step 3: Fence release
        tma_c[].tensormap_cp_fence_release(self.smem.desc_c)

        # Step 4: Sync within warp
        syncwarp()


# =============================================================================
# Validation Utilities (matching NVIDIA CuTe DSL constraints)
# =============================================================================


fn is_valid_dtypes_and_scale_factor_vec_size(
    ab_dtype: DType,
    sf_dtype: DType,
    sf_vec_size: Int,
    c_dtype: DType,
) -> Bool:
    """Check if dtypes and sf_vec_size are valid combinations.

    Valid combinations (from NVIDIA CuTe DSL grouped_blockscaled_gemm.py):
    - MXF8: Float8E5M2/Float8E4M3FN + Float8E8M0FNU + sf_vec_size=32
    - MXF4: Float4E2M1FN + Float8E8M0FNU + sf_vec_size=32
    - NVF4: Float4E2M1FN + Float8E8M0FNU/Float8E4M3FN + sf_vec_size=16

    Args:
        ab_dtype: The data type of A and B matrices.
        sf_dtype: The data type of scale factors.
        sf_vec_size: The vector size of scale factors (16 or 32).
        c_dtype: The data type of the output matrix.

    Returns:
        True if the combination is valid.
    """
    # Check valid ab_dtype (FP8 or FP4 types)
    var valid_ab = ab_dtype in (
        DType.float4_e2m1fn,  # NVF4, MXF4
        DType.float8_e5m2,  # MXF8
        DType.float8_e4m3fn,  # MXF8
    )
    if not valid_ab:
        return False

    # Check valid sf_vec_size (16 for NVF4, 32 for MXF8/MXF4)
    if sf_vec_size not in (16, 32):
        return False

    # Check valid sf_dtype
    var valid_sf = sf_dtype in (
        DType.float8_e8m0fnu,  # MXF8, MXF4, NVF4
        DType.float8_e4m3fn,  # NVF4 only
    )
    if not valid_sf:
        return False

    # Check sf_dtype and sf_vec_size combinations
    # Float8E4M3FN scale factors only valid with sf_vec_size=16 (NVF4)
    if sf_dtype == DType.float8_e4m3fn and sf_vec_size == 32:
        return False

    # MXF8 (Float8) requires sf_vec_size=32
    if (
        ab_dtype in (DType.float8_e5m2, DType.float8_e4m3fn)
        and sf_vec_size == 16
    ):
        return False

    # Check valid c_dtype
    var valid_c = c_dtype in (
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float8_e5m2,
        DType.float8_e4m3fn,
    )
    if not valid_c:
        return False

    return True


fn is_valid_mma_tiler_and_cluster_shape(
    mma_tiler_m: Int,
    mma_tiler_n: Int,
    cluster_m: Int,
    cluster_n: Int,
) -> Bool:
    """Check if MMA tiler and cluster shape are valid.

    Constraints (from NVIDIA CuTe DSL):
    - MMA tiler M: 128 or 256
    - MMA tiler N: 128 or 256
    - Cluster M must be multiple of 2 if MMA tiler M is 256
    - Cluster M/N: Power of 2, <=4 per axis (for SF multicast)
    - Total cluster size: <=16
    """
    # Check MMA tiler
    if mma_tiler_m not in (128, 256):
        return False
    if mma_tiler_n not in (128, 256):
        return False

    # Check cluster constraints
    if mma_tiler_m == 256 and cluster_m % 2 != 0:
        return False

    fn is_power_of_2(x: Int) -> Bool:
        return x > 0 and (x & (x - 1)) == 0

    if not is_power_of_2(cluster_m) or not is_power_of_2(cluster_n):
        return False

    # SF multicast constraint
    if cluster_m > 4 or cluster_n > 4:
        return False

    # Total cluster size
    if cluster_m * cluster_n > 16:
        return False

    return True


# =============================================================================
# GroupedBlockScaledMatmulKernel - Main kernel struct
# =============================================================================


struct GroupedBlockScaledMatmulKernel[
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
    # Grouped GEMM parameters
    max_groups: Int,
    # Cluster shape (for LLVM metadata)
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Epilogue fusion parameters
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
]:
    """Grouped block-scaled matmul kernel with dynamic tensormap updates.

    This kernel extends BlackwellBlockScaledMatmulKernel to support grouped GEMM:
    - Uses GroupedTileScheduler for linear tile iteration across groups
    - Uses GroupedTensormapManager for per-block updatable TMA descriptors
    - Updates tensormaps when transitioning between groups

    Architecture (aligned with NVIDIA CuTe DSL grouped_blockscaled_gemm.py):
    - TMA warp: Initializes A/B/SFA/SFB tensormaps, handles group transitions
    - MMA warp: Waits for tensormap init, consumes tiles, performs block-scaled MMA
    - Epilogue warps: Initializes C tensormap, handles C group transitions
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

    # ========== Tensormap Synchronization ==========

    # Named barrier for TMA->MMA tensormap init synchronization
    comptime TENSORMAP_AB_INIT_BARRIER_ID: Int = 3
    comptime TENSORMAP_AB_INIT_THREADS: Int = Self.TMA_LOAD_THREADS + Self.MMA_THREADS

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = Self.num_pipeline_stages // Self.config.k_group_size
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # TMEM configuration
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = Self.config.num_sf_k_tiles * (Self.BM // 32)
    comptime SFB_NUM_COLS = Self.config.num_sf_k_tiles * (Self.MMA_N // 32)
    comptime stage_stride_cols = Self.MMA_N

    # ========== Barrier Arrival Counts ==========

    comptime accum_pipeline_producer_arv_count = 1
    comptime accum_pipeline_consumer_arv_count = Self.cta_group * Self.EPILOGUE_THREADS

    # ========== CLC Configuration for 2SM ==========
    # These are used by run_2sm() for CLC-based work distribution

    comptime num_clc_pipeline_stages_2sm: Int = 2  # Use 2 stages for 2SM
    comptime clc_producer_arv_count = 1  # Scheduler warp produces work
    # Only MMA warp calls wait_and_advance() which signals empty barrier.
    # Each thread in MMA warp on all CTAs calls arrive_cluster(0).
    comptime clc_consumer_arv_count = Self.MMA_THREADS * Self.cta_group
    comptime clc_throttle_producer_arv_count = Self.TMA_LOAD_THREADS
    comptime clc_throttle_consumer_arv_count = Self.SCHEDULER_THREADS

    # ========== Grouped Tensormap Manager Type ==========
    # The manager stores only SMEM state. TMA descriptor references are
    # passed to methods with explicit GLOBAL address space.

    comptime TensormapManagerType = GroupedTensormapManager

    # ========== Grouped Tile Scheduler Type ==========

    comptime SchedulerType = GroupedTileScheduler[
        tile_m = Self.BM,
        tile_n = Self.BN,
        tile_k = Self.BK,
        max_groups = Self.max_groups,
    ]

    # ========== TMA Descriptor Array Types ==========
    # Per-block updatable tensormaps (not grid constants)

    comptime TMATensorTileArrayA = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.a_type,
        Self.ATmaOp.layout,
        Self.ATmaOp.desc_layout,
    ]
    comptime TMATensorTileArrayB = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.b_type,
        Self.BTmaOp.layout,
        Self.BTmaOp.desc_layout,
    ]
    comptime TMATensorTileArraySFA = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.sfa_dtype,
        Self.SFATmaOp.layout,
        Self.SFATmaOp.desc_layout,
    ]
    comptime TMATensorTileArraySFB = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.sfb_dtype,
        Self.SFBTmaOp.layout,
        Self.SFBTmaOp.desc_layout,
    ]
    comptime TMATensorTileArrayC = TMATensorTileArray[
        Self.CLUSTER_SIZE,
        Self.c_type,
        Self.CTmaOp.layout,
        Self.CTmaOp.desc_layout,
    ]

    # ========== Per-Group Pointer Layout ==========
    # Layout for arrays of per-group tensor pointers

    comptime GroupPtrLayout = _GroupPtrLayout[Self.max_groups]
    comptime GroupPtrTile = _GroupPtrTile[Self.max_groups]

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

    # SF_K_GROUP_SIZE = SF_ATOM_K * vec_sf_size
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

    # ========== Shared Memory Type ==========
    # Use GroupedBlockScaledSmem which includes SMEM storage for TMA descriptors.
    # This allows runtime updates to TMA base pointers for multi-group support.

    comptime SmemType = GroupedBlockScaledSmem[
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

    comptime Context = KernelContext[
        0,  # num_clc_pipeline_stages = 0 for grouped (no CLC)
        Self.cta_group,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
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
        Self.MMA_THREADS + Self.EPILOGUE_THREADS, 1
    ]

    # Barrier for tensormap init synchronization between MMA and TMA warps.
    # MMA warp initializes SMEM tensormaps, TMA warp waits before using them.
    # Following CuTe DSL pattern: tensormap_ab_init_barrier (64 threads)
    comptime TensormapAbInitBarrier = WarpGroupBarrier[
        Self.TENSORMAP_AB_INIT_THREADS, Self.TENSORMAP_AB_INIT_BARRIER_ID
    ]

    comptime MmaCtx = MmaWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.config.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.MMA_THREADS,
        Self.EPILOGUE_THREADS,
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
        elementwise_compute_lambda_fn = Self.elementwise_compute_lambda_fn,
        register_based_epilogue = Self.register_based_epilogue,
        batched=True,
    ]

    # ========== TMA Load Size Constants ==========

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
    # 3D batched layouts for A, B, C (batch dim = 1 for per-group updates)

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

    # C tile dims -- same AB_swapped-aware logic as other kernels
    comptime c_tile_dim1 = Self.OutputM if (
        Self.MMA_M == 256 or Self.cta_group == 1 or Self.config.AB_swapped
    ) else 64
    comptime c_tile_dim2 = Self.c_swizzle_elems if (
        Self.config.AB_swapped
    ) else Self.OutputN

    # A, B, C: 3D TMA layouts (batch=1, rows, cols)
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
    comptime CTileLayout = RowMajorLayout[
        *_IntToComptimeInt[1, Self.c_tile_dim1, Self.c_tile_dim2]
    ]
    comptime CDescLayout = tma_desc_layout_3d[
        Self.c_type, 1, Self.c_tile_dim1, Self.config.c_swizzle
    ]

    # SFA, SFB: 5D TMA layouts (batch=1, then 4D scale factor dims)
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

    # TMA load size constants
    comptime a_tma_load_size = Self.a_tile_dim1 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim1 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim1
    comptime b_tma_rows = Self.b_tile_dim1

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
        constrained[
            Self.max_groups >= 1,
            "max_groups must be at least 1",
        ]()
        constrained[Self.transpose_b, "Only support transposed B"]()

    # ========== TMA Update Helper ==========

    @staticmethod
    @always_inline
    fn _update_tensormap_address[
        dtype: DType,
        cta_tile_layout: Layout,
        desc_layout: Layout,
    ](
        tma_desc_ptr: UnsafePointer[
            TMATensorTile[dtype, cta_tile_layout, desc_layout], MutAnyOrigin
        ],
        new_addr: Int,
    ):
        """Update TMA descriptor global address using inline assembly.

        Args:
            tma_desc_ptr: Pointer to the TMA descriptor in device memory.
            new_addr: New global memory address for the tensor.
        """
        # Get pointer to the descriptor bytes
        var desc_byte_ptr = tma_desc_ptr.bitcast[NoneType]()

        # Use PTX instruction to update the global address
        inlined_assembly[
            "tensormap.replace.tile.global_address.global.b1024.b64 [$0], $1;",
            NoneType,
            constraints="l,l",
            has_side_effect=True,
        ](desc_byte_ptr, new_addr)

    # ========== Kernel Entry Point ==========

    # ========== Problem Sizes Layout ==========
    # Layout for problem_sizes tensor: (max_groups, 4) with [M, N, K, L] per group
    comptime ProblemSizesLayout = _ProblemSizesLayout[Self.max_groups]
    comptime ProblemSizesTile = _ProblemSizesTile[Self.max_groups]

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_template, `nvvm.grid_constant`)
    fn run(
        # Template tensormaps for SMEM initialization
        a_tma_template: Self.ATmaOp,
        b_tma_template: Self.BTmaOp,
        c_tma_template: Self.CTmaOp,
        sfa_tma_template: Self.SFATmaOp,
        sfb_tma_template: Self.SFBTmaOp,
        # Per-block updatable tensormaps
        device_tma_a: Self.TMATensorTileArrayA,
        device_tma_b: Self.TMATensorTileArrayB,
        device_tma_sfa: Self.TMATensorTileArraySFA,
        device_tma_sfb: Self.TMATensorTileArraySFB,
        device_tma_c: Self.TMATensorTileArrayC,
        # Per-group pointer arrays (uint64 addresses)
        group_a_ptrs_lt: Self.GroupPtrTile,
        group_b_ptrs_lt: Self.GroupPtrTile,
        group_c_ptrs_lt: Self.GroupPtrTile,
        group_sfa_ptrs_lt: Self.GroupPtrTile,
        group_sfb_ptrs_lt: Self.GroupPtrTile,
        # Per-group problem sizes: (num_groups, 4) with [M, N, K, L]
        problem_sizes_lt: Self.ProblemSizesTile,
        # Number of active groups
        num_groups: Int,
    ):
        """Grouped block-scaled GEMM kernel entry point.

        This kernel processes multiple GEMM problems (groups) with dynamic
        tensormap updates at group boundaries.
        """
        Self.validate_config()

        # Alias kernel args for internal methods
        var group_a_ptrs = group_a_ptrs_lt
        var group_b_ptrs = group_b_ptrs_lt
        var group_c_ptrs = group_c_ptrs_lt
        var group_sfa_ptrs = group_sfa_ptrs_lt
        var group_sfb_ptrs = group_sfb_ptrs_lt
        var problem_sizes = problem_sizes_lt

        # ===== Shared Memory Setup =====
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

        # ===== Kernel Context =====
        var ctx = Self.Context(tmem_addr_storage)

        # ===== Grouped Tile Scheduler =====
        var scheduler = Self.SchedulerType(problem_sizes, num_groups)

        # Per-warp work iterator
        var work_iter = scheduler.work_iterator()

        # ===== Barrier Initialization =====
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_template.prefetch_descriptor()
            b_tma_template.prefetch_descriptor()
            c_tma_template.prefetch_descriptor()
            sfa_tma_template.prefetch_descriptor()
            sfb_tma_template.prefetch_descriptor()

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
                Int32(Self.EPILOGUE_THREADS * Self.cta_group)
            )

        fence_mbarrier_init()
        cluster_sync()

        var mma_op = Self.MmaOp()

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():
            var blk = Int(block_idx.x)
            var tensormap_init_done = False

            # Tensormap manager for SMEM descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a),
                    UnsafePointer(to=smem.tensormap_b),
                    UnsafePointer(to=smem.tensormap_sfa),
                    UnsafePointer(to=smem.tensormap_sfb),
                    UnsafePointer(to=smem.tensormap_c),
                ),
            )

            with input_pipeline.producer() as producer:
                while work_iter.has_work():
                    with work_iter.next() as current:
                        # Wait for MMA warp to init tensormaps (first time only)
                        if not tensormap_init_done:
                            Self.TensormapAbInitBarrier.sync()
                            tensormap_init_done = True

                        # === LOOKAHEAD PATTERN (CuteDSL style) ===
                        # Initialize to "ready" (True), only peek if there's work.
                        # This avoids wasted try_acquire when num_k_iters == 0.
                        var num_k_iters = Int(current.k_tile_count)
                        var next_ready = True
                        if num_k_iters > 0:
                            next_ready = producer.try_acquire()

                        # Update tensormaps on group change (overlaps with peek)
                        if current.group_changed:
                            tensormap_mgr.update_ab_for_group(
                                current.group_idx,
                                group_a_ptrs,
                                group_b_ptrs,
                                group_sfa_ptrs,
                                group_sfb_ptrs,
                                device_tma_a[blk],
                                device_tma_b[blk],
                                device_tma_sfa[blk],
                                device_tma_sfb[blk],
                            )

                        # Load tiles using lookahead pattern
                        for k_tile in range(num_k_iters):
                            with producer.acquire_if_needed(
                                next_ready
                            ) as tiles:
                                Self.load_input_tiles(
                                    device_tma_a[blk][],
                                    device_tma_b[blk][],
                                    device_tma_sfa[blk][],
                                    device_tma_sfb[blk][],
                                    tiles,
                                    ctx.peer_cta_coord,
                                    (
                                        UInt(current.m),
                                        UInt(current.n),
                                        UInt(0),  # batch = 0 for grouped
                                    ),
                                    ctx.a_multicast_mask,
                                    ctx.b_multicast_mask,
                                    UInt32(k_tile),
                                    ctx.elect_one_cta,
                                )
                            # Peek for next iteration (CuteDSL style):
                            # Reset to ready, then conditionally peek.
                            next_ready = True
                            if k_tile + 1 < num_k_iters:
                                next_ready = producer.try_acquire()

                        syncwarp()

                producer.drain()

        # ===== SCHEDULER WARP =====
        # For grouped GEMM, no CLC scheduling is needed (num_clc_pipeline_stages=0).
        # The scheduler warp just returns immediately, matching working kernel pattern.
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            # No CLC for grouped GEMM - just return
            return

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            # Initialize SMEM tensormaps from templates (MMA warp, per CuTe DSL)
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a),
                    UnsafePointer(to=smem.tensormap_b),
                    UnsafePointer(to=smem.tensormap_sfa),
                    UnsafePointer(to=smem.tensormap_sfb),
                    UnsafePointer(to=smem.tensormap_c),
                ),
            )
            tensormap_mgr.init_ab_tensormaps(
                a_tma_template,
                b_tma_template,
                sfa_tma_template,
                sfb_tma_template,
            )

            # Barrier sync with TMA warp - signal init complete
            Self.TensormapAbInitBarrier.sync()

            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            with mma_ctx:
                while work_iter.has_work():
                    # Use wait_and_advance() like working kernel
                    with work_iter.wait_and_advance() as current:
                        if ctx.elect_one_cta:
                            with mma_ctx.output_pipeline.producer() as output_stage:
                                var tmem_offset = UInt32(
                                    output_stage.tmem.offset()
                                )

                                with input_pipeline.consumer() as consumer:
                                    var num_k_iters = Int(current.k_tile_count)

                                    # === LOOKAHEAD PATTERN (CuteDSL style) ===
                                    # Initialize to "ready" (True), only peek if there's work.
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
                                                0,  # k_start = 0 for each group
                                            )
                                        # Peek for next iteration (CuteDSL style):
                                        # Reset to ready, then conditionally peek.
                                        next_ready = True
                                        if k_tile + 1 < num_k_iters:
                                            next_ready = consumer.try_acquire()

        # ===== EPILOGUE WARPS =====
        if WarpRole.is_epilogue():
            var blk = Int(block_idx.x)

            # Tensormap manager for C descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a),
                    UnsafePointer(to=smem.tensormap_b),
                    UnsafePointer(to=smem.tensormap_sfa),
                    UnsafePointer(to=smem.tensormap_sfb),
                    UnsafePointer(to=smem.tensormap_c),
                ),
            )

            # Init C tensormap in SMEM (epilogue warp, per CuTe DSL)
            tensormap_mgr.init_c_tensormap(c_tma_template)
            syncwarp()

            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with epi_ctx:
                while work_iter.has_work():
                    with work_iter.next() as current:
                        # Update C tensormap on group change
                        if current.group_changed:
                            tensormap_mgr.update_c_for_group(
                                current.group_idx,
                                group_c_ptrs,
                                device_tma_c[blk],
                            )

                        # Get current group's M, N dimensions
                        var g = Int(current.group_idx)
                        var group_m = UInt32(Int(problem_sizes[g, 0]))
                        var group_n = UInt32(Int(problem_sizes[g, 1]))

                        # Use per-block GMEM tensormap for epilogue
                        with epi_ctx.output_pipeline.consumer() as output_stage:
                            Self.epilogue(
                                c_tiles,
                                device_tma_c[blk][],
                                output_stage,
                                (current.m, current.n, UInt32(0)),
                                group_m,
                                group_n,
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
        work_tile_coord: Tuple[UInt, UInt, UInt],
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        iter_idx: UInt32,
        elect_one_cta: Bool,
    ):
        """Load A, B, SFA, SFB tiles using TMA with InputProducerStage."""
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
        var batch_coord = work_tile_coord[2]

        if elect_one_sync():
            if elect_one_cta:
                tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()

            @parameter
            for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Self.config.k_group_size](
                        tiles.stage(), jj
                    )
                )

                # Peer CTA slicing using TileTensor pattern (ptr + layout)
                var a_peer_tile = type_of(a_tile)(
                    a_tile.ptr + peer_m_rank * UInt(Self.a_tma_load_size),
                    a_tile.layout,
                )
                var b_peer_tile = type_of(b_tile)(
                    b_tile.ptr + peer_rank_m * UInt(Self.b_tma_load_size),
                    b_tile.layout,
                )

                var k_coord = UInt(iter_idx + j) * UInt(Self.BK)

                # TileTensor directly to TMA (uses TileTensor overload)
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
        tmem_region: Self.TmemRegion,
        iter_idx: UInt32,
        k_start: UInt32,
    ):
        """Execute MMA operations using InputConsumerStage."""
        if elect_one_sync():

            @parameter
            for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor (native SMEM storage)
                var a_tile, b_tile, sfa_tile, sfb_tile = (
                    tiles.payload().get_tile[Self.config.k_group_size](
                        tiles.stage(), jj
                    )
                )

                var tile_idx = (
                    Int(tiles.stage()) * Self.config.k_group_size + jj
                )

                var sfa_tmem_offset = UInt32(tmem_region.sfa(tile_idx).col_addr)
                var sfb_tmem_offset = UInt32(tmem_region.sfb(tile_idx).col_addr)

                var is_first_k = (iter_idx + j) == k_start

                # MMA has TileTensor overload - pass tiles directly
                # (layout is extracted from TileTensor type parameters)
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

    # ========== Epilogue ==========

    @staticmethod
    @always_inline
    fn epilogue(
        c_tiles: Self.SmemType.CTileArray,
        c_tma_op: Self.CTmaOp,
        stage: Self.TileWriterType.Stage,
        work_tile_coord: Tuple[UInt32, UInt32, UInt32],
        M: UInt32,
        N: UInt32,
        alpha: Float32 = Float32(1.0),
    ):
        """Execute epilogue to store accumulated results."""
        var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))
        tile_writer.write_batched(
            c_tiles,
            stage,
            work_tile_coord,
            (M, N),
            alpha,
        )

    # ========== 2SM Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 1, 1))
    @__llvm_arg_metadata(a_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_template, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_template, `nvvm.grid_constant`)
    fn run_2sm(
        # Template tensormaps for SMEM initialization
        a_tma_template: Self.ATmaOp,
        b_tma_template: Self.BTmaOp,
        c_tma_template: Self.CTmaOp,
        sfa_tma_template: Self.SFATmaOp,
        sfb_tma_template: Self.SFBTmaOp,
        # Per-block updatable tensormaps
        device_tma_a: Self.TMATensorTileArrayA,
        device_tma_b: Self.TMATensorTileArrayB,
        device_tma_sfa: Self.TMATensorTileArraySFA,
        device_tma_sfb: Self.TMATensorTileArraySFB,
        device_tma_c: Self.TMATensorTileArrayC,
        # Per-group pointer arrays (uint64 addresses)
        group_a_ptrs_lt: Self.GroupPtrTile,
        group_b_ptrs_lt: Self.GroupPtrTile,
        group_c_ptrs_lt: Self.GroupPtrTile,
        group_sfa_ptrs_lt: Self.GroupPtrTile,
        group_sfb_ptrs_lt: Self.GroupPtrTile,
        # Per-group problem sizes: (num_groups, 4) with [M, N, K, L]
        problem_sizes_lt: Self.ProblemSizesTile,
        # Number of active groups
        num_groups: Int,
    ):
        """Grouped block-scaled GEMM kernel with 2SM (cta_group=2) support.

        This entry point uses CLC-based work distribution for proper 2SM
        synchronization between CTAs in a cluster. Both CTAs cooperate on
        each tile, with one CTA doing MMA work and both doing TMA loads.

        Architecture matches the working block_scaled_matmul_kernel:
        - Scheduler warp: Produces work items via CLC barriers
        - TMA warp: Loads tiles with tensormap updates on group change
        - MMA warp: Waits on CLC, executes MMA (elected CTA only)
        - Epilogue warps: Stores results with tensormap updates
        """
        Self.validate_config()

        # Alias kernel args for internal methods
        var group_a_ptrs = group_a_ptrs_lt
        var group_b_ptrs = group_b_ptrs_lt
        var group_c_ptrs = group_c_ptrs_lt
        var group_sfa_ptrs = group_sfa_ptrs_lt
        var group_sfb_ptrs = group_sfb_ptrs_lt
        var problem_sizes = problem_sizes_lt

        # ===== Shared Memory Setup =====
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

        # Get typed barrier arrays
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var clc_full = smem.pipelines.clc_full()
        var clc_empty = smem.pipelines.clc_empty()
        var clc_throttle = smem.pipelines.clc_throttle()
        var clc_response = smem.pipelines.clc_response()
        var tmem_addr_storage = smem.pipelines.tmem_addr().ptr

        # Create input pipeline with tile payload
        var tile_payload = Self.TilePayload(
            a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )
        var input_pipeline = Self.InputTilePipelineType(
            input_barriers, tile_payload
        )

        # ===== Kernel Context =====
        var ctx = Self.Context(tmem_addr_storage)

        # ===== Initial Work Info =====
        # Compute initial work from first cluster's tile
        var initial_linear_idx = UInt32(block_idx.x // 2)  # 2SM: cta_group=2
        var initial_work = Self._compute_initial_work(
            problem_sizes, num_groups, initial_linear_idx
        )

        # ===== Barrier Initialization =====
        if ctx.elect_one_warp and ctx.elect_one_thread:
            a_tma_template.prefetch_descriptor()
            b_tma_template.prefetch_descriptor()
            c_tma_template.prefetch_descriptor()
            sfa_tma_template.prefetch_descriptor()
            sfb_tma_template.prefetch_descriptor()

            # Initialize input pipeline barriers (for 2SM cluster)
            Self.InputTilePipelineType.init_barriers(
                input_barriers.ptr,
                Int32(1),
                Int32(
                    Self.config.cluster_shape[0] // 2  # cta_group=2
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

            # Initialize CLC barriers
            @parameter
            for i in range(Self.num_clc_pipeline_stages_2sm):
                clc_full.ptr[i].init(Self.clc_producer_arv_count)
                clc_empty.ptr[i].init(Int32(Self.clc_consumer_arv_count))

            # Initialize throttle barriers
            @parameter
            for i in range(Self.num_clc_pipeline_stages_2sm * 2):
                clc_throttle.ptr[i].init(
                    Int32(
                        Self.clc_throttle_producer_arv_count if i
                        < Self.num_clc_pipeline_stages_2sm else Self.clc_throttle_consumer_arv_count
                    )
                )

            # Initialize TMEM deallocation barrier (for cta_group=2)
            smem.pipelines.tmem_dealloc().ptr[].init(
                Int32(Self.EPILOGUE_THREADS * 2)
            )

        fence_mbarrier_init()
        cluster_sync()

        var mma_op = Self.MmaOp()

        # ===== TMA LOAD WARP =====
        if WarpRole.is_main_load():
            var blk = Int(block_idx.x)
            var tensormap_init_done = False

            # Tensormap manager for SMEM descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a),
                    UnsafePointer(to=smem.tensormap_b),
                    UnsafePointer(to=smem.tensormap_sfa),
                    UnsafePointer(to=smem.tensormap_sfb),
                    UnsafePointer(to=smem.tensormap_c),
                ),
            )

            # Create CLC work iterator for TMA warp
            var work_iter = GroupedCLCWorkIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            with input_pipeline.producer() as producer:
                while work_iter.has_work():
                    with work_iter.next() as current:
                        # Signal throttle (first CTA only)
                        work_iter.throttle_signal(ctx.is_first_cta_in_cluster)

                        # Wait for MMA warp to init tensormaps (first time only)
                        if not tensormap_init_done:
                            Self.TensormapAbInitBarrier.sync()
                            tensormap_init_done = True

                        # === LOOKAHEAD PATTERN (CuteDSL style) ===
                        # Initialize to "ready" (True), only peek if there's work.
                        var num_k_iters = Int(current.k_tile_count)
                        var next_ready = True
                        if num_k_iters > 0:
                            next_ready = producer.try_acquire()

                        # Update tensormaps on group change (overlaps with peek)
                        if current.group_changed:
                            tensormap_mgr.update_ab_for_group(
                                current.group_idx,
                                group_a_ptrs,
                                group_b_ptrs,
                                group_sfa_ptrs,
                                group_sfb_ptrs,
                                device_tma_a[blk],
                                device_tma_b[blk],
                                device_tma_sfa[blk],
                                device_tma_sfb[blk],
                            )

                        # Load tiles using lookahead pattern
                        for k_tile in range(num_k_iters):
                            with producer.acquire_if_needed(
                                next_ready
                            ) as tiles:
                                Self.load_input_tiles(
                                    device_tma_a[blk][],
                                    device_tma_b[blk][],
                                    device_tma_sfa[blk][],
                                    device_tma_sfb[blk][],
                                    tiles,
                                    ctx.peer_cta_coord,
                                    (
                                        UInt(current.m),
                                        UInt(current.n),
                                        UInt(0),
                                    ),
                                    ctx.a_multicast_mask,
                                    ctx.b_multicast_mask,
                                    UInt32(k_tile),
                                    ctx.elect_one_cta,
                                )
                            # Peek for next iteration (CuteDSL style):
                            # Reset to ready, then conditionally peek.
                            next_ready = True
                            if k_tile + 1 < num_k_iters:
                                next_ready = producer.try_acquire()
                        syncwarp()

                producer.drain()

        # ===== SCHEDULER WARP =====
        if WarpRole.is_scheduler() and ctx.is_first_cta_in_cluster:
            # Create scheduler iterator for CLC work production
            var sched_iter = GroupedCLCSchedulerIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            while sched_iter.has_work():
                with sched_iter.next():
                    sched_iter.signal_and_advance()

            sched_iter.drain()

        # ===== MMA WARP =====
        if WarpRole.is_mma():
            # Initialize SMEM tensormaps from templates (MMA warp, per CuTe DSL)
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a),
                    UnsafePointer(to=smem.tensormap_b),
                    UnsafePointer(to=smem.tensormap_sfa),
                    UnsafePointer(to=smem.tensormap_sfb),
                    UnsafePointer(to=smem.tensormap_c),
                ),
            )
            tensormap_mgr.init_ab_tensormaps(
                a_tma_template,
                b_tma_template,
                sfa_tma_template,
                sfb_tma_template,
            )

            # Barrier sync with TMA warp
            Self.TensormapAbInitBarrier.sync()

            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            # Create CLC work iterator for MMA warp
            var work_iter = GroupedCLCWorkIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            with mma_ctx:
                while work_iter.has_work():
                    # Wait on CLC for next work (synchronizes both CTAs)
                    with work_iter.wait_and_advance() as current:
                        if ctx.elect_one_cta:
                            with mma_ctx.output_pipeline.producer() as output_stage:
                                var tmem_offset = UInt32(
                                    output_stage.tmem.offset()
                                )

                                with input_pipeline.consumer() as consumer:
                                    var num_k_iters = Int(current.k_tile_count)

                                    # === LOOKAHEAD PATTERN (CuteDSL style) ===
                                    # Initialize to "ready" (True), only peek if there's work.
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
                                        # Peek for next iteration (CuteDSL style):
                                        # Reset to ready, then conditionally peek.
                                        next_ready = True
                                        if k_tile + 1 < num_k_iters:
                                            next_ready = consumer.try_acquire()

        # ===== EPILOGUE WARPS =====
        if WarpRole.is_epilogue():
            var blk = Int(block_idx.x)

            # Tensormap manager for C descriptor updates
            var tensormap_mgr = Self.TensormapManagerType(
                smem=GroupedTensormapSmem.from_smem(
                    UnsafePointer(to=smem.tensormap_a),
                    UnsafePointer(to=smem.tensormap_b),
                    UnsafePointer(to=smem.tensormap_sfa),
                    UnsafePointer(to=smem.tensormap_sfb),
                    UnsafePointer(to=smem.tensormap_c),
                ),
            )

            # Init C tensormap in SMEM
            tensormap_mgr.init_c_tensormap(c_tma_template)
            syncwarp()

            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers, tmem, UInt16(ctx.mma_complete_mask)
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            # Create work iterator for epilogue (uses simple advance, not CLC)
            var work_iter = GroupedCLCWorkIterator[
                Self.BM,
                Self.BN,
                Self.BK,
                Self.max_groups,
                Self.num_clc_pipeline_stages_2sm,
                2,  # cta_group=2
            ](
                problem_sizes,
                num_groups,
                clc_full.ptr,
                clc_empty.ptr,
                clc_response.ptr,
                clc_throttle.ptr,
                initial_work,
            )

            with epi_ctx:
                while work_iter.has_work():
                    with work_iter.next() as current:
                        # Update C tensormap on group change
                        if current.group_changed:
                            tensormap_mgr.update_c_for_group(
                                current.group_idx,
                                group_c_ptrs,
                                device_tma_c[blk],
                            )

                        # Get current group's M, N dimensions
                        var g = Int(current.group_idx)
                        var group_m = UInt32(Int(problem_sizes[g, 0]))
                        var group_n = UInt32(Int(problem_sizes[g, 1]))

                        with epi_ctx.output_pipeline.consumer() as output_stage:
                            Self.epilogue(
                                c_tiles,
                                device_tma_c[blk][],
                                output_stage,
                                (current.m, current.n, UInt32(0)),
                                group_m,
                                group_n,
                            )

    @staticmethod
    @always_inline
    fn _compute_initial_work(
        problem_sizes: Self.ProblemSizesTile,
        num_groups: Int,
        linear_idx: UInt32,
    ) -> GroupedWorkInfo:
        """Compute initial work info from linear tile index."""
        # Build cumulative tiles
        var cumulative = StaticTuple[UInt32, Self.max_groups + 1]()

        @parameter
        for i in range(Self.max_groups + 1):
            cumulative[i] = 0

        var cumsum: UInt32 = 0
        for g in range(num_groups):
            var m = UInt32(Int(problem_sizes[g, 0]))
            var n = UInt32(Int(problem_sizes[g, 1]))
            var m_tiles = ceildiv(Int(m), Self.BM)
            var n_tiles = ceildiv(Int(n), Self.BN)
            cumsum += UInt32(m_tiles * n_tiles)
            cumulative[g + 1] = cumsum

        if linear_idx >= cumsum:
            return GroupedWorkInfo()

        # Binary search for group
        var lo: UInt32 = 0
        var hi: UInt32 = UInt32(num_groups)
        while lo < hi:
            var mid = (lo + hi) / 2
            if linear_idx < cumulative[Int(mid + 1)]:
                hi = mid
            else:
                lo = mid + 1
        var group_idx = lo

        var local_idx = linear_idx - cumulative[Int(group_idx)]
        var m = UInt32(Int(problem_sizes[Int(group_idx), 0]))
        var k = UInt32(Int(problem_sizes[Int(group_idx), 2]))
        var m_tiles = ceildiv(Int(m), Self.BM)
        var k_tiles = ceildiv(Int(k), Self.BK)

        var m_tile = local_idx % UInt32(m_tiles)
        var n_tile = local_idx / UInt32(m_tiles)

        return GroupedWorkInfo(
            m=m_tile,
            n=n_tile,
            k_start=0,
            is_valid_tile=True,
            group_idx=group_idx,
            k_tile_count=UInt32(k_tiles),
            group_changed=True,  # First tile always triggers update
        )
