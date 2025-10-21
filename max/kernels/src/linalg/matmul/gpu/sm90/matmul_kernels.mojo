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
from sys import simd_width_of, size_of

from buffer.buffer import NDBuffer
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.device_context import DeviceBuffer
from gpu.id import (
    block_id_in_cluster,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu.id import warp_id as get_warp_id
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import (
    AddressSpace,
    fence_mbarrier_init,
    fence_async_view_proxy,
)
from gpu.mma import st_matrix
from gpu.sync import named_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import (
    copy_local_to_dram,
    copy_sram_to_dram,
)
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout, RuntimeTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    st_matrix_n_layout,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
)
from memory import bitcast, stack_allocation
from stdlib.bit import log2_floor

from utils.index import Index, IndexList
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from ....utils_gpu import block_swizzle
from ..tile_scheduler import MatmulSchedule, TileScheduler, RasterOrder
from ..tile_scheduler_splitk import SplitKTileScheduler
from ....structuring import NVIDIASharedMemoryManager as SharedMemoryManager
from ....structuring import (
    SMemTileType,
    RegTileType,
)
from .ring_buffer import RingBuffer, RingBufferConsumer, RingBufferProducer
from .scatter_gather import (
    ScatterGatherTMA,
    ScatterGatherCPAsync,
    ScatterGather,
)


# Shared memory structure for Hopper SM90 kernel
@register_passable("trivial")
struct HopperMatmulSM90Kernel_SMem[
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    c_type: DType,
    c_layout: Layout,
    num_pipeline_stages: Int,
]:
    """Shared memory layout for Hopper SM90 matrix multiplication kernel.

    This struct manages the shared memory allocation for:
    - Input tiles (A and B matrices) with multi-stage pipelining
    - Output tile (C matrix) for accumulation
    - Synchronization barriers for producer-consumer coordination

    The memory is organized to support asynchronous loads and efficient
    bank-conflict-free access patterns for tensor core operations.
    """

    alias SMM = SharedMemoryManager[]

    # Tile iterator types - manage cycling through pipeline stages
    alias ATileArray = Self.SMM.TileArray[a_type, a_layout, num_pipeline_stages]
    alias BTileArray = Self.SMM.TileArray[b_type, b_layout, num_pipeline_stages]
    alias CTile = Self.SMM.Tile[c_type, c_layout]

    # Pipeline barrier types - for producer/consumer synchronization
    alias PipelineBarrier = Self.SMM.Array[
        SharedMemBarrier, num_pipeline_stages
    ]

    # Tile iterators - cycle through pipeline stages
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var c_tile: Self.CTile

    # Pipeline barriers:
    # - full_mbar: Signals when tiles are loaded and ready for consumption
    # - empty_mbar: Signals when tiles have been consumed and can be reused
    var full_mbar: Self.PipelineBarrier
    var empty_mbar: Self.PipelineBarrier

    fn __init__(out self):
        var smem_mgr = Self.SMM()
        # Initialize tile iterators
        self.a_tiles = smem_mgr.build[T = Self.ATileArray]()
        self.b_tiles = smem_mgr.build[T = Self.BTileArray]()
        self.c_tile = smem_mgr.build[T = Self.CTile]()
        # Initialize barriers
        self.full_mbar = smem_mgr.build[T = Self.PipelineBarrier]()
        self.empty_mbar = smem_mgr.build[T = Self.PipelineBarrier]()

    @staticmethod
    @always_inline
    fn pipeline_storage_size() -> Int:
        """Calculate the memory size for all pipeline stages."""
        var a_size = Self.ATileArray.storage_size
        var b_size = Self.BTileArray.storage_size

        return (
            # A and B tile iterators with padding
            a_size
            + b_size
            # Pipeline barriers (full + empty)
            + 2 * Self.PipelineBarrier.storage_size
        )

    @staticmethod
    @always_inline
    fn output_storage_size() -> Int:
        """Calculate the memory size for output tile."""
        return Self.CTile.storage_size

    @staticmethod
    @always_inline
    fn storage_size() -> Int:
        """Calculate the total storage size."""
        return Self.pipeline_storage_size() + Self.output_storage_size()


struct HopperMatmulSM90Kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    c_smem_layout: Layout,
    block_tile_shape: IndexList[3],
    wgmma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    num_pipeline_stages: Int,
    num_threads: Int = 128,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE,
    partitioned_multicast: Bool = False,
    use_tma_store: Bool = False,
    promotion_frequency: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
    hilbert_swizzle: Bool = False,
]:
    """Hopper SM90 Matrix Multiplication kernel optimized for NVIDIA H100 GPUs.

    This kernel implements a highly optimized matrix multiplication (GEMM) using:
    - Tensor Memory Accelerator (TMA) for efficient global-to-shared memory transfers
    - Warp Group Matrix Multiply Accumulate (WGMMA) instructions for tensor cores
    - Multi-stage software pipelining for overlapping compute and memory operations
    - Producer-consumer model with separate warp groups for loading and computing

    Template Parameters:
        a_type, b_type, c_type: Data types for input and output matrices
        a_layout, b_layout, c_layout: Memory layouts for matrices
        c_smem_layout: Shared memory layout for output tile
        block_tile_shape: Tile dimensions [M, N, K] processed by each thread block
        wgmma_shape: Dimensions for each WGMMA instruction [M, N, K]
        cluster_shape: Thread block cluster dimensions for distributed shared memory
        num_pipeline_stages: Number of stages in the software pipeline (typically 3-7)
        num_threads: Number of threads per block (must be multiple of 128)
        transpose_b: Whether B matrix is transposed (required to be True)
        a_swizzle, b_swizzle: Memory swizzling for bank-conflict-free access
        c_swizzle: Swizzling for output writes
        partitioned_multicast: Enable partitioned multicast for large tiles
        use_tma_store: Use TMA for storing output (vs regular stores)
        promotion_frequency: How often to promote FP8 accumulation to higher precision
        pdl_level: Programmatic Dependency Launch (PDL) level
        elementwise_lambda_fn: Optional epilogue function
        elementwise_compute_lambda_fn: Optional compute function
        hilbert_swizzle: Use Hilbert curve for thread block scheduling
    """

    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

    alias num_consumer = (num_threads // 128) - 1
    alias num_consumer_threads = Self.num_consumer * 128

    alias num_m_mmas = Self.BM // wgmma_shape[0] // Self.num_consumer
    alias num_n_mmas = Self.BN // wgmma_shape[1]

    alias accum_type = get_accum_type[a_type]()
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // 128

    alias a_smem_layout = tile_layout_k_major[
        a_type, Self.BM, Self.BK, a_swizzle
    ]()
    alias b_smem_layout = tile_layout_k_major[
        b_type, Self.BN, Self.BK, b_swizzle
    ]()

    alias AccumRegTileType = RegTileType[
        Self.accum_type,
        Layout.row_major(Self.num_m_mmas * Self.num_n_mmas, Self.c_frag_size),
    ]

    alias cluster_size = Int(
        cluster_shape[0] * cluster_shape[1] * cluster_shape[2]
    )

    alias SMem = HopperMatmulSM90Kernel_SMem[
        a_type,
        Self.a_smem_layout,
        b_type,
        Self.b_smem_layout,
        c_type,
        c_smem_layout,
        num_pipeline_stages,
    ]

    alias RingBuffer[tma_transfer: Bool = True] = RingBuffer[
        a_type,
        b_type,
        Self.a_smem_layout,
        Self.b_smem_layout,
        num_pipeline_stages,
        Self.num_consumer,
        Self.cluster_size,
        tma_transfer,
    ]

    alias RingBufferConsumer[
        origin: Origin[True], tma_transfer: Bool
    ] = RingBufferConsumer[origin, Self.RingBuffer[tma_transfer]]

    alias WgmmaOp = TensorCoreAsync[
        Self.accum_type,
        a_type,
        b_type,
        wgmma_shape,
        a_swizzle,
        b_swizzle,
        transpose_b,
    ]

    @staticmethod
    @always_inline
    fn validate_constraints():
        """Validate common constraints for all kernel variants."""
        constrained[a_type == b_type, "A and B must have the same type"]()

        constrained[transpose_b, "Only support transposed B in layout"]()

        constrained[
            not partitioned_multicast
            or a_swizzle.bytes() // size_of[a_type]() == Self.BK,
            (
                "Currently partitioned multi-casting is only supported when BK"
                " == (a_swizzle.bytes // size_of[a_type])"
            ),
        ]()
        constrained[
            not partitioned_multicast
            or b_swizzle.bytes() // size_of[b_type]() == Self.BK,
            (
                "Currently partitioned multi-casting is only supported when BK"
                " == (b_swizzle.bytes // size_of[b_type])"
            ),
        ]()

    @always_inline
    @staticmethod
    fn pipeline_init():
        """Initialize pipeline synchronization barriers.

        This function ensures that all pipeline initialization (barriers, shared memory)
        is visible to all thread blocks in the cluster before proceeding. This is
        critical for correct producer-consumer synchronization.

        For multi-cluster configurations, uses fence and cluster sync.
        For single block, uses a simple barrier.
        """

        @parameter
        if Self.cluster_size > 1:
            fence_mbarrier_init()
            cluster_sync_relaxed()
        else:
            barrier()

    @staticmethod
    @always_inline
    fn finalize_kernel():
        """Common finalization for all kernel variants."""

        @parameter
        if pdl_level >= PDLLevel.OVERLAP_AT_END:
            launch_dependent_grids()

        # Synchronize all thread blocks in the cluster before kernel exit
        # to ensure shared memory isn't deallocated while other blocks are still using it
        @parameter
        if Self.cluster_size > 1:
            cluster_sync()

    @staticmethod
    @always_inline
    fn multicast_mask(rank_m: UInt, rank_n: UInt) -> Tuple[Int32, Int32]:
        alias CLUSTER_N = cluster_shape[0]
        alias CLUSTER_M = cluster_shape[1]

        # Setup multicast masks for cluster-wide data distribution
        var multicast_column_mask = 0

        @parameter
        for i in range(CLUSTER_M):
            multicast_column_mask |= Int(1 << (i * CLUSTER_N))
        multicast_column_mask <<= rank_n

        var multicast_row_mask = ((1 << CLUSTER_N) - 1) << (rank_m * CLUSTER_N)
        return (multicast_row_mask, multicast_column_mask)

    @staticmethod
    @always_inline
    fn common_kernel_init() -> (
        Tuple[
            UInt,
            UInt,
            UInt,
            UInt,
            UInt,
            Bool,
        ]
    ):
        """Common initialization for all kernel variants.

        Returns:
            Tuple of (warp_group_idx, warp_group_thread_idx,
                     rank_m, rank_n, warp_id, lane_predicate).
        """
        Self.validate_constraints()

        var warp_group_idx, warp_group_thread_idx = divmod(
            thread_idx.x, UInt(WARPGROUP_SIZE)
        )

        var rank_m = block_id_in_cluster.y
        var rank_n = block_id_in_cluster.x

        # Check and wait for PDL grids if needed
        @parameter
        if (
            pdl_level > PDLLevel.OFF
            and pdl_level != PDLLevel.NO_WAIT_OVERLAP_AT_END
        ):
            wait_on_dependent_grids()

        var warp_id = get_warp_id()
        var lane_predicate = elect_one_sync()

        return (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        )

    @staticmethod
    @always_inline
    fn build_ring_buffer[
        tma_transfer: Bool = True
    ](
        smem: Self.SMem,
        warp_group_thread_idx: UInt,
    ) -> Self.RingBuffer[
        tma_transfer
    ]:
        """Create ring buffer for producer-consumer synchronization."""
        return Self.RingBuffer[tma_transfer](
            smem.full_mbar.ptr,
            smem.empty_mbar.ptr,
            warp_group_thread_idx,
            smem.a_tiles,
            smem.b_tiles,
        )

    @staticmethod
    @always_inline
    fn setup_producer() -> Int:
        """Setup producer warp group by deallocating registers.

        Returns:
            Number of registers deallocated.
        """
        alias num_regs = 24 if Self.num_consumer <= 2 else 32
        warpgroup_reg_dealloc[num_regs]()
        return num_regs

    @staticmethod
    @always_inline
    fn setup_consumer(
        warp_group_idx: UInt,
    ) -> Tuple[UInt, Self.AccumRegTileType, Self.AccumRegTileType]:
        """Setup consumer warp group.

        Returns:
            Tuple of (local_warp_group_idx, c_reg_tile, final_c_reg_tile).
        """

        @parameter
        fn num_regs() -> Int:
            if Self.num_consumer == 1:
                return 256
            if Self.num_consumer == 2:
                return 240
            return 160

        warpgroup_reg_alloc[num_regs()]()

        var local_warp_group_idx = warp_group_idx - 1
        var c_reg_tile = Self.AccumRegTileType.stack_allocation()
        var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

        return (local_warp_group_idx, c_reg_tile, final_c_reg_tile)

    @staticmethod
    @always_inline
    fn get_block_swizzle(
        lut_ptr: UnsafePointer[UInt32] = UnsafePointer[UInt32](),
    ) -> IndexList[2, element_type = DType.uint32]:
        """Calculate block swizzle for better L2 cache locality.

        Args:
            lut_ptr: Lookup table for Hilbert curve block scheduling (optional).

        Returns:
            Swizzled block indices.
        """
        alias use_cluster = Self.cluster_size > 1

        @parameter
        if not use_cluster:

            @parameter
            if hilbert_swizzle:
                # Hilbert curve ordering maximizes spatial locality
                var linear = UInt32(block_idx.y * grid_dim.x + block_idx.x)
                var packed = lut_ptr[linear]
                var new_x = packed & 0xFFFF
                var new_y = packed >> 16
                return Index[dtype = DType.uint32](new_x, new_y)
            else:
                # Default swizzling pattern for L2 cache optimization
                return block_swizzle(
                    Index[dtype = DType.uint32](block_idx.x, block_idx.y),
                    Index[dtype = DType.uint32](grid_dim.x, grid_dim.y),
                )
        else:
            # Multi-cluster mode: no swizzling (handled by hardware)
            return Index[dtype = DType.uint32](block_idx.x, block_idx.y)

    @staticmethod
    @always_inline
    fn consumer_output[
        custom_elementwise_lambda_fn: OptionalReg[
            elementwise_epilogue_type
        ] = elementwise_lambda_fn
    ](
        c_tma_op: TMATensorTile[c_type, _, _],
        c: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
        c_tile: Self.SMem.CTile,
        output_reg_tile: Self.AccumRegTileType,
        warp_group_thread_idx: UInt,
        local_warp_group_idx: UInt,
        local_thread_idx: UInt,
        block_y: Int,
        block_x: Int,
    ):
        """Handle consumer output using warp specialized GEMM output."""
        warp_specialized_gemm_output[
            c_tile_shape = Index(Self.BM, Self.BN),
            c_swizzle=c_swizzle,
            wgmma_shape=wgmma_shape,
            num_consumer = Self.num_consumer,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=custom_elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ](
            c_tma_op,
            c,
            c_tile,
            output_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            local_thread_idx,
            block_y,
            block_x,
        )

    @staticmethod
    @always_inline
    fn build_tma_loaders[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout, //,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        rank_m: UInt,
        rank_n: UInt,
    ) -> Tuple[
        ScatterGatherTMA[
            origin_of(a_tma_op),
            a_type,
            a_tile_layout,
            a_desc_layout,
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[0],
            use_partitioned_multicast=partitioned_multicast,
        ],
        ScatterGatherTMA[
            origin_of(b_tma_op),
            b_type,
            b_tile_layout,
            b_desc_layout,
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[1],
            use_partitioned_multicast=partitioned_multicast,
        ],
    ]:
        # Prefetch TMA descriptors if on thread 0.
        if thread_idx.x == 0:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()

        var multicast_mask = Self.multicast_mask(rank_m, rank_n)
        var a_loader = ScatterGatherTMA[
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[0],
            use_partitioned_multicast=partitioned_multicast,
        ](Pointer(to=a_tma_op), rank_n, UInt16(multicast_mask[0]))
        var b_loader = ScatterGatherTMA[
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[1],
            use_partitioned_multicast=partitioned_multicast,
        ](Pointer(to=b_tma_op), rank_m, UInt16(multicast_mask[1]))
        return (a_loader, b_loader)

    @always_inline
    @staticmethod
    fn build_cpasync_loaders[
        k_align: Int,
        vector_size: Int = k_align // size_of[a_type](),
        num_threads_per_row: Int = Self.BK // vector_size,
        thread_layout: Layout = Layout.row_major(
            WARPGROUP_SIZE // num_threads_per_row, num_threads_per_row
        ),
    ](
        a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
    ) -> Tuple[
        ScatterGatherCPAsync[
            a_type,
            a_layout,
            thread_layout,
            a_swizzle,
            vector_size,
        ],
        ScatterGatherCPAsync[
            b_type,
            b_layout,
            thread_layout,
            b_swizzle,
            vector_size,
        ],
    ]:
        var a_loader = ScatterGatherCPAsync[
            a_type,
            a_layout,
            thread_layout,
            a_swizzle,
            vector_size,
        ](a)
        var b_loader = ScatterGatherCPAsync[
            b_type,
            b_layout,
            thread_layout,
            b_swizzle,
            vector_size,
        ](b)
        return (a_loader, b_loader)

    @staticmethod
    @always_inline
    fn producer_main_loop[
        a_loader_type: ScatterGather,
        b_loader_type: ScatterGather, //,
        num_k_iters: Int,
    ](
        m_coord: UInt,
        n_coord: UInt,
        k_coord: UInt,
        a_loader: a_loader_type,
        b_loader: b_loader_type,
        mut ring_buffer: RingBuffer[
            a_loader_type._dtype,  # RingBuffer and ScatterGather must agree on dtypes
            b_loader_type._dtype,
            _,
            _,
            num_pipeline_stages,
            _,
            _,
            _,
        ],
    ):
        """Polymorphic A and B Tile Loader, works with both TMA and CPAsync."""

        @always_inline
        @parameter
        fn producer_loop[
            num_pipeline_stages_to_unroll: Int,
        ](k_iter: Int):
            @parameter
            for j in range(num_pipeline_stages_to_unroll):
                var k_offset = UInt(
                    k_coord + UInt(k_iter * num_pipeline_stages + j)
                )

                # Get the next available tile slot from the ring buffer.
                # The context manager ensures proper barrier synchronization.
                with ring_buffer.producer() as producer:
                    with producer.get_tiles() as tiles:
                        a_loader.load_tile(
                            tiles.a_tile,
                            tiles.barrier,
                            (m_coord, k_offset),
                        )
                        b_loader.load_tile(
                            tiles.b_tile,
                            tiles.barrier,
                            (n_coord, k_offset),
                        )

        # Calculate how many full pipeline iterations we need
        alias num_full_k_iters = ceildiv(num_k_iters, num_pipeline_stages)
        # Handle uneven division: the last iteration may have fewer stages
        alias num_remaining_k_iters = num_k_iters % num_pipeline_stages

        @parameter
        if num_remaining_k_iters == 0:
            for k_iter in range(num_full_k_iters):
                producer_loop[num_pipeline_stages](k_iter)
        else:
            for k_iter in range(num_full_k_iters - 1):
                producer_loop[num_pipeline_stages](k_iter)
            producer_loop[num_remaining_k_iters](num_full_k_iters - 1)

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        lut_ptr: UnsafePointer[UInt32],
    ):
        """Main kernel entry point for matrix multiplication.

        This kernel implements a producer-consumer pattern where:
        - One warp group (producer) loads tiles from global memory using TMA
        - Multiple warp groups (consumers) perform matrix multiplication using tensor cores

        The kernel uses software pipelining to overlap memory transfers with computation,
        achieving high throughput on Hopper GPUs.

        Args:
            a_tma_op: TMA descriptor for matrix A.
            b_tma_op: TMA descriptor for matrix B.
            c_tma_op: TMA descriptor for matrix C.
            a: Input matrix A.
            b: Input matrix B.
            c: Output matrix C.
            lut_ptr: Lookup table for Hilbert curve block scheduling (optional).
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create ring buffer
        var ring_buffer = Self.build_ring_buffer(smem, warp_group_thread_idx)

        # Create ScatterGatherTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle(lut_ptr)
        var m_coord = block_idx_swizzle[1] * Self.BM
        var n_coord = block_idx_swizzle[0] * Self.BN

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate:
                Self.producer_main_loop[num_k_iters=num_k_iters](
                    UInt(m_coord),
                    UInt(n_coord),
                    0,  # k_start,
                    a_loader,
                    b_loader,
                    ring_buffer,
                )
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            # Enter consumer mode
            with ring_buffer.consumer() as consumer:
                Self.consumer_main_loop[num_k_iters=num_k_iters](
                    wgmma_op,
                    local_warp_group_idx,
                    final_c_reg_tile,
                    c_reg_tile,
                    consumer,
                )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            Self.consumer_output(
                c_tma_op,
                c,
                smem.c_tile,
                output_reg_tile,
                warp_group_thread_idx,
                local_warp_group_idx,
                UInt(thread_idx.x - UInt(WARPGROUP_SIZE)),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_persistent[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
        grid_shape: IndexList[2],
        schedule: MatmulSchedule,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        problem_shape: IndexList[3],
    ):
        alias K = b_layout.shape[1].value()
        alias num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create ring buffer
        var ring_buffer = Self.build_ring_buffer(smem, warp_group_thread_idx)

        # Create ScatterGatherTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        alias N = b_layout.shape[0].value()
        alias M = a_layout.shape[0].value()
        var scheduler = TileScheduler[
            Index(M, N, K), block_tile_shape, grid_shape, schedule=schedule
        ](problem_shape)
        var work_info = scheduler.get_current_work_info()

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate:
                while work_info.is_valid():
                    var m_coord = work_info.m
                    var n_coord = work_info.n

                    Self.producer_main_loop[num_k_iters=num_k_iters](
                        UInt(m_coord),
                        UInt(n_coord),
                        0,
                        a_loader,
                        b_loader,
                        ring_buffer,
                    )
                    work_info = scheduler.fetch_next_work()
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            @parameter
            if a_type is DType.float8_e4m3fn:
                _ = final_c_reg_tile.fill(0.0)
            else:
                _ = c_reg_tile.fill(0.0)

            # Enter consumer mode
            with ring_buffer.consumer() as consumer:
                while work_info.is_valid():
                    Self.consumer_main_loop[num_k_iters=num_k_iters](
                        wgmma_op,
                        local_warp_group_idx,
                        final_c_reg_tile,
                        c_reg_tile,
                        consumer,
                    )

                    var block_y = UInt(ceildiv(work_info.m, Self.BM))
                    var block_x = UInt(ceildiv(work_info.n, Self.BN))
                    var output_reg_tile = (
                        final_c_reg_tile if a_type
                        is DType.float8_e4m3fn else c_reg_tile
                    )

                    Self.consumer_output(
                        c_tma_op,
                        c,
                        smem.c_tile,
                        output_reg_tile,
                        UInt(warp_group_thread_idx),
                        UInt(local_warp_group_idx),
                        thread_idx.x - UInt(WARPGROUP_SIZE),
                        Int(block_y),
                        Int(block_x),
                    )
                    work_info = scheduler.fetch_next_work()

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_unaligned[
        c_desc_layout: Layout,
        c_tma_layout: Layout,
    ](
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        a: LayoutTensor[a_type, a_layout, MutableAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    ):
        """Kernel using cp.async for A/B loading when K alignment doesn't meet TMA requirements.
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = ceildiv(K, Self.BK)

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create RingBuffer for cp.async operations
        var ring_buffer = Self.build_ring_buffer[tma_transfer=False](
            smem, warp_group_thread_idx
        )

        # Create ScatterGatherCPAsync loaders
        alias k_align = find_K_alignment_upto_16B(K * size_of[a_type]())
        var a_loader, b_loader = Self.build_cpasync_loaders[k_align](a, b)

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle()

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            warpgroup_reg_dealloc[32]()

            Self.producer_main_loop[num_k_iters=num_k_iters](
                UInt(block_idx_swizzle[1]),
                UInt(block_idx_swizzle[0]),
                0,
                a_loader,
                b_loader,
                ring_buffer,
            )
        else:
            # Consumer warp groups
            constrained[
                Self.num_consumer <= 2, "Only support 1 or 2 consumer"
            ]()
            warpgroup_reg_alloc[232]()

            var local_warp_group_idx = warp_group_idx - 1
            var c_reg_tile = Self.AccumRegTileType.stack_allocation()
            var final_c_reg_tile = Self.AccumRegTileType.stack_allocation()

            # Enter consumer mode
            with ring_buffer.consumer() as consumer:
                Self.consumer_main_loop[num_k_iters=num_k_iters](
                    wgmma_op,
                    local_warp_group_idx,
                    final_c_reg_tile,
                    c_reg_tile,
                    consumer,
                )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            Self.consumer_output(
                c_tma_op,
                c,
                smem.c_tile,
                output_reg_tile,
                UInt(warp_group_thread_idx),
                UInt(local_warp_group_idx),
                thread_idx.x - UInt(WARPGROUP_SIZE),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_splitk[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tma_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
        splits: Int,
        raster_order: RasterOrder,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
        workspace_buffer: NDBuffer[Self.accum_type, 3, MutableAnyOrigin],
        locks_ptr: UnsafePointer[UInt8],
        problem_shape: IndexList[3],
    ):
        """Split-K variant of the kernel for better load balancing on small problems.
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = K // Self.BK

        constrained[(K % Self.BK) == 0, "K must be divisible by BK"]()

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create ring buffer
        var ring_buffer = Self.build_ring_buffer(smem, warp_group_thread_idx)

        # Create ScatterGatherTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        alias N = b_layout.shape[0].value()
        alias M = a_layout.shape[0].value()
        alias NUM_TILES = ceildiv(M, Self.BM) * ceildiv(N, Self.BN)

        alias workspace_layout = Layout.row_major(NUM_TILES, Self.BM, Self.BN)
        var reduction_workspace = LayoutTensor(
            workspace_buffer.data,
            RuntimeLayout[workspace_layout].row_major(
                IndexList[3](NUM_TILES, Self.BM, Self.BN)
            ),
        )

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])

        var scheduler = SplitKTileScheduler[
            Index(N, K),
            block_tile_shape,
            splits,
            Self.num_consumer,
            num_pipeline_stages,
            Index(CLUSTER_M, CLUSTER_N),
            raster_order,
        ](
            problem_shape,
            Index(rank_m, rank_n),
            locks_ptr,
        )

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()
            var work_tile_info = scheduler.initial_work_tile_info()

            if warp_id == 0 and lane_predicate:
                while work_tile_info.is_valid():
                    var m_coord = work_tile_info.m * Self.BM
                    var n_coord = work_tile_info.n * Self.BN

                    alias work_k_tile_count = num_k_iters // splits
                    var work_k_tile_start = work_tile_info.get_k_start()

                    Self.producer_main_loop[num_k_iters=work_k_tile_count](
                        UInt(m_coord),
                        UInt(n_coord),
                        UInt(work_k_tile_start),
                        a_loader,
                        b_loader,
                        ring_buffer,
                    )

                    # Get next work tile
                    work_tile_info = scheduler.fetch_next_work(work_tile_info)
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            var work_tile_info = scheduler.initial_work_tile_info()

            @parameter
            if a_type is DType.float8_e4m3fn:
                _ = final_c_reg_tile.fill(0.0)
            else:
                _ = c_reg_tile.fill(0.0)

            # Enter consumer mode
            with ring_buffer.consumer() as consumer:
                while work_tile_info.is_valid():
                    alias work_k_tile_count = num_k_iters // splits

                    Self.consumer_main_loop[num_k_iters=work_k_tile_count](
                        wgmma_op,
                        local_warp_group_idx,
                        final_c_reg_tile,
                        c_reg_tile,
                        consumer,
                    )

                    var output_reg_tile = (
                        final_c_reg_tile if a_type
                        is DType.float8_e4m3fn else c_reg_tile
                    )

                    scheduler.reduction(
                        reduction_workspace,
                        output_reg_tile,
                        work_tile_info,
                        Self.num_consumer,
                        local_warp_group_idx,
                    )

                    # check if this is the reduction tile
                    if scheduler.is_last_split(work_tile_info):
                        var block_y = UInt(work_tile_info.m)
                        var block_x = UInt(work_tile_info.n)

                        Self.consumer_output(
                            c_tma_op,
                            c,
                            smem.c_tile,
                            output_reg_tile,
                            UInt(warp_group_thread_idx),
                            UInt(local_warp_group_idx),
                            thread_idx.x - UInt(WARPGROUP_SIZE),
                            Int(block_y),
                            Int(block_x),
                        )

                    # Get next work tile
                    work_tile_info = scheduler.fetch_next_work(work_tile_info)

        Self.finalize_kernel()

    @staticmethod
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads),
        `nvvm.cluster_dim`=cluster_shape,
    )
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    fn run_grouped[
        a_tile_layout: Layout,
        b_tile_layout: Layout,
        c_tile_layout: Layout,
        a_desc_layout: Layout,
        b_desc_layout: Layout,
        c_desc_layout: Layout,
    ](
        a_tma_op: TMATensorTile[a_type, a_tile_layout, a_desc_layout],
        b_tma_op: TMATensorTile[b_type, b_tile_layout, b_desc_layout],
        c_tma_op: TMATensorTile[c_type, c_tile_layout, c_desc_layout],
        a_offsets: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
        expert_ids: NDBuffer[DType.int32, 1, MutableAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutableAnyOrigin],
    ):
        """Grouped matmul variant for MoE (Mixture of Experts) models.

        This variant handles multiple experts where each expert processes a subset of tokens.
        The a_offsets array indicates token boundaries for each expert.
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = K // Self.BK

        constrained[(K % Self.BK) == 0, "K must be divisible by BK"]()

        # Initialize WgmmaOp and SMem first
        var wgmma_op = Self.WgmmaOp()
        var smem = Self.SMem()

        # Common initialization
        var (
            warp_group_idx,
            warp_group_thread_idx,
            rank_m,
            rank_n,
            warp_id,
            lane_predicate,
        ) = Self.common_kernel_init()

        # Create ring buffer
        var ring_buffer = Self.build_ring_buffer(smem, warp_group_thread_idx)

        # Create ScatterGatherTMA loaders
        var a_loader, b_loader = Self.build_tma_loaders(
            a_tma_op, b_tma_op, rank_m, rank_n
        )

        Self.pipeline_init()

        # Calculate block swizzle
        var block_idx_swizzle = Self.get_block_swizzle()

        # The block may be OOB because we create blocks based the maximum
        # number of tokens per expert.
        var M = a_offsets[Int(block_idx.z + 1)] - a_offsets[Int(block_idx.z)]
        if UInt32(block_idx_swizzle[1] * Self.BM) >= M:
            return

        var a_start_row = a_offsets[Int(block_idx.z)]

        var expert = expert_ids[Int(block_idx.z)]
        # We use -1 to indicate that the block is not active for LoRA use cases.
        # but we still need to zero out the output for this case.
        var skip_matmul = expert < 0

        alias N = c_layout.shape[1].value()
        var b_start_row = expert * N

        alias CLUSTER_N = UInt(cluster_shape[0])
        alias CLUSTER_M = UInt(cluster_shape[1])

        # Split thread blocks into producer and consumer warp groups
        if warp_group_idx == 0:
            # Producer warp group
            _ = Self.setup_producer()

            if warp_id == 0 and lane_predicate and not skip_matmul:
                var m_coord = block_idx.y * UInt(
                    Self.BM
                ) if CLUSTER_N > 1 else UInt(Int(a_start_row)) + UInt(
                    block_idx_swizzle[1]
                ) * UInt(
                    Self.BM
                )

                var n_coord = block_idx.x * UInt(
                    Self.BN
                ) if CLUSTER_M > 1 else UInt(Int(b_start_row)) + UInt(
                    block_idx_swizzle[0]
                ) * UInt(
                    Self.BN
                )

                if warp_id == 0 and lane_predicate:
                    Self.producer_main_loop[num_k_iters=num_k_iters](
                        m_coord,
                        n_coord,
                        0,  # k_start,
                        a_loader,
                        b_loader,
                        ring_buffer,
                    )
        else:
            # Consumer warp groups
            var local_warp_group_idx, c_reg_tile, final_c_reg_tile = (
                Self.setup_consumer(warp_group_idx)
            )

            _ = c_reg_tile.fill(0.0)

            if not skip_matmul:
                # Enter consumer mode
                with ring_buffer.consumer() as consumer:
                    Self.consumer_main_loop[num_k_iters=num_k_iters](
                        wgmma_op,
                        local_warp_group_idx,
                        final_c_reg_tile,
                        c_reg_tile,
                        consumer,
                    )

            var output_reg_tile = (
                final_c_reg_tile if a_type
                is DType.float8_e4m3fn else c_reg_tile
            )

            # C layout for current expert
            alias c_gmem_layout = Layout(
                IntTuple(UNKNOWN_VALUE, N), IntTuple(N, 1)
            )
            alias c_gmem_type = LayoutTensor[
                c_type,
                c_gmem_layout,
                MutableAnyOrigin,
                layout_int_type = DType.int32,
                address_space = AddressSpace.GENERIC,
            ]

            var c_gmem_runtime_layout = RuntimeLayout[c_gmem_layout](
                Index(M, N), Index(N, 1)
            )

            var c_by_expert = c_gmem_type(
                c.ptr + a_start_row * N, c_gmem_runtime_layout
            )

            @parameter
            fn elementwise_epilogue_fn_wrapper[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](idx: IndexList[2], val: SIMD[dtype, width]):
                @parameter
                if elementwise_lambda_fn:
                    alias elementwise_epilogue = elementwise_lambda_fn.value()
                    var batch_idx = IndexList[2](
                        Int(a_start_row + idx[0]), idx[1]
                    )
                    elementwise_epilogue(batch_idx, val)

            Self.consumer_output[
                OptionalReg[elementwise_epilogue_type](
                    elementwise_epilogue_fn_wrapper
                ) if elementwise_lambda_fn else None
            ](
                c_tma_op,
                c_by_expert,
                smem.c_tile,
                output_reg_tile,
                warp_group_thread_idx,
                local_warp_group_idx,
                UInt(thread_idx.x - UInt(WARPGROUP_SIZE)),
                block_idx_swizzle[1],
                block_idx_swizzle[0],
            )

        Self.finalize_kernel()

    @staticmethod
    @always_inline
    fn consumer_main_loop[
        ring_buffer_origin: Origin[True], //,
        num_k_iters: Int,
    ](
        wgmma_op: Self.WgmmaOp,
        local_warp_group_idx: UInt,
        final_c_reg_tile: Self.AccumRegTileType,
        c_reg_tile: Self.AccumRegTileType,
        mut ring_buffer: Self.RingBufferConsumer[ring_buffer_origin, _],
    ):
        """Main computation loop for consumer warp groups.

        This function implements the core matrix multiplication using tensor cores.
        It consumes tiles from the ring buffer and accumulates results using WGMMA
        (Warp Group Matrix Multiply Accumulate) instructions.

        For FP8 data types, it periodically promotes intermediate results to higher
        precision to maintain accuracy.

        Args:
            wgmma_op: Tensor core operator for matrix multiplication.
            local_warp_group_idx: Index of this consumer warp group (0-based).
            final_c_reg_tile: Final accumulation register tile (for FP8 promotion).
            c_reg_tile: Working accumulation register tile.
            ring_buffer: Consumer handle for synchronized tile access.
        """

        @parameter
        if a_type is DType.float8_e4m3fn:
            _ = final_c_reg_tile.fill(0.0)
        else:
            _ = c_reg_tile.fill(0.0)

        var fp8_promotion_iter = 0

        alias num_full_k_iters = ceildiv(num_k_iters, num_pipeline_stages)
        alias num_remaining_k_iters = num_k_iters % num_pipeline_stages

        # `num_pipeline_stages_to_unroll` determines how many pipeline stages should be unroll in the consumer loop;
        # if num_k_iters % pipeline_stages != 0 then for the last loop, we only unroll (num_k_iters % pipeline_stages) pipeline stages
        @always_inline
        @parameter
        fn consumer_loop[
            num_pipeline_stages_to_unroll: Int,
        ](k_iter: Int):
            @parameter
            for j in range(num_pipeline_stages_to_unroll):
                # Get the next available tile slot from the ring buffer.
                # The context manager ensures proper barrier synchronization.
                with ring_buffer.get_tiles() as tiles:
                    Self.wgmma(
                        wgmma_op,
                        local_warp_group_idx,
                        tiles.a_tile,
                        tiles.b_tile,
                        c_reg_tile,
                    )

                @parameter
                if a_type is DType.float8_e4m3fn:
                    fp8_promotion_iter += 1
                    if fp8_promotion_iter == promotion_frequency:
                        Self.promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)
                        fp8_promotion_iter -= promotion_frequency

        @parameter
        if num_remaining_k_iters == 0:
            for k_iter in range(num_full_k_iters):
                consumer_loop[num_pipeline_stages](k_iter)
        else:
            for k_iter in range(num_full_k_iters - 1):
                consumer_loop[num_pipeline_stages](k_iter)
            consumer_loop[num_remaining_k_iters](num_full_k_iters - 1)

        # Final promotion for fp8 data type if num_k_iters % promotion_frequency != 0
        @parameter
        if a_type is DType.float8_e4m3fn:
            if fp8_promotion_iter != 0:
                Self.promote_to_cuda_cores(c_reg_tile, final_c_reg_tile)

    @staticmethod
    @always_inline
    fn promote_to_cuda_cores(
        c_reg_tile: Self.AccumRegTileType,
        final_c_reg_tile: Self.AccumRegTileType,
    ):
        """Promote FP8 accumulation to higher precision using CUDA cores.

        When using FP8 data types, tensor cores accumulate in limited precision.
        To maintain accuracy over many accumulations, we periodically add the
        intermediate results to a higher-precision accumulator using CUDA cores.

        This technique is commonly used in production libraries like cuBLAS to
        achieve both high performance (from FP8 tensor cores) and good accuracy.

        Args:
            c_reg_tile: Current accumulation from tensor cores.
            final_c_reg_tile: Higher-precision accumulator (updated in place).
        """
        constrained[
            c_reg_tile.dtype in (DType.float32, DType.float16),
            "Only support fp32 and fp16 data type in CUDA Core promotion",
        ]()
        constrained[
            len(c_reg_tile.layout) == 2,
            "Only support 2D layout in CUDA Core promotion",
        ]()

        alias num_mma = c_reg_tile.layout.shape[0].value()
        alias c_frag_size = c_reg_tile.layout.shape[1].value()

        # Add tensor core results to higher-precision accumulator
        @parameter
        for mma_id in range(num_mma):

            @parameter
            for i in range(c_frag_size):
                final_c_reg_tile[mma_id, i] = rebind[Scalar[Self.accum_type]](
                    final_c_reg_tile[mma_id, i]
                ) + rebind[Scalar[Self.accum_type]](c_reg_tile[mma_id, i])

    @always_inline
    @staticmethod
    fn wgmma(
        wgmma_op: Self.WgmmaOp,
        local_warp_group_idx: UInt,
        a_tile: SMemTileType[a_type, _, _],
        b_tile: SMemTileType[b_type, _, _],
        c_reg_tile: Self.AccumRegTileType,
    ):
        warpgroup_fence(c_reg_tile)
        wgmma_op.arrive()
        alias scale_c = 0 if a_type is DType.float8_e4m3fn else 1
        wgmma_op.wgmma[Self.num_consumer, scale_c=scale_c](
            a_tile,
            b_tile,
            c_reg_tile,
            local_warp_group_idx,
        )
        wgmma_op.commit_group()
        warpgroup_fence(c_reg_tile)
        wgmma_op.wait_group()


@always_inline
fn find_K_alignment_upto_16B(row_bytes_arg: Int) -> Int:
    """Find alignment among 1B, 2B, 4B, 16B based on the row's bytes.

    This function determines the largest power-of-2 alignment (up to 16 bytes)
    that evenly divides the given row size. This is used to determine the
    optimal vector size for cp.async operations when K dimension alignment
    doesn't meet TMA requirements.

    Args:
        row_bytes_arg: Number of bytes in a row (K * sizeof(element)).

    Returns:
        Alignment in bytes (1, 2, 4, 8, or 16).
    """

    var row_bytes = row_bytes_arg
    var alignment = 1

    @parameter
    for i in range(4):
        # Check if current alignment divides evenly
        if row_bytes & 1 == 1:
            return alignment
        row_bytes >>= 1
        alignment <<= 1

    return alignment


# Helper functions for st.matrix output processing
# These functions break down the complex handle_stmatrix_output function into
# smaller, more manageable pieces for better readability and maintainability.


@always_inline
fn _calculate_output_bounds[
    c_layout: Layout, //, BM: Int, BN: Int
](
    c: LayoutTensor[_, c_layout, MutableAnyOrigin, *_, **_],
    block_y: Int,
    block_x: Int,
) -> Tuple[UInt32, UInt32]:
    """Calculate the output bounds for the current thread block."""
    alias N = c_layout.shape[1].value()
    var M = c.dim[0]()
    var M_bound = min(UInt32((block_y + 1) * BM), UInt32(M))
    var N_bound = min(UInt32((block_x + 1) * BN), UInt32(N))
    return M_bound, N_bound


@always_inline
fn _store_fragments_to_smem[
    c_type: DType,
    c_tile_layout: Layout,
    accum_type: DType,
    c_reg_layout: Layout, //,
    wgmma_shape: IndexList[3],
    BN: Int,
    WG_BM: Int,
    WG_BN: Int,
    TMA_BN: Int,
    num_m_mmas: Int,
    sub_wg_bn_id: Int,
    use_x2_for_last_iter: Bool,
    num_consumer: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    st_matrix_swizzle: Swizzle,
    st_matrix_rt_layout: RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ],
):
    """Store fragments from registers to shared memory using st.matrix instructions.
    """

    @parameter
    for tma_n in range(WG_BN // TMA_BN):

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for i in range(TMA_BN // 16):
                var st_matrix_args = RuntimeTuple[
                    IntTuple(
                        UNKNOWN_VALUE,
                        IntTuple(
                            i,
                            m_mma,
                            UNKNOWN_VALUE,
                        ),
                    )
                ](
                    Int(warp_group_thread_idx),
                    i,
                    m_mma,
                    Int(local_warp_group_idx),
                )
                var n_mma = (
                    i + tma_n * (TMA_BN // 16) + sub_wg_bn_id * (WG_BN // 16)
                )
                var offset = c_tile.ptr.offset(
                    st_matrix_swizzle(st_matrix_rt_layout(st_matrix_args))
                    + WG_BM * TMA_BN * tma_n
                )

                @always_inline
                @parameter
                fn st_matrix_frag[x2: Bool = False]():
                    alias xn = 1 if x2 else 2
                    alias xf = 2 if x2 else 1
                    var c_frag = c_reg_tile.tile[1, 4 * xf](
                        m_mma,
                        xn * n_mma,
                    )
                    var d_reg = c_frag.load[4 * xf](0, 0).cast[DType.bfloat16]()
                    var d_reg_f32_packed = bitcast[DType.float32, 2 * xf](d_reg)
                    st_matrix[simd_width = 2 * xf](offset, d_reg_f32_packed)

                @parameter
                if use_x2_for_last_iter:
                    st_matrix_frag()
                else:
                    st_matrix_frag[True]()


# Mutable fragment lambda applicator used for both compute and non-compute cases

alias elementwise_lambda_type = fn[
    dtype: DType, width: Int, *, alignment: Int = 1
] (IndexList[2], mut SIMD[dtype, width]) capturing -> None


@always_inline
fn _apply_epilogue_lambda[
    c_type: DType,
    c_tile_layout: Layout, //,
    elementwise_lambda_fn: elementwise_lambda_type,
    N: Int,
    WG_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_wg_coord_m: Int,
    c_gmem_wg_coord_n: Int,
    local_thread_idx: UInt,
    M_bound: UInt32,
    N_bound: UInt32,
):
    """Apply the epilogue lambda function to the output data."""
    alias epilogue = elementwise_lambda_fn
    alias st_matrix_vec_swizzle = make_ldmatrix_swizzle[c_type, WG_BN]()
    alias thread_layout = Layout.row_major(
        num_consumer_threads // (WG_BN // simd_size),
        WG_BN // simd_size,
    )

    var c_gmem_frag_with_offsets = c_gmem_wg_tile.vectorize[
        1, simd_size
    ]().distribute_with_offset[thread_layout](local_thread_idx)
    var c_gmem_frag = c_gmem_frag_with_offsets[0]
    var c_gmem_offset_coords = c_gmem_frag_with_offsets[1]
    var coord_m = c_gmem_wg_coord_m + c_gmem_offset_coords[0]
    var coord_n = c_gmem_wg_coord_n + c_gmem_offset_coords[1] * simd_size

    var c_smem_frag = c_tile.vectorize[1, simd_size]().distribute[
        thread_layout, swizzle=st_matrix_vec_swizzle
    ](local_thread_idx)

    alias num_stores_per_thread = c_gmem_frag.layout.size()

    @parameter
    for i in range(num_stores_per_thread):
        alias src_idx = c_smem_frag.layout(i)
        alias dst_idx = c_gmem_frag.layout(i)
        alias dst_m_offset = dst_idx // N
        alias dst_n_offset = dst_idx % N
        var m = UInt32(coord_m + dst_m_offset)
        var n = UInt32(coord_n + dst_n_offset)
        alias alignment = align_of[SIMD[c_type, simd_size]]()

        if m < M_bound and n < N_bound:
            epilogue(
                (Int(m), Int(n)),
                c_smem_frag[i, 0],
            )


@always_inline
fn _perform_output_store[
    c_type: DType,
    c_tma_layout: Layout,
    c_tile_layout: Layout, //,
    use_tma_store: Bool,
    use_x2_for_last_iter: Bool,
    BM: Int,
    BN: Int,
    WG_BM: Int,
    WG_BN: Int,
    TMA_BN: Int,
    num_consumer_threads: Int,
    simd_size: Int,
    st_matrix_swizzle: Swizzle,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, _],
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_gmem_wg_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    local_thread_idx: UInt,
    block_x: Int,
    block_y: Int,
    sub_wg_bn_id: Int,
):
    """Perform the final output store operation."""

    @parameter
    if use_tma_store and not use_x2_for_last_iter:
        fence_async_view_proxy()

        if local_thread_idx < UInt(WG_BN // TMA_BN):
            var smem_offset = c_tile.ptr.offset(
                WG_BM * TMA_BN * local_thread_idx
            )
            var c_tma_tile = SMemTileType[
                c_type,
                c_tma_layout,
                alignment=128,
            ](smem_offset)

            c_tma_op.async_store(
                c_tma_tile,
                (
                    UInt(
                        block_x * BN
                        + sub_wg_bn_id * WG_BN
                        + local_thread_idx * UInt(TMA_BN)
                    ),
                    UInt(block_y * BM),
                ),
            )
            c_tma_op.commit_group()
            c_tma_op.wait_group()

    else:
        alias thread_layout = Layout.row_major(
            num_consumer_threads // (WG_BN // simd_size),
            WG_BN // simd_size,
        )

        @parameter
        if use_x2_for_last_iter:
            var masked_c_tile = c_tile.slice[
                Slice(0, Int(c_tile.layout.shape[0])),
                Slice(0, Int(c_tile.layout.shape[1]) // 2),
            ]()
            var masked_c_gmem_wg_tile = c_gmem_wg_tile.slice[
                Slice(0, Int(c_gmem_wg_tile.layout.shape[0])),
                Slice(0, Int(c_gmem_wg_tile.layout.shape[1]) // 2),
            ]()
            alias thread_layout_v2 = Layout.row_major(
                num_consumer_threads // (WG_BN // simd_size),
                WG_BN // simd_size // 2,
            )
            if local_thread_idx < UInt(num_consumer_threads // 2):
                copy_sram_to_dram[
                    thread_layout=thread_layout_v2,
                    swizzle=st_matrix_swizzle,
                ](
                    masked_c_gmem_wg_tile.vectorize[1, simd_size](),
                    masked_c_tile.vectorize[1, simd_size](),
                )
        else:
            copy_sram_to_dram[
                thread_layout=thread_layout,
                swizzle=st_matrix_swizzle,
            ](
                c_gmem_wg_tile.vectorize[1, simd_size](),
                c_tile.vectorize[1, simd_size](),
            )


@always_inline
fn handle_stmatrix_output[
    c_type: DType,
    c_tma_layout: Layout,
    c_desc_layout: Layout,
    accum_type: DType,
    c_layout: Layout,
    c_tile_layout: Layout,
    c_reg_layout: Layout, //,
    wgmma_shape: IndexList[3],
    num_consumer: Int,
    use_tma_store: Bool,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type],
    elementwise_compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type],
    BM: Int,
    BN: Int,
    num_m_mmas: Int,
    num_consumer_threads: Int,
    simd_size: Int,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
    c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
    c_gmem_tile: LayoutTensor[c_type, _, MutableAnyOrigin, *_, **_],
    c_gmem_corner_coords: c.CornerCoordsType,
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Handle output using st.matrix instructions for optimized bf16 output."""
    # Calculate output bounds
    var M_bound, N_bound = _calculate_output_bounds[BM, BN](c, block_y, block_x)

    # Layout dimensions
    alias WG_BM = c_tile.layout.shape[0].value()
    alias WG_BN = c_tile.layout.shape[1].value()
    alias TMA_BN = c_tma_op.layout.shape[1].value() if use_tma_store else WG_BN

    var st_matrix_rt_layout = RuntimeLayout[
        st_matrix_n_layout[c_type, TMA_BN, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]()
    alias st_matrix_swizzle = make_ldmatrix_swizzle[
        c_type, TMA_BN, log2_floor(16 // size_of[c_type]())
    ]()

    alias num_sub_wg_bn_iters = ceildiv(BN, WG_BN)
    alias last_iter = BN // WG_BN
    alias needs_x2 = BN % WG_BN != 0
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE

    constrained[
        needs_x2 == (c_frag_size % 4 == 0 and c_frag_size % 8 != 0),
        "stmatrix and wgmma register count conflict: needs_x2 = "
        + String(needs_x2)
        + " c_frag_size ="
        + String(c_frag_size),
    ]()

    @parameter
    for sub_wg_bn_id in range(num_sub_wg_bn_iters):
        alias use_x2_for_last_iter = needs_x2 and sub_wg_bn_id == last_iter

        # Store fragments to shared memory
        _store_fragments_to_smem[
            wgmma_shape,
            BN,
            WG_BM,
            WG_BN,
            TMA_BN,
            num_m_mmas,
            sub_wg_bn_id,
            use_x2_for_last_iter,
            num_consumer,
        ](
            c_tile,
            c_reg_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            st_matrix_swizzle,
            st_matrix_rt_layout,
        )

        named_barrier[num_consumer_threads](10)

        alias thread_layout = Layout.row_major(
            num_consumer_threads // (WG_BN // simd_size),
            WG_BN // simd_size,
        )

        var c_gmem_wg_tile_crd_idx = c_gmem_tile.tile_with_offset[BM, WG_BN](
            0, sub_wg_bn_id
        )
        var c_gmem_wg_tile = c_gmem_wg_tile_crd_idx[0]
        var c_gmem_wg_coords = rebind[c.CornerCoordsType](
            c_gmem_wg_tile_crd_idx[1]
        )
        c_gmem_wg_coords = c_gmem_wg_coords + c_gmem_corner_coords

        # Handle compute lambda
        @parameter
        if elementwise_compute_lambda_fn:
            alias lambda_fn = elementwise_compute_lambda_fn.value()

            @parameter
            fn _compute_lambda[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](
                index: IndexList[2], mut val: SIMD[dtype, width]
            ) capturing -> None:
                var res = lambda_fn[alignment=alignment](index, val)
                val = res

            _apply_epilogue_lambda[
                _compute_lambda,
                c_layout.shape[1].value(),
                WG_BN,
                num_consumer_threads,
                simd_size,
            ](
                c_tile,
                c_gmem_wg_tile,
                c_gmem_wg_coords[0],
                c_gmem_wg_coords[1],
                local_thread_idx,
                M_bound,
                N_bound,
            )
            named_barrier[num_consumer_threads](10)

        # Handle epilogue lambda
        @parameter
        if elementwise_lambda_fn:
            alias lambda_fn = elementwise_lambda_fn.value()

            @parameter
            fn _epilogue_lambda[
                dtype: DType, width: Int, *, alignment: Int = 1
            ](
                index: IndexList[2], mut val: SIMD[dtype, width]
            ) capturing -> None:
                _ = lambda_fn[alignment=alignment](index, val)

            _apply_epilogue_lambda[
                _epilogue_lambda,
                c_layout.shape[1].value(),
                WG_BN,
                num_consumer_threads,
                simd_size,
            ](
                c_tile,
                c_gmem_wg_tile,
                c_gmem_wg_coords[0],
                c_gmem_wg_coords[1],
                local_thread_idx,
                M_bound,
                N_bound,
            )

        else:
            # Regular store path
            _perform_output_store[
                use_tma_store,
                use_x2_for_last_iter,
                BM,
                BN,
                WG_BM,
                WG_BN,
                TMA_BN,
                num_consumer_threads,
                simd_size,
                st_matrix_swizzle,
            ](
                c_tma_op,
                c_tile,
                c_gmem_wg_tile,
                local_thread_idx,
                block_x,
                block_y,
                sub_wg_bn_id,
            )

        named_barrier[num_consumer_threads](10)


@always_inline
fn warp_specialized_gemm_output[
    c_type: DType,
    accum_type: DType,
    c_layout: Layout,
    c_tile_layout: Layout,
    c_tma_layout: Layout,
    c_reg_layout: Layout,
    c_desc_layout: Layout, //,
    /,
    *,
    c_tile_shape: IndexList[2],
    c_swizzle: TensorMapSwizzle,
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: OptionalReg[
        elementwise_compute_lambda_type
    ] = None,
](
    c_tma_op: TMATensorTile[c_type, c_tma_layout, c_desc_layout],
    c: LayoutTensor[c_type, c_layout, MutableAnyOrigin, *_, **_],
    c_tile: SMemTileType[
        c_type,
        c_tile_layout,
        alignment=128,
    ],
    c_reg_tile: RegTileType[
        accum_type,
        c_reg_layout,
        _,
    ],
    warp_group_thread_idx: UInt,
    local_warp_group_idx: UInt,
    local_thread_idx: UInt,
    block_y: Int,
    block_x: Int,
):
    """Write matrix multiplication output from registers to global memory.

    This function handles three main paths:
    1. Using st.matrix instructions for optimized bf16 output
    2. Aligned case when N is divisible by tile size
    3. General case with bounds checking
    """
    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_m_mmas = BM // wgmma_shape[0] // num_consumer
    alias num_n_mmas = BN // wgmma_shape[1]
    alias num_consumer_threads = num_consumer * WARPGROUP_SIZE
    alias simd_size = simd_width_of[c_type]()
    alias BM = c_tile_shape[0]
    alias BN = c_tile_shape[1]

    # Use helper to compute tile coordinates
    var tile_crd_idx = c.tile_with_offset[BM, BN](block_y, block_x)
    var c_gmem_tile = tile_crd_idx[0]
    var c_gmem_corner_coords = tile_crd_idx[1]
    var c_gmem_offset = tile_crd_idx[2]
    var c_gmem_split_crd_idx = c_gmem_tile.tile_with_offset[
        BM // num_consumer, BN
    ](Int(local_warp_group_idx), 0)
    var c_gmem_split = c_gmem_split_crd_idx[0]

    alias c_coord_type = c.CornerCoordsType
    var warp_id = warp_group_thread_idx // UInt(WARP_SIZE)

    alias N = c_layout.shape[1].value()
    alias is_N_multiple_of_16B = N * size_of[c_type]() % 16 == 0
    alias WG_BM = c_tile.layout.shape[0].value()
    alias WG_BN = c_tile.layout.shape[1].value()
    alias TMA_BN = c_tma_op.layout.shape[1].value() if use_tma_store else WG_BN
    # fmt: off
    alias use_stmatrix = accum_type is DType.float32 \
            and c_type is DType.bfloat16 \
            and c_frag_size % 4 == 0 \
            and BM % wgmma_shape[0] == 0 \
            and WG_BN % 16 == 0 \
            and num_consumer <= 2 \
            and BN == wgmma_shape[1] \
            and BM == WG_BM \
            and N * size_of[c_type]() % 16 == 0
    # fmt: on

    @parameter
    if use_stmatrix:
        # Use the extracted st_matrix handler
        handle_stmatrix_output[
            wgmma_shape,
            num_consumer,
            use_tma_store,
            elementwise_lambda_fn,
            elementwise_compute_lambda_fn,
            BM,
            BN,
            num_m_mmas,
            num_consumer_threads,
            simd_size,
        ](
            c_tma_op,
            c,
            c_tile,
            c_reg_tile,
            c_gmem_tile,
            c_gmem_corner_coords,
            warp_group_thread_idx,
            local_warp_group_idx,
            local_thread_idx,
            block_y,
            block_x,
        )

    # When N is evenly divisible by BN, we don't need bounds checking on the N dimension
    elif N % BN == 0:

        @parameter
        if (
            elementwise_lambda_fn is not None
            or elementwise_compute_lambda_fn is not None
        ):
            # Output dimensions in global memory.
            alias N = c_layout.shape[1].value()
            var M = UInt(c.dim[0]())

            var lane = lane_id()

            var c_frag_vec2 = c_reg_tile.vectorize[1, 2]()

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    var warp_tile_crd_idx = c_gmem_split.tile_with_offset[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](Int(m_mma * 4 + warp_id), n_mma)
                    var warp_tile = warp_tile_crd_idx[0]
                    var warp_tile_coords = rebind[c_coord_type](
                        warp_tile_crd_idx[1]
                    )
                    warp_tile_coords = (
                        warp_tile_coords
                        + c_gmem_corner_coords
                        + rebind[c_coord_type](c_gmem_split_crd_idx[1])
                    )
                    var warp_tile_offset = (
                        warp_tile_crd_idx[2]
                        + c_gmem_offset
                        + c_gmem_split_crd_idx[2]
                    )

                    var gmem_frag_with_offsets = warp_tile.vectorize[
                        1, 2
                    ]().distribute_with_offset[Layout.row_major(8, 4)](lane)
                    var gmem_frag = gmem_frag_with_offsets[0]
                    var gmem_offset_coords = rebind[c_coord_type](
                        gmem_frag_with_offsets[1]
                    )
                    gmem_offset_coords[1] *= 2
                    var coords = gmem_offset_coords + warp_tile_coords
                    var c_block_offset = (
                        gmem_frag_with_offsets[2] + warp_tile_offset
                    )

                    alias num_vecs = gmem_frag.layout.size()

                    @parameter
                    for i in range(num_vecs):
                        alias dst_idx = gmem_frag.layout(i)
                        alias dst_m_offset = dst_idx // N
                        alias dst_n_offset = dst_idx % N
                        var m = Int(coords[0] + dst_m_offset)
                        var n = Int(coords[1] + dst_n_offset)

                        alias alignment = align_of[SIMD[c_type, 2]]()
                        if m < Int(M) and n < N:

                            @parameter
                            if elementwise_lambda_fn:
                                alias epilogue = elementwise_lambda_fn.value()
                                epilogue[alignment=alignment](
                                    (m, n),
                                    c_frag_vec2[mma_id, i].cast[c_type](),
                                )
                            else:
                                alias compute_lambda = elementwise_compute_lambda_fn.value()
                                var reg_val = compute_lambda[
                                    alignment=alignment
                                ](
                                    (m, n),
                                    c_frag_vec2[mma_id, i].cast[c_type](),
                                )
                                c.ptr.store[alignment=alignment](
                                    c_block_offset + dst_idx, reg_val
                                )

        else:

            @parameter
            for m_mma in range(num_m_mmas):

                @parameter
                for n_mma in range(num_n_mmas):
                    alias mma_id = n_mma * num_m_mmas + m_mma

                    # (m_mma, n_mma) is coordinates for a warp group's tile.
                    # A warp group is 4x1 warps.
                    warp_tile = c_gmem_split.tile[
                        wgmma_shape[0] // 4, wgmma_shape[1]
                    ](Int(m_mma * 4 + warp_id), n_mma)

                    # Tile at (mma_id, 0) is a long vector containing all fragments
                    # for this warp.
                    c_frag = c_reg_tile.tile[1, c_frag_size](mma_id, 0)

                    # A warp is organized as row_major(8, 4) and each thread owns 2 contiguous
                    # elementwise. This pattern repeats to fill the warp tile.
                    copy_local_to_dram[Layout.row_major(8, 4)](
                        warp_tile.vectorize[1, 2](),
                        c_frag.vectorize[1, 2](),
                    )

    else:
        constrained[
            elementwise_compute_lambda_fn is None,
            (
                "compute_lambda_fn is not supported when N % BN != 0 and TMA is"
                " not used"
            ),
        ]()

        # Lane's coordinate is in 8x4 warp.
        var lane_coord0 = UInt32(lane_id() // 4)
        var lane_coord1 = UInt32(lane_id() % 4)

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                alias mma_id = n_mma * num_m_mmas + m_mma

                # (m_mma, n_mma) is coordinates for a warp group's tile.
                # A warp group is 4x1 warps.
                var warp_tile_crd_idx = c_gmem_split.tile_with_offset[
                    wgmma_shape[0] // 4, wgmma_shape[1]
                ](Int(m_mma * 4 + warp_id), n_mma, 0, 0)
                var warp_tile = warp_tile_crd_idx[0]
                var warp_tile_coords = rebind[c_coord_type](
                    warp_tile_crd_idx[1]
                )
                warp_tile_coords = (
                    warp_tile_coords
                    + c_gmem_corner_coords
                    + c_coord_type(
                        wgmma_shape[0] * Int(local_warp_group_idx), 0
                    )
                )

                # A single fragment matrix is the 8x8 output shard by a warp.
                alias num_n_frag_mat = wgmma_shape[1] // 8
                alias num_m_frag_mat = wgmma_shape[0] // 4 // 8

                @parameter
                for n_frag in range(num_n_frag_mat):

                    @parameter
                    for m_frag in range(num_m_frag_mat):
                        alias frag_mat_id = n_frag * num_m_frag_mat + m_frag

                        var frag_mat_gmem = warp_tile.tile[8, 8](m_frag, n_frag)

                        var bound0 = UInt32(
                            frag_mat_gmem.runtime_layout.shape[0].value[0]
                        )
                        var bound1 = UInt32(
                            frag_mat_gmem.runtime_layout.shape[1].value[0]
                        )

                        @parameter
                        for i in range(2):
                            if (
                                lane_coord0 < bound0
                                and lane_coord1 * 2 + i < bound1
                            ):

                                @parameter
                                if elementwise_lambda_fn:
                                    alias epilogue = elementwise_lambda_fn.value()

                                    var frag_mat_coords = (
                                        warp_tile_coords
                                        + c_coord_type(
                                            Int(m_frag * 8), Int(n_frag * 8)
                                        )
                                    )
                                    var frag_coords = (
                                        frag_mat_coords
                                        + c_coord_type(
                                            Int(lane_coord0),
                                            Int(lane_coord1 * 2 + i),
                                        )
                                    )

                                    epilogue[
                                        alignment = align_of[Scalar[c_type]]()
                                    ](
                                        Index(frag_coords[0], frag_coords[1]),
                                        c_reg_tile[
                                            mma_id, frag_mat_id * 2 + i
                                        ].cast[c_type](),
                                    )

                                else:
                                    frag_mat_gmem[
                                        Int(lane_coord0),
                                        Int(lane_coord1 * 2 + i),
                                    ] = rebind[frag_mat_gmem.element_type](
                                        c_reg_tile[
                                            mma_id, frag_mat_id * 2 + i
                                        ].cast[c_type]()
                                    )
