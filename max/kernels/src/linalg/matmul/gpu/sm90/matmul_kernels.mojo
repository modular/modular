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
from sys import size_of

from buffer.buffer import NDBuffer
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier
from gpu.cluster import cluster_sync, cluster_sync_relaxed, elect_one_sync
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from gpu.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.device_context import DeviceBuffer
from gpu import (
    block_id_in_cluster,
    block_idx,
    grid_dim,
    thread_idx,
)
from gpu import warp_id as get_warp_id
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import (
    AddressSpace,
    fence_mbarrier_init,
)
from layout import IntTuple, Layout, LayoutTensor
from layout.runtime_layout import UNKNOWN_VALUE, RuntimeLayout
from layout.swizzle import Swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    warpgroup_fence,
)
from layout.tma_async import (
    TMATensorTile,
)
from memory import stack_allocation

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
    PipelineBarrier,
)
from .ring_buffer import RingBuffer, RingBufferConsumer, RingBufferProducer
from .tile_loader import (
    TileLoaderTMA,
    TileLoaderCPAsync,
    TileLoader,
)
from .matmul_output import MatmulTileWriter


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
    alias PipelineBarrier = PipelineBarrier[num_pipeline_stages]

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
        multicast_column_mask <<= Int(rank_n)

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
            smem.full_mbar,
            smem.empty_mbar,
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
        c: LayoutTensor[c_type, _, MutAnyOrigin, *_, **_],
        c_tile: Self.SMem.CTile,
        output_reg_tile: Self.AccumRegTileType,
        warp_group_thread_idx: UInt,
        local_warp_group_idx: UInt,
        local_thread_idx: UInt,
        block_y: Int,
        block_x: Int,
    ):
        """Handle consumer output by writing GEMM results to global memory."""
        var matmul_tile_writer = MatmulTileWriter[
            BM = Self.BM,
            BN = Self.BN,
            swizzle=c_swizzle,
            wgmma_shape=wgmma_shape,
            num_consumer = Self.num_consumer,
            use_tma_store=use_tma_store,
            elementwise_lambda_fn=custom_elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
        ](
            # Pointer(to=c_tma_op),
            c,
            c_tile,
            warp_group_thread_idx,
            local_warp_group_idx,
            local_thread_idx,
            block_y,
            block_x,
        )
        matmul_tile_writer.write_tile(c_tma_op, output_reg_tile)

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
        TileLoaderTMA[
            origin_of(a_tma_op),
            a_type,
            a_tile_layout,
            a_desc_layout,
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[0],
            use_partitioned_multicast=partitioned_multicast,
        ],
        TileLoaderTMA[
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

        var a_multicast_mask, b_multicast_mask = Self.multicast_mask(
            rank_m, rank_n
        )
        var a_loader = TileLoaderTMA[
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[0],
            use_partitioned_multicast=partitioned_multicast,
        ](Pointer(to=a_tma_op), rank_n, UInt16(a_multicast_mask))
        var b_loader = TileLoaderTMA[
            BK = UInt(Self.BK),
            cluster_size = cluster_shape[1],
            use_partitioned_multicast=partitioned_multicast,
        ](Pointer(to=b_tma_op), rank_m, UInt16(b_multicast_mask))
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
        a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
    ) -> Tuple[
        TileLoaderCPAsync[
            a_type,
            a_layout,
            thread_layout,
            a_swizzle,
            vector_size,
        ],
        TileLoaderCPAsync[
            b_type,
            b_layout,
            thread_layout,
            b_swizzle,
            vector_size,
        ],
    ]:
        var a_loader = TileLoaderCPAsync[
            a_type,
            a_layout,
            thread_layout,
            a_swizzle,
            vector_size,
        ](a)
        var b_loader = TileLoaderCPAsync[
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
        a_loader_type: TileLoader,
        b_loader_type: TileLoader, //,
        num_k_iters: Int,
    ](
        m_coord: UInt,
        n_coord: UInt,
        k_coord: UInt,
        a_loader: a_loader_type,
        b_loader: b_loader_type,
        mut ring_buffer: RingBuffer[
            a_loader_type._dtype,  # RingBuffer and TileLoader must agree on dtypes
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
        a: LayoutTensor[a_type, a_layout, MutAnyOrigin],
        b: LayoutTensor[b_type, b_layout, MutAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
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

        # Create TileLoaderTMA loaders
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
        c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
        workspace_buffer: NDBuffer[Self.accum_type, 3, MutAnyOrigin],
        locks_ptr: UnsafePointer[UInt8],
        problem_shape: IndexList[3],
    ):
        """Split-K variant of the kernel for better load balancing on small problems.
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = K // Self.BK

        # FIXME: this seems to trip some logits tests
        # constrained[(K % Self.BK) == 0, "K must be divisible by BK"]()

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

        # Create TileLoaderTMA loaders
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
        a_offsets: NDBuffer[DType.uint32, 1, MutAnyOrigin],
        expert_ids: NDBuffer[DType.int32, 1, MutAnyOrigin],
        c: LayoutTensor[c_type, c_layout, MutAnyOrigin],
    ):
        """Grouped matmul variant for MoE (Mixture of Experts) models.

        This variant handles multiple experts where each expert processes a subset of tokens.
        The a_offsets array indicates token boundaries for each expert.
        """
        alias K = b_layout.shape[1].value()
        alias num_k_iters = K // Self.BK

        # FIXME: this seems to trip some logits tests
        # constrained[(K % Self.BK) == 0, "K must be divisible by BK"]()

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

        # Create TileLoaderTMA loaders
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
                MutAnyOrigin,
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
        a_tile: SMemTileType[a_type, _, **_],
        b_tile: SMemTileType[b_type, _, **_],
        c_reg_tile: Self.AccumRegTileType,
    ):
        warpgroup_fence(c_reg_tile)
        wgmma_op.arrive()
        alias scale_c = 0 if a_type is DType.float8_e4m3fn else 1
        wgmma_op.wgmma[Self.num_consumer, scale_c=scale_c](
            a_tile,
            b_tile,
            c_reg_tile,
            Int(local_warp_group_idx),
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
