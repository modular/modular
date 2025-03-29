# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import InlineArray, OptionalReg
from gpu import (
    barrier,
    lane_id,
    block_dim,
    block_idx,
    global_idx,
    thread_idx,
    WARP_SIZE,
    grid_dim,
    MAX_THREADS_PER_BLOCK_METADATA,
)
from gpu.sync import (
    schedule_barrier as amd_schedule_barrier,
    schedule_group_barrier,
    AMDScheduleBarrierMask,
)
import gpu.warp as warp
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor, IntTuple
from layout.layout_tensor import (
    copy_dram_to_sram,
    copy_local_to_dram,
    copy_dram_to_local,
    copy,
    ThreadScope,
    _tile_is_masked,
)
from layout.runtime_layout import RuntimeLayout
from layout.tensor_core import TensorCore
from linalg.utils import GemmShape
from math import align_down, ceildiv, align_up
from memory import UnsafePointer
from sys import simdwidthof, alignof
from utils import Index, IndexList, StaticTuple
from utils.numerics import get_accum_type
from .utils_gpu import MatmulConfig
from .utils import apply_epilogue, elementwise_epilogue_type
from layout.swizzle import Swizzle


struct MMATileBuffers[
    layout: Layout,
    /,
    type: DType,
    alignment: Int,
    thread_layout: Layout,
    swizzle: Swizzle,
    num_k_tiles: Int,
    block_k_dim: Int,
    warp_tile_size: Int,
    warp_tile_mmas: Int,
    mma_warp_dim: Int,
    num_mmas: Int,
]:
    """Manages memory for a single matrix (A or B) in GEMM computation.

    This struct encapsulates all memory handling for a matrix, including:
    - Shared memory allocation and tiling
    - Register buffer allocation
    - Data movement between memory levels (DRAM→local→shared)
    """

    alias simd_width = simdwidthof[type]()

    # Tensor types for different memory regions

    # Shared memory allocation for matrix data shared across the block
    alias SharedMemTileType = LayoutTensor[
        type,
        layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=alignment,
    ]
    var shared_mem_tile: Self.SharedMemTileType

    # Tile view into shared memory for the current warp's portion of the matrix
    alias WarpTileType = __type_of(
        Self.SharedMemTileType.tile_type[warp_tile_size, block_k_dim]()
    )
    var warp_tile: Self.WarpTileType

    # Tile view optimized for matrix multiplication acceleration (MMA) operations
    alias MMATileType = __type_of(
        Self.SharedMemTileType.tile_type[mma_warp_dim, block_k_dim]()
    )
    var mma_tile: Self.MMATileType

    # Buffer for loading data from global memory before transferring to shared memory
    alias LoadTileType = LayoutTensor[
        type,
        Layout.row_major(warp_tile_mmas * num_k_tiles, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var load_tile: Self.LoadTileType

    # Register-level storage for matrix data during computation
    alias RegisterTileType = LayoutTensor[
        type,
        Layout.row_major(num_mmas * num_k_tiles, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var register_buffer: __type_of(
        Self.RegisterTileType.stack_allocation().split[2]()
    )

    @always_inline
    fn __init__(
        out self,
        warp_id: Int,
        warp_idx: Int,
    ):
        """Initialize memory regions for a matrix based on warp coordinates.

        Args:
            warp_id: The global warp ID (used for warp tiling).
            warp_idx: The warp index within the computation grid (used for MMA operations).
        """
        self.shared_mem_tile = Self.SharedMemTileType.stack_allocation()
        self.warp_tile = self.shared_mem_tile.tile[warp_tile_size, block_k_dim](
            warp_id, 0
        )
        self.mma_tile = self.shared_mem_tile.tile[mma_warp_dim, block_k_dim](
            warp_idx, 0
        )
        self.load_tile = Self.LoadTileType.stack_allocation()
        self.register_buffer = Self.RegisterTileType.stack_allocation().split[
            2
        ]()

    @always_inline
    fn copy_to_shared(self):
        """Copy data from thread-local memory to shared memory.

        Uses structured thread cooperation to efficiently transfer data.
        """
        copy[
            thread_layout=thread_layout,
            swizzle=swizzle,
            thread_scope = ThreadScope.WARP,
            row_major=True,
        ](
            self.warp_tile.vectorize[1, Self.simd_width](),
            self.load_tile.vectorize[1, Self.simd_width](),
        )

    @always_inline
    fn load_from_dram(
        self, gmem_iter: LayoutTensor, source_tensor: LayoutTensor
    ):
        """Load data from global memory (DRAM) to thread-local memory.

        Args:
            gmem_iter: Iterator for accessing global memory.
            source_tensor: The source tensor in global memory.
        """
        copy_dram_to_local[
            src_thread_layout=thread_layout,
            thread_scope = ThreadScope.WARP,
        ](
            self.load_tile.vectorize[1, Self.simd_width](),
            gmem_iter.vectorize[1, Self.simd_width](),
            source_tensor,
        )

    @always_inline
    fn get_register_tile[
        k_group: Int, mma_idx: Int, k: Int, elements_per_thread: Int
    ](
        self,
        out result: __type_of(
            self.register_buffer[k_group].tile_type[
                num_mmas, elements_per_thread
            ]()
        ),
    ):
        """Get a specific K-dimension tile from the register buffer.

        Parameters:
            k_group: The K-dimension tile index.
            mma_idx: The MMA tile index in K dimension.
            k: The sub-tile index within the MMA tile.
            elements_per_thread: The number of elements per thread.

        Returns:
            A tile view for the specified location in the register buffer.
        """

        return self.register_buffer[k_group].tile[
            num_mmas, elements_per_thread
        ](mma_idx, k)


struct AMD_MMA[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int,
    k_tiles_count: Int,
    mmas_per_warp_m: Int,
    mmas_per_warp_n: Int,
    simd_width: Int,
    swizzle: Swizzle,
]:
    alias mma_op = TensorCore[
        out_type,
        in_type,
        shape,
        transpose_b,
    ]()

    @always_inline
    @staticmethod
    fn load_tiles[
        k_group: Int
    ](a_tiles: MMATileBuffers, b_tiles: MMATileBuffers):
        Self.mma_op.load_a[swizzle=swizzle](
            a_tiles.mma_tile,
            a_tiles.register_buffer[k_group]
            .tile[mmas_per_warp_m, simd_width](k_group, 0)
            .vectorize[1, simd_width](),
            k_group,
        )

        Self.mma_op.load_b[swizzle=swizzle](
            b_tiles.mma_tile,
            b_tiles.register_buffer[k_group]
            .tile[mmas_per_warp_n, simd_width](k_group, 0)
            .vectorize[1, simd_width](),
            k_group,
        )

    @always_inline
    @staticmethod
    fn mma[
        k_group: Int
    ](
        a_tiles: MMATileBuffers,
        b_tiles: MMATileBuffers,
        c_reg_tile: LayoutTensor,
    ):
        @parameter
        for k in range(k_group_size):
            alias elements_per_thread = simd_width // k_group_size

            var a_reg_tile = a_tiles.get_register_tile[
                k_group, 0, k, elements_per_thread
            ]()
            var b_reg_tile = b_tiles.get_register_tile[
                k_group, 0, k, elements_per_thread
            ]()

            Self.mma_op.mma(
                a_reg_tile.vectorize[1, elements_per_thread](),
                b_reg_tile.vectorize[1, elements_per_thread](),
                c_reg_tile.vectorize[1, 4](),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn gemm_kernel[
    c_type: DType,
    c_layout: Layout,
    a_type: DType,
    a_layout: Layout,
    b_type: DType,
    b_layout: Layout,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[
        c_type, c_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    a: LayoutTensor[
        a_type, a_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
    b: LayoutTensor[
        b_type, b_layout, MutableAnyOrigin, address_space = AddressSpace.GLOBAL
    ],
):
    """AMD-optimized GEMM kernel for matrix multiplication C = A * B.

    This kernel implements an efficient matrix multiplication algorithm optimized
    for AMD GPUs, with hierarchical tiling and structured memory access patterns.

    Parameters:
        c_type: Data type for the output matrix C.
        c_layout: Memory layout for matrix C.
        a_type: Data type for the input matrix A.
        a_layout: Memory layout for matrix A.
        b_type: Data type for the input matrix B.
        b_layout: Memory layout for matrix B.
        transpose_b: Whether matrix B should be transposed.
        config: GEMM configuration parameters (tile sizes, etc.).
        elementwise_lambda_fn: Optional function to apply to output elements.

    Args:
        c: Output matrix C (result).
        a: Input matrix A.
        b: Input matrix B (must be transposed).
    """
    # Validate input constraints
    constrained[transpose_b, "Transpose b must be true"]()
    constrained[a_type == b_type, "a and b must have same type"]()
    constrained[b_layout.all_dims_known(), "b_layout must be known"]()

    # Type and shape aliases
    alias accum_type = get_accum_type[a_type]()

    # Block-level tile dimensions
    alias block_m = config.block_tile_shape[0]
    alias block_n = config.block_tile_shape[1]
    alias block_k = config.block_tile_shape[2]

    # Warp-level tile dimensions
    alias warp_m = config.warp_tile_shape[0]
    alias warp_n = config.warp_tile_shape[1]

    # Matrix multiply instruction dimensions
    alias mma_m = config.mma_shape[0]
    alias mma_n = config.mma_shape[1]
    alias mma_k = config.mma_shape[2]

    # SIMD and vectorization parameters
    alias simd_width = simdwidthof[a_type]()

    # AMD specific parameters
    # TODO: Document the logic behind these magic numbers
    alias k_group_size = 16 // simd_width
    alias smem_alignment = alignof[SIMD[a_type, simd_width]]()
    alias swizzle = Swizzle(2, 0, 2)
    alias thread_layout = Layout.row_major(16, 4)

    # Warp organization
    alias warps_m = block_m // warp_m
    alias warps_n = block_n // warp_n
    alias warps_per_block = warps_m * warps_n

    # MMA instruction tiling
    alias mmas_per_warp_m = warp_m // mma_m
    alias mmas_per_warp_n = warp_n // mma_n

    # K dimension tiling
    alias k_tile_size = mma_k * k_group_size
    alias k_tiles_count = block_k // k_tile_size

    # Thread tile dimensions within warp
    alias warp_tile_m = block_m // warps_per_block
    alias warp_tile_n = block_n // warps_per_block
    alias warp_tile_m_mmas = warp_tile_m // mma_m
    alias warp_tile_n_mmas = warp_tile_n // mma_n

    # Validate tiling constraints
    constrained[
        warp_tile_m % 16 == 0,
        "Block M dimension per warp ("
        + String(warp_tile_m)
        + ") must be divisible by 16",
    ]()
    constrained[
        warp_tile_n % 16 == 0,
        "Block N dimension per warp ("
        + String(warp_tile_n)
        + ") must be divisible by 16",
    ]()

    # Matrix dimensions from input tensors
    var m_dim = a.dim(0)
    alias n_dim = b.shape[0 if transpose_b else 1]()
    alias k_dim = b.shape[1 if transpose_b else 0]()

    # Thread and warp indices
    var warp_id = thread_idx.x // WARP_SIZE

    # Helper function for shared memory layout
    @always_inline
    @parameter
    fn get_smem_layout[tile_size: Int, block_size: Int]() -> Layout:
        # Calculate the number of tiles in each dimension
        alias num_tiles = block_size // tile_size

        # K-dimension tiling
        alias k_tile_dim = k_group_size * mma_k
        alias num_k_tiles = block_k // k_tile_dim

        # Construct shape dimensions (logical 2D grid of elements)
        alias shape_dims = IntTuple(
            IntTuple(tile_size, num_tiles), IntTuple(k_tile_dim, num_k_tiles)
        )

        # Calculate strides for memory layout (physical arrangement)
        alias k_stride = k_tile_dim
        alias tile_stride = block_k * tile_size
        alias k_tile_stride = 1
        alias main_tile_stride = k_tile_dim * tile_size

        alias stride_dims = IntTuple(
            IntTuple(k_stride, tile_stride),
            IntTuple(k_tile_stride, main_tile_stride),
        )

        return Layout(shape_dims, stride_dims)

    # A (input matrix) memory
    var a_tiles = MMATileBuffers[
        get_smem_layout[mma_m, block_m](),
        type=a_type,
        alignment=smem_alignment,
        thread_layout=thread_layout,
        swizzle=swizzle,
        num_k_tiles=k_tiles_count,
        block_k_dim=block_k,
        warp_tile_size=warp_tile_m,
        warp_tile_mmas=warp_tile_m_mmas,
        mma_warp_dim=warp_m,
        num_mmas=mmas_per_warp_m,
    ](warp_id, warp_id // warps_n)

    # Global memory iterator for matrix A
    var a_gmem_iter = a.tile[block_m, k_dim](block_idx.y, 0).tiled_iterator[
        warp_tile_m, block_k, axis=1
    ](warp_id, 0)

    # B (weights matrix) memory
    var b_tiles = MMATileBuffers[
        get_smem_layout[mma_n, block_n](),
        type=b_type,
        alignment=smem_alignment,
        thread_layout=thread_layout,
        swizzle=swizzle,
        num_k_tiles=k_tiles_count,
        block_k_dim=block_k,
        warp_tile_size=warp_tile_n,
        warp_tile_mmas=warp_tile_n_mmas,
        mma_warp_dim = block_n // warps_n,
        num_mmas=mmas_per_warp_n,
    ](warp_id, warp_id % warps_n)

    # Global memory iterator for matrix B
    var b_gmem_iter = b.tile[block_n, k_dim](block_idx.x, 0).tiled_iterator[
        warp_tile_n, block_k, axis=1
    ](warp_id, 0)

    # Accumulation registers for result
    var c_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(mmas_per_warp_m * mmas_per_warp_n, 4),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation().fill(0)

    # AMD TensorCore operator for matrix multiplication
    alias mma = AMD_MMA[
        out_type=accum_type,
        in_type=a_type,
        shape = config.mma_shape,
        transpose_b=True,
        k_group_size=k_group_size,
        k_tiles_count=k_tiles_count,
        mmas_per_warp_m=mmas_per_warp_m,
        mmas_per_warp_n=mmas_per_warp_n,
        simd_width=simd_width,
        swizzle=swizzle,
    ]

    # --- Helper functions for matrix operations ---

    @always_inline
    @parameter
    fn load_tiles_from_dram():
        a_tiles.load_from_dram(a_gmem_iter[], a)
        b_tiles.load_from_dram(b_gmem_iter[], b)
        a_gmem_iter._incr()
        b_gmem_iter._incr()

    @always_inline
    @parameter
    fn copy_tiles_to_shared():
        a_tiles.copy_to_shared()
        b_tiles.copy_to_shared()

    # Function to handle AMD-specific scheduling
    @always_inline
    @parameter
    fn amd_scheduling_hints():
        alias threads_per_row = block_k // simd_width
        alias rows_per_thread_block = config.num_threads() // threads_per_row
        alias a_loads_per_thread = block_m // rows_per_thread_block
        alias b_loads_per_thread = block_n // rows_per_thread_block

        @parameter
        for i in range(
            (
                warp_tile_m_mmas * k_tiles_count
                + warp_tile_n_mmas * k_tiles_count
            )
            // k_tiles_count
        ):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, config.scheduler_hint[2], 0
            )

        @parameter
        for i in range(a_loads_per_thread + b_loads_per_thread):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_WRITE, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, config.scheduler_hint[0], 0
            )
            schedule_group_barrier(AMDScheduleBarrierMask.VMEM_READ, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, config.scheduler_hint[1], 0
            )

        @parameter
        for i in range(
            (
                warp_tile_m_mmas * k_tiles_count
                + warp_tile_n_mmas * k_tiles_count
            )
            // k_tiles_count
        ):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, config.scheduler_hint[2], 0
            )

    # GEMM Computation Pipeline
    # This kernel implements a pipelined approach optimized for AMD GPUs:
    # 1. Load: Transfer first tiles from global to shared memory
    # 2. Prepare: Load shared memory data to registers, prefetch next tiles
    # 3. Main Loop: Process tiles with overlapped computation and data movement
    # 4. Finalize: Process remaining tiles and write results back

    # Stage 1: Initial data loading - Global→Local→Shared memory transfer
    load_tiles_from_dram()
    copy_tiles_to_shared()

    barrier()

    # Stage 2: First tile preparation - Register loading and prefetching
    load_tiles_from_dram()
    mma.load_tiles[0](a_tiles, b_tiles)

    amd_schedule_barrier()

    # Stage 3: Main computation loop - Pipelined execution with double buffering
    for _ in range(0, k_dim - 2 * block_k, block_k):

        @parameter
        for k_group in range(1, k_tiles_count):
            mma.load_tiles[k_group](a_tiles, b_tiles)

        mma.mma[0](a_tiles, b_tiles, c_reg_tile)

        barrier()

        copy_tiles_to_shared()
        load_tiles_from_dram()

        @parameter
        for k_group in range(1, k_tiles_count):
            mma.mma[k_group](a_tiles, b_tiles, c_reg_tile)

        barrier()

        mma.load_tiles[0](a_tiles, b_tiles)

        amd_scheduling_hints()

    amd_schedule_barrier()

    @parameter
    for k_group in range(1, k_tiles_count):
        mma.load_tiles[k_group](a_tiles, b_tiles)

    barrier()

    copy_tiles_to_shared()

    @parameter
    for k_group in range(0, k_tiles_count):
        mma.mma[k_group](a_tiles, b_tiles, c_reg_tile)

    amd_schedule_barrier()

    barrier()

    @parameter
    for k_group in range(0, k_tiles_count):
        mma.load_tiles[k_group](a_tiles, b_tiles)

    @parameter
    for k_group in range(0, k_tiles_count):
        mma.mma[k_group](a_tiles, b_tiles, c_reg_tile)

    amd_schedule_barrier()

    # --- Write results to output tensor ---
    # Output stage: Transfer results from registers to global memory
    var c_block_tile = c.tile[block_m, block_n](block_idx.y, block_idx.x)
    var c_warp_tile = c_block_tile.tile[warp_m, warp_n](
        warp_id // warps_n, warp_id % warps_n
    )

    alias output_thread_layout = Layout.row_major(4, 16)

    @parameter
    if elementwise_lambda_fn:
        # Apply custom elementwise operation to each output element
        constrained[
            elementwise_lambda_fn is not None,
            "elementwise_lambda_fn is not valid",
        ]()
        alias epilogue_fn = elementwise_lambda_fn.value()

        var c_gmem_fragment = c_warp_tile.vectorize[4, 1]().distribute[
            output_thread_layout
        ](lane_id())
        var c_reg_fragment = c_reg_tile.vectorize[1, 4]()

        var thread_offset = c_gmem_fragment.distance(c.ptr)

        @parameter
        for i in range(__type_of(c_gmem_fragment).layout.size()):
            alias src_idx = c_reg_fragment.layout(i)
            alias dst_static_idx: UInt = __type_of(c_gmem_fragment).layout(i)
            var dst_idx = 0

            @parameter
            if c_gmem_fragment.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = Int(c_gmem_fragment.runtime_layout(i))

            var global_offset = Int(thread_offset) + dst_idx
            var m = global_offset // n_dim
            var n = global_offset % n_dim

            if m < m_dim and n < n_dim:
                var result_vec = c_reg_fragment.ptr.offset(src_idx).load[
                    width=4,
                    alignment = alignof[SIMD[c_type, 4]](),
                ]()

                @parameter
                for j in range(4):
                    if m + j < m_dim:
                        epilogue_fn[alignment = alignof[SIMD[c_type, 1]]()](
                            (m + j, n), result_vec[j].cast[c_type]()
                        )
    else:
        # Direct copy to global memory
        copy_local_to_dram[
            output_thread_layout, thread_scope = ThreadScope.WARP
        ](c_warp_tile.vectorize[4, 1](), c_reg_tile.vectorize[1, 4](), c)
