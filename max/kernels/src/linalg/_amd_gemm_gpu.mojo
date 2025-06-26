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
from sys import alignof, simdwidthof

import gpu.warp as warp
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from gpu import warp_id as get_warp_id
from gpu.memory import AddressSpace
from gpu.mma import mma as mma_simd
from gpu.sync import AMDScheduleBarrierMask
from gpu.sync import schedule_barrier as amd_schedule_barrier
from gpu.sync import schedule_group_barrier
from layout import IntTuple, Layout, LayoutTensor
from layout.layout_tensor import (
    UNKNOWN_VALUE,
    ThreadScope,
    _tile_is_masked,
    copy,
    copy_dram_to_local,
    copy_dram_to_sram,
    copy_local_to_dram,
)
from layout.swizzle import Swizzle
from layout._utils import TensorCoreKGroup

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig


# Function to handle AMD-specific scheduling
@always_inline
fn amd_scheduling_hints[
    BM: Int,
    BN: Int,
    BK: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_k_tiles: Int,
    simd_width: Int,
    num_threads: Int,
    scheduler_hint: IndexList[3],
]():
    alias threads_per_row = BK // simd_width
    alias rows_per_thread_block = num_threads // threads_per_row
    alias a_loads_per_thread = BM // rows_per_thread_block
    alias b_loads_per_thread = BN // rows_per_thread_block

    @parameter
    for i in range(
        (num_m_mmas * num_k_tiles + num_n_mmas * num_k_tiles) // num_k_tiles
    ):
        schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, scheduler_hint[2], 0
        )

    @parameter
    for i in range(a_loads_per_thread + b_loads_per_thread):
        schedule_group_barrier(AMDScheduleBarrierMask.DS_WRITE, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, scheduler_hint[0], 0
        )
        schedule_group_barrier(AMDScheduleBarrierMask.VMEM_READ, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, scheduler_hint[1], 0
        )

    @parameter
    for i in range(
        (num_m_mmas * num_k_tiles + num_n_mmas * num_k_tiles)
        // num_k_tiles
        * (num_k_tiles - 1)
    ):
        schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
        schedule_group_barrier(
            AMDScheduleBarrierMask.MFMA, scheduler_hint[2], 0
        )


struct MMATileBuffers[
    tensor_origin: ImmutableOrigin, //,
    smem_layout: Layout,
    /,
    tensor_type: __type_of(LayoutTensor),
    thread_layout: Layout,
    swizzle: Swizzle,
    block_rows: Int,
    warp_rows: Int,
    BK: Int,
    num_k_tiles: Int,
    stride: Int,
    num_mmas: Int,
]:
    """Manages memory for a single matrix (A or B) in GEMM computation.

    This struct encapsulates all memory handling for a matrix, including:
    - Shared memory allocation and tiling
    - Register buffer allocation
    - Data movement between memory levels (DRAM→local→shared)
    """

    alias simd_width = simdwidthof[tensor_type.dtype]()
    alias type_alignment = alignof[SIMD[tensor_type.dtype, Self.simd_width]]()

    # Tensor types for different memory regions

    # Shared memory allocation for matrix data shared across the block
    alias SharedMemTileType = LayoutTensor[
        tensor_type.dtype,
        smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = Self.type_alignment,
    ]
    var shared_mem_tile: Self.SharedMemTileType

    # Tile view optimized for matrix multiplication acceleration (MMA) operations
    alias SharedMemWarpTileType = Self.SharedMemTileType.TileType[warp_rows, BK]

    var shared_mem_warp_tile: Self.SharedMemWarpTileType

    # Buffer for loading data from global memory before transferring to shared memory
    alias LoadRegTileType = LayoutTensor[
        tensor_type.dtype,
        Layout.row_major(num_mmas * num_k_tiles, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var load_reg_tile: Self.LoadRegTileType

    # Register-level storage for matrix data during computation
    alias MMARegTileType = LayoutTensor[
        tensor_type.dtype,
        Layout.row_major(num_mmas * num_k_tiles, Self.simd_width),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var mma_reg_tile: Self.MMARegTileType.StaticSplitType[num_k_tiles]

    # Global memory iterator for input tensor
    alias iter_type = tensor_type.TileType[
        block_rows, stride
    ].TiledIteratorType[block_rows, BK, axis=1]
    var gmem_iter: Self.iter_type

    var global_offset: UInt

    var tensor: Pointer[tensor_type, tensor_origin]

    @always_inline
    fn __init__(
        out self,
        ref [tensor_origin]tensor: tensor_type,
        warp_idx: Int,
        block_idx: Int,
    ):
        """Initialize memory regions for a matrix based on warp coordinates.

        Args:
            tensor: The tensor to load from global memory.
            warp_idx: The warp index within the computation grid (used for MMA operations).
            block_idx: The block index within the computation grid (used for warp tiling).
        """
        self.shared_mem_tile = Self.SharedMemTileType.stack_allocation()
        self.shared_mem_warp_tile = self.shared_mem_tile.tile[warp_rows, BK](
            warp_idx, 0
        )
        self.load_reg_tile = Self.LoadRegTileType.stack_allocation()
        self.mma_reg_tile = Self.MMARegTileType.stack_allocation().split[
            num_k_tiles
        ]()
        self.gmem_iter = tensor.tile[block_rows, stride](
            block_idx, 0
        ).tiled_iterator[block_rows, BK, axis=1](0, 0)
        self.global_offset = stride * (block_rows * block_idx)
        # TODO: remove rebind once MOCO-1905 is fixed
        self.tensor = rebind[Pointer[tensor_type, tensor_origin]](
            Pointer(to=tensor)
        )

    @always_inline
    fn copy_to_shared(self):
        """Copy data from thread-local memory to shared memory.

        Uses structured thread cooperation to efficiently transfer data.
        """
        copy[
            thread_layout=thread_layout,
            swizzle=swizzle,
            thread_scope = ThreadScope.BLOCK,
            row_major=True,
        ](
            self.shared_mem_tile.vectorize[1, Self.simd_width](),
            self.load_reg_tile.vectorize[1, Self.simd_width](),
        )

    @always_inline
    fn load_from_dram(mut self) -> None:
        """Load data from global memory (DRAM) to thread-local memory."""
        copy_dram_to_local[
            src_thread_layout=thread_layout,
            thread_scope = ThreadScope.BLOCK,
        ](
            self.load_reg_tile.vectorize[1, Self.simd_width](),
            self.gmem_iter[].vectorize[1, Self.simd_width](),
            self.tensor[],
            self.global_offset,
        )
        self.global_offset += BK
        self.gmem_iter._incr()

    @always_inline
    fn get_reg_tile[
        k_tile_idx: Int
    ](self) -> Self.MMARegTileType.SplitElementType[num_k_tiles]:
        """Get a specific K-dimension tile from the register buffer.

        Parameters:
            k_tile_idx: The K-dimension tile index.

        Returns:
            A tile view for the specified location in the register buffer.
        """
        return self.mma_reg_tile[k_tile_idx]


struct AMD_MMA[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int,
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    simd_width: Int,
    swizzle: Swizzle,
]:
    alias tensor_core_mma = TensorCoreKGroup[
        out_type,
        in_type,
        shape,
        k_group_size,
        transpose_b,
    ]()

    @always_inline
    @staticmethod
    fn load_tiles[
        k_tile_idx: Int
    ](a_tiles: MMATileBuffers, b_tiles: MMATileBuffers):
        Self.tensor_core_mma.mma_op.load_a[swizzle=swizzle](
            a_tiles.shared_mem_warp_tile,
            a_tiles.mma_reg_tile[k_tile_idx]
            .tile[num_m_mmas, simd_width](k_tile_idx, 0)
            .vectorize[1, simd_width](),
            k_tile_idx,
        )

        Self.tensor_core_mma.mma_op.load_b[swizzle=swizzle](
            b_tiles.shared_mem_warp_tile,
            b_tiles.mma_reg_tile[k_tile_idx]
            .tile[num_n_mmas, simd_width](k_tile_idx, 0)
            .vectorize[1, simd_width](),
            k_tile_idx,
        )

    @always_inline
    @staticmethod
    fn mma[
        k_tile_idx: Int,
        swap_a_b: Bool,
    ](
        a_tiles: MMATileBuffers,
        b_tiles: MMATileBuffers,
        c_reg_tile: LayoutTensor,
    ):
        var a_reg_tile = a_tiles.get_reg_tile[k_tile_idx]()
        var b_reg_tile = b_tiles.get_reg_tile[k_tile_idx]()

        Self.tensor_core_mma.mma[swap_a_b=swap_a_b](
            a_reg_tile,
            b_reg_tile,
            c_reg_tile,
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
    c_layout_int_type: DType,
    a_layout_int_type: DType,
    b_layout_int_type: DType,
    c_linear_idx_type: DType,
    a_linear_idx_type: DType,
    b_linear_idx_type: DType,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c: LayoutTensor[
        c_type,
        c_layout,
        MutableAnyOrigin,
        layout_int_type=c_layout_int_type,
        linear_idx_type=c_linear_idx_type,
    ],
    a: LayoutTensor[
        a_type,
        a_layout,
        MutableAnyOrigin,
        layout_int_type=a_layout_int_type,
        linear_idx_type=a_linear_idx_type,
    ],
    b: LayoutTensor[
        b_type,
        b_layout,
        MutableAnyOrigin,
        layout_int_type=b_layout_int_type,
        linear_idx_type=b_linear_idx_type,
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
        c_layout_int_type: Data type for the integer part of matrix C.
        a_layout_int_type: Data type for the integer part of matrix A.
        b_layout_int_type: Data type for the integer part of matrix B.
        c_linear_idx_type: Data type for the linear index of matrix C.
        a_linear_idx_type: Data type for the linear index of matrix A.
        b_linear_idx_type: Data type for the linear index of matrix B.
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

    # Type and shape aliases
    alias accum_type = get_accum_type[a_type]()

    # Block-level tile dimensions
    alias BM = config.block_tile_shape[0]
    alias BN = config.block_tile_shape[1]
    alias BK = config.block_tile_shape[2]

    # Warp-level tile dimensions
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]

    # Matrix multiply instruction dimensions
    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    # SIMD and vectorization parameters
    alias simd_width = simdwidthof[a_type]()

    # AMD specific parameters
    # TODO: Document the logic behind these magic numbers
    alias k_group_size = 16 // simd_width
    alias swizzle = Swizzle(2, 0, 2)
    alias thread_layout = Layout(
        IntTuple(IntTuple(8, 4), IntTuple(4, 2)),
        IntTuple(IntTuple(4, 64), IntTuple(1, 32)),
    )

    # Warp organization
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    alias warps_per_block = num_warps_m * num_warps_n

    # MMA instruction tiling
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N

    # K dimension tiling
    alias k_tile_size = MMA_K * k_group_size
    alias num_k_tiles = BK // k_tile_size

    # Matrix dimensions from input tensors
    var M = a.dim[0]()
    var N = b.dim[0 if transpose_b else 1]()
    var K = b.dim[1 if transpose_b else 0]()
    alias stride = b.stride[0]()

    # Thread and warp indices
    var warp_id = get_warp_id()
    var warp_m = warp_id // num_warps_n
    var warp_n = warp_id % num_warps_n

    # Helper function for shared memory layout
    @always_inline
    @parameter
    fn get_smem_layout[block_rows: Int]() -> Layout:
        # TODO: Document the logic behind these magic numbers
        return Layout(
            IntTuple(block_rows, IntTuple(32, 2)),
            IntTuple(32, IntTuple(1, block_rows * 32)),
        )

    var a_tiles = MMATileBuffers[
        get_smem_layout[BM](),
        tensor_type = __type_of(a),
        thread_layout=thread_layout,
        swizzle=swizzle,
        block_rows=BM,
        warp_rows=WM,
        BK=BK,
        num_k_tiles=num_k_tiles,
        stride=stride,
        num_mmas=num_m_mmas,
    ](a, warp_m, block_idx.y)

    # B (weights matrix) memory
    var b_tiles = MMATileBuffers[
        get_smem_layout[BN](),
        tensor_type = __type_of(b),
        thread_layout=thread_layout,
        swizzle=swizzle,
        block_rows=BN,
        warp_rows=WN,
        BK=BK,
        num_k_tiles=num_k_tiles,
        stride=stride,
        num_mmas=num_n_mmas,
    ](b, warp_n, block_idx.x)

    # Accumulation registers for result
    alias c_reg_tile_type = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, 4),
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ]
    var c_reg_tile = c_reg_tile_type.stack_allocation().fill(0)

    # AMD TensorCore operator for matrix multiplication
    alias mma = AMD_MMA[
        out_type=accum_type,
        in_type=a_type,
        shape = config.mma_shape,
        transpose_b=True,
        k_group_size=k_group_size,
        num_k_tiles=num_k_tiles,
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        simd_width=simd_width,
        swizzle=swizzle,
    ]

    # --- Helper functions for matrix operations ---

    @always_inline
    @parameter
    fn load_tiles_from_dram():
        a_tiles.load_from_dram()
        b_tiles.load_from_dram()

    @always_inline
    @parameter
    fn copy_tiles_to_shared():
        a_tiles.copy_to_shared()
        b_tiles.copy_to_shared()

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
    for _ in range(2, K // BK):

        @parameter
        for k_tile_idx in range(1, num_k_tiles):
            mma.load_tiles[k_tile_idx](a_tiles, b_tiles)

        mma.mma[0, swap_a_b=True](a_tiles, b_tiles, c_reg_tile)

        barrier()

        copy_tiles_to_shared()
        load_tiles_from_dram()

        @parameter
        for k_tile_idx in range(1, num_k_tiles):
            mma.mma[k_tile_idx, swap_a_b=True](a_tiles, b_tiles, c_reg_tile)

        barrier()

        mma.load_tiles[0](a_tiles, b_tiles)

        amd_scheduling_hints[
            BM=BM,
            BN=BN,
            BK=BK,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            num_k_tiles=num_k_tiles,
            simd_width=simd_width,
            num_threads = config.num_threads(),
            scheduler_hint = config.scheduler_hint,
        ]()

    amd_schedule_barrier()

    @parameter
    for k_tile_idx in range(1, num_k_tiles):
        mma.load_tiles[k_tile_idx](a_tiles, b_tiles)

    barrier()

    copy_tiles_to_shared()

    @parameter
    for k_tile_idx in range(0, num_k_tiles):
        mma.mma[k_tile_idx, swap_a_b=True](a_tiles, b_tiles, c_reg_tile)

    amd_schedule_barrier()

    barrier()

    @parameter
    for k_tile_idx in range(0, num_k_tiles):
        mma.load_tiles[k_tile_idx](a_tiles, b_tiles)

    @parameter
    for k_tile_idx in range(0, num_k_tiles):
        mma.mma[k_tile_idx, swap_a_b=True](a_tiles, b_tiles, c_reg_tile)

    amd_schedule_barrier()

    # --- Write results to output tensor ---
    # Output stage: Transfer results from registers to global memory
    var c_block_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    var c_warp_tile = c_block_tile.tile[WM, WN](warp_m, warp_n)

    alias static_N = b.shape[0]()
    constrained[
        static_N != UNKNOWN_VALUE, "N should be known at compile time"
    ]()

    alias output_thread_layout = Layout.col_major(16, 4)

    @parameter
    if Bool(elementwise_lambda_fn) or (static_N % BN != 0):
        var c_gmem_fragment = c_warp_tile.vectorize[1, 4]().distribute[
            output_thread_layout
        ](lane_id())
        var c_reg_fragment = c_reg_tile.vectorize[1, 4]()

        var thread_offset = c_gmem_fragment.distance(c.ptr)

        @parameter
        for i in range(c_gmem_fragment.layout.size()):
            alias src_idx = c_reg_fragment.layout(i)
            alias dst_static_idx: UInt = c_gmem_fragment.layout(i)
            var dst_idx = 0

            @parameter
            if c_gmem_fragment.layout.all_dims_known():
                dst_idx = dst_static_idx
            else:
                dst_idx = Int(c_gmem_fragment.runtime_layout(i))

            var global_offset = Int(thread_offset) + dst_idx

            var m = (
                (i % num_m_mmas) * MMA_M
                + lane_id() % 16
                + warp_m * WM
                + block_idx.y * BM
            )
            var n = (
                (i // num_m_mmas) * MMA_N
                + (lane_id() // 16) * 4
                + warp_n * WN
                + block_idx.x * BN
            )

            if m < M and n < N:
                var result_vec = (
                    c_reg_fragment.ptr.offset(src_idx)
                    .load[
                        width=4,
                        alignment = alignof[SIMD[c_type, 4]](),
                    ]()
                    .cast[c_type]()
                )

                alias alignment = alignof[SIMD[c_type, 4]]()

                @parameter
                if elementwise_lambda_fn:
                    # Apply custom elementwise operation to each output element
                    constrained[
                        elementwise_lambda_fn is not None,
                        "elementwise_lambda_fn is not valid",
                    ]()
                    alias epilogue_fn = elementwise_lambda_fn.value()

                    epilogue_fn[alignment = alignof[SIMD[c_type, 4]]()](
                        (m, n), result_vec
                    )
                else:
                    c.ptr.offset(global_offset).store[alignment=alignment](
                        result_vec
                    )
    else:
        # Direct copy to global memory
        copy_local_to_dram[
            output_thread_layout, thread_scope = ThreadScope.WARP
        ](c_warp_tile.vectorize[1, 4](), c_reg_tile.vectorize[1, 4](), c)
