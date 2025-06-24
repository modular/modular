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
from layout.tensor_core import TensorCore

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig


struct MMATileBuffers[
    layout: Layout,
    /,
    type: DType,
    alignment: Int,
    thread_layout: Layout,
    swizzle: Swizzle,
    num_k_tiles: Int,
    block_dim: Int,
    block_k_dim: Int,
    stride: Int,
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

    var global_offset: UInt

    @always_inline
    fn __init__(
        out self,
        warp_id: Int,
        warp_idx: Int,
        block_idx: Int,
    ):
        """Initialize memory regions for a matrix based on warp coordinates.

        Args:
            warp_id: The global warp ID (used for warp tiling).
            warp_idx: The warp index within the computation grid (used for MMA operations).
            block_idx: The block index within the computation grid (used for warp tiling).
        """
        self.shared_mem_tile = Self.SharedMemTileType.stack_allocation()
        self.mma_tile = self.shared_mem_tile.tile[mma_warp_dim, block_k_dim](
            warp_idx, 0
        )
        self.load_tile = Self.LoadTileType.stack_allocation()
        self.register_buffer = Self.RegisterTileType.stack_allocation().split[
            2
        ]()
        self.global_offset = stride * (block_dim * block_idx)

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
            self.load_tile.vectorize[1, Self.simd_width](),
        )

    @always_inline
    fn load_from_dram(
        mut self, gmem_iter: LayoutTensor, source_tensor: LayoutTensor
    ) -> None:
        """Load data from global memory (DRAM) to thread-local memory.

        Args:
            gmem_iter: Iterator for accessing global memory.
            source_tensor: The source tensor in global memory.
        """
        copy_dram_to_local[
            src_thread_layout=thread_layout,
            thread_scope = ThreadScope.BLOCK,
        ](
            self.load_tile.vectorize[1, Self.simd_width](),
            gmem_iter.vectorize[1, Self.simd_width](),
            source_tensor,
            self.global_offset,
        )
        self.global_offset += block_k_dim

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

            Self.mma_alt(
                a_reg_tile.vectorize[1, elements_per_thread](),
                b_reg_tile.vectorize[1, elements_per_thread](),
                c_reg_tile.vectorize[1, 4](),
            )

    @staticmethod
    @always_inline
    fn mma_alt(
        a_frag: LayoutTensor, b_frag: LayoutTensor, c_frag: LayoutTensor
    ):
        # TODO: Hack for hackathon, merge this with
        # TODO: Assume that fragments are all vectorized layout tensor with
        # dims num_vectors x 1. Consider using TensorCore to allocate fragments
        # so the caller don't explicitly maintain the shape.
        alias num_m_mmas = a_frag.shape[0]()
        alias num_n_mmas = b_frag.shape[0]()

        constrained[
            c_frag.shape[0]() == num_m_mmas * num_n_mmas,
            "Fragments size mismatch. Expected c_frag shape[0] to be num_m_mmas"
            " * num_n_mmas = "
            + String(num_m_mmas * num_n_mmas)
            + ", got "
            + String(c_frag.shape[0]()),
        ]()

        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                mma_simd(
                    c_frag[m_mma * num_n_mmas + n_mma, 0],
                    a_frag[m_mma, 0],
                    b_frag[n_mma, 0],
                    c_frag[m_mma * num_n_mmas + n_mma, 0],
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
    alias thread_layout = Layout(
        IntTuple(IntTuple(8, 4), IntTuple(4, 2)),
        IntTuple(IntTuple(4, 64), IntTuple(1, 32)),
    )

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
    alias warp_tile_m_mmas = warp_m // mma_m
    alias warp_tile_n_mmas = warp_n // mma_n

    # Matrix dimensions from input tensors
    var m_dim = a.dim[0]()
    var n_dim = b.dim[0 if transpose_b else 1]()
    var k_dim = b.dim[1 if transpose_b else 0]()
    alias stride = b.stride[0]()

    alias num_threads = config.num_threads()

    # Thread and warp indices
    var warp_id = get_warp_id()

    # Helper function for shared memory layout
    @always_inline
    @parameter
    fn get_smem_layout[
        tile_size: Int, block_size: Int, simd_width: Int, threads: Int
    ]() -> Layout:
        # Calculate the number of tiles in each dimension
        alias threads_per_row = block_k // simd_width
        alias threads_per_block = block_size // warps_per_block
        alias blocks = warps_per_block

        alias ro = blocks
        alias ri = threads_per_block
        alias co = 1
        alias ci = block_k

        alias shape_dims = IntTuple(IntTuple(ri, ro), ci)

        alias ro_stride = ri * block_k
        alias ri_stride = block_k
        alias co_stride = threads
        alias ci_stride = 1

        alias stride_dims = IntTuple(IntTuple(ri_stride, ro_stride), ci_stride)
        return Layout(
            IntTuple(block_size, IntTuple(32, 2)),
            IntTuple(32, IntTuple(1, block_size * 32)),
        )

    var a_tiles = MMATileBuffers[
        get_smem_layout[mma_m, block_m, simd_width, num_threads](),
        type=a_type,
        alignment=smem_alignment,
        thread_layout=thread_layout,
        swizzle=swizzle,
        num_k_tiles=k_tiles_count,
        block_dim=block_m,
        block_k_dim=block_k,
        stride=stride,
        warp_tile_mmas=warp_tile_m_mmas,
        mma_warp_dim=warp_m,
        num_mmas=mmas_per_warp_m,
    ](warp_id, warp_id // warps_n, block_idx.y)

    # Global memory iterator for matrix A
    var a_gmem_iter = a.tile[block_m, stride](block_idx.y, 0).tiled_iterator[
        block_m, block_k, axis=1
    ](0, 0)

    # B (weights matrix) memory
    var b_tiles = MMATileBuffers[
        get_smem_layout[mma_n, block_n, simd_width, num_threads](),
        type=b_type,
        alignment=smem_alignment,
        thread_layout=thread_layout,
        swizzle=swizzle,
        num_k_tiles=k_tiles_count,
        block_dim=block_n,
        block_k_dim=block_k,
        stride=stride,
        warp_tile_mmas=warp_tile_n_mmas,
        mma_warp_dim=warp_n,
        num_mmas=mmas_per_warp_n,
    ](warp_id, warp_id % warps_n, block_idx.x)

    # Global memory iterator for matrix B
    var b_gmem_iter = b.tile[block_n, stride](block_idx.x, 0).tiled_iterator[
        block_n, block_k, axis=1
    ](0, 0)

    # Accumulation registers for result
    var c_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(mmas_per_warp_m * mmas_per_warp_n, 4),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

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
            (mmas_per_warp_m * k_tiles_count + mmas_per_warp_n * k_tiles_count)
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
            (mmas_per_warp_m * k_tiles_count + mmas_per_warp_n * k_tiles_count)
            // k_tiles_count
            * (k_tiles_count - 1)
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
    for _ in range(2, k_dim // block_k):

        @parameter
        for k_group in range(1, k_tiles_count):
            mma.load_tiles[k_group](a_tiles, b_tiles)

        mma.mma[0](b_tiles, a_tiles, c_reg_tile)

        barrier()

        copy_tiles_to_shared()
        load_tiles_from_dram()

        @parameter
        for k_group in range(1, k_tiles_count):
            mma.mma[k_group](b_tiles, a_tiles, c_reg_tile)

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
        mma.mma[k_group](b_tiles, a_tiles, c_reg_tile)

    amd_schedule_barrier()

    barrier()

    @parameter
    for k_group in range(0, k_tiles_count):
        mma.load_tiles[k_group](a_tiles, b_tiles)

    @parameter
    for k_group in range(0, k_tiles_count):
        mma.mma[k_group](b_tiles, a_tiles, c_reg_tile)

    amd_schedule_barrier()

    # --- Write results to output tensor ---
    # Output stage: Transfer results from registers to global memory
    var c_block_tile = c.tile[block_m, block_n](block_idx.y, block_idx.x)
    var c_warp_tile = c_block_tile.tile[warp_m, warp_n](
        warp_id // warps_n, warp_id % warps_n
    )

    alias static_N = b_layout.shape[0].value()
    constrained[
        static_N != UNKNOWN_VALUE, "N should be known at compile time"
    ]()

    alias output_thread_layout = Layout.col_major(16, 4)

    @parameter
    if Bool(elementwise_lambda_fn) or (static_N % block_n != 0):
        var c_gmem_fragment = c_warp_tile.vectorize[1, 4]().distribute[
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

            var m = (
                (i % mmas_per_warp_m) * mma_m
                + lane_id() % 16
                + (warp_id // warps_n) * warp_m
                + block_idx.y * block_m
            )
            var n = (
                (i // mmas_per_warp_m) * mma_n
                + (lane_id() // 16) * 4
                + (warp_id % warps_n) * warp_n
                + block_idx.x * block_n
            )

            if m < m_dim and n < n_dim:
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
