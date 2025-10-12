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
from sys import align_of, simd_width_of

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
)
from gpu import warp_id as get_warp_id
from gpu.memory import AddressSpace
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from layout import Layout, LayoutTensor, IntTuple
from layout.layout import blocked_product
from layout.layout_tensor import (
    UNKNOWN_VALUE,
    LayoutTensorIter,
    ThreadScope,
    copy_local_to_shared,
)
from layout._utils import idx2crd
from layout.swizzle import Swizzle
from layout.tensor_core import TiledTensorCore
from memory import stack_allocation

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from ....structuring import (
    IteratorScatterGatherAmd,
    RegTileType,
    ScatterGatherAmd,
    SMemTileType,
    SMemWarpTileType,
)
from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig
from .._multistage_gemm_gpu import (
    warp_split_k_reduction,
    WarpSplitKReductionSMem,
)
from itertools import product


struct MmaOpAMD[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int,
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    out_frag_size: Int,
    swizzle: Swizzle,
]:
    alias simd_width = simd_width_of[in_type]()
    alias alignment = align_of[SIMD[in_type, Self.simd_width]]()
    alias tensor_core_mma = TiledTensorCore[
        out_type,
        in_type,
        shape,
        k_group_size,
        transpose_b,
    ]()

    alias reg_tile_layout[num_mmas: Int] = Layout.row_major(
        num_mmas * num_k_tiles, Self.simd_width
    )

    alias RegTileType[num_mmas: Int] = RegTileType[
        in_type, Self.reg_tile_layout[num_mmas]
    ]

    # Register-level storage for matrix data during computation
    var _a_reg_tile: Self.RegTileType[num_m_mmas]
    var _b_reg_tile: Self.RegTileType[num_n_mmas]

    # FIXME: We didn't use to support 3D layouts, now we do.
    # This should really be Layout.row_major(num_m_mmas, num_n_mmas, out_frag_size)
    alias out_reg_layout = Layout.row_major(
        num_m_mmas * num_n_mmas, out_frag_size
    )
    alias OutRegTileType = RegTileType[out_type, Self.out_reg_layout]

    # Accumulation registers for result
    var out_reg_tile: Self.OutRegTileType

    @always_inline
    fn __init__(out self):
        self._a_reg_tile = Self.RegTileType[num_m_mmas].stack_allocation()
        self._b_reg_tile = Self.RegTileType[num_n_mmas].stack_allocation()
        self.out_reg_tile = Self.OutRegTileType.stack_allocation()

    @always_inline
    fn a_reg_tile(
        self, k_tile_idx: Int
    ) -> Self.RegTileType[num_m_mmas].SIMDTileType[num_m_mmas]:
        return self._a_reg_tile.simd_tile[num_m_mmas](k_tile_idx)

    @always_inline
    fn b_reg_tile(
        self, k_tile_idx: Int
    ) -> Self.RegTileType[num_n_mmas].SIMDTileType[num_n_mmas]:
        return self._b_reg_tile.simd_tile[num_n_mmas](k_tile_idx)

    @always_inline
    fn mma[k_tile_idx: Int](self):
        Self.tensor_core_mma.mma[swap_a_b=True](
            self.a_reg_tile(k_tile_idx),
            self.b_reg_tile(k_tile_idx),
            self.out_reg_tile,
        )

    @always_inline
    fn load_tile_fragment[
        k_tile_idx: Int
    ](self, a_smem_tiles: SMemWarpTileType, b_smem_tiles: SMemWarpTileType):
        Self.tensor_core_mma.mma_op.load_a[swizzle=swizzle](
            a_smem_tiles,
            self.a_reg_tile(k_tile_idx).vectorize(),
            UInt(k_tile_idx),
        )
        Self.tensor_core_mma.mma_op.load_b[swizzle=swizzle](
            b_smem_tiles,
            self.b_reg_tile(k_tile_idx).vectorize(),
            UInt(k_tile_idx),
        )

    @always_inline
    fn reset_accumulator(self):
        _ = self.out_reg_tile.fill(0)


struct MMATileBuffers[
    _dtype: DType,
    /,
    smem_layout: Layout,
    reg_tile_layout: Layout,
    tensor_type: __type_of(LayoutTensor),
    thread_layout: Layout,
    warp_rows: Int,
    warp_cols: Int,
    swizzle: Swizzle,
]:
    """Manages memory for a single matrix (A or B) in GEMM computation.

    This struct encapsulates all memory handling for a matrix, including:
    - Shared memory allocation and tiling
    - Register buffer allocation
    - Data movement between memory levels (DRAM→local→shared)
    """

    # Tensor types for different memory regions

    # Shared memory tiles
    alias SMemTileType = SMemTileType[_dtype, smem_layout]
    var smem_tile: Self.SMemTileType

    # View on Shared memory tiles optimized for MmaOp
    var smem_warp_tile: SMemWarpTileType[
        _dtype, smem_layout, warp_rows, warp_cols
    ]

    # Register tile fragments for data movement from GMEM to SMEM
    alias MMARegTileType = RegTileType[_dtype, reg_tile_layout]
    var load_reg_tile: Self.MMARegTileType

    @always_inline
    fn __init__(
        out self,
        tensor: tensor_type,
        warp_idx: Int,
        warp_k_idx: Int,
        block_idx: Int,
    ):
        """Initialize memory regions for a matrix based on warp coordinates.

        Args:
            tensor: The tensor to load from global memory.
            warp_idx: The warp index within the computation grid (used for MMA operations).
            warp_k_idx: The warp index within the computation grid (used for MMA operations).
            block_idx: The block index within the computation grid (used for warp tiling).
        """
        self.smem_tile = Self.SMemTileType.stack_allocation()
        self.smem_warp_tile = self.smem_tile.tile[warp_rows, warp_cols](
            warp_idx, warp_k_idx
        )
        self.load_reg_tile = Self.MMARegTileType.stack_allocation()

    @always_inline
    fn copy_to_smem(self):
        """Copy data from thread-local memory to shared memory.

        Uses structured thread cooperation to efficiently transfer data.
        """
        alias simd_width = simd_width_of[_dtype]()
        copy_local_to_shared[thread_layout, swizzle, row_major=True](
            self.smem_tile.vectorize(),
            self.load_reg_tile.vectorize(),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](config.num_threads())
)
fn gemm_kernel_amd[
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
    alias BK = config.block_tile_shape[2] * config.num_warp_k_partitions

    # Warp-level tile dimensions
    alias WM = config.warp_tile_shape[0]
    alias WN = config.warp_tile_shape[1]
    alias WK = config.warp_tile_shape[2]

    # Matrix multiply instruction dimensions
    alias MMA_M = config.mma_shape[0]
    alias MMA_N = config.mma_shape[1]
    alias MMA_K = config.mma_shape[2]

    # SIMD and vectorization parameters
    alias simd_width = simd_width_of[a_type]()

    # Warp organization
    alias num_warps_m = BM // WM
    alias num_warps_n = BN // WN
    alias num_warps_k = BK // WK

    # MMA instruction tiling
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    alias num_k_mmas = WK // MMA_K

    # K dimension tiling
    alias frag_size = MMA_M * MMA_K // WARP_SIZE
    alias c_frag_size = MMA_M * MMA_N // WARP_SIZE
    alias k_group_size = simd_width // frag_size
    alias k_tile_size = MMA_K * k_group_size
    alias num_k_tiles = WK // k_tile_size

    # Matrix dimensions from input tensors
    var M = a.dim[0]()

    alias N = b.shape[0 if transpose_b else 1]()
    constrained[N != UNKNOWN_VALUE, "N should be known at compile time"]()

    var K = b.dim[1 if transpose_b else 0]()

    alias stride = b.stride[0 if transpose_b else 1]()
    constrained[
        stride != UNKNOWN_VALUE, "stride should be known at compile time"
    ]()

    alias c_stride = c.stride[0]()
    constrained[
        c_stride != UNKNOWN_VALUE, "c_stride should be known at compile time"
    ]()

    # Thread and warp indices
    var warp_id = Int(get_warp_id())
    var warp_km, warp_n = divmod(warp_id, num_warps_n)
    var warp_k, warp_m = divmod(warp_km, num_warps_m)

    # Swizzle pattern for SMEM tiles
    alias swizzle = Swizzle(3, 0, 1)

    # SMEM tile layout
    @always_inline
    fn smem_tile_layout[block_rows: Int, block_cols: Int]() -> Layout:
        # Shared memory layout
        #
        # - base_layout: Layout.row_major(block_rows, k_tile_size) -> block_rows x k_tile_size tiles
        # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
        # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
        #
        # Resulting shape: block_rowsx(k_tile_size x num_repeats) = block_rows x block_cols tensor
        # Where block_cols = k_tile_size x num_repeats, k_tile_size = MMA_K x k_group_size
        #
        # This creates num_repeats blocks of block_rows x k_tile_size arranged horizontally:
        # Within each k_tile_size-column block, elements are consecutive (stride 1)
        # Between blocks: stride = block_rows x k_tile_size
        #
        # ASCII diagram for block_rows=64, k_tile_size=32, block_cols=64 (showing first 2 of 2 blocks):
        # ┌─────────────────────────────────────────────────────────────────────────┐
        # │         Block 0 (64x32)             │         Block 1 (64x32)           │
        # ├─────────────────────────────────────┼───────────────────────────────────┤
        # │   0    1    2  ...   30   31        │ 2048 2049 2050 ... 2078 2079      │
        # │  32   33   34  ...   62   63        │ 2080 2081 2082 ... 2110 2111      │
        # │  64   65   66  ...   94   95        │ 2112 2113 2114 ... 2142 2143      │
        # │  96   97   98  ...  126  127        │ 2144 2145 2146 ... 2174 2175      │
        # │ ...                                 │  ...                              │
        # │2016 2017 2018  ... 2046 2047        │ 4064 4065 4066 ... 4094 4095      │
        # └─────────────────────────────────────────────────────────────────────────┘
        # stride between blocks = block_rows x k_tile_size = 64 x 32 = 2048

        alias base_layout = Layout.row_major(block_rows, k_tile_size)
        alias num_repeats = block_cols // k_tile_size
        alias tiler_layout = Layout.row_major(1, num_repeats)
        return blocked_product(base_layout, tiler_layout, coalesce_output=True)

    # Helper function for thread layout
    @parameter
    fn get_thread_layout() -> Layout:
        # TODO: Document the logic behind this layout
        # Define a layout that corresponds to the below pattern:
        #
        # | T00 T01 T02 T03 | T16 T17 T18 T19 | ...
        # | T04 T05 T06 T07 | T20 T21 T22 T23 |
        # | T08 T09 T10 T11 | T24 T25 T26 T27 |
        # | T12 T13 T14 T15 | T28 T29 T30 T31 |
        # | T64 T65 T66 T67 | T80 T81 T82 T83 | ...
        # | T68 T69 T70 T71 | T84 T85 T86 T87 |
        # | T72 T73 T74 T75 | T88 T89 T90 T91 |
        # | T76 T77 T78 T79 | T92 T93 T94 T95 |
        alias inner_block_size = 16
        alias inner_block_cols = k_tile_size // simd_width
        alias inner_block_rows = inner_block_size // inner_block_cols

        alias base_layout = Layout.row_major(inner_block_rows, inner_block_cols)

        alias num_repeats_col = BK // k_tile_size
        alias outer_block_size = num_repeats_col * inner_block_size
        alias num_repeats_row = config.num_threads() // UInt(outer_block_size)

        alias tiler_layout = Layout.row_major(
            num_repeats_row,
            num_repeats_col,
        )
        return blocked_product(base_layout, tiler_layout)

    # AMD TensorCore operator for matrix multiplication
    var mma_op = MmaOpAMD[
        out_type=accum_type,
        in_type=a_type,
        shape = config.mma_shape,
        transpose_b=True,
        k_group_size=k_group_size,
        num_k_tiles=num_k_tiles,
        num_m_mmas=num_m_mmas,
        num_n_mmas=num_n_mmas,
        out_frag_size=c_frag_size,
        swizzle=swizzle,
    ]()

    alias thread_layout = get_thread_layout()

    # A tensor tiles manager
    var a_tiles = MMATileBuffers[
        mma_op.in_type,
        smem_layout = smem_tile_layout[BM, BK](),
        reg_tile_layout = mma_op.reg_tile_layout[num_m_mmas],
        tensor_type = __type_of(a),
        thread_layout=thread_layout,
        warp_rows=WM,
        warp_cols=WK,
        swizzle=swizzle,
    ](a, warp_m, warp_k, block_idx.y)

    # A tensor tile iterator
    var a_gmem_iter = a.tile[BM, stride](block_idx.y, 0).tiled_iterator[
        BM, BK, axis=1
    ](0, 0)
    # A tensor data movement delegate
    var a_scatter_gather = IteratorScatterGatherAmd[thread_layout](
        a, a_gmem_iter
    )

    # B tensor tiles manager
    var b_tiles = MMATileBuffers[
        mma_op.in_type,
        smem_layout = smem_tile_layout[BN, BK](),
        reg_tile_layout = mma_op.reg_tile_layout[num_n_mmas],
        tensor_type = __type_of(b),
        thread_layout=thread_layout,
        warp_rows=WN,
        warp_cols=WK,
        swizzle=swizzle,
    ](b, warp_n, warp_k, block_idx.x)

    # B tensor tile iterator
    var b_gmem_iter = b.tile[BN, stride](block_idx.x, 0).tiled_iterator[
        BN, BK, axis=1
    ](0, 0)
    # B tensor data movement delegate
    var b_scatter_gather = IteratorScatterGatherAmd[thread_layout](
        b, b_gmem_iter
    )

    # SMem storage for Split-K Reduction
    alias SplitKReductionSMem = WarpSplitKReductionSMem[
        accum_type,
        BM,
        BN,
        num_warps_k,
    ]
    var reduction_smem: SplitKReductionSMem

    if num_warps_k > 1:
        reduction_smem = SplitKReductionSMem.stack_allocation()
    else:
        reduction_smem = SplitKReductionSMem.null()

    # --- Helper functions for matrix operations ---

    @always_inline
    @parameter
    fn load_tiles_from_dram():
        """Load tiles from input tensors to register tiles."""
        a_scatter_gather.copy(
            a_tiles.load_reg_tile.vectorize(),
            a_gmem_iter,
        )
        a_gmem_iter._incr()

        b_scatter_gather.copy(
            b_tiles.load_reg_tile.vectorize(),
            b_gmem_iter,
        )
        b_gmem_iter._incr()

    @always_inline
    @parameter
    fn copy_tiles_to_smem():
        a_tiles.copy_to_smem()
        b_tiles.copy_to_smem()

    @always_inline
    @parameter
    fn schedule_loop_body():
        alias threads_per_row = BK // simd_width
        alias rows_per_thread_block = config.num_threads() // UInt(
            threads_per_row
        )
        alias a_loads_per_thread = BM // rows_per_thread_block
        alias b_loads_per_thread = BN // rows_per_thread_block

        alias num_mn_mmas = num_m_mmas + num_n_mmas

        # Compute the number of MMA and smem load/store operations for the loop body.
        alias num_mma_ops = num_m_mmas * num_n_mmas * num_k_mmas
        alias num_smem_store_ops = a_loads_per_thread + b_loads_per_thread
        alias num_smem_load_ops = num_mn_mmas * num_k_tiles

        # Compute the number of MMA operations to distribute across the smem loads.
        # The distribution is dependent on the latency of the MMA operation: MMA operations
        # that have a shape 32x32x8 execute in twice the cycles of 16x16x16, so account
        # for that here. Also defensively guard against underflow of the remaining MMA
        # operations.
        alias mmas_per_smem_load = min(
            1 if MMA_M == MMA_N == 32 else 2, num_mma_ops // num_smem_load_ops
        )
        alias num_remaining_mma_ops = num_mma_ops - num_smem_load_ops * mmas_per_smem_load

        # Distribute the remaining MMA operations across the smem stores and global
        # memory loads.
        alias mmas_per_smem_store = num_remaining_mma_ops // num_smem_store_ops
        alias mmas_per_smem_store_extra = num_remaining_mma_ops % num_smem_store_ops

        @parameter
        for i in range(num_mn_mmas * (num_k_tiles - 1)):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, mmas_per_smem_load, 0
            )

        @parameter
        for i in range(num_smem_store_ops):
            alias mmas_this_smem_store = (
                mmas_per_smem_store + 1
            ) if i < mmas_per_smem_store_extra else mmas_per_smem_store

            schedule_group_barrier(AMDScheduleBarrierMask.DS_WRITE, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, mmas_this_smem_store // 2, 0
            )
            schedule_group_barrier(AMDScheduleBarrierMask.VMEM_READ, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA,
                mmas_this_smem_store - mmas_this_smem_store // 2,
                0,
            )

        @parameter
        for i in range(num_mn_mmas):
            schedule_group_barrier(AMDScheduleBarrierMask.DS_READ, 1, 0)
            schedule_group_barrier(
                AMDScheduleBarrierMask.MFMA, mmas_per_smem_load, 0
            )

    # GEMM Computation Pipeline
    # This kernel implements a pipelined approach optimized for AMD GPUs:
    # 1. Load: Transfer first tiles from global to shared memory
    # 2. Prepare: Load shared memory data to registers, prefetch next tiles
    # 3. Main Loop: Process tiles with overlapped computation and data movement
    # 4. Finalize: Process remaining tiles and write results back

    # Set output accumulator to zero
    mma_op.reset_accumulator()

    # Stage 1: Initial data loading - Global→Local→Shared memory transfer
    load_tiles_from_dram()
    copy_tiles_to_smem()

    barrier()

    # Stage 2: First tile preparation - Register loading and prefetching
    load_tiles_from_dram()
    mma_op.load_tile_fragment[0](a_tiles.smem_warp_tile, b_tiles.smem_warp_tile)

    schedule_barrier()

    # Stage 3: Main computation loop - Pipelined execution with double buffering
    for _ in range(2, K // BK):

        @parameter
        for k_tile_idx in range(1, num_k_tiles):
            mma_op.load_tile_fragment[k_tile_idx](
                a_tiles.smem_warp_tile, b_tiles.smem_warp_tile
            )

        mma_op.mma[0]()

        barrier()

        copy_tiles_to_smem()
        load_tiles_from_dram()

        @parameter
        for k_tile_idx in range(1, num_k_tiles):
            mma_op.mma[k_tile_idx]()

        barrier()

        mma_op.load_tile_fragment[0](
            a_tiles.smem_warp_tile, b_tiles.smem_warp_tile
        )

        schedule_loop_body()

    schedule_barrier()

    @parameter
    for k_tile_idx in range(1, num_k_tiles):
        mma_op.load_tile_fragment[k_tile_idx](
            a_tiles.smem_warp_tile, b_tiles.smem_warp_tile
        )

    barrier()

    copy_tiles_to_smem()

    @parameter
    for k_tile_idx in range(0, num_k_tiles):
        mma_op.mma[k_tile_idx]()

    schedule_barrier()

    barrier()

    @parameter
    for k_tile_idx in range(0, num_k_tiles):
        mma_op.load_tile_fragment[k_tile_idx](
            a_tiles.smem_warp_tile, b_tiles.smem_warp_tile
        )

    @parameter
    for k_tile_idx in range(0, num_k_tiles):
        mma_op.mma[k_tile_idx]()

    schedule_barrier()

    # Accumulate the warp-k tiles via shared memory.
    @parameter
    if num_warps_k > 1:
        warp_split_k_reduction[
            BM, BN, config.num_threads() // UInt(num_warps_k), num_warps_k
        ](warp_k, mma_op.out_reg_tile, reduction_smem.ptr)

        if warp_k != 0:
            return

    # --- Write results to output tensor ---
    # Output stage: Transfer results from registers to global memory
    var c_block_tile = c.tile[BM, BN](block_idx.y, block_idx.x)
    var c_warp_tile = c_block_tile.tile[WM, WN](warp_m, warp_n)

    # Equivalent to Layout.col_major(MMA_M, WARP_SIZE // MMA_M)
    alias output_thread_layout = Layout.col_major(MMA_M, MMA_N // c_frag_size)

    constrained[
        c_warp_tile.layout.all_dims_known(),
        "c_warp_tile layout must be fully static",
    ]()

    @parameter
    if Bool(elementwise_lambda_fn) or (N % BN != 0):
        # 3D view on the output register fragments, see FIXME note on out_reg_layout
        alias out_frag_layout = Layout(
            IntTuple(num_m_mmas, num_n_mmas, c_frag_size),
            IntTuple(c_frag_size, num_m_mmas * c_frag_size, 1),
        )

        # mma output register fragment
        var c_reg_fragment = mma_op.out_reg_tile.reshape[
            out_frag_layout
        ]().vectorize[1, 1, c_frag_size]()

        # Output tensor tile
        var c_gmem_fragment = c_warp_tile.vectorize[
            1, c_frag_size
        ]().distribute[output_thread_layout](lane_id())

        # Warp tile coordinates
        var warp_tile_m = block_idx.y * UInt(BM) + UInt(warp_m * WM)
        var warp_tile_n = block_idx.x * UInt(BN) + UInt(warp_n * WN)

        # Write output fragments
        write_output_fragments[
            c_type,
            c_frag_size,
            MMA_M,
            MMA_N,
            output_thread_layout,
            elementwise_lambda_fn,
        ](
            c_reg_fragment,
            c_gmem_fragment,
            warp_tile_m,
            warp_tile_n,
            M,
            N,
        )

    else:
        # Direct tile copy to global memory
        var c_scatter_gather = ScatterGatherAmd[
            output_thread_layout, thread_scope = ThreadScope.WARP
        ](c)

        c_scatter_gather.copy(
            c_warp_tile.vectorize[1, c_frag_size](),
            mma_op.out_reg_tile.vectorize[1, c_frag_size](),
        )


@always_inline
fn write_output_fragments[
    c_type: DType,
    c_frag_size: Int,
    MMA_M: Int,
    MMA_N: Int,
    output_thread_layout: Layout,
    elementwise_lambda_fn: OptionalReg[elementwise_epilogue_type] = None,
](
    c_reg_fragment: LayoutTensor,
    c_gmem_fragment: LayoutTensor,
    warp_tile_m: Int,
    warp_tile_n: Int,
    M: Int,
    N: Int,
):
    """Write output fragments from registers to global memory with optional elementwise operations.

    Parameters:
        c_type: Data type for the output matrix C.
        c_frag_size: Size of each output fragment.
        MMA_M: Matrix multiply instruction M dimension.
        MMA_N: Matrix multiply instruction N dimension.
        output_thread_layout: Thread layout for output operations.
        elementwise_lambda_fn: Optional elementwise operation to apply.

    Args:
        c_reg_fragment: Register fragments containing computation results.
        c_gmem_fragment: Global memory fragment for output.
        warp_tile_m: M coordinate of the warp tile.
        warp_tile_n: N coordinate of the warp tile.
        M: Total M dimension of the output matrix.
        N: Total N dimension of the output matrix.
    """
    # Warp lane coordinates
    var lane_crd = idx2crd[output_thread_layout](lane_id())

    # c_gmem_fragment tile coordinates for this thread (vectorized)
    var thread_tile_m: Int = warp_tile_m + lane_crd[0]
    var thread_tile_n: Int = warp_tile_n + lane_crd[1] * c_frag_size

    # Valid fragment bounds
    var max_valid_frag_m = (M - thread_tile_m + MMA_M - 1) // MMA_M
    var max_valid_frag_n = (N - thread_tile_n + MMA_N - 1) // MMA_N

    # Output fragment dimensions
    alias frag_height = c_gmem_fragment.layout.shape[0].value()
    alias frag_width = c_gmem_fragment.layout.shape[1].value()

    @parameter
    for frag_m, frag_n in product(range(frag_height), range(frag_width)):
        if frag_m < max_valid_frag_m and frag_n < max_valid_frag_n:
            # Load result vector, cast to output tensor data type
            var result_vec = c_reg_fragment[frag_m, frag_n, 0].cast[c_type]()

            @parameter
            if elementwise_lambda_fn:
                # Apply custom elementwise operation to each output element
                constrained[
                    elementwise_lambda_fn is not None,
                    "elementwise_lambda_fn is not valid",
                ]()
                alias epilogue_fn = elementwise_lambda_fn.value()

                # Compute global coordinates
                var m = thread_tile_m + frag_m * MMA_M
                var n = thread_tile_n + frag_n * MMA_N

                epilogue_fn[alignment = align_of[SIMD[c_type, c_frag_size]]()](
                    (m, n), result_vec
                )
            else:
                # Store output fragment
                # FIXME: why do we need to rebind to c_gmem_fragment.element_type?
                c_gmem_fragment[frag_m, frag_n] = rebind[
                    c_gmem_fragment.element_type
                ](result_vec)
