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
from sys import align_of, simd_width_of, size_of

from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    lane_id,
    warp_id as get_warp_id,
)
from gpu.memory import AddressSpace
from gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from gpu.intrinsics import buffer_store
from layout.element import Element
from layout import IntTuple, Layout, LayoutTensor
from layout.layout import blocked_product
from layout.layout_tensor import (
    UNKNOWN_VALUE,
    ThreadScope,
    copy_local_to_shared,
    copy_dram_to_local,
    copy_local_to_dram,
)
from layout.swizzle import Swizzle
from layout._utils import TensorCoreKGroup, get_amd_buffer_descriptor
from memory import stack_allocation

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from ._multistage_gemm_gpu import warp_split_k_reduction
from .utils import elementwise_epilogue_type
from .utils_gpu import MatmulConfig


@always_inline("nodebug")
fn copy_local_to_dram_row_major[
    dst_thread_layout: Layout,
](dst: LayoutTensor, src: LayoutTensor, dst_base: LayoutTensor):
    # TODO: This is a temporary hack, we need to support this in copy_local_to_dram instead.
    # write c in row major order
    var worker_idx = lane_id()

    var dst_fragments = dst.distribute[dst_thread_layout](worker_idx)

    var offset = (Int(dst.ptr) - Int(dst_base.ptr)) // size_of[dst.dtype]()
    var descriptor = get_amd_buffer_descriptor(dst_base)
    var dst_frag_offset = dst_fragments.distance(dst.ptr) + offset
    alias num_stores_per_thread = dst_fragments.layout.size()
    alias m = dst_fragments.shape[0]()
    alias n = dst_fragments.shape[1]()

    @parameter
    for i in range(m):

        @parameter
        for j in range(n):
            alias idx = Layout.col_major(m, n)([i, j])
            alias src_idx = src.layout(idx)
            alias dst_static_idx = dst_fragments.layout(idx)
            var dst_idx = dst_frag_offset

            constrained[
                dst_fragments.layout.all_dims_known(),
                "dst_fragments.layout must have known dimensions",
            ]()
            dst_idx += dst_static_idx

            var src_element = Element[index_type = src.linear_idx_type].load(
                src.ptr.offset(src_idx),
                src.runtime_element_layout,
            )

            alias element_stride = dst_fragments.element_layout.stride[
                1
            ].value()
            constrained[element_stride == 1, "element_stride must be 1"]()
            buffer_store(
                descriptor,
                Int32(dst_idx),
                src_element.element_data.cast[dst.dtype](),
            )


# Dummy ScatterGather implementation that just calls the original copy_dram_to_local and copy_local_to_dram_row_major
# The "real" ScatterGather with _buffer_resource descriptor caching will be added in a subsequent PR
struct ScatterGatherAmd[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
]:
    @always_inline
    fn __init__(out self, tensor: LayoutTensor):
        pass

    # copy_dram_to_local
    @always_inline
    fn copy(
        self,
        dst_reg_tile: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
        src_gmem_tile: LayoutTensor,
        src_tensor: LayoutTensor,
        offset: OptionalReg[UInt] = None,
    ):
        copy_dram_to_local[
            src_thread_layout=thread_layout,
            num_threads=num_threads,
            thread_scope=thread_scope,
            block_dim_count=block_dim_count,
        ](dst_reg_tile, src_gmem_tile, src_tensor, offset)

    # copy_local_to_dram
    @always_inline("nodebug")
    fn copy(
        self,
        dst_gmem_tile: LayoutTensor,
        src_reg_tile: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
        dst_tensor: LayoutTensor,
    ):
        copy_local_to_dram_row_major[dst_thread_layout=thread_layout](
            dst_gmem_tile, src_reg_tile, dst_tensor
        )


# SMEM and REG tiles type declarations, shared by MmaOpAMD and MMATileBuffers
alias SMemTileType[_dtype: DType, layout: Layout] = LayoutTensor[
    _dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.SHARED,
    alignment = align_of[SIMD[_dtype, simd_width_of[_dtype]()]](),
]

alias SMemWarpTileType[
    _dtype: DType, layout: Layout, warp_rows: Int, warp_cols: Int
] = SMemTileType[_dtype, layout].TileType[warp_rows, warp_cols]

alias RegTileType[_dtype: DType, layout: Layout] = LayoutTensor[
    _dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.LOCAL,
    alignment = align_of[SIMD[_dtype, simd_width_of[_dtype]()]](),
]


struct MmaOpAMD[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    transpose_b: Bool,
    k_group_size: Int,
    num_k_tiles: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    BK: Int,
    WK: Int,
]:
    alias swizzle = Swizzle(3, 0, 1)
    alias simd_width = simd_width_of[in_type]()
    alias alignment = align_of[SIMD[in_type, Self.simd_width]]()
    alias tensor_core_mma = TensorCoreKGroup[
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

    alias RegTileFragType[num_mmas: Int] = Self.RegTileType[
        num_mmas
    ].StaticSplitType[num_k_tiles]

    # Register-level storage for matrix data during computation
    var a_reg_tile: Self.RegTileFragType[num_m_mmas]
    var b_reg_tile: Self.RegTileFragType[num_n_mmas]

    alias out_reg_layout = Layout.row_major(num_m_mmas * num_n_mmas, 4)
    alias OutRegTileType = RegTileType[out_type, Self.out_reg_layout]

    # Accumulation registers for result
    var out_reg_tile: Self.OutRegTileType

    @always_inline
    @staticmethod
    fn smem_tile_layout[block_rows: Int, k_tile_size: Int]() -> Layout:
        # Shared memory layout
        #
        # - base_layout: Layout.row_major(block_rows, k_tile_size) -> block_rows×k_tile_size tiles
        # - tiler_layout: Layout.row_major(1, num_repeats) -> repeat tiles num_repeats times horizontally
        # - smem_layout: blocked_product(base_layout, tiler_layout) -> tiled blocked layout
        #
        # Resulting shape: block_rows×(k_tile_size × num_repeats) = block_rows×BK tensor
        # Where BK = k_tile_size × num_repeats, k_tile_size = MMA_K × k_group_size
        #
        # This creates num_repeats blocks of block_rows×k_tile_size arranged horizontally:
        # Within each k_tile_size-column block, elements are consecutive (stride 1)
        # Between blocks: stride = block_rows × k_tile_size
        #
        # ASCII diagram for block_rows=64, k_tile_size=32, BK=64 (showing first 2 of 2 blocks):
        # ┌─────────────────────────────────────────────────────────────────────────┐
        # │         Block 0 (64×32)             │         Block 1 (64×32)           │
        # ├─────────────────────────────────────┼───────────────────────────────────┤
        # │   0    1    2  ...   30   31        │ 2048 2049 2050 ... 2078 2079      │
        # │  32   33   34  ...   62   63        │ 2080 2081 2082 ... 2110 2111      │
        # │  64   65   66  ...   94   95        │ 2112 2113 2114 ... 2142 2143      │
        # │  96   97   98  ...  126  127        │ 2144 2145 2146 ... 2174 2175      │
        # │ ...                                 │  ...                              │
        # │2016 2017 2018  ... 2046 2047        │ 4064 4065 4066 ... 4094 4095      │
        # └─────────────────────────────────────────────────────────────────────────┘
        # stride between blocks = block_rows × k_tile_size = 64 × 32 = 2048

        alias base_layout = Layout.row_major(block_rows, k_tile_size)
        alias num_repeats = BK // k_tile_size
        alias tiler_layout = Layout.row_major(1, num_repeats)
        return blocked_product(base_layout, tiler_layout, coalesce_output=True)

    @always_inline
    fn __init__(out self):
        self.a_reg_tile = (
            Self.RegTileType[num_m_mmas].stack_allocation().split[num_k_tiles]()
        )
        self.b_reg_tile = (
            Self.RegTileType[num_n_mmas].stack_allocation().split[num_k_tiles]()
        )
        self.out_reg_tile = Self.OutRegTileType.stack_allocation()

    @always_inline
    fn mma[k_tile_idx: Int](self):
        Self.tensor_core_mma.mma[swap_a_b=True](
            self.a_reg_tile[k_tile_idx],
            self.b_reg_tile[k_tile_idx],
            self.out_reg_tile,
        )

    @always_inline
    fn load_tile_fragment[
        k_tile_idx: Int
    ](self, a_smem_tiles: SMemWarpTileType, b_smem_tiles: SMemWarpTileType,):
        Self.tensor_core_mma.mma_op.load_a[swizzle = Self.swizzle](
            a_smem_tiles,
            self.a_reg_tile[k_tile_idx]
            .tile[num_m_mmas, Self.simd_width](k_tile_idx, 0)
            .vectorize[1, Self.simd_width](),
            UInt(k_tile_idx),
        )
        Self.tensor_core_mma.mma_op.load_b[swizzle = Self.swizzle](
            b_smem_tiles,
            self.b_reg_tile[k_tile_idx]
            .tile[num_n_mmas, Self.simd_width](k_tile_idx, 0)
            .vectorize[1, Self.simd_width](),
            UInt(k_tile_idx),
        )

    @always_inline
    fn reset_accumulator(self):
        _ = self.out_reg_tile.fill(0)


struct MMATileBuffers[
    tensor_origin: ImmutableOrigin, //,
    _dtype: DType,
    /,
    smem_layout: Layout,
    reg_tile_layout: Layout,
    swizzle: Swizzle,
    tensor_type: __type_of(LayoutTensor),
    thread_layout: Layout,
    block_rows: Int,
    block_cols: Int,
    warp_rows: Int,
    warp_cols: Int,
    stride: Int,
]:
    """Manages memory for a single matrix (A or B) in GEMM computation.

    This struct encapsulates all memory handling for a matrix, including:
    - Shared memory allocation and tiling
    - Register buffer allocation
    - Data movement between memory levels (DRAM→local→shared)
    """

    # Tensor types for different memory regions

    # Shared memory allocation for matrix data shared across the block
    alias SMemTileType = SMemTileType[_dtype, smem_layout]
    var smem_tile: Self.SMemTileType

    # Tile view optimized for matrix multiplication acceleration (MMA) operations
    var smem_warp_tile: SMemWarpTileType[
        _dtype, smem_layout, warp_rows, warp_cols
    ]

    # Buffer for loading data from global memory before transferring to shared memory
    alias MMARegTileType = RegTileType[_dtype, reg_tile_layout]
    var load_reg_tile: Self.MMARegTileType

    # Global memory iterator for input tensor
    alias iter_type = tensor_type.TileType[
        block_rows, stride
    ].TiledIteratorType[block_rows, block_cols, axis=1]
    var gmem_iter: Self.iter_type

    var scatter_gather: ScatterGatherAmd[
        thread_layout=thread_layout,
        thread_scope = ThreadScope.BLOCK,
    ]

    var global_offset: UInt

    var tensor: Pointer[tensor_type, tensor_origin]

    @always_inline
    fn __init__(
        out self,
        ref [tensor_origin]tensor: tensor_type,
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
        self.gmem_iter = tensor.tile[block_rows, stride](
            block_idx, 0
        ).tiled_iterator[block_rows, block_cols, axis=1](0, 0)
        self.scatter_gather = ScatterGatherAmd[
            thread_layout=thread_layout,
            thread_scope = ThreadScope.BLOCK,
        ](tensor)
        self.global_offset = UInt(stride * (block_rows * block_idx))
        self.tensor = rebind[Pointer[tensor_type, tensor_origin]](
            Pointer(to=tensor)
        )

    @always_inline
    fn copy_to_smem(self):
        """Copy data from thread-local memory to shared memory.

        Uses structured thread cooperation to efficiently transfer data.
        """
        alias simd_width = simd_width_of[_dtype]()
        copy_local_to_shared[
            thread_layout=thread_layout,
            swizzle=swizzle,
            thread_scope = ThreadScope.BLOCK,
            row_major=True,
        ](
            self.smem_tile.vectorize[1, simd_width](),
            self.load_reg_tile.vectorize[1, simd_width](),
        )

    @always_inline
    fn load_from_dram(mut self) -> None:
        """Load data from global memory (DRAM) to thread-local memory."""
        alias simd_width = simd_width_of[_dtype]()
        self.scatter_gather.copy(
            self.load_reg_tile.vectorize[1, simd_width](),
            self.gmem_iter[].vectorize[1, simd_width](),
            self.tensor[],
            self.global_offset,
        )
        self.global_offset += UInt(block_cols)
        self.gmem_iter._incr()


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
    alias num_warps_m = UInt(BM // WM)
    alias num_warps_n = UInt(BN // WN)
    alias num_warps_k = UInt(BK // WK)

    # MMA instruction tiling
    alias num_m_mmas = WM // MMA_M
    alias num_n_mmas = WN // MMA_N
    alias num_k_mmas = WK // MMA_K

    # K dimension tiling
    alias frag_size = MMA_M * MMA_K // WARP_SIZE
    alias k_group_size = simd_width // frag_size
    alias k_tile_size = MMA_K * k_group_size
    alias num_k_tiles = WK // k_tile_size

    # Matrix dimensions from input tensors
    var M = a.dim[0]()
    var N = b.dim[0 if transpose_b else 1]()
    var K = b.dim[1 if transpose_b else 0]()
    alias stride = b.stride[0]()

    # Thread and warp indices
    var warp_id = get_warp_id()
    var warp_km, warp_n = divmod(warp_id, num_warps_n)
    var warp_k, warp_m = divmod(warp_km, num_warps_m)

    # Helper function for thread layout
    @parameter
    fn thread_layout() -> Layout:
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
        alias num_repeats_row = config.num_threads() // outer_block_size

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
        BK=BK,
        WK=WK,
    ]()

    # A tensor tiles manager
    var a_tiles = MMATileBuffers[
        mma_op.in_type,
        smem_layout = mma_op.smem_tile_layout[BM, k_tile_size](),
        reg_tile_layout = mma_op.reg_tile_layout[num_m_mmas],
        swizzle = mma_op.swizzle,
        tensor_type = __type_of(a),
        thread_layout = thread_layout(),
        block_rows=BM,
        block_cols=BK,
        warp_rows=WM,
        warp_cols=WK,
        stride=stride,
    ](a, Int(warp_m), Int(warp_k), Int(block_idx.y))

    # B tensor tiles manager
    var b_tiles = MMATileBuffers[
        mma_op.in_type,
        smem_layout = mma_op.smem_tile_layout[BN, k_tile_size](),
        reg_tile_layout = mma_op.reg_tile_layout[num_n_mmas],
        swizzle = mma_op.swizzle,
        tensor_type = __type_of(b),
        thread_layout = thread_layout(),
        block_rows=BN,
        block_cols=BK,
        warp_rows=WN,
        warp_cols=WK,
        stride=stride,
    ](b, Int(warp_n), Int(warp_k), Int(block_idx.x))

    # --- Helper functions for matrix operations ---

    @always_inline
    @parameter
    fn load_tiles_from_dram():
        a_tiles.load_from_dram()
        b_tiles.load_from_dram()

    @always_inline
    @parameter
    fn copy_tiles_to_smem():
        a_tiles.copy_to_smem()
        b_tiles.copy_to_smem()

    @always_inline
    @parameter
    fn schedule_loop_body():
        alias threads_per_row = BK // simd_width
        alias rows_per_thread_block = config.num_threads() // threads_per_row
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
        var reduction_smem = stack_allocation[
            Int(BM * BN * (num_warps_k // 2)),
            accum_type,
            address_space = AddressSpace.SHARED,
            alignment = align_of[SIMD[accum_type, 4]](),
        ]()

        warp_split_k_reduction[
            BM, BN, Int(config.num_threads() // num_warps_k), Int(num_warps_k)
        ](Int(warp_k), mma_op.out_reg_tile, reduction_smem)

        if warp_k != 0:
            return

    alias output_thread_layout = Layout.col_major(16, 4)

    var c_scatter_gather = ScatterGatherAmd[
        output_thread_layout, thread_scope = ThreadScope.WARP
    ](c)

    # --- Write results to output tensor ---
    # Output stage: Transfer results from registers to global memory
    var c_block_tile = c.tile[BM, BN](Int(block_idx.y), Int(block_idx.x))
    var c_warp_tile = c_block_tile.tile[WM, WN](Int(warp_m), Int(warp_n))

    alias static_N = b.shape[0]()
    constrained[
        static_N != UNKNOWN_VALUE, "N should be known at compile time"
    ]()

    @parameter
    if Bool(elementwise_lambda_fn) or (static_N % BN != 0):
        var c_gmem_fragment = c_warp_tile.vectorize[1, 4]().distribute[
            output_thread_layout
        ](lane_id())
        var c_reg_fragment = mma_op.out_reg_tile.vectorize[1, 4]()

        var thread_offset = c_gmem_fragment.distance(c.ptr)

        @parameter
        for i in range(c_gmem_fragment.layout.size()):
            alias src_idx = c_reg_fragment.layout(i)
            alias dst_static_idx: UInt = UInt(c_gmem_fragment.layout(i))
            var dst_idx: Int

            @parameter
            if c_gmem_fragment.layout.all_dims_known():
                dst_idx = Int(dst_static_idx)
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
                alias alignment = align_of[SIMD[c_type, 4]]()

                var result_vec = (
                    c_reg_fragment.ptr.offset(src_idx)
                    .load[
                        width=4,
                        alignment=alignment,
                    ]()
                    .cast[c_type]()
                )

                @parameter
                if elementwise_lambda_fn:
                    # Apply custom elementwise operation to each output element
                    constrained[
                        elementwise_lambda_fn is not None,
                        "elementwise_lambda_fn is not valid",
                    ]()
                    alias epilogue_fn = elementwise_lambda_fn.value()

                    epilogue_fn[alignment=alignment](
                        (Int(m), Int(n)), result_vec
                    )
                else:
                    c.ptr.offset(global_offset).store[alignment=alignment](
                        result_vec
                    )
    else:
        # Direct copy to global memory
        c_scatter_gather.copy(
            c_warp_tile.vectorize[1, 4](),
            mma_op.out_reg_tile.vectorize[1, 4](),
            c,
        )
