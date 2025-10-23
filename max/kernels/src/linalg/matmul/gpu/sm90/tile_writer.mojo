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

"""TileWriter module for efficient tile writing in GPU matrix multiplication.

This module provides utilities for writing matrix tiles from shared memory to
global memory using two different mechanisms:

1. TMA (Tensor Memory Accelerator): Hardware-accelerated stores for efficient
   2D tile transfers from shared to global memory.

2. Regular stores: Software-based synchronous stores with manual thread
   distribution and swizzling for optimal memory access patterns.

The TileWriter trait abstracts these writing mechanisms to provide a unified
interface for the matmul kernel's consumer threads.
"""

from layout.tma_async import TMATensorTile
from layout.layout_tensor import LayoutTensor, copy_sram_to_dram
from gpu.memory import fence_async_view_proxy
from ....structuring import (
    SharedMemBarrier,
    SMemBarrier,
    SMemTileType,
    RegTileType,
)
from layout.swizzle import Swizzle
from gpu.id import thread_idx, lane_id
from sys import simd_width_of
from gpu.host._nvidia_cuda import TensorMapSwizzle
from layout.layout import coalesce
from layout import Layout
from memory import AddressSpace
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE

# from memory.pointer import _GPUAddressSpace

from gpu.mma import st_matrix
from memory import bitcast
from layout import RuntimeLayout, RuntimeTuple, IntTuple
from layout.tensor_core_async import st_matrix_n_layout
from layout.runtime_layout import UNKNOWN_VALUE
from ....utils import elementwise_epilogue_type, elementwise_compute_lambda_type
from utils.index import IndexList
from sys import align_of, size_of


# Import ThreadInfo from matmul_output
@register_passable("trivial")
struct ThreadInfo:
    """Thread identification within the warp group."""

    var warp_id: UInt
    var lane_id: UInt
    var lane_row: UInt32
    var lane_col: UInt32

    fn __init__(
        out self,
        warp_id: UInt,
        lane_id: UInt,
        lane_row: UInt32,
        lane_col: UInt32,
    ):
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.lane_row = lane_row
        self.lane_col = lane_col

    @always_inline
    @staticmethod
    fn from_warp_group_idx(warp_group_thread_idx: UInt) -> ThreadInfo:
        """Create ThreadInfo from a warp group thread index.

        Args:
            warp_group_thread_idx: Thread index within the warp group.

        Returns:
            ThreadInfo struct with computed warp_id, lane_id, lane_row, and lane_col.
        """
        var warp_id = warp_group_thread_idx // UInt(WARP_SIZE)
        var lid = UInt(lane_id())
        var lane_row = UInt32(lid // 4)
        var lane_col = UInt32(lid % 4)
        return ThreadInfo(warp_id, lid, lane_row, lane_col)


@register_passable("trivial")
struct TileCoordinates:
    """Helper struct for managing tile coordinate offsets.

    This struct encapsulates corner and split coordinates used in epilogue
    processing and provides a clean interface for coordinate transformations.
    """

    var corner: IndexList[2]
    var split: IndexList[2]

    @always_inline
    fn __init__(out self, corner: IndexList[2], split: IndexList[2]):
        """Initialize tile coordinates.

        Args:
            corner: Corner coordinates offset.
            split: Split coordinates offset.
        """
        self.corner = corner
        self.split = split

    @always_inline
    fn adjust(self, base_coords: IndexList[2]) -> IndexList[2]:
        """Add corner and split offsets to base coordinates.

        Args:
            base_coords: Base tile coordinates.

        Returns:
            Adjusted coordinates with corner and split offsets applied.
        """
        return IndexList[2](
            base_coords[0] + self.corner[0] + self.split[0],
            base_coords[1] + self.corner[1] + self.split[1],
        )


@register_passable("trivial")
struct MMATileCoords:
    """Coordinates for an MMA tile in the output."""

    var m_mma: Int
    var n_mma: Int
    var mma_id: Int

    @always_inline
    fn __init__(out self, m_mma: Int, n_mma: Int, mma_id: Int):
        """Initialize MMA tile coordinates.

        Args:
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.
            mma_id: Linearized MMA tile ID.
        """
        self.m_mma = m_mma
        self.n_mma = n_mma
        self.mma_id = mma_id

    @always_inline
    @staticmethod
    fn compute(m_mma: Int, n_mma: Int, num_m_mmas: Int) -> MMATileCoords:
        """Compute MMA tile coordinates from indices.

        Args:
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.
            num_m_mmas: Total number of MMA tiles in M dimension.

        Returns:
            MMATileCoords with computed mma_id.
        """
        return MMATileCoords(m_mma, n_mma, n_mma * num_m_mmas + m_mma)


@register_passable("trivial")
trait TileWriter:
    """Base trait for tile writing mechanisms in matrix multiplication.

    This trait defines the interface for writing tiles from shared memory to global memory,
    abstracting over different hardware mechanisms.
    """

    alias _dtype: DType

    @always_inline
    fn write_tile(
        self,
        src: SMemTileType[Self._dtype, _, alignment=128, *_, **_],
        coords: Tuple[UInt, UInt],
    ):
        """Write a tile from shared memory to global memory.

        Args:
            src: Source tile in shared memory (must be 128-byte aligned).
            coords: Tile coordinates (row, column) in the destination matrix.
        """
        ...


@register_passable("trivial")
struct TileWriterTMA[
    tma_origin: Origin[False],
    dtype: DType,
    tma_layout: Layout,
    desc_layout: Layout,
](TileWriter):
    """TMA-based tile writer for hardware-accelerated memory transfers.

    This writer uses NVIDIA's Tensor Memory Accelerator (TMA) for efficient
    2D tile transfers from shared to global memory.

    Parameters:
        tma_origin: Origin type for the TMA operation.
        dtype: Data type of the elements being written.
        tma_layout: Layout of the TMA tile for async store operations.
        desc_layout: Layout described by the TMA descriptor.
    """

    alias _dtype = Self.dtype

    alias TMATensorTilePtr = Pointer[
        TMATensorTile[dtype, tma_layout, desc_layout], tma_origin
    ]
    var tma_op: Self.TMATensorTilePtr

    @always_inline
    fn __init__(
        out self,
        tma_op: Self.TMATensorTilePtr,
    ):
        """Initialize the TMA tile writer.

        Args:
            tma_op: Pointer to the TMA tensor descriptor.
        """
        self.tma_op = tma_op

    @always_inline
    fn write_tile(
        self,
        src: SMemTileType[Self._dtype, _, alignment=128, *_, **_],
        coords: Tuple[UInt, UInt],
    ):
        """Write a tile using TMA hardware acceleration.

        Performs an asynchronous TMA store from shared memory to global memory.
        The operation includes proper fencing and synchronization.

        Args:
            src: Source tile in shared memory.
            coords: Tile coordinates (col, row) in element space.

        Note:
            Coordinates are expected in (N, M) order for column-major output.
        """
        # Ensure all prior async operations are visible before TMA store
        fence_async_view_proxy()

        # Perform the async store
        self.tma_op[].async_store(src, coords)

        # Commit and wait for completion
        self.tma_op[].commit_group()
        self.tma_op[].wait_group()


@register_passable("trivial")
struct TileWriterRegular[
    dtype: DType,
    dst_layout: Layout,
    dst_address_space: AddressSpace,
    dst_element_layout: Layout,
    dst_layout_int_type: DType,
    dst_linear_idx_type: DType,
    dst_masked: Bool,
    dst_alignment: Int, //,
    thread_layout: Layout,
    swizzle: Swizzle,
    simd_size: Int,
    use_x2_for_last_iter: Bool = False,  # Handle masked x2 case
](TileWriter):
    alias _dtype = Self.dtype

    var dst: LayoutTensor[
        dtype,
        dst_layout,
        MutableAnyOrigin,
        address_space=dst_address_space,
        element_layout=dst_element_layout,
        layout_int_type=dst_layout_int_type,
        linear_idx_type=dst_linear_idx_type,
        masked=dst_masked,
        alignment=dst_alignment,
    ]
    var thread_idx: UInt

    @always_inline
    fn __init__(
        out self,
        dst: LayoutTensor[
            dtype,
            dst_layout,
            MutableAnyOrigin,
            address_space=dst_address_space,
            element_layout=dst_element_layout,
            layout_int_type=dst_layout_int_type,
            linear_idx_type=dst_linear_idx_type,
            masked=dst_masked,
            alignment=dst_alignment,
        ],
        thread_idx: UInt,
    ):
        """Initialize the regular tile writer.

        Args:
            dst: Destination tensor in global memory.
            thread_idx: Thread index within the consumer warp group.
        """
        self.dst = dst
        self.thread_idx = thread_idx

    @always_inline
    fn write_tile(
        self,
        src: SMemTileType[Self._dtype, _, alignment=128, *_, **_],
        coords: Tuple[UInt, UInt],
    ):
        """Write a tile using regular stores.

        Distributes the write operation across threads with proper swizzling
        for optimal memory access patterns.

        Args:
            src: Source tile in shared memory.
            coords: Tile indices (row_tile, col_tile) in the destination matrix.
        """
        # For the regular writer used in _perform_output_store, coords are always (0, 0)
        # because the destination tile is already pre-extracted as c_gmem_wg_tile.
        # We directly use self.dst as the destination.

        @parameter
        if use_x2_for_last_iter:
            # Handle masked x2 case - write only half the tile width
            # Get compile-time layout dimensions
            alias dst_height = dst_layout.shape[0].value()
            alias dst_width = dst_layout.shape[1].value()
            alias half_width = dst_width // 2

            # Slice both source and destination to half width internally
            var masked_src = src.slice[
                Slice(0, dst_height),
                Slice(0, half_width),
            ]()
            var masked_dst = self.dst.slice[
                Slice(0, dst_height),
                Slice(0, half_width),
            ]()

            # Compute half-width thread layout
            alias half_thread_layout = Layout.row_major(
                thread_layout.shape[0].value(),
                thread_layout.shape[1].value() // 2,
            )

            # Only first half of threads participate
            alias num_threads = thread_layout.size()
            if self.thread_idx < UInt(num_threads // 2):
                copy_sram_to_dram[
                    thread_layout=half_thread_layout,
                    swizzle=swizzle,
                ](
                    masked_dst.vectorize[1, simd_size](),
                    masked_src.vectorize[1, simd_size](),
                )
        else:
            # Normal case - write full tile
            copy_sram_to_dram[thread_layout=thread_layout, swizzle=swizzle,](
                self.dst.vectorize[1, simd_size](),
                src.vectorize[1, simd_size](),
            )


@register_passable("trivial")
struct FragmentToSMemWriter[
    c_type: DType,
    c_tile_layout: Layout, //,
    tile_n_size: Int,  # Size of each tile in N dimension (e.g., TMA_BN)
    num_m_mmas: Int,
    num_consumer: Int,
    use_x2_for_last_iter: Bool,
    WG_BM: Int,  # Warp group M dimension
    WG_BN: Int,  # Warp group N dimension
    sub_wg_bn_id: Int,  # Sub warp group ID in N dimension
]:  # (TileWriter):
    """Writer for storing accumulator fragments from registers to shared memory.

    Uses st.matrix instructions for efficient bf16 storage with proper swizzling
    for bank conflict avoidance.

    Parameters:
        c_type: Output data type (e.g., bfloat16).
        c_tile_layout: Shared memory tile layout.
        tile_n_size: Size of each tile in N dimension.
        num_m_mmas: Number of MMA tiles in M dimension.
        num_consumer: Number of consumer warp groups.
        use_x2_for_last_iter: Whether to use x2 mode for the last iteration.
        WG_BM: Warp group M dimension.
        WG_BN: Warp group N dimension.
        sub_wg_bn_id: Sub warp group ID in N dimension.
    """

    alias _dtype = Self.c_type

    var c_tile: SMemTileType[c_type, c_tile_layout, alignment=128]
    var warp_group_thread_idx: UInt
    var local_warp_group_idx: UInt
    var st_matrix_swizzle: Swizzle
    var st_matrix_rt_layout: RuntimeLayout[
        st_matrix_n_layout[c_type, tile_n_size, num_m_mmas, num_consumer](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    @always_inline
    fn __init__(
        out self,
        c_tile: SMemTileType[c_type, c_tile_layout, alignment=128],
        warp_group_thread_idx: UInt,
        local_warp_group_idx: UInt,
        st_matrix_swizzle: Swizzle,
        st_matrix_rt_layout: RuntimeLayout[
            st_matrix_n_layout[c_type, tile_n_size, num_m_mmas, num_consumer](),
            element_type = DType.int32,
            linear_idx_type = DType.int32,
        ],
    ):
        """Initialize the fragment writer.

        Args:
            c_tile: Shared memory tile to write to.
            warp_group_thread_idx: Thread index within the warp group.
            local_warp_group_idx: Warp group index within the consumer groups.
            st_matrix_swizzle: Swizzle pattern for bank conflict avoidance.
            st_matrix_rt_layout: Runtime layout for st.matrix operations.
        """
        self.c_tile = c_tile
        self.warp_group_thread_idx = warp_group_thread_idx
        self.local_warp_group_idx = local_warp_group_idx
        self.st_matrix_swizzle = st_matrix_swizzle
        self.st_matrix_rt_layout = st_matrix_rt_layout

    @always_inline
    @staticmethod
    fn _compute_tile_offset(row_tile_idx: UInt, col_tile_idx: UInt) -> Int:
        """Compute shared memory offset from tile coordinates.

        Args:
            row_tile_idx: Row tile index.
            col_tile_idx: Column tile index.

        Returns:
            The linearized tile offset in shared memory.
        """
        return (
            Int(row_tile_idx) * WG_BM * tile_n_size
            + Int(col_tile_idx) * WG_BM * tile_n_size
        )

    @always_inline
    fn write_tile(
        self,
        c_reg_tile: RegTileType[_, _, _],
        coords: Tuple[UInt, UInt],  # (row_tile_idx, col_tile_idx) in the output
    ):
        """Write accumulator fragments to shared memory at specified tile coordinates.

        Args:
            c_reg_tile: Source register tile containing accumulator values.
            coords: Tile coordinates (row_tile_idx, col_tile_idx) where to write.
        """
        # Unpack coordinates
        var row_tile_idx = coords[0]
        var col_tile_idx = coords[1]

        # Compute internal offsets from coordinates
        # col_tile_idx maps to tma_n (which TMA-sized chunk in N dimension)
        var tile_offset = Self._compute_tile_offset(row_tile_idx, col_tile_idx)
        var n_mma_base = Int(col_tile_idx) * (
            tile_n_size // 16
        ) + sub_wg_bn_id * (WG_BN // 16)

        @parameter
        for m_mma in range(num_m_mmas):
            # Each st.matrix instruction handles 16 bytes at a time
            @parameter
            for i in range(tile_n_size // 16):
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
                    Int(self.warp_group_thread_idx),
                    i,
                    m_mma,
                    Int(self.local_warp_group_idx),
                )

                # Calculate which MMA tile we're storing in the N dimension
                var n_mma = n_mma_base + i

                # Calculate shared memory offset with swizzling for bank conflict avoidance
                var offset = self.c_tile.ptr.offset(
                    self.st_matrix_swizzle(
                        self.st_matrix_rt_layout(st_matrix_args)
                    )
                    + tile_offset
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


from collections import OptionalReg


@register_passable("trivial")
struct RegisterToGMemWriter[
    c_type: DType,
    dst_layout: Layout,
    dst_address_space: AddressSpace,
    dst_element_layout: Layout,
    dst_layout_int_type: DType,
    dst_linear_idx_type: DType,
    dst_masked: Bool,
    dst_alignment: Int, //,
    wgmma_shape: IndexList[3],
    num_consumer: Int,
    N: Int,  # Matrix N dimension
    epilogue_fn: OptionalReg[elementwise_epilogue_type] = None,
    compute_lambda_fn: OptionalReg[elementwise_compute_lambda_type] = None,
    check_n_bounds: Bool = False,  # New parameter for N-dimension bounds checking
]:  # (TileWriter):
    """Writer for transferring accumulator registers directly to global memory.

    This writer handles the direct copy from register tiles to global memory
    tiles, with proper thread distribution and alignment. It supports optional
    epilogue processing, compute lambda transformations, and bounds checking.

    Parameters:
        c_type: Output data type.
        dst_layout: Layout of the destination tensor.
        dst_address_space: Address space of the destination tensor.
        dst_element_layout: Element layout of the destination tensor.
        dst_layout_int_type: Integer type for destination layout indices.
        dst_linear_idx_type: Linear index type for destination tensor.
        dst_masked: Whether the destination tensor is masked.
        dst_alignment: Alignment requirement for destination tensor.
        wgmma_shape: Shape of the WGMMA operation [M, N, K].
        num_consumer: Number of consumer warp groups.
        N: Matrix N dimension.
        epilogue_fn: Optional epilogue function (mutates value in place).
        compute_lambda_fn: Optional compute lambda function (returns new value).
        check_n_bounds: Whether to perform bounds checking on N dimension.

    Note:
        At most one of epilogue_fn or compute_lambda_fn should be set.
    """

    alias _dtype = Self.c_type

    alias c_frag_size = wgmma_shape[0] * wgmma_shape[1] // WARPGROUP_SIZE
    alias num_n_frag_mat = wgmma_shape[1] // 8
    alias num_m_frag_mat = wgmma_shape[0] // 4 // 8
    alias num_frag_mats = Self.num_n_frag_mat * Self.num_m_frag_mat

    var thread_info: ThreadInfo
    var dst: LayoutTensor[
        c_type,
        dst_layout,
        MutableAnyOrigin,
        address_space=dst_address_space,
        element_layout=dst_element_layout,
        layout_int_type=dst_layout_int_type,
        linear_idx_type=dst_linear_idx_type,
        masked=dst_masked,
        alignment=dst_alignment,
    ]
    var num_m_mmas: Int
    var tile_coords: OptionalReg[TileCoordinates]
    var M_bound: OptionalReg[UInt]
    var N_bound: OptionalReg[UInt32]  # New field for N-dimension bounds

    @always_inline
    fn __init__(
        out self,
        dst: LayoutTensor[
            c_type,
            dst_layout,
            MutableAnyOrigin,
            address_space=dst_address_space,
            element_layout=dst_element_layout,
            layout_int_type=dst_layout_int_type,
            linear_idx_type=dst_linear_idx_type,
            masked=dst_masked,
            alignment=dst_alignment,
        ],
        warp_group_thread_idx: UInt,
        num_m_mmas: Int,
        tile_coords: OptionalReg[TileCoordinates] = None,
        M_bound: OptionalReg[UInt] = None,
        N_bound: OptionalReg[UInt32] = None,
    ):
        """Initialize the register-to-global-memory writer.

        Args:
            dst: Destination tensor in global memory.
            warp_group_thread_idx: Thread index within the warp group.
            num_m_mmas: Number of MMA tiles in M dimension.
            tile_coords: Optional tile coordinates for epilogue processing.
            M_bound: Optional maximum valid M coordinate (for epilogue).
            N_bound: Optional maximum valid N coordinate (for bounds checking).
        """
        constrained[
            (epilogue_fn is None) or (compute_lambda_fn is None),
            "Only one of epilogue_fn or compute_lambda_fn should be set",
        ]()

        # Store destination tensor
        self.dst = dst
        self.num_m_mmas = num_m_mmas
        self.tile_coords = tile_coords
        self.M_bound = M_bound
        self.N_bound = N_bound

        # Extract thread information
        self.thread_info = ThreadInfo.from_warp_group_idx(warp_group_thread_idx)

    @always_inline
    fn _get_mma_id(self, m_mma: Int, n_mma: Int) -> Int:
        """Calculate MMA tile ID from M and N indices.

        Args:
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.

        Returns:
            The linearized MMA tile ID.
        """
        return n_mma * self.num_m_mmas + m_mma

    @always_inline
    fn write_tile(
        self,
        c_reg_tile: RegTileType[_, _, _],
        coords: Tuple[UInt, UInt],
    ):
        """Write a single MMA tile from registers to global memory.

        Args:
            c_reg_tile: Register tile containing accumulator values.
            coords: Tile coordinates (row, column) in the destination matrix.
        """

        @parameter
        if epilogue_fn is not None or compute_lambda_fn is not None:
            # With compute lambda: transform and store (returns new value)
            self._write_tile_with_epilogue(
                c_reg_tile,
                coords[0],
                coords[1],
            )
        else:
            # Without epilogue or compute lambda: direct copy
            self._write_tile_direct(
                c_reg_tile,
                coords[0],
                coords[1],
            )

    @always_inline
    fn _write_tile_direct(
        self,
        c_reg_tile: RegTileType[_, _, _],
        m_mma: Int,
        n_mma: Int,
    ):
        """Internal method for direct tile write without epilogue."""

        @parameter
        if check_n_bounds:
            # With bounds checking
            self._write_tile_with_runtime_bounds[with_epilogue=False](
                c_reg_tile, m_mma, n_mma
            )
        else:
            # Fast path without bounds checking
            from layout.layout_tensor import copy_local_to_dram

            # Calculate MMA tile ID
            var mma_id = self._get_mma_id(m_mma, n_mma)

            # Get the warp's portion of the tile
            var warp_tile = self.dst.tile[wgmma_shape[0] // 4, wgmma_shape[1]](
                Int(m_mma * 4 + self.thread_info.warp_id), n_mma
            )

            # Get the corresponding register fragment
            var c_frag = c_reg_tile.tile[1, Self.c_frag_size](mma_id, 0)

            # Direct copy using hardware layout
            copy_local_to_dram[Layout.row_major(8, 4)](
                warp_tile.vectorize[1, 2](),
                c_frag.vectorize[1, 2](),
            )

    @always_inline
    fn _write_tile_with_epilogue(
        self,
        c_reg_tile: RegTileType[_, _, _],
        m_mma: Int,
        n_mma: Int,
    ):
        """Internal method for tile write with epilogue processing."""

        # If we need bounds checking, use the runtime bounds version
        @parameter
        if check_n_bounds:
            self._write_tile_with_runtime_bounds[with_epilogue=True](
                c_reg_tile, m_mma, n_mma
            )
            return

        # Calculate MMA tile ID
        var mma_id = self._get_mma_id(m_mma, n_mma)

        # Get warp tile and coordinates
        var warp_tile, warp_tile_coords, warp_tile_offset = (
            self.dst.tile_with_offset[wgmma_shape[0] // 4, wgmma_shape[1]](
                Int(m_mma * 4 + self.thread_info.warp_id), n_mma
            )
        )

        # Calculate final coordinates using tile_coords
        var warp_coords_base = IndexList[2](
            warp_tile_coords[0], warp_tile_coords[1]
        )
        var warp_coords = self.tile_coords.value().adjust(warp_coords_base)

        # Process each element in the warp tile with vectorization
        var c_frag_vec2 = c_reg_tile.vectorize[1, 2]()
        var gmem_frag, gmem_offset_coords_raw, gmem_offset = (
            warp_tile.vectorize[1, 2]().distribute_with_offset[
                Layout.row_major(8, 4)
            ](self.thread_info.lane_id)
        )

        var gmem_offset_coords = IndexList[2](
            gmem_offset_coords_raw[0], gmem_offset_coords_raw[1] * 2
        )
        var coords = gmem_offset_coords + warp_coords

        alias num_vecs = gmem_frag.layout.size()

        @parameter
        for i in range(num_vecs):
            alias dst_idx = gmem_frag.layout(i)
            alias dst_m_offset = dst_idx // N
            alias dst_n_offset = dst_idx % N
            var m = Int(coords[0] + dst_m_offset)
            var n = Int(coords[1] + dst_n_offset)

            alias alignment = align_of[SIMD[c_type, 2]]()

            @parameter
            if epilogue_fn is not None:
                alias epilogue = epilogue_fn.value()

                @parameter
                if check_n_bounds:
                    # Full bounds checking including N dimension
                    if m < Int(self.M_bound.value()) and n < Int(
                        self.N_bound.value()
                    ):
                        epilogue[alignment=alignment](
                            (m, n),
                            c_frag_vec2[mma_id, i].cast[c_type](),
                        )
                else:
                    # Only M bounds checking (N is known to be aligned)
                    if m < Int(self.M_bound.value()) and n < N:
                        epilogue[alignment=alignment](
                            (m, n),
                            c_frag_vec2[mma_id, i].cast[c_type](),
                        )
            else:
                alias compute_lambda = compute_lambda_fn.value()

                if m < Int(self.M_bound.value()) and n < N:
                    var reg_val = compute_lambda[alignment=alignment](
                        (m, n),
                        c_frag_vec2[mma_id, i].cast[c_type](),
                    )
                    # Use warp_tile pointer directly - offsets are relative to dst
                    warp_tile.ptr.store[alignment=alignment](
                        warp_tile_offset + gmem_offset + dst_idx, reg_val
                    )

    @always_inline
    fn _write_tile_with_runtime_bounds[
        with_epilogue: Bool
    ](self, c_reg_tile: RegTileType[_, _, _], m_mma: Int, n_mma: Int,):
        """Unified method for tile write with runtime bounds checking.

        Parameters:
            with_epilogue: Whether to apply epilogue function.

        Args:
            c_reg_tile: Register tile containing accumulator values.
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.
        """
        # Calculate MMA tile ID
        var mma_id = self._get_mma_id(m_mma, n_mma)

        # Get warp tile with bounds checking
        var warp_tile, warp_tile_coords_raw, _ = self.dst.tile_with_offset[
            wgmma_shape[0] // 4, wgmma_shape[1]
        ](Int(m_mma * 4 + self.thread_info.warp_id), n_mma, 0, 0)

        # Convert coordinates
        var warp_tile_coords = rebind[IndexList[2]](warp_tile_coords_raw)

        # Add corner and split coordinates if available
        if self.tile_coords:
            warp_tile_coords = self.tile_coords.value().adjust(warp_tile_coords)

        # Fragment matrix handling
        @parameter
        for n_frag in range(Self.num_n_frag_mat):

            @parameter
            for m_frag in range(Self.num_m_frag_mat):
                alias frag_mat_id = n_frag * Self.num_m_frag_mat + m_frag
                var frag_mat_gmem = warp_tile.tile[8, 8](m_frag, n_frag)

                # Get runtime bounds for this fragment
                var bound0 = UInt32(
                    frag_mat_gmem.runtime_layout.shape[0].value[0]
                )
                var bound1 = UInt32(
                    frag_mat_gmem.runtime_layout.shape[1].value[0]
                )

                # Process each element with runtime bounds checking
                @parameter
                for i in range(2):
                    if (
                        self.thread_info.lane_row < bound0
                        and self.thread_info.lane_col * 2 + i < bound1
                    ):
                        # FIXME: should this support compute_lambda as well?
                        @parameter
                        if with_epilogue:
                            alias epilogue = epilogue_fn.value()
                            var frag_m = Int(warp_tile_coords[0]) + Int(
                                m_frag * 8 + self.thread_info.lane_row
                            )
                            var frag_n = Int(warp_tile_coords[1]) + Int(
                                n_frag * 8 + self.thread_info.lane_col * 2 + i
                            )
                            epilogue[alignment = align_of[Scalar[c_type]]()](
                                (frag_m, frag_n),
                                c_reg_tile[mma_id, frag_mat_id * 2 + i].cast[
                                    c_type
                                ](),
                            )
                        else:
                            frag_mat_gmem[
                                Int(self.thread_info.lane_row),
                                Int(self.thread_info.lane_col * 2 + i),
                            ] = rebind[frag_mat_gmem.element_type](
                                c_reg_tile[mma_id, frag_mat_id * 2 + i].cast[
                                    c_type
                                ]()
                            )
