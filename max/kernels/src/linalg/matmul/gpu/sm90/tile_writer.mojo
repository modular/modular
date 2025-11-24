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

This module provides utilities for writing tiles to memory using different
mechanisms and destinations:

1. Register → Shared Memory: Uses st.matrix hardware instruction for efficient
   storage of WGMMA accumulator results to shared memory with swizzling.

2. Register → Global Memory: Direct stores from register tiles to global memory
   with optional epilogue processing and bounds checking.

3. Shared Memory → Global Memory: Hardware-accelerated TMA stores or regular
   stores for efficient 2D tile transfers from shared to global memory.

Two main traits abstract these writing mechanisms:
- TileWriter: For shared memory → global memory transfers
- RegTileWriter: For register → memory (shared or global) transfers
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
from gpu import thread_idx, lane_id
from sys import simd_width_of
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout.layout import coalesce
from layout import Layout
from gpu.globals import WARP_SIZE, WARPGROUP_SIZE

from gpu.mma import st_matrix
from memory import bitcast
from layout import RuntimeLayout, RuntimeTuple, IntTuple
from layout.tensor_core_async import st_matrix_n_layout
from layout.runtime_layout import UNKNOWN_VALUE
from ....utils import elementwise_epilogue_type, elementwise_compute_lambda_type
from utils.index import IndexList
from sys import align_of, size_of
from collections import OptionalReg
from layout.layout_tensor import copy_local_to_dram
import itertools
from memory.pointer import _GPUAddressSpace
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core_async import st_matrix_n_layout
from stdlib.bit import log2_floor


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
        var lane_row, lane_col = divmod(UInt32(lid), 4)
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
trait SMemTileWriter:
    """Base trait for tile writing mechanisms in matrix multiplication.

    This trait defines the interface for writing tiles from shared memory to global memory,
    abstracting over different hardware mechanisms.
    """

    alias _dtype: DType

    @always_inline
    fn write_tile(
        self,
        src: SMemTileType[Self._dtype, _, alignment=128, **_],
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
    desc_layout: Layout, //,
](SMemTileWriter):
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
        TMATensorTile[Self.dtype, Self.tma_layout, Self.desc_layout],
        Self.tma_origin,
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
        src: SMemTileType[Self._dtype, _, alignment=128, **_],
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
struct TileWriterThreadwise[
    dtype: DType,
    dst_layout: Layout,
    dst_address_space: AddressSpace,
    dst_element_layout: Layout,
    dst_layout_int_type: DType,
    dst_linear_idx_type: DType,
    dst_masked: Bool,
    dst_alignment: Int, //,
    thread_layout: Layout,
    simd_size: Int,
    half_tile: Bool = False,  # Handle masked x2 case
](SMemTileWriter):
    alias _dtype = Self.dtype

    alias DstType = LayoutTensor[
        Self.dtype,
        Self.dst_layout,
        MutAnyOrigin,
        address_space = Self.dst_address_space,
        element_layout = Self.dst_element_layout,
        layout_int_type = Self.dst_layout_int_type,
        linear_idx_type = Self.dst_linear_idx_type,
        masked = Self.dst_masked,
        alignment = Self.dst_alignment,
    ]
    var dst: Self.DstType
    var thread_idx: UInt

    @always_inline
    fn __init__(
        out self,
        dst: Self.DstType,
        thread_idx: UInt,
    ):
        """Initialize the threadwise tile writer.

        Args:
            dst: Destination tensor in global memory.
            thread_idx: Thread index within the consumer warp group.
        """
        self.dst = dst
        self.thread_idx = thread_idx

    @always_inline
    fn write_tile(
        self,
        src: SMemTileType[Self._dtype, _, alignment=128, **_],
        coords: Tuple[UInt, UInt],
    ):
        """Write a tile using thread-distributed stores.

        Each thread writes a portion of the tile with proper swizzling
        for optimal memory access patterns.

        Args:
            src: Source tile in shared memory.
            coords: Tile indices (row_tile, col_tile) in the destination matrix.
        """
        # For the threadwise writer, coords are always (0, 0) because the
        # destination tile is already pre-extracted as the workgroup tile.
        # We directly use self.dst as the destination.

        alias swizzle = make_ldmatrix_swizzle[
            Self._dtype,
            src.stride[0](),
            log2_floor(16 // size_of[Self._dtype]()),
        ]()

        @parameter
        if Self.half_tile:
            # Handle masked x2 case - write only half the tile width
            # Get compile-time layout dimensions
            alias dst_height = Self.dst_layout.shape[0].value()
            alias dst_width = Self.dst_layout.shape[1].value()
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
                Self.thread_layout.shape[0].value(),
                Self.thread_layout.shape[1].value() // 2,
            )

            # Only first half of threads participate
            alias num_threads = Self.thread_layout.size()
            if self.thread_idx < UInt(num_threads // 2):
                copy_sram_to_dram[
                    thread_layout=half_thread_layout,
                    swizzle=swizzle,
                ](
                    masked_dst.vectorize[1, Self.simd_size](),
                    masked_src.vectorize[1, Self.simd_size](),
                )
        else:
            # Normal case - write full tile
            copy_sram_to_dram[
                thread_layout = Self.thread_layout,
                swizzle=swizzle,
            ](
                self.dst.vectorize[1, Self.simd_size](),
                src.vectorize[1, Self.simd_size](),
            )


@register_passable("trivial")
trait RegTileWriter:
    """Base trait for tile writing mechanisms in matrix multiplication.

    This trait defines the interface for writing register tiles to memory
    (either shared memory or global memory).
    """

    @always_inline
    fn write_tile(
        self,
        c_reg_tile: RegTileType,
        coords: Tuple[UInt, UInt],
    ) capturing -> None:
        """Write a register tile to memory.

        Args:
            c_reg_tile: Source register tile containing accumulator values.
            coords: Tile coordinates (row, column) in the destination matrix.
        """
        ...


@register_passable("trivial")
struct FragmentToSMemWriter[
    c_type: DType,
    c_tile_layout: Layout, //,
    tile_n_size: Int,  # Size of each tile in N dimension (e.g., TMA_BN)
    num_m_mmas: Int,
    num_consumer: Int,
    half_tile: Bool,
    WG_BM: Int,  # Warp group M dimension
    WG_BN: Int,  # Warp group N dimension
    sub_wg_id: Int,  # Sub warp group ID in N dimension
](RegTileWriter):
    """Writes WGMMA accumulator results from registers to shared memory using st.matrix.

    Stores 16-byte fragments with swizzling to avoid bank conflicts. Sub-warp groups
    divide N-dimension work, each handling a portion of WG_BN output tiles.

    Parameters:
        c_type: Output data type (must be bfloat16 for st.matrix).
        c_tile_layout: Layout of the entire shared memory region.
        tile_n_size: Width of each output tile (typically TMA_BN).
        num_m_mmas: Number of MMA operations in M dimension.
        num_consumer: Number of consumer warp groups.
        half_tile: Special mode for handling partial tiles.
        WG_BM: Warp group tile height.
        WG_BN: Warp group tile width.
        sub_wg_id: Which portion of WG_BN this instance handles.
    """

    alias st_matrix_swizzle = make_ldmatrix_swizzle[
        Self.c_type, Self.tile_n_size, log2_floor(16 // size_of[Self.c_type]())
    ]()
    alias st_matrix_rt_layout_type = RuntimeLayout[
        st_matrix_n_layout[
            Self.c_type, Self.tile_n_size, Self.num_m_mmas, Self.num_consumer
        ](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    var c_tile: SMemTileType[Self.c_type, Self.c_tile_layout, alignment=128]
    var warp_group_thread_idx: UInt
    var local_warp_group_idx: UInt
    var st_matrix_rt_layout: RuntimeLayout[
        st_matrix_n_layout[
            Self.c_type, Self.tile_n_size, Self.num_m_mmas, Self.num_consumer
        ](),
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    @always_inline
    fn __init__(
        out self,
        c_tile: SMemTileType[Self.c_type, Self.c_tile_layout, alignment=128],
        warp_group_thread_idx: UInt,
        local_warp_group_idx: UInt,
    ):
        """Initialize the fragment writer.

        Args:
            c_tile: Shared memory tile to write to.
            warp_group_thread_idx: Thread index within the warp group.
            local_warp_group_idx: Sub-warp group index (divides N-dimension work).
        """
        self.c_tile = c_tile
        self.warp_group_thread_idx = warp_group_thread_idx
        self.local_warp_group_idx = local_warp_group_idx
        self.st_matrix_rt_layout = Self.st_matrix_rt_layout_type()

    @always_inline
    fn _compute_swizzled_offset[n_frag: Int, m_frag: Int](self) -> Int32:
        """Compute swizzled offset for st.matrix to avoid bank conflicts.

        Parameters:
            n_frag: Fragment index in N dimension within tile.
            m_frag: Fragment index in M dimension (MMA tile index).

        Returns:
            Swizzled offset for st.matrix instruction.
        """
        var layout_coords = RuntimeTuple[
            IntTuple(
                UNKNOWN_VALUE,
                IntTuple(n_frag, m_frag, UNKNOWN_VALUE),
            )
        ](
            Int(self.warp_group_thread_idx),
            n_frag,
            m_frag,
            Int(self.local_warp_group_idx),
        )
        var linear_idx = self.st_matrix_rt_layout(layout_coords)
        return self.st_matrix_swizzle(linear_idx)

    alias st_matrix_layout = Layout.row_major(Self.WG_BM, Self.tile_n_size)

    @always_inline
    fn _store_fragment[
        elements_per_op: Int,  # 8 for normal mode, 4 for x2 mode
        m_frag: Int,
        n_frag: Int,
    ](
        self,
        smem_tile: SMemTileType[Self.c_type, Self.st_matrix_layout, **_],
        data: SIMD[Self.c_type, elements_per_op],
    ) -> None:
        """Store register data to shared memory using st.matrix instruction.

        Parameters:
            elements_per_op: Elements per st.matrix operation (4 or 8).
            m_frag: Fragment index in M dimension (0 to num_m_mmas-1).
            n_frag: Fragment index in N dimension within tile.

        Args:
            smem_tile: Target shared memory tile.
            data: Register data to store.
        """
        alias packed_width = elements_per_op // 2  # BF16 pairs packed as float32

        # Pack BF16 pairs into float32 (hardware requirement)
        var packed_data = bitcast[DType.float32, packed_width](data)

        # Get swizzled offset for bank conflict avoidance
        var swizzled_offset = self._compute_swizzled_offset[n_frag, m_frag]()

        # Execute st.matrix hardware instruction
        st_matrix[simd_width=packed_width](
            smem_tile.ptr.offset(swizzled_offset), packed_data
        )

    @always_inline
    fn write_tile(
        self,
        c_reg_tile: RegTileType,
        coords: Tuple[UInt, UInt],
    ) capturing -> None:
        """Write accumulator tile from registers to shared memory.

        Args:
            c_reg_tile: Register tile containing MMA results.
            coords: Tile position (row_idx, col_idx) in output.
        """
        # Locate destination tile in shared memory
        var tile_linear_idx = Int(coords[0]) + Int(coords[1])
        alias elements_per_tile = Self.WG_BM * Self.tile_n_size
        alias total_tiles = Self.c_tile_layout.size() // elements_per_tile

        # Reshape shared memory to access individual tiles
        var smem_tiles = self.c_tile.reshape[
            Layout.row_major(total_tiles, elements_per_tile)
        ]()
        var dest_tile_flat = smem_tiles.tile[1, elements_per_tile](
            tile_linear_idx, 0
        )

        # SMem fragment view for st.matrix operations
        var smem_frag = dest_tile_flat.reshape[Self.st_matrix_layout]()

        # st.matrix configuration
        alias elements_per_store = 4 * (1 if Self.half_tile else 2)
        alias reg_fragment_scale = 2 if Self.half_tile else 1

        alias ST_MATRIX_WIDTH_BYTES = 16  # Fragment size: st.matrix operates on 16-byte chunks
        var n_fragment_base = (
            Int(coords[1]) * Self.tile_n_size
            + Self.sub_wg_id * Self.WG_BN  # Sub-warp handles portion of WG_BN
        ) // ST_MATRIX_WIDTH_BYTES

        # Store all fragments using st.matrix
        @parameter
        for m_frag, n_frag in itertools.product(
            range(Self.num_m_mmas),
            range(Self.tile_n_size // ST_MATRIX_WIDTH_BYTES),
        ):
            # Load fragment from registers
            var reg_fragment_idx = reg_fragment_scale * (
                n_fragment_base + n_frag
            )
            var reg_fragment = c_reg_tile.tile[1, elements_per_store](
                m_frag, reg_fragment_idx
            )
            var frag_data = reg_fragment.load[elements_per_store](0, 0).cast[
                Self.c_type
            ]()
            self._store_fragment[elements_per_store, m_frag, n_frag](
                smem_frag, frag_data
            )


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
    check_runtime_bounds: Bool = False,  # New parameter for N-dimension bounds checking
](RegTileWriter):
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
        check_runtime_bounds: Whether to perform bounds checking on N dimension.

    Note:
        At most one of epilogue_fn or compute_lambda_fn should be set.
    """

    alias c_frag_size = Self.wgmma_shape[0] * Self.wgmma_shape[
        1
    ] // WARPGROUP_SIZE
    alias num_n_frag_mat = Self.wgmma_shape[1] // 8
    alias num_m_frag_mat = Self.wgmma_shape[0] // 4 // 8
    alias num_frag_mats = Self.num_n_frag_mat * Self.num_m_frag_mat

    var thread_info: ThreadInfo

    alias DstType = LayoutTensor[
        Self.c_type,
        Self.dst_layout,
        MutAnyOrigin,
        address_space = Self.dst_address_space,
        element_layout = Self.dst_element_layout,
        layout_int_type = Self.dst_layout_int_type,
        linear_idx_type = Self.dst_linear_idx_type,
        masked = Self.dst_masked,
        alignment = Self.dst_alignment,
    ]
    var dst: Self.DstType
    var num_m_mmas: Int
    var tile_coords: OptionalReg[TileCoordinates]
    var max_row: OptionalReg[UInt32]

    @always_inline
    fn __init__(
        out self,
        dst: Self.DstType,
        warp_group_thread_idx: UInt,
        num_m_mmas: Int,
        tile_coords: OptionalReg[TileCoordinates] = None,
        max_row: OptionalReg[UInt32] = None,
    ):
        """Initialize the register-to-global-memory writer.

        Args:
            dst: Destination tensor in global memory.
            warp_group_thread_idx: Thread index within the warp group.
            num_m_mmas: Number of MMA tiles in M dimension.
            tile_coords: Optional tile coordinates for epilogue processing.
            max_row: Optional maximum valid M coordinate (for epilogue).
        """
        constrained[
            (Self.epilogue_fn is None) or (Self.compute_lambda_fn is None),
            "Only one of epilogue_fn or compute_lambda_fn should be set",
        ]()

        # Store destination tensor
        self.dst = dst
        self.num_m_mmas = num_m_mmas
        self.tile_coords = tile_coords
        self.max_row = max_row

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
        c_reg_tile: RegTileType,
        coords: Tuple[UInt, UInt],
    ) capturing -> None:
        """Write a single MMA tile from registers to global memory.

        Args:
            c_reg_tile: Register tile containing accumulator values.
            coords: Tile coordinates (row, column) in the destination matrix.
        """
        var m_mma = Int(coords[0])
        var n_mma = Int(coords[1])
        var mma_id = self._get_mma_id(m_mma, n_mma)

        @parameter
        if Self.check_runtime_bounds:
            # Element-by-element with runtime bounds checking
            self._write_with_runtime_bounds(c_reg_tile, m_mma, n_mma, mma_id)
        elif Self.epilogue_fn is not None or Self.compute_lambda_fn is not None:
            # Vectorized with epilogue/compute_lambda
            self._write_with_transform(c_reg_tile, m_mma, n_mma, mma_id)
        else:
            # Direct vectorized copy
            self._write_direct_vectorized(c_reg_tile, m_mma, n_mma, mma_id)

    @always_inline
    fn _write_direct_vectorized(
        self,
        c_reg_tile: RegTileType,
        m_mma: Int,
        n_mma: Int,
        mma_id: Int,
    ):
        """Direct vectorized copy without transformations."""
        # Get the warp's portion of the tile
        var warp_tile = self.dst.tile[
            Self.wgmma_shape[0] // 4, Self.wgmma_shape[1]
        ](Int(m_mma * 4 + Int(self.thread_info.warp_id)), n_mma)

        # Get the corresponding register fragment
        var c_frag = c_reg_tile.tile[1, Self.c_frag_size](mma_id, 0)

        # Direct copy using hardware layout
        copy_local_to_dram[Layout.row_major(8, 4)](
            warp_tile.vectorize[1, 2](),
            c_frag.vectorize[1, 2](),
        )

    @always_inline
    fn _write_with_transform(
        self,
        c_reg_tile: RegTileType,
        m_mma: Int,
        n_mma: Int,
        mma_id: Int,
    ):
        """Vectorized write with epilogue or compute_lambda transformation."""
        # Get warp tile and coordinates
        var warp_tile, warp_tile_coords, warp_tile_offset = (
            self.dst.tile_with_offset[
                Self.wgmma_shape[0] // 4, Self.wgmma_shape[1]
            ](Int(m_mma * 4 + Int(self.thread_info.warp_id)), n_mma)
        )

        # Calculate global coordinates
        var warp_coords_base = IndexList[2](
            warp_tile_coords[0], warp_tile_coords[1]
        )
        var warp_coords = self.tile_coords.value().adjust(warp_coords_base)

        # Distribute fragments across threads
        var c_reg_frag = c_reg_tile.vectorize[1, 2]()
        var gmem_frag, gmem_offset_coords_raw, gmem_offset = (
            warp_tile.vectorize[1, 2]().distribute_with_offset[
                Layout.row_major(8, 4)
            ](self.thread_info.lane_id)
        )

        var gmem_offset_coords = IndexList[2](
            gmem_offset_coords_raw[0], gmem_offset_coords_raw[1] * 2
        )
        var coords = gmem_offset_coords + warp_coords
        var max_row = self.max_row.value()

        alias num_vecs = gmem_frag.layout.size()

        # Process all vectors
        @parameter
        for frag_idx in range(num_vecs):
            alias dst_idx = gmem_frag.layout(frag_idx)
            alias dst_m_offset = dst_idx // Self.N
            alias dst_n_offset = dst_idx % Self.N
            var m = Int(coords[0] + dst_m_offset)
            var n = Int(coords[1] + dst_n_offset)

            # Bounds check and apply transformation
            if m < Int(max_row) and n < Self.N:
                self._apply_transform_and_store[frag_idx](
                    gmem_frag, c_reg_frag, mma_id, m, n
                )

    @always_inline
    fn _apply_transform_and_store[
        frag_idx: Int
    ](
        self,
        gmem_frag: LayoutTensor[
            Self.c_type, _, MutAnyOrigin, address_space=_, *_, **_
        ],
        c_reg_frag: RegTileType,
        mma_id: Int,
        m: Int,
        n: Int,
    ) capturing:
        """Apply epilogue or compute_lambda and store result."""
        alias alignment = align_of[SIMD[Self.c_type, 2]]()

        @parameter
        if Self.epilogue_fn:
            alias epilogue = Self.epilogue_fn.value()
            epilogue[alignment=alignment](
                (m, n),
                c_reg_frag[mma_id, frag_idx].cast[Self.c_type](),
            )
        else:  # compute_lambda
            alias compute_lambda = Self.compute_lambda_fn.value()
            var reg_val = compute_lambda[alignment=alignment](
                (m, n),
                c_reg_frag[mma_id, frag_idx].cast[Self.c_type](),
            )
            gmem_frag[frag_idx, 0] = rebind[gmem_frag.element_type](reg_val)

    @always_inline
    fn _write_with_runtime_bounds(
        self,
        c_reg_tile: RegTileType,
        m_mma: Int,
        n_mma: Int,
        mma_id: Int,
    ):
        """Element-by-element with full runtime bounds checking.

        Args:
            c_reg_tile: Register tile containing accumulator values.
            m_mma: MMA tile index in M dimension.
            n_mma: MMA tile index in N dimension.
            mma_id: Linearized MMA tile ID.
        """
        # Get warp tile with bounds checking
        var warp_tile, warp_tile_coords_raw, _ = self.dst.tile_with_offset[
            Self.wgmma_shape[0] // 4, Self.wgmma_shape[1]
        ](Int(m_mma * 4 + Int(self.thread_info.warp_id)), n_mma, 0, 0)

        var warp_tile_coords = rebind[IndexList[2]](warp_tile_coords_raw)
        if self.tile_coords:
            warp_tile_coords = self.tile_coords.value().adjust(warp_tile_coords)

        # Process fragment matrices
        @parameter
        for m_frag, n_frag in itertools.product(
            range(Self.num_m_frag_mat), range(Self.num_n_frag_mat)
        ):
            alias frag_mat_id = n_frag * Self.num_m_frag_mat + m_frag
            var frag_mat_gmem = warp_tile.tile[8, 8](m_frag, n_frag)

            # Get runtime bounds
            var max_row = UInt32(frag_mat_gmem.runtime_layout.shape[0].value[0])
            var max_col = UInt32(frag_mat_gmem.runtime_layout.shape[1].value[0])

            @parameter
            for i in range(2):
                if (
                    self.thread_info.lane_row < max_row
                    and self.thread_info.lane_col * 2 + i < max_col
                ):
                    var reg_val = c_reg_tile[mma_id, frag_mat_id * 2 + i].cast[
                        Self.c_type
                    ]()

                    @parameter
                    fn epilogue_coordinates() -> Tuple[Int, Int]:
                        return (
                            Int(warp_tile_coords[0])
                            + Int(m_frag * 8 + self.thread_info.lane_row),
                            Int(warp_tile_coords[1])
                            + Int(
                                n_frag * 8 + self.thread_info.lane_col * 2 + i
                            ),
                        )

                    @parameter
                    if Self.epilogue_fn:
                        alias epilogue = Self.epilogue_fn.value()
                        var frag_m, frag_n = epilogue_coordinates()
                        epilogue[alignment = align_of[Scalar[Self.c_type]]()](
                            (frag_m, frag_n), reg_val
                        )
                    else:

                        @parameter
                        if Self.compute_lambda_fn:
                            alias compute_lambda = Self.compute_lambda_fn.value()
                            var frag_m, frag_n = epilogue_coordinates()
                            reg_val = compute_lambda[
                                alignment = align_of[Scalar[Self.c_type]]()
                            ]((frag_m, frag_n), reg_val)

                        frag_mat_gmem[
                            Int(self.thread_info.lane_row),
                            Int(self.thread_info.lane_col * 2 + i),
                        ] = rebind[frag_mat_gmem.element_type](reg_val)
