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

from gpu.intrinsics import AMDBufferResource
from gpu.memory import AddressSpace, external_memory
from layout import Layout, LayoutTensor
from layout.layout import coalesce
from layout._utils import _get_bounds, make_amd_buffer_resource
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    _copy_dram_to_local,
    _copy_local_to_dram,
)
from layout.int_tuple import _get_index_type
from layout.tma_async import SharedMemBarrier
from layout.layout import blocked_product, logical_product


struct ScatterGatherAmd[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
]:
    """Tile-based AMD data movement delegate for scatter-gather operations.

    This struct facilitates data movement between DRAM and registers on AMD GPUs
    using tile-based operations.

    Parameters:
        thread_layout: The layout defining thread organization.
        num_threads: Total number of threads (defaults to thread_layout size).
        thread_scope: The scope of thread execution (block or warp).
        block_dim_count: Number of block dimensions.
    """

    var buffer: AMDBufferResource

    @always_inline
    fn __init__(out self, tensor: LayoutTensor):
        """Initialize the scatter-gather delegate with a tensor.

        Args:
            tensor: The layout tensor to create an AMD buffer resource from.
        """
        self.buffer = make_amd_buffer_resource(tensor)

    @always_inline
    fn copy(
        self,
        dst_reg_tile: LayoutTensor[
            mut=True, *_, address_space = AddressSpace.LOCAL, **_
        ],
        src_gmem_tile: LayoutTensor,
        src_tensor: LayoutTensor,
        offset: OptionalReg[UInt] = None,
    ):
        """Copy data from DRAM to registers (local memory).

        Args:
            dst_reg_tile: Destination register tile in local address space.
            src_gmem_tile: Source global memory tile.
            src_tensor: Source tensor for the copy operation.
            offset: Optional offset for the copy operation.
        """
        _copy_dram_to_local[
            thread_layout, num_threads, thread_scope, block_dim_count
        ](dst_reg_tile, src_gmem_tile, self.buffer)

    @always_inline("nodebug")
    fn copy(
        self,
        dst_gmem_tile: LayoutTensor[mut=True, *_, **_],
        src_reg_tile: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
    ):
        """Copy data from registers (local memory) to DRAM.

        Args:
            dst_gmem_tile: Destination global memory tile.
            src_reg_tile: Source register tile in local address space.
        """
        _copy_local_to_dram[
            thread_layout, num_threads, thread_scope, block_dim_count
        ](dst_gmem_tile, src_reg_tile, self.buffer)


struct IteratorScatterGatherAmd[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
]:
    """Iterator-based AMD data movement delegate for scatter-gather operations.

    This struct provides an iterator-based approach for data movement between
    DRAM and registers on AMD GPUs.

    Parameters:
        thread_layout: The layout defining thread organization.
        num_threads: Total number of threads (defaults to thread_layout size).
        thread_scope: The scope of thread execution (block or warp).
        block_dim_count: Number of block dimensions.
    """

    var buffer: AMDBufferResource

    @always_inline
    fn __init__(out self, tensor: LayoutTensor, tensor_iter: LayoutTensorIter):
        """Initialize the iterator-based scatter-gather delegate.

        Args:
            tensor: The layout tensor for bounds information.
            tensor_iter: The layout tensor iterator for creating the AMD buffer resource.
        """
        self.buffer = make_amd_buffer_resource(tensor_iter, _get_bounds(tensor))

    @always_inline
    fn copy(
        self,
        dst_reg_tile: LayoutTensor[mut=True, *_, **_],
        src_gmem_tile_iter: LayoutTensorIter,
    ):
        """Copy data from DRAM to registers using an iterator.

        Args:
            dst_reg_tile: Destination register tile.
            src_gmem_tile_iter: Source global memory tile iterator.
        """
        _copy_dram_to_local[
            thread_layout, num_threads, thread_scope, block_dim_count
        ](dst_reg_tile, src_gmem_tile_iter, self.buffer)


# Shared Memory and Register tiles type declarations, shared by TileOps and Tile Buffer objects

alias SMemTileType[
    _dtype: DType,
    layout: Layout,
    alignment: Int = align_of[SIMD[_dtype, simd_width_of[_dtype]()]](),
] = LayoutTensor[
    _dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.SHARED,
    alignment=alignment,
]
"""Type alias for shared memory tile tensors."""

alias SMemWarpTileType[
    _dtype: DType, layout: Layout, warp_rows: Int, warp_cols: Int
] = SMemTileType[_dtype, layout].TileType[warp_rows, warp_cols]
"""Type alias for warp-level shared memory tiles with specified dimensions."""

alias RegTileType[
    _dtype: DType,
    layout: Layout,
    alignment: Int = align_of[SIMD[_dtype, simd_width_of[_dtype]()]](),
] = LayoutTensor[
    _dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.LOCAL,
    alignment=alignment,
]
"""Type alias for register (local memory) tile tensors."""

alias SMemBarrier = UnsafePointer[
    SharedMemBarrier, address_space = AddressSpace.SHARED
]
"""Type alias for shared memory barrier pointer."""


@register_passable("trivial")
struct SMemTileArrayType[
    dtype: DType,
    layout: Layout,
    num_tiles: Int,
    alignment: Int,
]:
    """Shared memory tile array for managing multiple tiles in shared memory.

    Parameters:
        dtype: Data type of the tiles.
        layout: Layout configuration for each tile.
        num_tiles: Number of tiles in the array.
        alignment: Memory alignment requirement.
    """

    alias TileType = SMemTileType[
        dtype,
        Self.layout,
        alignment=alignment,
    ]

    var ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED]

    fn __init__[
        mut: Bool, //, origin: Origin[mut]
    ](
        out self,
        unsafe_ptr: UnsafePointer[
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
            mut=mut,
            origin=origin,
        ],
    ):
        """Initialize the shared memory tile array.

        Args:
            unsafe_ptr: Pointer to the shared memory location.
        """
        constrained[
            layout.all_dims_known(), "Layout must be known at compile time."
        ]()

        self.ptr = unsafe_ptr

    fn __getitem__[T: Intable](self, index: T) -> Self.TileType:
        """Access a tile at the specified index.

        Args:
            index: The index of the tile to access.

        Returns:
            The tile at the specified index.
        """
        return Self.TileType(self.ptr + eval[layout.size()] * Int(index))


alias eval[T: AnyType, //, val: T] = val
"""Helper alias to force evaluation of expressions at compile time."""


struct SMemArray[type: AnyType, size: Int]:
    """Helper struct for allocating shared memory arrays.

    Parameters:
        type: The type of elements in the array.
        size: The number of elements in the array.
    """

    alias T = UnsafePointer[type, address_space = AddressSpace.SHARED]
    alias storage_size = size * size_of[type]()

    @staticmethod
    @always_inline
    fn build(mut smem_mgr: NVIDIASharedMemoryManager) -> Self.T:
        """Build a shared memory array using the memory manager.

        Args:
            smem_mgr: The NVIDIA shared memory manager.

        Returns:
            Pointer to the allocated shared memory array.
        """
        return rebind[Self.T](smem_mgr.array[type, size]())


struct SMemTile[dtype: DType, layout: Layout, alignment: Int]:
    """Helper struct for allocating shared memory tiles.

    Parameters:
        dtype: Data type of the tile elements.
        layout: Layout configuration for the tile.
        alignment: Memory alignment requirement.
    """

    alias T = SMemTileType[dtype, layout, alignment=alignment]
    alias storage_size = layout.size() * size_of[dtype]()

    @staticmethod
    @always_inline
    fn build(mut smem_mgr: NVIDIASharedMemoryManager) -> Self.T:
        """Build a shared memory tile using the memory manager.

        Args:
            smem_mgr: The NVIDIA shared memory manager.

        Returns:
            The allocated shared memory tile.
        """
        return rebind[Self.T](smem_mgr.tile[dtype, layout]())


struct SMemTileArray[
    dtype: DType, layout: Layout, alignment: Int, num_tiles: Int
]:
    """Helper struct for allocating arrays of shared memory tiles.

    Parameters:
        dtype: Data type of the tile elements.
        layout: Layout configuration for each tile.
        alignment: Memory alignment requirement.
        num_tiles: Number of tiles in the array.
    """

    alias T = SMemTileArrayType[dtype, layout, num_tiles, alignment=alignment]
    alias storage_size = Self.layout.size() * size_of[dtype]() * num_tiles

    @staticmethod
    @always_inline
    fn build(mut smem_mgr: NVIDIASharedMemoryManager) -> Self.T:
        """Build a shared memory tile array using the memory manager.

        Args:
            smem_mgr: The NVIDIA shared memory manager.

        Returns:
            The allocated shared memory tile array.
        """
        return rebind[Self.T](smem_mgr.tile_array[dtype, layout, num_tiles]())


struct NVIDIASharedMemoryManager[
    name: StaticString = "extern_ptr_syml",
    alignment: Int = 128,
    memory_alignment: Int = 8,
]:
    """Manager for allocating and organizing shared memory on NVIDIA GPUs.

    This struct provides a unified interface for allocating tiles, tile arrays,
    and regular arrays in shared memory with proper alignment.

    Parameters:
        name: Symbol name for external shared memory.
        alignment: Default alignment for memory allocations (default: 128).
        memory_alignment: Base memory alignment (default: 8).
    """

    alias Tile[dtype: DType, layout: Layout] = SMemTile[
        dtype, layout, alignment=alignment
    ]
    alias TileArray[
        dtype: DType, layout: Layout, num_tiles: Int
    ] = SMemTileArray[dtype, layout, alignment, num_tiles=num_tiles]
    alias Array[type: AnyType, num_tiles: Int] = SMemArray[type, num_tiles]

    var base_ptr: UnsafePointer[Int8, address_space = AddressSpace.SHARED]
    var offset: Int

    @always_inline
    fn __init__(out self):
        """Initialize the shared memory manager."""
        self.base_ptr = external_memory[
            Int8,
            address_space = AddressSpace.SHARED,
            alignment=memory_alignment,
            name=name,
        ]()
        self.offset = 0

    @always_inline
    fn tile[
        dtype: DType, layout: Layout
    ](mut self) -> SMemTileType[dtype, layout, alignment=alignment]:
        """Allocate a single tile in shared memory.

        Returns:
            A shared memory tile with the specified configuration.
        """
        var result = SMemTileType[dtype, layout, alignment=alignment](
            self.base_ptr.offset(self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += eval[layout.size()] * size_of[dtype]()
        return result

    @always_inline
    fn tile_array[
        dtype: DType, layout: Layout, num_tiles: Int
    ](mut self) -> SMemTileArrayType[
        dtype, layout, num_tiles, alignment=alignment
    ]:
        """Allocate an array of tiles in shared memory.

        Returns:
            A shared memory tile array with the specified configuration.
        """
        var result = SMemTileArrayType[
            dtype, layout, num_tiles, alignment=alignment
        ](
            self.base_ptr.offset(self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += eval[layout.size()] * size_of[dtype]() * num_tiles
        return result

    @always_inline
    fn array[
        type: AnyType, size: Int
    ](mut self) -> UnsafePointer[type, address_space = AddressSpace.SHARED]:
        """Allocate a regular array in shared memory.

        Returns:
            A pointer to the allocated shared memory array.
        """
        var result = self.base_ptr.offset(self.offset).bitcast[type]()
        self.offset += size * size_of[type]()
        return result
