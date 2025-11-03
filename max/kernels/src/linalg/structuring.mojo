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
from gpu.memory import external_memory
from layout import Layout, LayoutTensor
from layout.layout import coalesce
from layout._utils import _get_bounds, make_amd_buffer_resource
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    _copy_dram_to_local,
    _copy_local_to_dram,
)
from layout.int_tuple import (
    _get_index_type,
    _get_layout_type,
    _get_unsigned_type,
)
from layout.tma_async import SharedMemBarrier
from layout.layout import blocked_product, logical_product
from memory import stack_allocation


struct ScatterGatherAmd[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
]:
    """AMD tile-based scatter-gather for DRAM-register data movement.

    Parameters:
        thread_layout: Thread organization layout.
        num_threads: Total threads (defaults to thread_layout size).
        thread_scope: Thread execution scope (block or warp).
        block_dim_count: Number of block dimensions.
    """

    var buffer: AMDBufferResource

    @always_inline
    fn __init__(out self, tensor: LayoutTensor):
        """Initialize with a tensor.

        Args:
            tensor: Layout tensor for AMD buffer resource creation.
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
        """Copy DRAM to registers.

        Args:
            dst_reg_tile: Destination register tile.
            src_gmem_tile: Source global memory tile.
            src_tensor: Source tensor.
            offset: Optional copy offset.
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
        """Copy registers to DRAM.

        Args:
            dst_gmem_tile: Destination global memory tile.
            src_reg_tile: Source register tile.
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
    """Iterator-based AMD scatter-gather for DRAM-register data movement.

    Parameters:
        thread_layout: Thread organization layout.
        num_threads: Total threads (defaults to thread_layout size).
        thread_scope: Thread execution scope (block or warp).
        block_dim_count: Number of block dimensions.
    """

    var buffer: AMDBufferResource

    @always_inline
    fn __init__(out self, tensor: LayoutTensor, tensor_iter: LayoutTensorIter):
        """Initialize with tensor and iterator.

        Args:
            tensor: Layout tensor for bounds.
            tensor_iter: Iterator for AMD buffer resource.
        """
        self.buffer = make_amd_buffer_resource(tensor_iter, _get_bounds(tensor))

    @always_inline
    fn copy(
        self,
        dst_reg_tile: LayoutTensor[mut=True, *_, **_],
        src_gmem_tile_iter: LayoutTensorIter,
    ):
        """Copy DRAM to registers via iterator.

        Args:
            dst_reg_tile: Destination register tile.
            src_gmem_tile_iter: Source memory iterator.
        """
        _copy_dram_to_local[
            thread_layout, num_threads, thread_scope, block_dim_count
        ](dst_reg_tile, src_gmem_tile_iter, self.buffer)


# Shared Memory and Register tiles type declarations, shared by TileOps and Tile Buffer objects

alias SMemTileType[
    _dtype: DType,
    layout: Layout,
    /,
    *,
    element_layout: Layout = Layout(1, 1),
    layout_int_type: DType = _get_layout_type(layout, AddressSpace.SHARED),
    linear_idx_type: DType = _get_index_type(layout, AddressSpace.SHARED),
    masked: Bool = False,
    alignment: Int = align_of[_dtype](),
] = LayoutTensor[
    _dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    element_layout=element_layout,
    layout_int_type=layout_int_type,
    linear_idx_type=linear_idx_type,
    masked=masked,
    alignment=alignment,
]
"""Type alias for shared memory tile tensors."""

alias RegTileType[
    _dtype: DType,
    layout: Layout,
    /,
    *,
    element_layout: Layout = Layout(1, 1),
    layout_int_type: DType = _get_layout_type(layout, AddressSpace.LOCAL),
    linear_idx_type: DType = _get_index_type(layout, AddressSpace.LOCAL),
    masked: Bool = False,
    alignment: Int = align_of[_dtype](),
] = LayoutTensor[
    _dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.LOCAL,
    element_layout=element_layout,
    layout_int_type=layout_int_type,
    linear_idx_type=linear_idx_type,
    masked=masked,
    alignment=alignment,
]
"""Type alias for register (local memory) tile tensors."""

alias SMemBarrier = UnsafePointer[
    SharedMemBarrier, address_space = AddressSpace.SHARED
]
"""Type alias for shared memory barrier pointer."""

alias PipelineBarrier[num_pipeline_stages: Int] = SMemArrayType[
    SharedMemBarrier, num_pipeline_stages
]
"""Type alias for shared memory pipeline barrier array."""


@register_passable("trivial")
struct SMemTileArrayType[
    dtype: DType,
    layout: Layout,
    num_tiles: Int,
    alignment: Int,
]:
    """Array of tiles in shared memory.

    Parameters:
        dtype: Tile data type.
        layout: Tile layout configuration.
        num_tiles: Number of tiles.
        alignment: Memory alignment.
    """

    alias Tile = SMemTileType[
        dtype,
        Self.layout,
        alignment=alignment,
    ]

    alias storage_size = eval[layout.size()] * size_of[dtype]() * num_tiles

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
        """Initialize with shared memory pointer.

        Args:
            unsafe_ptr: Shared memory pointer.
        """
        constrained[
            layout.all_dims_known(), "Layout must be known at compile time."
        ]()

        self.ptr = unsafe_ptr

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at index.

        Args:
            index: Tile index.

        Returns:
            Tile at index.
        """
        return Self.Tile(self.ptr + eval[layout.size()] * Int(index))

    @always_inline
    @staticmethod
    fn stack_allocation() -> Self:
        var ptr = stack_allocation[
            Self.storage_size,
            dtype,
            alignment=alignment,
            address_space = AddressSpace.SHARED,
        ]()
        return Self(ptr)


@register_passable("trivial")
struct SMemArrayType[type: AnyTrivialRegType, size: Int]:
    """Shared memory array of fixed size.

    Parameters:
        type: Element type.
        size: Number of elements.
    """

    alias ptr_type = UnsafePointer[type, address_space = AddressSpace.SHARED]
    alias storage_size = size * size_of[type]()

    var ptr: Self.ptr_type

    @always_inline
    fn __init__(
        out self,
        unsafe_ptr: Self.ptr_type,
    ):
        """Initialize with shared memory pointer.

        Args:
            unsafe_ptr: Shared memory pointer.
        """
        self.ptr = unsafe_ptr

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.ptr_type:
        """Get a pointer to the element at index.

        Args:
            index: Element index.

        Returns:
            Pointer to element.
        """
        return self.ptr.offset(Int(index))

    @always_inline
    @staticmethod
    fn len() -> Int:
        """Get array length in bytes.

        Returns:
            Total size in bytes.
        """
        return size * size_of[type]()

    @always_inline
    @staticmethod
    fn stack_allocation[alignment: Int = align_of[type]()]() -> Self:
        var ptr = stack_allocation[
            Self.len(),
            type,
            alignment=alignment,
            address_space = AddressSpace.SHARED,
        ]()
        return Self(ptr)


alias eval[T: AnyType, //, val: T] = val
"""Helper alias to force evaluation of expressions at compile time."""


trait SharedMemoryBasePtr:
    alias alignment: Int

    @always_inline
    @staticmethod
    fn ptr() -> UnsafePointer[Int8, address_space = AddressSpace.SHARED]:
        ...


struct NVIDIASharedMemoryBasePtr[
    name: StaticString = "extern_ptr_syml",
    memory_alignment: Int = 8,
](SharedMemoryBasePtr):
    alias alignment: Int = 128

    @always_inline
    @staticmethod
    fn ptr() -> UnsafePointer[Int8, address_space = AddressSpace.SHARED]:
        return external_memory[
            Int8,
            address_space = AddressSpace.SHARED,
            alignment=memory_alignment,
            name=name,
        ]()


struct SharedMemoryManager[SMBP: SharedMemoryBasePtr]:
    alias Tile[dtype: DType, layout: Layout] = SMemTileType[
        dtype, layout, alignment = SMBP.alignment
    ]

    alias TileArray[
        dtype: DType, layout: Layout, num_tiles: Int
    ] = SMemTileArrayType[dtype, layout, num_tiles, SMBP.alignment]

    alias Array[type: AnyTrivialRegType, size: Int] = SMemArrayType[type, size]

    var base_ptr: UnsafePointer[Int8, address_space = AddressSpace.SHARED]
    var offset: Int

    @always_inline
    fn __init__(out self):
        """Initialize the shared memory manager."""
        self.base_ptr = SMBP.ptr()
        self.offset = 0

    @always_inline
    fn build[
        dtype: DType,
        layout: Layout, //,
        T: type_of(Self.Tile[dtype, layout]),
    ](mut self) -> T:
        """Allocate a single tile.

        Returns:
            Allocated tile.
        """
        var result = T(
            self.base_ptr.offset(self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += T.storage_size
        return result

    @always_inline
    fn build[
        dtype: DType,
        layout: Layout,
        num_tiles: Int, //,
        T: type_of(Self.TileArray[dtype, layout, num_tiles]),
    ](mut self) -> T:
        """Allocate a tile array.

        Returns:
            Allocated tile array.
        """
        var result = T(
            self.base_ptr.offset(self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += T.storage_size
        return result

    @always_inline
    fn build[
        type: AnyTrivialRegType,
        size: Int, //,
        T: type_of(Self.Array[type, size]),
    ](mut self) -> T:
        """Allocate a regular array.

        Returns:
            Allocated array.
        """
        var result = self.base_ptr.offset(self.offset).bitcast[type]()
        self.offset += T.storage_size
        return T(result)


alias NVIDIASharedMemoryManager = SharedMemoryManager[NVIDIASharedMemoryBasePtr]
