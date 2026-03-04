# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Shared memory type aliases for LayoutTensor-based GPU kernels.

This module defines the core SMEM type aliases used across SM90, SM100, and
other GPU kernel implementations. They depend only on `layout` and stdlib,
making them safe to import without pulling in higher-level kernel packages.

Types:
- SMemTile: Shared memory tile (LayoutTensor alias)
- RegTile: Register tile (LayoutTensor alias)
- SMemBarrier: Pointer to shared memory barrier
- PipelineBarrier: Array of pipeline barriers
- SMemTileIter: Iterator over shared memory tiles
- SMemTileArray: Array of shared memory tiles
- SMemArray: Generic shared memory array
- SMemPtr: Typed pointer into shared memory
- eval: Compile-time expression evaluator
"""

from std.sys import align_of, size_of

from std.gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.int_tuple import _get_index_type, _get_layout_type
from layout.layout_tensor import LayoutTensorIter
from layout.tma_async import SharedMemBarrier
from std.memory import LegacyUnsafePointer, stack_allocation

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


comptime SMemTile[
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

comptime RegTile[
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

comptime SMemBarrier = UnsafePointer[
    SharedMemBarrier, address_space = AddressSpace.SHARED
]
"""Type alias for shared memory barrier pointer."""

comptime PipelineBarrier[num_pipeline_stages: Int] = SMemArray[
    SharedMemBarrier, num_pipeline_stages
]
"""Type alias for shared memory pipeline barrier array."""

comptime SMemTileIter[
    dtype: DType,
    layout: Layout,
] = LayoutTensorIter[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    alignment=128,
]


struct SMemTileArray[
    dtype: DType,
    layout: Layout,
    num_tiles: Int,
    alignment: Int,
](TrivialRegisterPassable):
    """Array of tiles in shared memory.

    Parameters:
        dtype: Tile data type.
        layout: Tile layout configuration.
        num_tiles: Number of tiles.
        alignment: Memory alignment.
    """

    comptime Tile = SMemTile[
        Self.dtype,
        Self.layout,
        alignment = Self.alignment,
    ]

    comptime num_elements = Self.layout.size() * Self.num_tiles

    comptime storage_size = Self.num_elements * size_of[Self.dtype]()

    comptime Storage = InlineArray[Scalar[Self.dtype], Self.num_elements]

    var ptr: UnsafePointer[
        Scalar[Self.dtype], address_space = AddressSpace.SHARED
    ]

    fn __init__(
        ref[AddressSpace.SHARED] storage: Self.Storage,
    ) -> Self:
        """Initialize with Storage.

        Args:
            storage: Storage.
        """
        return Self(storage.unsafe_ptr())

    fn __init__[
        mut: Bool, //, origin: Origin[mut=mut]
    ](
        out self,
        unsafe_ptr: LegacyUnsafePointer[
            Scalar[Self.dtype],
            address_space = AddressSpace.SHARED,
            origin=origin,
        ],
    ):
        """Initialize with shared memory pointer.

        Args:
            unsafe_ptr: Shared memory pointer.
        """
        comptime assert (
            Self.layout.all_dims_known()
        ), "Layout must be known at compile time."

        self.ptr = unsafe_ptr

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.Tile:
        """Get tile at index.

        Args:
            index: Tile index.

        Returns:
            Tile at index.
        """
        return Self.Tile(self.ptr + eval[Self.layout.size()] * Int(index))

    fn slice[
        length: Int
    ](
        self,
        start: Int,
        out result: SMemTileArray[
            Self.dtype, Self.layout, length, Self.alignment
        ],
    ):
        return type_of(result)(self.ptr + eval[Self.layout.size()] * start)

    @always_inline
    @staticmethod
    fn stack_allocation() -> Self:
        var ptr = stack_allocation[
            Self.storage_size,
            Self.dtype,
            alignment = Self.alignment,
            address_space = AddressSpace.SHARED,
        ]()
        return Self(ptr)


struct SMemArray[type: __TypeOfAllTypes, size: Int](TrivialRegisterPassable):
    """Shared memory array of fixed size.

    Parameters:
        type: Element type.
        size: Number of elements.
    """

    comptime ptr_type = UnsafePointer[
        Self.type, address_space = AddressSpace.SHARED
    ]
    comptime storage_size = Self.size * size_of[Self.type]()
    comptime Storage = InlineArray[Self.type, Self.size]

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

    fn __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from Storage."""
        return Self(rebind[Self.ptr_type](storage.unsafe_ptr()))

    @always_inline
    fn __getitem__[T: Intable](self, index: T) -> Self.ptr_type:
        """Get a pointer to the element at index.

        Args:
            index: Element index.

        Returns:
            Pointer to element.
        """
        return self.ptr + Int(index)

    @always_inline
    @staticmethod
    fn len() -> Int:
        """Get array length in bytes.

        Returns:
            Total size in bytes.
        """
        return Self.size * size_of[Self.type]()

    @always_inline
    @staticmethod
    fn stack_allocation[alignment: Int = align_of[Self.type]()]() -> Self:
        var ptr = stack_allocation[
            Self.len(),
            Self.type,
            alignment=alignment,
            address_space = AddressSpace.SHARED,
        ]()
        return Self(ptr)


comptime eval[T: AnyType, //, val: T] = val
"""Helper alias to force evaluation of expressions at compile time."""

comptime SMemPtr[type: AnyType] = UnsafePointer[
    type, address_space = AddressSpace.SHARED
]
