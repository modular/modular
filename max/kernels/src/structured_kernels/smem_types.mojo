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
"""Shared memory type aliases for GPU kernels.

This module defines the core SMEM type aliases used across SM90, SM100, and
other GPU kernel implementations. They depend only on `layout` and stdlib,
making them safe to import without pulling in higher-level kernel packages.

Types:
- SMemBarrier: Pointer to shared memory barrier
- PipelineBarrier: Array of pipeline barriers
- SMemArray: Generic shared memory array
- SMemPtr: Typed pointer into shared memory
- eval: Compile-time expression evaluator

Shared-memory and register tiles are `TileTensor`s; see
`structured_kernels.tile_types` for the tile and tile-array helpers.
"""

from std.sys import align_of, size_of

from layout.tma_async import SharedMemBarrier
from std.memory import unsafe_stack_allocation


comptime SMemBarrier = MutPointer[SharedMemBarrier, _, address_space=.SHARED]
"""Type alias for shared memory barrier pointer."""

comptime PipelineBarrier[num_pipeline_stages: Int] = SMemArray[
    SharedMemBarrier, num_pipeline_stages
]
"""Type alias for shared memory pipeline barrier array."""


struct SMemArray[type: TrivialRegisterPassable, size: Int](
    TrivialRegisterPassable
):
    """Shared memory array of fixed size.

    Parameters:
        type: Element type.
        size: Number of elements.
    """

    comptime ptr_type = MutPointer[
        Self.type, MutUntrackedOrigin, address_space=.SHARED
    ]
    comptime storage_size = Self.size * size_of[Self.type]()
    comptime Storage = Array[Self.type, Self.size]

    var ptr: Self.ptr_type

    @inline(.always)
    def __init__(
        out self,
        unsafe_ptr: Self.ptr_type,
    ):
        """Initialize with shared memory pointer.

        Args:
            unsafe_ptr: Shared memory pointer.
        """
        self.ptr = unsafe_ptr

    def __init__(ref[AddressSpace.SHARED] storage: Self.Storage) -> Self:
        """Initialize from Storage."""
        return Self(rebind[Self.ptr_type](storage.unsafe_ptr()))

    @inline(.always)
    def __getitem__[T: Intable](self, index: T) -> Self.ptr_type:
        """Get a pointer to the element at index.

        Args:
            index: Element index.

        Returns:
            Pointer to element.
        """
        return self.ptr.unsafe_offset(Int(index))

    @inline(.always)
    @staticmethod
    def len() -> Int:
        """Get array length in bytes.

        Returns:
            Total size in bytes.
        """
        return Self.size * size_of[Self.type]()

    @inline(.always)
    @staticmethod
    def stack_allocation[alignment: Int = align_of[Self.type]()]() -> Self:
        var ptr = unsafe_stack_allocation[
            Self.len(),
            Self.type,
            alignment=alignment,
            address_space=.SHARED,
        ]()
        return Self(ptr)


comptime eval[T: AnyType, //, val: T] = val
"""Helper alias to force evaluation of expressions at compile time."""

comptime SMemPtr[type: AnyType] = MutPointer[
    type, MutUntrackedOrigin, address_space=.SHARED
]
