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

"""Provides shared memory tile types and a bump allocator for GPU kernels."""

from max.gpu.memory import external_memory
from layout import Layout


# Shared Memory and Register tiles type declarations.
# Canonical definitions live in structured_kernels.smem_types; re-exported
# here for backward compatibility so existing
# `from linalg.structuring import ...` statements continue to work.
from structured_kernels.smem_types import (
    SMemTile,
    RegTile,
    SMemBarrier,
    SMemTileArray,
    SMemArray,
    SMemPtr,
)


trait SharedMemoryBasePtr:
    """Defines a base pointer into GPU shared memory with a fixed alignment.

    Implementations provide a statically-aligned pointer to shared memory
    that a `SharedMemoryManager` allocates tiles and arrays from.

    Parameters:
        alignment: Required byte alignment of the shared memory base pointer.
    """

    comptime alignment: Int

    @inline(.always)
    @staticmethod
    def ptr() -> UnsafePointer[Int8, MutUntrackedOrigin, address_space=.SHARED]:
        ...


struct NVIDIASharedMemoryBasePtr[
    name: StaticString = "extern_ptr_syml",
    memory_alignment: Int = 8,
](SharedMemoryBasePtr):
    """NVIDIA implementation of `SharedMemoryBasePtr` using external memory.

    Exposes a shared memory base pointer backed by NVIDIA's
    `external_memory` intrinsic, parameterized by a symbolic name and
    alignment.

    Parameters:
        name: Symbolic name for the external memory allocation.
        memory_alignment: Byte alignment of the external memory allocation.
    """

    comptime alignment: Int = 128

    @inline(.always)
    @staticmethod
    def ptr() -> UnsafePointer[Int8, MutUntrackedOrigin, address_space=.SHARED]:
        return external_memory[
            Int8,
            address_space=.SHARED,
            alignment=Self.memory_alignment,
            name=Self.name,
        ]()


struct SharedMemoryManager[SMBP: SharedMemoryBasePtr]:
    """Manages bump allocation of tiles and arrays from a shared memory base pointer.

    Allocates `SMemTile`, `SMemTileArray`, and `SMemArray` instances by
    advancing an offset over a shared memory base pointer provided by the
    `SMBP` parameter.

    Parameters:
        SMBP: Shared memory base pointer provider implementing `SharedMemoryBasePtr`.
    """

    comptime Tile[dtype: DType, layout: Layout] = SMemTile[
        dtype, layout, alignment=Self.SMBP.alignment
    ]

    comptime TileArray[
        dtype: DType, layout: Layout, num_tiles: Int
    ] = SMemTileArray[dtype, layout, num_tiles, Self.SMBP.alignment]

    comptime Array[type: TrivialRegisterPassable, size: Int] = SMemArray[
        type, size
    ]

    var base_ptr: UnsafePointer[Int8, MutUntrackedOrigin, address_space=.SHARED]
    var offset: Int

    @inline(.always)
    def __init__(out self):
        """Initialize the shared memory manager."""
        self.base_ptr = Self.SMBP.ptr()
        self.offset = 0

    @inline(.always)
    def build[
        dtype: DType,
        layout: Layout,
        //,
        T: type_of(Self.Tile[dtype, layout]),
    ](mut self) -> T:
        """Allocate a single tile.

        Parameters:
            dtype: Element type of the allocated tile.
            layout: Memory layout of the allocated tile.
            T: The allocated `SMemTile` type (inferred).

        Returns:
            Allocated tile.
        """
        var result = T(
            (self.base_ptr + self.offset)
            .bitcast[Scalar[dtype]]()
            .as_unsafe_any_origin(),
        )
        self.offset += T.storage_size
        return result

    @inline(.always)
    def build[
        dtype: DType,
        layout: Layout,
        num_tiles: Int,
        //,
        T: type_of(Self.TileArray[dtype, layout, num_tiles]),
    ](mut self) -> T:
        """Allocate a tile array.

        Parameters:
            dtype: Element type of each tile in the array.
            layout: Memory layout of each tile in the array.
            num_tiles: Number of tiles in the array.
            T: The allocated `SMemTileArray` type (inferred).

        Returns:
            Allocated tile array.
        """
        var result = T(
            (self.base_ptr + self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += T.storage_size
        return result

    @inline(.always)
    def build[
        type: TrivialRegisterPassable,
        size: Int,
        //,
        T: type_of(Self.Array[type, size]),
    ](mut self) -> T:
        """Allocate a regular array.

        Parameters:
            type: Element type stored in the array.
            size: Number of elements in the array.
            T: The allocated `SMemArray` type (inferred).

        Returns:
            Allocated array.
        """
        var result = (self.base_ptr + self.offset).bitcast[type]()
        self.offset += T.storage_size
        return T(result)


comptime NVIDIASharedMemoryManager = SharedMemoryManager[
    NVIDIASharedMemoryBasePtr[]
]
