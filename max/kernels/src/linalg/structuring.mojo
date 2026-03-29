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

from std.collections import Optional

from std.gpu.intrinsics import AMDBufferResource
from std.gpu.memory import external_memory
from layout import Layout, LayoutTensor
from layout._utils import _get_bounds, make_amd_buffer_resource
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    _copy_dram_to_local,
    _copy_local_to_dram,
)


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
    def __init__(out self, tensor: LayoutTensor):
        """Initialize with a tensor.

        Args:
            tensor: Layout tensor for AMD buffer resource creation.
        """
        self.buffer = make_amd_buffer_resource(tensor)

    @always_inline
    def copy(
        self,
        dst_reg_tile: LayoutTensor[
            mut=True, address_space=AddressSpace.LOCAL, ...
        ],
        src_gmem_tile: LayoutTensor,
        offset: Optional[UInt] = None,
    ):
        """Copy DRAM to registers.

        Args:
            dst_reg_tile: Destination register tile.
            src_gmem_tile: Source global memory tile.
            offset: Optional copy offset.
        """
        _copy_dram_to_local[
            Self.thread_layout,
            Self.num_threads,
            Self.thread_scope,
            Self.block_dim_count,
        ](dst_reg_tile, src_gmem_tile, self.buffer)

    @always_inline("nodebug")
    def copy(
        self,
        dst_gmem_tile: LayoutTensor[mut=True, ...],
        src_reg_tile: LayoutTensor[address_space=AddressSpace.LOCAL, ...],
    ):
        """Copy registers to DRAM.

        Args:
            dst_gmem_tile: Destination global memory tile.
            src_reg_tile: Source register tile.
        """
        _copy_local_to_dram[
            Self.thread_layout,
            Self.num_threads,
            Self.thread_scope,
            Self.block_dim_count,
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
    def __init__(out self, tensor: LayoutTensor, tensor_iter: LayoutTensorIter):
        """Initialize with tensor and iterator.

        Args:
            tensor: Layout tensor for bounds.
            tensor_iter: Iterator for AMD buffer resource.
        """
        self.buffer = make_amd_buffer_resource(tensor_iter, _get_bounds(tensor))

    @always_inline
    def copy(
        self,
        dst_reg_tile: LayoutTensor[mut=True, ...],
        src_gmem_tile_iter: LayoutTensorIter,
    ):
        """Copy DRAM to registers via iterator.

        Args:
            dst_reg_tile: Destination register tile.
            src_gmem_tile_iter: Source memory iterator.
        """
        _copy_dram_to_local[
            Self.thread_layout,
            Self.num_threads,
            Self.thread_scope,
            Self.block_dim_count,
        ](dst_reg_tile, src_gmem_tile_iter, self.buffer)


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
    comptime alignment: Int

    @always_inline
    @staticmethod
    def ptr() -> (
        UnsafePointer[Int8, MutAnyOrigin, address_space=AddressSpace.SHARED]
    ):
        ...


struct NVIDIASharedMemoryBasePtr[
    name: StaticString = "extern_ptr_syml",
    memory_alignment: Int = 8,
](SharedMemoryBasePtr):
    comptime alignment: Int = 128

    @always_inline
    @staticmethod
    def ptr() -> (
        UnsafePointer[Int8, MutAnyOrigin, address_space=AddressSpace.SHARED]
    ):
        return external_memory[
            Int8,
            address_space=AddressSpace.SHARED,
            alignment=Self.memory_alignment,
            name=Self.name,
        ]()


struct SharedMemoryManager[SMBP: SharedMemoryBasePtr]:
    comptime Tile[dtype: DType, layout: Layout] = SMemTile[
        dtype, layout, alignment=Self.SMBP.alignment
    ]

    comptime TileArray[
        dtype: DType, layout: Layout, num_tiles: Int
    ] = SMemTileArray[dtype, layout, num_tiles, Self.SMBP.alignment]

    comptime Array[type: TrivialRegisterPassable, size: Int] = SMemArray[
        type, size
    ]

    var base_ptr: UnsafePointer[
        Int8, MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var offset: Int

    @always_inline
    def __init__(out self):
        """Initialize the shared memory manager."""
        self.base_ptr = Self.SMBP.ptr()
        self.offset = 0

    @always_inline
    def build[
        dtype: DType,
        layout: Layout,
        //,
        T: type_of(Self.Tile[dtype, layout]),
    ](mut self) -> T:
        """Allocate a single tile.

        Returns:
            Allocated tile.
        """
        var result = T(
            (self.base_ptr + self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += T.storage_size
        return result

    @always_inline
    def build[
        dtype: DType,
        layout: Layout,
        num_tiles: Int,
        //,
        T: type_of(Self.TileArray[dtype, layout, num_tiles]),
    ](mut self) -> T:
        """Allocate a tile array.

        Returns:
            Allocated tile array.
        """
        var result = T(
            (self.base_ptr + self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += T.storage_size
        return result

    @always_inline
    def build[
        type: TrivialRegisterPassable,
        size: Int,
        //,
        T: type_of(Self.Array[type, size]),
    ](mut self) -> T:
        """Allocate a regular array.

        Returns:
            Allocated array.
        """
        var result = (self.base_ptr + self.offset).bitcast[type]()
        self.offset += T.storage_size
        return T(result)


comptime NVIDIASharedMemoryManager = SharedMemoryManager[
    NVIDIASharedMemoryBasePtr[]
]
