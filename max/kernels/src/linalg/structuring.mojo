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
from layout._utils import _get_bounds, make_amd_buffer_resource
from layout.layout_tensor import (
    LayoutTensorIter,
    ThreadScope,
    _copy_dram_to_local,
    _copy_local_to_dram,
)


# Tile based AMD Data Movement Delegate
struct ScatterGatherAmd[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
]:
    var buffer: AMDBufferResource

    @always_inline
    fn __init__(out self, tensor: LayoutTensor):
        self.buffer = make_amd_buffer_resource(tensor)

    # DRAM -> Registers (Local)
    @always_inline
    fn copy(
        self,
        dst_reg_tile: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
        src_gmem_tile: LayoutTensor,
        src_tensor: LayoutTensor,
        offset: OptionalReg[UInt] = None,
    ):
        _copy_dram_to_local[
            thread_layout, num_threads, thread_scope, block_dim_count
        ](dst_reg_tile, src_gmem_tile, self.buffer)

    # Registers (Local) -> DRAM
    @always_inline("nodebug")
    fn copy(
        self,
        dst_gmem_tile: LayoutTensor,
        src_reg_tile: LayoutTensor[*_, address_space = AddressSpace.LOCAL, **_],
    ):
        _copy_local_to_dram[
            thread_layout, num_threads, thread_scope, block_dim_count
        ](dst_gmem_tile, src_reg_tile, self.buffer)


# Tile Iterator based AMD Data Movement Delegate
struct IteratorScatterGatherAmd[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
    block_dim_count: Int = 1,
]:
    var buffer: AMDBufferResource

    @always_inline
    fn __init__(out self, tensor: LayoutTensor, tensor_iter: LayoutTensorIter):
        self.buffer = make_amd_buffer_resource(tensor_iter, _get_bounds(tensor))

    # DRAM -> Registers (Local)
    @always_inline
    fn copy(
        self,
        dst_reg_tile: LayoutTensor,
        src_gmem_tile_iter: LayoutTensorIter,
    ):
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

alias SMemTileIterType[
    _dtype: DType,
    layout: Layout,
    alignment: Int = align_of[SIMD[_dtype, simd_width_of[_dtype]()]](),
    circular: Bool = False,
] = LayoutTensorIter[
    _dtype,
    layout,
    MutableAnyOrigin,
    address_space = AddressSpace.SHARED,
    alignment=alignment,
    circular=circular,
]

alias SMemWarpTileType[
    _dtype: DType, layout: Layout, warp_rows: Int, warp_cols: Int
] = SMemTileType[_dtype, layout].TileType[warp_rows, warp_cols]

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


# Helper "function" to evaluate expressions at compile time
alias eval[T: AnyType, //, val: T] = val


struct SMemArray[type: AnyType, size: Int]:
    alias T = UnsafePointer[type, address_space = AddressSpace.SHARED]
    alias storage_size = size * size_of[type]()

    @staticmethod
    @always_inline
    fn build(mut smem_mgr: NVIDIASharedMemoryManager) -> Self.T:
        return rebind[Self.T](smem_mgr.array[type, size]())


struct SMemTile[dtype: DType, layout: Layout, alignment: Int]:
    alias T = SMemTileType[dtype, layout, alignment=alignment]
    alias storage_size = layout.size() * size_of[dtype]()

    @staticmethod
    @always_inline
    fn build(mut smem_mgr: NVIDIASharedMemoryManager) -> Self.T:
        return rebind[Self.T](smem_mgr.tile[dtype, layout]())


struct SMemTileIter[
    dtype: DType, layout: Layout, alignment: Int, num_tiles: Int
]:
    alias T = SMemTileIterType[dtype, layout, alignment=alignment]
    alias storage_size = layout.size() * size_of[dtype]() * num_tiles
    alias tile_storage_size = layout.size() * size_of[dtype]()

    @staticmethod
    @always_inline
    fn build(mut smem_mgr: NVIDIASharedMemoryManager) -> Self.T:
        return rebind[Self.T](smem_mgr.tile_iter[dtype, layout, num_tiles]())


struct NVIDIASharedMemoryManager[
    name: StaticString = "extern_ptr_syml",
    alignment: Int = 128,
    memory_alignment: Int = 8,
]:
    alias TileIter[dtype: DType, layout: Layout, num_tiles: Int] = SMemTileIter[
        dtype, layout, alignment=alignment, num_tiles=num_tiles
    ]
    alias Tile[dtype: DType, layout: Layout] = SMemTile[
        dtype, layout, alignment=alignment
    ]
    alias Array[type: AnyType, num_tiles: Int] = SMemArray[type, num_tiles]

    var base_ptr: UnsafePointer[Int8, address_space = AddressSpace.SHARED]
    var offset: Int

    @always_inline
    fn __init__(out self):
        self.base_ptr = external_memory[
            Int8,
            address_space = AddressSpace.SHARED,
            alignment=memory_alignment,
            name=name,
        ]()
        self.offset = 0

    @always_inline
    fn tile_iter[
        dtype: DType, layout: Layout, num_tiles: Int
    ](mut self) -> SMemTileIterType[dtype, layout, alignment=alignment]:
        var smem_size = eval[layout.size()] * num_tiles
        var result = SMemTileIterType[dtype, layout, alignment=alignment](
            self.base_ptr.offset(self.offset).bitcast[Scalar[dtype]](),
            smem_size,
        )
        self.offset += smem_size * size_of[dtype]()
        return result

    @always_inline
    fn tile[
        dtype: DType, layout: Layout
    ](mut self) -> SMemTileType[dtype, layout, alignment=alignment]:
        var result = SMemTileType[dtype, layout, alignment=alignment](
            self.base_ptr.offset(self.offset).bitcast[Scalar[dtype]](),
        )
        self.offset += eval[layout.size()] * size_of[dtype]()
        return result

    @always_inline
    fn array[
        type: AnyType, size: Int
    ](mut self) -> UnsafePointer[type, address_space = AddressSpace.SHARED]:
        var result = self.base_ptr.offset(self.offset).bitcast[type]()
        self.offset += size * size_of[type]()
        return result
