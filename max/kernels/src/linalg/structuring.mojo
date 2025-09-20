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

from sys import align_of, simd_width_of, size_of
from gpu.memory import AddressSpace
from gpu.intrinsics import AMDBufferResource
from layout import Layout, LayoutTensor
from layout.layout_tensor import (
    ThreadScope,
    _copy_dram_to_local,
    _copy_local_to_dram,
    LayoutTensorIter,
)
from collections import OptionalReg
from layout._utils import _get_bounds, make_amd_buffer_resource


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
