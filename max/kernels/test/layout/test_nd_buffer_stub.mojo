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

import std.math

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from layout import Layout, LayoutTensor
from layout._fillers import arange
from layout._ndbuffer_stub import (
    ElementLayout,
    TileMask,
    _copy_layout_tensor_to_nd_buffer,
    _copy_nd_buffer_to_layout_tensor,
    _copy_nd_buffer_to_layout_tensor_masked,
    _distribute_mask,
    _tile_mask,
    _vectorize_mask,
    copy_from_nd_buffer,
    copy_from_nd_buffer_masked,
    copy_to_nd_buffer,
    copy_to_nd_buffer_masked,
    distribute,
    from_ndbuffer_row_major,
    vectorize,
)

from std.memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from std.utils import Index, IndexList, StaticTuple


fn linspace_fill[
    dtype: DType, rank: Int, shape: DimList
](mut buff: NDBuffer[mut=True, dtype, rank, _, shape]):
    for i in range(buff.size()):
        buff.data[i] = Scalar[dtype](i)


fn print_buff[
    dtype: DType, rank: Int, shape: DimList
](buff: NDBuffer[dtype, rank, _, shape]):
    comptime assert rank == 2, "rank-2 buffer is expected"
    for m in range(buff.dim(0)):
        for n in range(buff.dim(1)):
            print(buff[m, n], end=" ")
        print("")


fn print_tile_mask[*tile_sizes: Int](mask: TileMask):
    for i in range(tile_sizes[0]):
        for j in range(tile_sizes[1]):
            var mas_val = mask.access_mask((i, j))
            print(and_all(mas_val), end=" ")
        print("")


fn print_tile_mask_with_size[*tile_sizes: Int](mask: TileMask):
    for i in range(tile_sizes[0]):
        for j in range(tile_sizes[1]):
            var mas_val = mask.access_mask((i, j))
            var size = mask.access_size((i, j), mas_val)
            print(and_all(mas_val), ":", size[0], "x", size[1], end=" ")
        print("")


fn print_element[
    dtype: DType,
    rank: Int,
    element_shape: IndexList[rank],
](
    element_ptr: LegacyUnsafePointer[mut=False, Scalar[dtype]],
    element_layout: ElementLayout[rank, element_shape],
):
    var simd_element = SIMD[dtype, element_shape[0] * element_shape[1]](0)

    comptime for i in range(element_shape[0]):
        comptime for j in range(element_shape[1]):
            simd_element[i * element_shape[1] + j] = element_ptr[
                i * element_layout.stride[0] + j * element_layout.stride[1]
            ]

    print(simd_element, end=" ")


fn print_vectorized_buff[
    dtype: DType,
    shape: DimList,
    element_shape: IndexList[2],
](
    buff: NDBuffer[dtype, 2, _, shape],
    element_layout: ElementLayout[2, element_shape],
):
    for m in range(buff.dim(0)):
        for n in range(buff.dim(1)):
            print_element(buff._offset(IndexList[2](m, n)), element_layout)
        print("")


fn and_all[rank: Int](mask: StaticTuple[Bool, rank]) -> Bool:
    var res = True

    comptime for i in range(rank):
        res &= mask[i]

    return res


# CHECK-LABEL: test_copy_from_nd_buffer_scalars
fn test_copy_from_nd_buffer_scalars():
    print("== test_copy_from_nd_buffer_scalars")

    var buff_stack = InlineArray[Float32, 64](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, _, DimList(8, 8)](
        buff_stack.unsafe_ptr()
    )
    linspace_fill(buff)

    var tensor_stack = InlineArray[Float32, 64](uninitialized=True)
    var layout_tensor = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_stack
    ).fill(0)

    comptime threads_layout = Layout.row_major(4, 4)
    for th_id in range(16):
        var thread_local_layout_tensor = layout_tensor.distribute[
            threads_layout
        ](UInt(th_id))
        copy_from_nd_buffer[thread_layout=threads_layout](
            thread_local_layout_tensor, buff, th_id
        )
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    print(layout_tensor)


fn main():
    test_copy_from_nd_buffer_scalars()
