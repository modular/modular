# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

import math

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from layout import IntTuple, Layout, LayoutTensor
from layout.fillers import arange
from layout.layout import LayoutList
from layout.nd_buffer_stub import (
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
from memory import UnsafePointer
from testing import assert_equal

from utils import Index, IndexList, StaticTuple


fn linspace_fill[
    dtype: DType, rank: Int, shape: DimList
](mut buff: NDBuffer[dtype, rank, shape]):
    for i in range(buff.size()):
        buff.data[i] = i


fn print_buff[
    dtype: DType, rank: Int, shape: DimList
](buff: NDBuffer[dtype, rank, shape]):
    constrained[rank == 2, "rank-2 buffer is expected"]()
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
    element_ptr: UnsafePointer[Scalar[dtype]],
    element_layout: ElementLayout[rank, element_shape],
):
    var simd_element = SIMD[dtype, element_shape[0] * element_shape[1]](0)

    @parameter
    for i in range(element_shape[0]):

        @parameter
        for j in range(element_shape[1]):
            simd_element[i * element_shape[1] + j] = element_ptr[
                i * element_layout.stride[0] + j * element_layout.stride[1]
            ]

    print(simd_element, end=" ")


fn print_vectorized_buff[
    dtype: DType,
    rank: Int,
    shape: DimList,
    element_shape: IndexList[rank],
](
    buff: NDBuffer[dtype, rank, shape],
    element_layout: ElementLayout[rank, element_shape],
):
    for m in range(buff.dim(0)):
        for n in range(buff.dim(1)):
            print_element(buff._offset(VariadicList[Int](m, n)), element_layout)
        print("")


fn and_all[rank: Int](mask: StaticTuple[Bool, rank]) -> Bool:
    var res = True

    @parameter
    for i in range(rank):
        res &= mask[i]

    return res


# CHECK-LABEL: test_copy_from_nd_buffer_scalars
fn test_copy_from_nd_buffer_scalars():
    print("== test_copy_from_nd_buffer_scalars")

    var buff_stack = InlineArray[Float32, 64](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](
        buff_stack.unsafe_ptr()
    )
    linspace_fill(buff)

    var tensor_stack = InlineArray[Float32, 64](uninitialized=True)
    var layout_tensor = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_stack
    ).fill(0)

    alias threads_layout = Layout.row_major(4, 4)
    for th_id in range(16):
        var thread_local_layout_tensor = layout_tensor.distribute[
            threads_layout
        ](th_id)
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


# CHECK-LABEL: test_copy_to_nd_buffer_scalars
fn test_copy_to_nd_buffer_scalars():
    print("== test_copy_to_nd_buffer_scalars")

    var tensor_stack = InlineArray[Float32, 64](uninitialized=True)
    var layout_tensor = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_stack
    )
    arange(layout_tensor)

    var buff_stack = InlineArray[Float32, 64](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](
        buff_stack.unsafe_ptr()
    )
    buff.zero()

    alias threads_layout = Layout.row_major(4, 4)
    for th_id in range(16):
        var thread_local_layout_tensor = layout_tensor.distribute[
            threads_layout
        ](th_id)
        copy_to_nd_buffer[thread_layout=threads_layout](
            buff, thread_local_layout_tensor, th_id
        )
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    print_buff(buff)


# CHECK-LABEL: test_copy_from_nd_buffer_vectors
fn test_copy_from_nd_buffer_vectors():
    print("== test_copy_from_nd_buffer_vectors")

    var buff_storage = UnsafePointer[Float32].alloc(16 * 16)
    var buff = NDBuffer[DType.float32, 2, DimList(16, 16)](buff_storage)
    linspace_fill(buff)

    var tensor_stack = InlineArray[Float32, 16 * 16](uninitialized=True)
    var layout_tensor = LayoutTensor[DType.float32, Layout.row_major(16, 16)](
        tensor_stack
    ).fill(0)

    alias threads_layout = Layout.row_major(4, 4)
    for th_id in range(16):
        var thread_local_layout_tensor = layout_tensor.vectorize[
            1, 4
        ]().distribute[threads_layout](th_id)
        copy_from_nd_buffer[thread_layout=threads_layout](
            thread_local_layout_tensor, buff, th_id
        )
    # [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0] [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0] [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0] [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0] [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    # [64.0, 65.0, 66.0, 67.0] [68.0, 69.0, 70.0, 71.0] [72.0, 73.0, 74.0, 75.0] [76.0, 77.0, 78.0, 79.0]
    # [80.0, 81.0, 82.0, 83.0] [84.0, 85.0, 86.0, 87.0] [88.0, 89.0, 90.0, 91.0] [92.0, 93.0, 94.0, 95.0]
    # [96.0, 97.0, 98.0, 99.0] [100.0, 101.0, 102.0, 103.0] [104.0, 105.0, 106.0, 107.0] [108.0, 109.0, 110.0, 111.0]
    # [112.0, 113.0, 114.0, 115.0] [116.0, 117.0, 118.0, 119.0] [120.0, 121.0, 122.0, 123.0] [124.0, 125.0, 126.0, 127.0]
    # [128.0, 129.0, 130.0, 131.0] [132.0, 133.0, 134.0, 135.0] [136.0, 137.0, 138.0, 139.0] [140.0, 141.0, 142.0, 143.0]
    # [144.0, 145.0, 146.0, 147.0] [148.0, 149.0, 150.0, 151.0] [152.0, 153.0, 154.0, 155.0] [156.0, 157.0, 158.0, 159.0]
    # [160.0, 161.0, 162.0, 163.0] [164.0, 165.0, 166.0, 167.0] [168.0, 169.0, 170.0, 171.0] [172.0, 173.0, 174.0, 175.0]
    # [176.0, 177.0, 178.0, 179.0] [180.0, 181.0, 182.0, 183.0] [184.0, 185.0, 186.0, 187.0] [188.0, 189.0, 190.0, 191.0]
    # [192.0, 193.0, 194.0, 195.0] [196.0, 197.0, 198.0, 199.0] [200.0, 201.0, 202.0, 203.0] [204.0, 205.0, 206.0, 207.0]
    # [208.0, 209.0, 210.0, 211.0] [212.0, 213.0, 214.0, 215.0] [216.0, 217.0, 218.0, 219.0] [220.0, 221.0, 222.0, 223.0]
    # [224.0, 225.0, 226.0, 227.0] [228.0, 229.0, 230.0, 231.0] [232.0, 233.0, 234.0, 235.0] [236.0, 237.0, 238.0, 239.0]
    # [240.0, 241.0, 242.0, 243.0] [244.0, 245.0, 246.0, 247.0] [248.0, 249.0, 250.0, 251.0] [252.0, 253.0, 254.0, 255.0]
    print(layout_tensor.vectorize[1, 4]())

    _ = layout_tensor.fill(0)

    for th_id in range(16):
        var thread_local_layout_tensor = layout_tensor.vectorize[
            4, 4
        ]().distribute[threads_layout](th_id)
        copy_from_nd_buffer[thread_layout=threads_layout](
            thread_local_layout_tensor, buff, th_id
        )

    # CHECK: [0.0, 1.0, 2.0, 3.0, 16.0, 17.0, 18.0, 19.0, 32.0, 33.0, 34.0, 35.0, 48.0, 49.0, 50.0, 51.0] [4.0, 5.0, 6.0, 7.0, 20.0, 21.0, 22.0, 23.0, 36.0, 37.0, 38.0, 39.0, 52.0, 53.0, 54.0, 55.0] [8.0, 9.0, 10.0, 11.0, 24.0, 25.0, 26.0, 27.0, 40.0, 41.0, 42.0, 43.0, 56.0, 57.0, 58.0, 59.0] [12.0, 13.0, 14.0, 15.0, 28.0, 29.0, 30.0, 31.0, 44.0, 45.0, 46.0, 47.0, 60.0, 61.0, 62.0, 63.0]
    # CHECK: [64.0, 65.0, 66.0, 67.0, 80.0, 81.0, 82.0, 83.0, 96.0, 97.0, 98.0, 99.0, 112.0, 113.0, 114.0, 115.0] [68.0, 69.0, 70.0, 71.0, 84.0, 85.0, 86.0, 87.0, 100.0, 101.0, 102.0, 103.0, 116.0, 117.0, 118.0, 119.0] [72.0, 73.0, 74.0, 75.0, 88.0, 89.0, 90.0, 91.0, 104.0, 105.0, 106.0, 107.0, 120.0, 121.0, 122.0, 123.0] [76.0, 77.0, 78.0, 79.0, 92.0, 93.0, 94.0, 95.0, 108.0, 109.0, 110.0, 111.0, 124.0, 125.0, 126.0, 127.0]
    # CHECK: [128.0, 129.0, 130.0, 131.0, 144.0, 145.0, 146.0, 147.0, 160.0, 161.0, 162.0, 163.0, 176.0, 177.0, 178.0, 179.0] [132.0, 133.0, 134.0, 135.0, 148.0, 149.0, 150.0, 151.0, 164.0, 165.0, 166.0, 167.0, 180.0, 181.0, 182.0, 183.0] [136.0, 137.0, 138.0, 139.0, 152.0, 153.0, 154.0, 155.0, 168.0, 169.0, 170.0, 171.0, 184.0, 185.0, 186.0, 187.0] [140.0, 141.0, 142.0, 143.0, 156.0, 157.0, 158.0, 159.0, 172.0, 173.0, 174.0, 175.0, 188.0, 189.0, 190.0, 191.0]
    # CHECK: [192.0, 193.0, 194.0, 195.0, 208.0, 209.0, 210.0, 211.0, 224.0, 225.0, 226.0, 227.0, 240.0, 241.0, 242.0, 243.0] [196.0, 197.0, 198.0, 199.0, 212.0, 213.0, 214.0, 215.0, 228.0, 229.0, 230.0, 231.0, 244.0, 245.0, 246.0, 247.0] [200.0, 201.0, 202.0, 203.0, 216.0, 217.0, 218.0, 219.0, 232.0, 233.0, 234.0, 235.0, 248.0, 249.0, 250.0, 251.0] [204.0, 205.0, 206.0, 207.0, 220.0, 221.0, 222.0, 223.0, 236.0, 237.0, 238.0, 239.0, 252.0, 253.0, 254.0, 255.0]
    print(layout_tensor.vectorize[4, 4]())

    buff_storage.free()


# CHECK-LABEL: test_copy_to_nd_buffer_vectors
fn test_copy_to_nd_buffer_vectors():
    print("== test_copy_to_nd_buffer_vectors")

    var tensor_stack = InlineArray[Float32, 16 * 16](uninitialized=True)
    var layout_tensor = LayoutTensor[DType.float32, Layout.row_major(16, 16)](
        tensor_stack
    )
    arange(layout_tensor)

    var buff_storage = UnsafePointer[Float32].alloc(16 * 16)
    var buff = NDBuffer[DType.float32, 2, DimList(16, 16)](buff_storage)
    buff.zero()

    alias threads_layout = Layout.row_major(4, 4)
    for th_id in range(threads_layout.size()):
        var thread_local_layout_tensor = layout_tensor.vectorize[
            1, 4
        ]().distribute[threads_layout](th_id)
        copy_to_nd_buffer[thread_layout=threads_layout](
            buff, thread_local_layout_tensor, th_id
        )

    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    print_buff(buff)
    buff.zero()

    for th_id in range(threads_layout.size()):
        var thread_local_layout_tensor = layout_tensor.vectorize[
            4, 4
        ]().distribute[threads_layout](th_id)
        copy_to_nd_buffer[thread_layout=threads_layout](
            buff, thread_local_layout_tensor, th_id
        )
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    # CHECK: 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0
    # CHECK: 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0
    # CHECK: 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0
    # CHECK: 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0 120.0 121.0 122.0 123.0 124.0 125.0 126.0 127.0
    # CHECK: 128.0 129.0 130.0 131.0 132.0 133.0 134.0 135.0 136.0 137.0 138.0 139.0 140.0 141.0 142.0 143.0
    # CHECK: 144.0 145.0 146.0 147.0 148.0 149.0 150.0 151.0 152.0 153.0 154.0 155.0 156.0 157.0 158.0 159.0
    # CHECK: 160.0 161.0 162.0 163.0 164.0 165.0 166.0 167.0 168.0 169.0 170.0 171.0 172.0 173.0 174.0 175.0
    # CHECK: 176.0 177.0 178.0 179.0 180.0 181.0 182.0 183.0 184.0 185.0 186.0 187.0 188.0 189.0 190.0 191.0
    # CHECK: 192.0 193.0 194.0 195.0 196.0 197.0 198.0 199.0 200.0 201.0 202.0 203.0 204.0 205.0 206.0 207.0
    # CHECK: 208.0 209.0 210.0 211.0 212.0 213.0 214.0 215.0 216.0 217.0 218.0 219.0 220.0 221.0 222.0 223.0
    # CHECK: 224.0 225.0 226.0 227.0 228.0 229.0 230.0 231.0 232.0 233.0 234.0 235.0 236.0 237.0 238.0 239.0
    # CHECK: 240.0 241.0 242.0 243.0 244.0 245.0 246.0 247.0 248.0 249.0 250.0 251.0 252.0 253.0 254.0 255.0
    print_buff(buff)

    buff_storage.free()


# CHECK-LABEL: test_distribute
fn test_distribute():
    print("== test_distribute")
    var buff_storage = InlineArray[Float32, 8 * 4](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 4)](
        buff_storage.unsafe_ptr()
    )
    linspace_fill(buff)

    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0 2.0
    # CHECK: 8.0 10.0
    # CHECK: 16.0 18.0
    # CHECK: 24.0 26.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 1.0 3.0
    # CHECK: 9.0 11.0
    # CHECK: 17.0 19.0
    # CHECK: 25.0 27.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 4.0 6.0
    # CHECK: 12.0 14.0
    # CHECK: 20.0 22.0
    # CHECK: 28.0 30.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0 7.0
    # CHECK: 13.0 15.0
    # CHECK: 21.0 23.0
    # CHECK: 29.0 31.0

    for th_i in range(4):
        print("----fragments-data[", th_i, "]----")
        var buff_th_local = distribute[thread_layout = Layout.row_major(2, 2)](
            buff, th_i
        )
        print_buff(buff_th_local)


# CHECK-LABEL: test_tile_and_distribute
fn test_tile_and_distribute():
    print("== test_tile_and_distribute")
    var buff_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](
        buff_storage.unsafe_ptr()
    )
    linspace_fill(buff)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 16.0 17.0 18.0 19.0
    # CHECK: 24.0 25.0 26.0 27.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0 2.0
    # CHECK: 16.0 18.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 1.0 3.0
    # CHECK: 17.0 19.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 8.0 10.0
    # CHECK: 24.0 26.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 9.0 11.0
    # CHECK: 25.0 27.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 12.0 13.0 14.0 15.0
    # CHECK: 20.0 21.0 22.0 23.0
    # CHECK: 28.0 29.0 30.0 31.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 4.0 6.0
    # CHECK: 20.0 22.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 5.0 7.0
    # CHECK: 21.0 23.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 12.0 14.0
    # CHECK: 28.0 30.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 13.0 15.0
    # CHECK: 29.0 31.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 32.0 33.0 34.0 35.0
    # CHECK: 40.0 41.0 42.0 43.0
    # CHECK: 48.0 49.0 50.0 51.0
    # CHECK: 56.0 57.0 58.0 59.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 32.0 34.0
    # CHECK: 48.0 50.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 33.0 35.0
    # CHECK: 49.0 51.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 40.0 42.0
    # CHECK: 56.0 58.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 41.0 43.0
    # CHECK: 57.0 59.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 36.0 37.0 38.0 39.0
    # CHECK: 44.0 45.0 46.0 47.0
    # CHECK: 52.0 53.0 54.0 55.0
    # CHECK: 60.0 61.0 62.0 63.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 36.0 38.0
    # CHECK: 52.0 54.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 37.0 39.0
    # CHECK: 53.0 55.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 44.0 46.0
    # CHECK: 60.0 62.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 45.0 47.0
    # CHECK: 61.0 63.0
    for tile_i in range(2):
        for tile_j in range(2):
            var tile_4x4 = buff.tile[4, 4](Index(tile_i, tile_j))
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            print_buff(tile_4x4)
            for th_i in range(4):
                var fragment_2x2 = distribute[
                    thread_layout = Layout.row_major(2, 2)
                ](
                    tile_4x4,
                    th_i,
                )
                print("----fragments-data[", th_i, "]----")
                print_buff(fragment_2x2)


# CHECK-LABEL: test_1d_2d_vectorize
fn test_1d_2d_vectorize():
    print("== test_1d_2d_vectorize")
    var buff_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](
        buff_storage.unsafe_ptr()
    )
    linspace_fill(buff)

    var buff_v_1_and_element_layout = vectorize[1, 4](buff)
    # CHECK: (1, 4):(8, 1)
    print(buff_v_1_and_element_layout[1])
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print_vectorized_buff(
        buff_v_1_and_element_layout[0], buff_v_1_and_element_layout[1]
    )

    var buff_v_4_4_and_element_layout = vectorize[4, 4](buff)
    # CHECK: (4, 4):(8, 1)
    print(buff_v_4_4_and_element_layout[1])
    # CHECK: [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0] [4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0, 28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 48.0, 49.0, 50.0, 51.0, 56.0, 57.0, 58.0, 59.0] [36.0, 37.0, 38.0, 39.0, 44.0, 45.0, 46.0, 47.0, 52.0, 53.0, 54.0, 55.0, 60.0, 61.0, 62.0, 63.0]
    print_vectorized_buff(
        buff_v_4_4_and_element_layout[0], buff_v_4_4_and_element_layout[1]
    )


# CHECK-LABEL: test_vectorize_and_distribute
fn test_vectorize_and_distribute():
    print("== test_vectorize_and_distribute")
    var buff_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](
        buff_storage.unsafe_ptr()
    )
    linspace_fill(buff)

    var buff_v_1_and_element_layout = vectorize[1, 4](buff)

    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0]
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: [36.0, 37.0, 38.0, 39.0]
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0]
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: [44.0, 45.0, 46.0, 47.0]
    # CHECK: ----fragments-data[ 4 ]----
    # CHECK: [16.0, 17.0, 18.0, 19.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0]
    # CHECK: ----fragments-data[ 5 ]----
    # CHECK: [20.0, 21.0, 22.0, 23.0]
    # CHECK: [52.0, 53.0, 54.0, 55.0]
    # CHECK: ----fragments-data[ 6 ]----
    # CHECK: [24.0, 25.0, 26.0, 27.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0]
    # CHECK: ----fragments-data[ 7 ]----
    # CHECK: [28.0, 29.0, 30.0, 31.0]
    # CHECK: [60.0, 61.0, 62.0, 63.0]
    for th_i in range(8):
        var buff_thread_local = distribute[
            thread_layout = Layout.row_major(4, 2)
        ](buff_v_1_and_element_layout[0], th_i)
        print("----fragments-data[", th_i, "]----")
        print_vectorized_buff(buff_thread_local, buff_v_1_and_element_layout[1])


# CHECK-LABEL: test_copy_nd_buffer_to_layout_tensor
fn test_copy_nd_buffer_to_layout_tensor():
    print("== test_copy_nd_buffer_to_layout_tensor")
    var buff_storage = UnsafePointer[Float32].alloc(8 * 8)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](buff_storage)
    # FIXME: This doesn't if _copy_nd_buffer_to_layout_tensor is inlined!
    # var buff = NDBuffer[DType.float32, 2, DimList(8, 8)].stack_allocation()

    linspace_fill(buff)

    var buff_v_1_1_and_element_layout = vectorize[1, 1](buff)

    var tensor_1_1_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var tensor_1_1 = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_1_1_storage
    ).fill(0)

    _copy_nd_buffer_to_layout_tensor(
        tensor_1_1,
        buff_v_1_1_and_element_layout[0],
        buff_v_1_1_and_element_layout[1],
    )
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    print(tensor_1_1)

    var buff_v_1_4_and_element_layout = vectorize[1, 4](buff)

    var tensor_4_1_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var tensor_4_1 = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_4_1_storage
    ).vectorize[1, 4]().fill(0)

    _copy_nd_buffer_to_layout_tensor(
        tensor_4_1,
        buff_v_1_4_and_element_layout[0],
        buff_v_1_4_and_element_layout[1],
    )

    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print(tensor_4_1)

    var buff_v_4_4_and_element_layout = vectorize[4, 4](buff)
    var tensor_4_4_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var tensor_4_4 = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_4_4_storage
    ).vectorize[4, 4]().fill(0)

    _copy_nd_buffer_to_layout_tensor(
        tensor_4_4,
        buff_v_4_4_and_element_layout[0],
        buff_v_4_4_and_element_layout[1],
    )

    # CHECK: [0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0] [4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0, 28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0, 40.0, 41.0, 42.0, 43.0, 48.0, 49.0, 50.0, 51.0, 56.0, 57.0, 58.0, 59.0] [36.0, 37.0, 38.0, 39.0, 44.0, 45.0, 46.0, 47.0, 52.0, 53.0, 54.0, 55.0, 60.0, 61.0, 62.0, 63.0]
    print(tensor_4_4)

    buff_storage.free()


# CHECK-LABEL: test_copy_layout_tensor_to_buffer
fn test_copy_layout_tensor_to_buffer():
    print("== test_copy_layout_tensor_to_buffer")
    var tensor_stack = InlineArray[Float32, 8 * 8](uninitialized=True)
    var tensor = LayoutTensor[DType.float32, Layout.row_major(8, 8)](
        tensor_stack
    )
    arange(tensor)

    var buff_storage = InlineArray[Float32, 8 * 8](uninitialized=True)
    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)](
        buff_storage.unsafe_ptr()
    )
    buff.zero()

    var buff_v_1_1_and_element_layout = vectorize[1, 1](buff)
    _copy_layout_tensor_to_nd_buffer(
        buff_v_1_1_and_element_layout[0],
        buff_v_1_1_and_element_layout[1],
        tensor.vectorize[1, 1](),
    )
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0
    print_vectorized_buff(
        buff_v_1_1_and_element_layout[0], buff_v_1_1_and_element_layout[1]
    )

    buff.zero()
    var buff_v_1_4_and_element_layout = vectorize[1, 4](buff)
    _copy_layout_tensor_to_nd_buffer(
        buff_v_1_4_and_element_layout[0],
        buff_v_1_4_and_element_layout[1],
        tensor.vectorize[1, 4](),
    )
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    print_vectorized_buff(
        buff_v_1_4_and_element_layout[0], buff_v_1_4_and_element_layout[1]
    )

    buff.zero()
    var buff_v_4_4_and_element_layout = vectorize[4, 4](buff)
    _copy_layout_tensor_to_nd_buffer(
        buff_v_4_4_and_element_layout[0],
        buff_v_4_4_and_element_layout[1],
        tensor.vectorize[4, 4](),
    )
    print_vectorized_buff(
        buff_v_4_4_and_element_layout[0], buff_v_4_4_and_element_layout[1]
    )


# CHECK-LABEL: test_tile_mask
fn test_tile_mask():
    print("test_tile_mask")
    # CHECK: ---tile[ 0 0 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: ---tile[ 0 1 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: ---tile[ 0 2 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: ---tile[ 0 3 ]---
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: ---tile[ 1 0 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: ---tile[ 1 1 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: ---tile[ 1 2 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: ---tile[ 1 3 ]---
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: ---tile[ 2 0 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: False False False False
    # CHECK: ---tile[ 2 1 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: False False False False
    # CHECK: ---tile[ 2 2 ]---
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: True True True True
    # CHECK: False False False False
    # CHECK: ---tile[ 2 3 ]---
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: True True True False
    # CHECK: False False False False
    for tile_i in range(math.ceildiv(11, 4)):
        for tile_j in range(math.ceildiv(15, 4)):
            print("---tile[", tile_i, tile_j, "]---")
            var tile_mas = _tile_mask[4, 4](
                IndexList[2](11, 15), IndexList[2](tile_i, tile_j)
            )
            print_tile_mask[4, 4](tile_mas)


# CHECK-LABEL: test_vectorize_mask
fn test_vectorize_mask():
    print("test_vectorize_mask")
    # CHECK: ---tile[ 0 0 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: ---tile[ 0 1 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: ---tile[ 0 2 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: ---tile[ 0 3 ]---
    # CHECK: True x (2, 2) False x (2, 1)
    # CHECK: True x (2, 2) False x (2, 1)
    # CHECK: ---tile[ 1 0 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: ---tile[ 1 1 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: ---tile[ 1 2 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: ---tile[ 1 3 ]---
    # CHECK: True x (2, 2) False x (2, 1)
    # CHECK: True x (2, 2) False x (2, 1)
    # CHECK: ---tile[ 2 0 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: False x (1, 2) False x (1, 2)
    # CHECK: ---tile[ 2 1 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: False x (1, 2) False x (1, 2)
    # CHECK: ---tile[ 2 2 ]---
    # CHECK: True x (2, 2) True x (2, 2)
    # CHECK: False x (1, 2) False x (1, 2)
    # CHECK: ---tile[ 2 3 ]---
    # CHECK: True x (2, 2) False x (2, 1)
    # CHECK: False x (1, 2) False x (1, 1)
    for tile_i in range(math.ceildiv(11, 4)):
        for tile_j in range(math.ceildiv(15, 4)):
            print("---tile[", tile_i, tile_j, "]---")
            var tile_mas = _tile_mask[4, 4](
                IndexList[2](11, 15), IndexList[2](tile_i, tile_j)
            )

            var vec_mas = _vectorize_mask[sizes= (2, 2)](tile_mas)
            for i in range(2):
                for j in range(2):
                    var mask = vec_mas.access_mask((i, j))
                    print(
                        and_all(mask),
                        "x",
                        vec_mas.access_size((i, j), mask),
                        end=" ",
                    )
                print("")


# CHECK-LABEL: test_distribute_mask
fn test_distribute_mask():
    print("test_distribute_mask")
    var tile_mask = TileMask[2]((3, 5))
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True True True
    # CHECK: True True True
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: True True False
    # CHECK: True True False
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True True True
    # CHECK: False False False
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: True True False
    # CHECK: False False False
    for th_id in range(4):
        var dist_mask = _distribute_mask[
            thread_layout = Layout.row_major(2, 2)
        ](tile_mask, th_id)
        print("---thread-[", th_id, "]-mask---")
        print_tile_mask[2, 3](dist_mask)


# CHECK-LABEL: test_composed_tile_vectorize_distribute
fn test_composed_tile_vectorize_distribute():
    print("test_composed_tile_vectorize_distribute")
    alias M = 19
    alias N = 21
    alias BM = 16
    alias BN = 16
    alias TM = 4
    alias TN = 4
    # CHECK: ---tile[ 0 0 ]---
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: vectorized-access:
    # CHECK: True  :  4 x 4 True  :  4 x 4 True  :  4 x 4 True  :  4 x 4
    # CHECK: True  :  4 x 4 True  :  4 x 4 True  :  4 x 4 True  :  4 x 4
    # CHECK: True  :  4 x 4 True  :  4 x 4 True  :  4 x 4 True  :  4 x 4
    # CHECK: True  :  4 x 4 True  :  4 x 4 True  :  4 x 4 True  :  4 x 4
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True True
    # CHECK: True True
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: True True
    # CHECK: True True
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True True
    # CHECK: True True
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: True True
    # CHECK: True True
    # CHECK: ---tile[ 0 1 ]---
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: True  :  4 x 4 False  :  4 x 1 False  :  4 x 0 False  :  4 x 0
    # CHECK: True  :  4 x 4 False  :  4 x 1 False  :  4 x 0 False  :  4 x 0
    # CHECK: True  :  4 x 4 False  :  4 x 1 False  :  4 x 0 False  :  4 x 0
    # CHECK: True  :  4 x 4 False  :  4 x 1 False  :  4 x 0 False  :  4 x 0
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True False
    # CHECK: True False
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True False
    # CHECK: True False
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---tile[ 1 0 ]---
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: True True True True True True True True True True True True True True True True
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: False  :  3 x 4 False  :  3 x 4 False  :  3 x 4 False  :  3 x 4
    # CHECK: False  :  0 x 4 False  :  0 x 4 False  :  0 x 4 False  :  0 x 4
    # CHECK: False  :  0 x 4 False  :  0 x 4 False  :  0 x 4 False  :  0 x 4
    # CHECK: False  :  0 x 4 False  :  0 x 4 False  :  0 x 4 False  :  0 x 4
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---tile[ 1 1 ]---
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: True True True True True False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: False False False False False False False False False False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: False  :  3 x 4 False  :  3 x 1 False  :  3 x 0 False  :  3 x 0
    # CHECK: False  :  0 x 4 False  :  0 x 1 False  :  0 x 0 False  :  0 x 0
    # CHECK: False  :  0 x 4 False  :  0 x 1 False  :  0 x 0 False  :  0 x 0
    # CHECK: False  :  0 x 4 False  :  0 x 1 False  :  0 x 0 False  :  0 x 0
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: False False
    # CHECK: False False
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: False False
    # CHECK: False False
    for tile_m in range(math.ceildiv(M, BM)):
        for tile_n in range(math.ceildiv(N, BN)):
            print("---tile[", tile_m, tile_n, "]---")
            var tile_mask = _tile_mask[BM, BN](
                IndexList[2](M, N), IndexList[2](tile_m, tile_n)
            )
            print_tile_mask[BM, BN](tile_mask)
            print("vectorized-access:")
            var vectoize_mask = _vectorize_mask[sizes= (TM, TN)](tile_mask)
            for i in range(BM // TM):
                for j in range(BN // TN):
                    var mask = vectoize_mask.access_mask((i, j))
                    print(
                        and_all(mask),
                        " : ",
                        vectoize_mask.access_size((i, j), mask)[0],
                        "x",
                        vectoize_mask.access_size((i, j), mask)[1],
                        end=" ",
                    )
                print("")
            print("thread-local-access:")
            for th_id in range(4):
                print("---thread-[", th_id, "]-mask---")
                var dist_mask = _distribute_mask[
                    thread_layout = Layout.row_major(2, 2)
                ](vectoize_mask, th_id)
                print_tile_mask[BM // TM, BN // TN](dist_mask)


# CHECK-LABEL: test_composed_tile_vectorize_distribute_small
fn test_composed_tile_vectorize_distribute_small():
    print("test_composed_tile_vectorize_distribute_small")
    alias M = 15
    alias N = 17
    alias BM = 8
    alias BN = 8
    alias TM = 4
    alias TN = 4
    # CHECK: ---tile[ 0 0 ]---
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: vectorized-access:
    # CHECK: True  :  4 x 4 True  :  4 x 4
    # CHECK: True  :  4 x 4 True  :  4 x 4
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---tile[ 0 1 ]---
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: vectorized-access:
    # CHECK: True  :  4 x 4 True  :  4 x 4
    # CHECK: True  :  4 x 4 True  :  4 x 4
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---tile[ 0 2 ]---
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: False  :  4 x 1 False  :  4 x 0
    # CHECK: False  :  4 x 1 False  :  4 x 0
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: False : 4 x 1
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: False : 4 x 0
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: False : 4 x 1
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: False : 4 x 0
    # CHECK: ---tile[ 1 0 ]---
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: False False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: True  :  4 x 4 True  :  4 x 4
    # CHECK: False  :  3 x 4 False  :  3 x 4
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---tile[ 1 1 ]---
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: True True True True True True True True
    # CHECK: False False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: True  :  4 x 4 True  :  4 x 4
    # CHECK: False  :  3 x 4 False  :  3 x 4
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: True : 4 x 4
    # CHECK: ---tile[ 1 2 ]---
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: True False False False False False False False
    # CHECK: False False False False False False False False
    # CHECK: vectorized-access:
    # CHECK: False  :  4 x 1 False  :  4 x 0
    # CHECK: False  :  3 x 1 False  :  3 x 0
    # CHECK: thread-local-access:
    # CHECK: ---thread-[ 0 ]-mask---
    # CHECK: False : 4 x 1
    # CHECK: ---thread-[ 1 ]-mask---
    # CHECK: False : 4 x 0
    # CHECK: ---thread-[ 2 ]-mask---
    # CHECK: False : 4 x 1
    # CHECK: ---thread-[ 3 ]-mask---
    # CHECK: False : 4 x 0

    for tile_m in range(math.ceildiv(M, BM)):
        for tile_n in range(math.ceildiv(N, BN)):
            print("---tile[", tile_m, tile_n, "]---")
            var tile_mask = _tile_mask[BM, BN](
                IndexList[2](M, N), IndexList[2](tile_m, tile_n)
            )
            print_tile_mask[BM, BN](tile_mask)
            print("vectorized-access:")
            var vectoize_mask = _vectorize_mask[sizes= (TM, TN)](tile_mask)
            for i in range(BM // TM):
                for j in range(BN // TN):
                    var mask = vectoize_mask.access_mask((i, j))
                    print(
                        and_all(mask),
                        " : ",
                        vectoize_mask.access_size((i, j), mask)[0],
                        "x",
                        vectoize_mask.access_size((i, j), mask)[1],
                        end=" ",
                    )
                print("")
            print("thread-local-access:")
            for th_id in range(4):
                print("---thread-[", th_id, "]-mask---")
                var dist_mask = _distribute_mask[
                    thread_layout = Layout.row_major(2, 2)
                ](vectoize_mask, th_id)
                print_tile_mask_with_size[1, 1](dist_mask)


# CHECK-LABLE: test_copy_nd_buffer_to_layout_tensor_masked_scalar
fn test_copy_nd_buffer_to_layout_tensor_masked_scalar():
    print("==test_copy_nd_buffer_to_layout_tensor_masked_scalar")
    var buff_stack = InlineArray[Float32, 7 * 9](uninitialized=True)
    var buff_7x9 = NDBuffer[DType.float32, 2, DimList(7, 9)](
        buff_stack.unsafe_ptr()
    )
    linspace_fill(buff_7x9)

    var tensor_stack = InlineArray[Float32, 8 * 12](uninitialized=True)
    var dst_tensor_8x12 = LayoutTensor[DType.float32, Layout.row_major(8, 12)](
        tensor_stack
    ).fill(0)

    # CHECK: --tile[ 0 , 0 ]---
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 9.0 10.0 11.0 12.0
    # CHECK: 18.0 19.0 20.0 21.0
    # CHECK: 27.0 28.0 29.0 30.0
    # CHECK: --tile[ 0 , 1 ]---
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 13.0 14.0 15.0 16.0
    # CHECK: 22.0 23.0 24.0 25.0
    # CHECK: 31.0 32.0 33.0 34.0
    # CHECK: --tile[ 0 , 2 ]---
    # CHECK: 8.0 0.0 0.0 0.0
    # CHECK: 17.0 0.0 0.0 0.0
    # CHECK: 26.0 0.0 0.0 0.0
    # CHECK: 35.0 0.0 0.0 0.0
    # CHECK: --tile[ 1 , 0 ]---
    # CHECK: 36.0 37.0 38.0 39.0
    # CHECK: 45.0 46.0 47.0 48.0
    # CHECK: 54.0 55.0 56.0 57.0
    # CHECK: 0.0 0.0 0.0 0.0
    # CHECK: --tile[ 1 , 1 ]---
    # CHECK: 40.0 41.0 42.0 43.0
    # CHECK: 49.0 50.0 51.0 52.0
    # CHECK: 58.0 59.0 60.0 61.0
    # CHECK: 0.0 0.0 0.0 0.0
    # CHECK: --tile[ 1 , 2 ]---
    # CHECK: 44.0 0.0 0.0 0.0
    # CHECK: 53.0 0.0 0.0 0.0
    # CHECK: 62.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0

    for tile_m in range(2):
        for tile_n in range(3):
            print("--tile[", tile_m, ",", tile_n, "]---")

            var buff_tile_4x4 = buff_7x9.tile[4, 4](Index(tile_m, tile_n))

            var tensor_tile_4x4 = dst_tensor_8x12.tile[4, 4](tile_m, tile_n)

            var tile_mask = _tile_mask[4, 4](
                buff_7x9.get_shape(), Index(tile_m, tile_n)
            )

            alias thread_layout = Layout.row_major(2, 2)
            for th_i in range(4):
                var buff_thread_local = distribute[thread_layout=thread_layout](
                    buff_tile_4x4, th_i
                )
                var tensor_thread_local = tensor_tile_4x4.distribute[
                    thread_layout
                ](th_i)

                var distribute_mask = _distribute_mask[
                    thread_layout=thread_layout
                ](tile_mask, th_i)
                _copy_nd_buffer_to_layout_tensor_masked(
                    tensor_thread_local,
                    buff_thread_local,
                    ElementLayout[2, (1, 1)](),
                    distribute_mask,
                )

            print(tensor_tile_4x4)


# CHECK-LABEL: test_copy_from_nd_buffer_masked_scalar
fn test_copy_from_nd_buffer_masked_scalar():
    print("test_copy_from_nd_buffer_masked_scalar")
    var buff_stack = InlineArray[Float32, 7 * 9](uninitialized=True)
    var buff_7x9 = NDBuffer[DType.float32, 2, DimList(7, 9)](
        buff_stack.unsafe_ptr()
    )
    linspace_fill(buff_7x9)

    var tensor_stack = InlineArray[Float32, 8 * 12](uninitialized=True)
    var dst_tensor_8x12 = LayoutTensor[DType.float32, Layout.row_major(8, 12)](
        tensor_stack
    ).fill(0)

    # CHECK: --tile[ 0 , 0 ]---
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 9.0 10.0 11.0 12.0
    # CHECK: 18.0 19.0 20.0 21.0
    # CHECK: 27.0 28.0 29.0 30.0
    # CHECK: --tile[ 0 , 1 ]---
    # CHECK: 4.0 5.0 6.0 7.0
    # CHECK: 13.0 14.0 15.0 16.0
    # CHECK: 22.0 23.0 24.0 25.0
    # CHECK: 31.0 32.0 33.0 34.0
    # CHECK: --tile[ 0 , 2 ]---
    # CHECK: 8.0 0.0 0.0 0.0
    # CHECK: 17.0 0.0 0.0 0.0
    # CHECK: 26.0 0.0 0.0 0.0
    # CHECK: 35.0 0.0 0.0 0.0
    # CHECK: --tile[ 1 , 0 ]---
    # CHECK: 36.0 37.0 38.0 39.0
    # CHECK: 45.0 46.0 47.0 48.0
    # CHECK: 54.0 55.0 56.0 57.0
    # CHECK: 0.0 0.0 0.0 0.0
    # CHECK: --tile[ 1 , 1 ]---
    # CHECK: 40.0 41.0 42.0 43.0
    # CHECK: 49.0 50.0 51.0 52.0
    # CHECK: 58.0 59.0 60.0 61.0
    # CHECK: 0.0 0.0 0.0 0.0
    # CHECK: --tile[ 1 , 2 ]---
    # CHECK: 44.0 0.0 0.0 0.0
    # CHECK: 53.0 0.0 0.0 0.0
    # CHECK: 62.0 0.0 0.0 0.0
    # CHECK: 0.0 0.0 0.0 0.0

    for tile_m in range(2):
        for tile_n in range(3):
            print("--tile[", tile_m, ",", tile_n, "]---")
            var buff_tile_4x4 = buff_7x9.tile[4, 4](Index(tile_m, tile_n))

            var tensor_tile_4x4 = dst_tensor_8x12.tile[4, 4](tile_m, tile_n)

            var tile_mask = _tile_mask[4, 4](
                buff_7x9.get_shape(), Index(tile_m, tile_n)
            )

            alias thread_layout = Layout.row_major(2, 2)
            for th_id in range(4):
                copy_from_nd_buffer_masked[thread_layout=thread_layout](
                    tensor_tile_4x4.distribute[thread_layout](th_id),
                    buff_tile_4x4,
                    tile_mask,
                    th_id,
                )

            print(tensor_tile_4x4)


# CHECK-LABEL: test_copy_to_nd_buffer_masked_scalar
fn test_copy_to_nd_buffer_masked_scalar():
    print("== test_copy_to_nd_buffer_masked_scalar")

    var buff_stack = InlineArray[Float32, 7 * 9](uninitialized=True)
    var buff_7x9 = NDBuffer[DType.float32, 2, DimList(7, 9)](
        buff_stack.unsafe_ptr()
    )
    buff_7x9.zero()

    var tensor_stack = InlineArray[Float32, 8 * 12](uninitialized=True)
    var dst_tensor_8x12 = LayoutTensor[DType.float32, Layout.row_major(8, 12)](
        tensor_stack
    )
    arange(dst_tensor_8x12)

    for tile_m in range(2):
        for tile_n in range(3):
            var buff_tile_4x4 = buff_7x9.tile[4, 4](Index(tile_m, tile_n))

            var tensor_tile_4x4 = dst_tensor_8x12.tile[4, 4](tile_m, tile_n)

            var tile_mask = _tile_mask[4, 4](
                buff_7x9.get_shape(), Index(tile_m, tile_n)
            )

            alias thread_layout = Layout.row_major(2, 2)
            for th_id in range(4):
                copy_to_nd_buffer_masked[thread_layout=thread_layout](
                    buff_tile_4x4,
                    tensor_tile_4x4.distribute[thread_layout](th_id),
                    tile_mask,
                    th_id,
                )
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
    # CHECK: 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0
    # CHECK: 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0
    # CHECK: 60.0 61.0 62.0 63.0 64.0 65.0 66.0 67.0 68.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 80.0
    print(dst_tensor_8x12.slice[:7, :9]())
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
    # CHECK: 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0
    # CHECK: 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0
    # CHECK: 60.0 61.0 62.0 63.0 64.0 65.0 66.0 67.0 68.0
    # CHECK: 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 80.0
    print_buff(buff_7x9)


fn test_from_ndbuffer_to_layout_tensor():
    print("== test_from_ndbuffer_to_layout_tensor")
    alias type = DType.float32
    alias ptr = UnsafePointer[Scalar[type]].alloc(64)
    alias rank = 4
    alias shape = DimList(2, 3, 2, 2)
    var buffer1 = NDBuffer[type, rank, shape=shape](ptr, shape)
    linspace_fill(buffer1)
    var tensor1 = from_ndbuffer_row_major(buffer1)

    alias static_shape = DimList(Dim(), 3, Dim(), 2)
    alias dynamic_shape = DimList(2, 3, 2, 2)

    var buffer2 = NDBuffer[type, rank, shape=static_shape](ptr, dynamic_shape)
    var tensor2 = from_ndbuffer_row_major(buffer2)

    print(tensor1.runtime_layout.shape)
    print(tensor1.runtime_layout.stride)
    print(tensor1.layout.shape)
    print(tensor1.layout.stride)
    print(tensor1[0, 1, 0, 1])
    print(tensor1[1, 0, 1, 0])
    # CHECK: (2, 3, 2, 2)
    # CHECK: (12, 4, 2, 1)
    # CHECK: (2, 3, 2, 2)
    # CHECK: (12, 4, 2, 1)
    # CHECK: 5.0
    # CHECK: 14.0
    print(tensor2.runtime_layout.shape)
    print(tensor2.runtime_layout.stride)
    print(tensor2.layout.shape)
    print(tensor2.layout.stride)
    print(tensor2[0, 1, 0, 0])
    print(tensor2[1, 0, 1, 1])
    # CHECK: (2, 3, 2, 2)
    # CHECK: (12, 4, 2, 1)
    # CHECK: (-1, 3, -1, 2)
    # CHECK: (-1, -1, 2, 1)
    # CHECK: 4.0
    # CHECK: 15.0

    ptr.free()


fn main():
    test_copy_from_nd_buffer_scalars()
    test_copy_to_nd_buffer_scalars()
    test_copy_from_nd_buffer_vectors()
    test_copy_to_nd_buffer_vectors()
    test_distribute()
    test_tile_and_distribute()
    test_1d_2d_vectorize()
    test_vectorize_and_distribute()
    test_copy_nd_buffer_to_layout_tensor()
    test_copy_layout_tensor_to_buffer()
    test_tile_mask()
    test_vectorize_mask()
    test_distribute_mask()
    test_composed_tile_vectorize_distribute()
    test_composed_tile_vectorize_distribute_small()
    test_copy_nd_buffer_to_layout_tensor_masked_scalar()
    test_copy_from_nd_buffer_masked_scalar()
    test_copy_to_nd_buffer_masked_scalar()
    test_from_ndbuffer_to_layout_tensor()
