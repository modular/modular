# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.list import DimList

from layout import LayoutTensor, Layout
from layout.nd_buffer_stub import copy_from_nd_buffer, copy_to_nd_buffer


fn linspace_fill[
    dtype: DType, rank: Int, shape: DimList
](inout buff: NDBuffer[dtype, rank, shape]):
    for i in range(buff.size()):
        buff.data[i] = i


fn zero_fill[
    dtype: DType, rank: Int, shape: DimList
](inout buff: NDBuffer[dtype, rank, shape]):
    for i in range(buff.size()):
        buff.data[i] = 0


fn print_buff[
    dtype: DType, rank: Int, shape: DimList
](buff: NDBuffer[dtype, rank, shape]):
    constrained[rank == 2, "rank-2 buffer is expected"]()
    for m in range(buff.dim(0)):
        for n in range(buff.dim(1)):
            print(buff[m, n], end=" ")
        print("")


# CHECK-LABEL: test_copy_from_nd_buffer_scalars
fn test_copy_from_nd_buffer_scalars():
    print("== test_copy_from_nd_buffer_scalars")

    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)].stack_allocation()
    linspace_fill(buff)

    var layout_tensor = LayoutTensor[
        DType.float32,
        Layout.row_major(8, 8),
    ].stack_allocation()
    layout_tensor.fill(0)

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
    layout_tensor.print()


# CHECK-LABEL: test_copy_to_nd_buffer_scalars
fn test_copy_to_nd_buffer_scalars():
    print("== test_copy_to_nd_buffer_scalars")

    var layout_tensor = LayoutTensor[
        DType.float32,
        Layout.row_major(8, 8),
    ].stack_allocation()
    layout_tensor.linspace()

    var buff = NDBuffer[DType.float32, 2, DimList(8, 8)].stack_allocation()
    zero_fill(buff)

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

    var buff = NDBuffer[DType.float32, 2, DimList(16, 16)].stack_allocation()
    linspace_fill(buff)

    var layout_tensor = LayoutTensor[
        DType.float32,
        Layout.row_major(16, 16),
    ].stack_allocation()
    layout_tensor.fill(0)

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
    layout_tensor.vectorize[1, 4]().print()

    layout_tensor.fill(0)

    for th_id in range(16):
        var thread_local_layout_tensor = layout_tensor.vectorize[
            4, 4
        ]().distribute[threads_layout](th_id)
        copy_from_nd_buffer[thread_layout=threads_layout](
            thread_local_layout_tensor, buff, th_id
        )

    # CHECK: [0.0, 16.0, 32.0, 48.0, 1.0, 17.0, 33.0, 49.0, 2.0, 18.0, 34.0, 50.0, 3.0, 19.0, 35.0, 51.0] [4.0, 20.0, 36.0, 52.0, 5.0, 21.0, 37.0, 53.0, 6.0, 22.0, 38.0, 54.0, 7.0, 23.0, 39.0, 55.0] [8.0, 24.0, 40.0, 56.0, 9.0, 25.0, 41.0, 57.0, 10.0, 26.0, 42.0, 58.0, 11.0, 27.0, 43.0, 59.0] [12.0, 28.0, 44.0, 60.0, 13.0, 29.0, 45.0, 61.0, 14.0, 30.0, 46.0, 62.0, 15.0, 31.0, 47.0, 63.0]
    # CHECK: [64.0, 80.0, 96.0, 112.0, 65.0, 81.0, 97.0, 113.0, 66.0, 82.0, 98.0, 114.0, 67.0, 83.0, 99.0, 115.0] [68.0, 84.0, 100.0, 116.0, 69.0, 85.0, 101.0, 117.0, 70.0, 86.0, 102.0, 118.0, 71.0, 87.0, 103.0, 119.0] [72.0, 88.0, 104.0, 120.0, 73.0, 89.0, 105.0, 121.0, 74.0, 90.0, 106.0, 122.0, 75.0, 91.0, 107.0, 123.0] [76.0, 92.0, 108.0, 124.0, 77.0, 93.0, 109.0, 125.0, 78.0, 94.0, 110.0, 126.0, 79.0, 95.0, 111.0, 127.0]
    # CHECK: [128.0, 144.0, 160.0, 176.0, 129.0, 145.0, 161.0, 177.0, 130.0, 146.0, 162.0, 178.0, 131.0, 147.0, 163.0, 179.0] [132.0, 148.0, 164.0, 180.0, 133.0, 149.0, 165.0, 181.0, 134.0, 150.0, 166.0, 182.0, 135.0, 151.0, 167.0, 183.0] [136.0, 152.0, 168.0, 184.0, 137.0, 153.0, 169.0, 185.0, 138.0, 154.0, 170.0, 186.0, 139.0, 155.0, 171.0, 187.0] [140.0, 156.0, 172.0, 188.0, 141.0, 157.0, 173.0, 189.0, 142.0, 158.0, 174.0, 190.0, 143.0, 159.0, 175.0, 191.0]
    # CHECK: [192.0, 208.0, 224.0, 240.0, 193.0, 209.0, 225.0, 241.0, 194.0, 210.0, 226.0, 242.0, 195.0, 211.0, 227.0, 243.0] [196.0, 212.0, 228.0, 244.0, 197.0, 213.0, 229.0, 245.0, 198.0, 214.0, 230.0, 246.0, 199.0, 215.0, 231.0, 247.0] [200.0, 216.0, 232.0, 248.0, 201.0, 217.0, 233.0, 249.0, 202.0, 218.0, 234.0, 250.0, 203.0, 219.0, 235.0, 251.0] [204.0, 220.0, 236.0, 252.0, 205.0, 221.0, 237.0, 253.0, 206.0, 222.0, 238.0, 254.0, 207.0, 223.0, 239.0, 255.0]

    layout_tensor.vectorize[4, 4]().print()


# CHECK-LABEL: test_copy_to_nd_buffer_vectors
fn test_copy_to_nd_buffer_vectors():
    print("== test_copy_to_nd_buffer_vectors")

    var layout_tensor = LayoutTensor[
        DType.float32,
        Layout.row_major(16, 16),
    ].stack_allocation()
    layout_tensor.linspace()

    var buff = NDBuffer[DType.float32, 2, DimList(16, 16)].stack_allocation()
    zero_fill(buff)

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
    zero_fill(buff)

    for th_id in range(threads_layout.size()):
        var thread_local_layout_tensor = layout_tensor.vectorize[
            4, 4
        ]().distribute[threads_layout](th_id)
        copy_to_nd_buffer[thread_layout=threads_layout](
            buff, thread_local_layout_tensor, th_id
        )
    # CHECK: 0.0 16.0 32.0 48.0 4.0 20.0 36.0 52.0 8.0 24.0 40.0 56.0 12.0 28.0 44.0 60.0
    # CHECK: 1.0 17.0 33.0 49.0 5.0 21.0 37.0 53.0 9.0 25.0 41.0 57.0 13.0 29.0 45.0 61.0
    # CHECK: 2.0 18.0 34.0 50.0 6.0 22.0 38.0 54.0 10.0 26.0 42.0 58.0 14.0 30.0 46.0 62.0
    # CHECK: 3.0 19.0 35.0 51.0 7.0 23.0 39.0 55.0 11.0 27.0 43.0 59.0 15.0 31.0 47.0 63.0
    # CHECK: 64.0 80.0 96.0 112.0 68.0 84.0 100.0 116.0 72.0 88.0 104.0 120.0 76.0 92.0 108.0 124.0
    # CHECK: 65.0 81.0 97.0 113.0 69.0 85.0 101.0 117.0 73.0 89.0 105.0 121.0 77.0 93.0 109.0 125.0
    # CHECK: 66.0 82.0 98.0 114.0 70.0 86.0 102.0 118.0 74.0 90.0 106.0 122.0 78.0 94.0 110.0 126.0
    # CHECK: 67.0 83.0 99.0 115.0 71.0 87.0 103.0 119.0 75.0 91.0 107.0 123.0 79.0 95.0 111.0 127.0
    # CHECK: 128.0 144.0 160.0 176.0 132.0 148.0 164.0 180.0 136.0 152.0 168.0 184.0 140.0 156.0 172.0 188.0
    # CHECK: 129.0 145.0 161.0 177.0 133.0 149.0 165.0 181.0 137.0 153.0 169.0 185.0 141.0 157.0 173.0 189.0
    # CHECK: 130.0 146.0 162.0 178.0 134.0 150.0 166.0 182.0 138.0 154.0 170.0 186.0 142.0 158.0 174.0 190.0
    # CHECK: 131.0 147.0 163.0 179.0 135.0 151.0 167.0 183.0 139.0 155.0 171.0 187.0 143.0 159.0 175.0 191.0
    # CHECK: 192.0 208.0 224.0 240.0 196.0 212.0 228.0 244.0 200.0 216.0 232.0 248.0 204.0 220.0 236.0 252.0
    # CHECK: 193.0 209.0 225.0 241.0 197.0 213.0 229.0 245.0 201.0 217.0 233.0 249.0 205.0 221.0 237.0 253.0
    # CHECK: 194.0 210.0 226.0 242.0 198.0 214.0 230.0 246.0 202.0 218.0 234.0 250.0 206.0 222.0 238.0 254.0
    # CHECK: 195.0 211.0 227.0 243.0 199.0 215.0 231.0 247.0 203.0 219.0 235.0 251.0 207.0 223.0 239.0 255.0
    print_buff(buff)


fn main():
    test_copy_from_nd_buffer_scalars()
    test_copy_to_nd_buffer_scalars()
    test_copy_from_nd_buffer_vectors()
    test_copy_to_nd_buffer_vectors()
