# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.list import DimList

from layout import LayoutTensor, Layout
from layout.nd_buffer_stub import copy_from_nd_buffer


fn linspace_fill[
    dtype: DType, rank: Int, shape: DimList
](inout buff: NDBuffer[dtype, rank, shape]):
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


fn main():
    test_copy_from_nd_buffer_scalars()
