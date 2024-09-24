# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.tensor_builder import LayoutTensorBuild as tb
from layout import LayoutTensor, Layout
from layout.fillers import *


fn print_tensor_info(tensor: LayoutTensor):
    print("---tensor-begin---")
    print("layout: ", tensor.layout)
    print("runtime_layout: ", tensor.runtime_layout)
    print("address_space: ", int(tensor.address_space))
    print("values:")
    print(tensor)
    print("---tensor-end---")


fn test_row_major():
    # CHECK-LABEL: test_tensor_builder_row_major
    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3):(3, 1))
    # CHECK: runtime_layout:  ((2, 3):(3, 1))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    print("== test_tensor_builder_row_major")
    var t = tb[DType.float32]().row_major[2, 3]().alloc()
    arange(t)
    print_tensor_info(t)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3):(3, 1))
    # CHECK: runtime_layout:  ((2, 3):(3, 1))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    var t_view = tb[DType.float32]().row_major[2, 3]().view(t.ptr)
    print_tensor_info(t_view)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((-1, -1):(-1, 1))
    # CHECK: runtime_layout:  ((2, 3):(3, 1))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    var t_dynamic = tb[DType.float32]().row_major[2]((2, 3)).view(t.ptr)
    print_tensor_info(t_dynamic)
    _ = t

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3, 4, 5):(60, 20, 5, 1))
    # CHECK: runtime_layout:  ((2, 3, 4, 5):(60, 20, 5, 1))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 60.0 20.0 80.0 40.0 100.0 5.0 65.0 25.0 85.0 45.0 105.0 10.0 70.0 30.0 90.0 50.0 110.0 15.0 75.0 35.0 95.0 55.0 115.0 1.0 61.0 21.0 81.0 41.0 101.0 6.0 66.0 26.0 86.0 46.0 106.0 11.0 71.0 31.0 91.0 51.0 111.0 16.0 76.0 36.0 96.0 56.0 116.0 2.0 62.0 22.0 82.0 42.0 102.0 7.0 67.0 27.0 87.0 47.0 107.0 12.0 72.0 32.0 92.0 52.0 112.0 17.0 77.0 37.0 97.0 57.0 117.0 3.0 63.0 23.0 83.0 43.0 103.0 8.0 68.0 28.0 88.0 48.0 108.0 13.0 73.0 33.0 93.0 53.0 113.0 18.0 78.0 38.0 98.0 58.0 118.0 4.0 64.0 24.0 84.0 44.0 104.0 9.0 69.0 29.0 89.0 49.0 109.0 14.0 74.0 34.0 94.0 54.0 114.0 19.0 79.0 39.0 99.0 59.0 119.0
    # CHECK: ---tensor-end---

    var t_4d = tb[DType.float32]().row_major[2, 3, 4, 5]().alloc()
    arange(t_4d)
    print_tensor_info(t_4d)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((-1, -1, -1, -1):(-1, -1, -1, 1))
    # CHECK: runtime_layout:  ((2, 3, 4, 5):(60, 20, 5, 1))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 60.0 20.0 80.0 40.0 100.0 5.0 65.0 25.0 85.0 45.0 105.0 10.0 70.0 30.0 90.0 50.0 110.0 15.0 75.0 35.0 95.0 55.0 115.0 1.0 61.0 21.0 81.0 41.0 101.0 6.0 66.0 26.0 86.0 46.0 106.0 11.0 71.0 31.0 91.0 51.0 111.0 16.0 76.0 36.0 96.0 56.0 116.0 2.0 62.0 22.0 82.0 42.0 102.0 7.0 67.0 27.0 87.0 47.0 107.0 12.0 72.0 32.0 92.0 52.0 112.0 17.0 77.0 37.0 97.0 57.0 117.0 3.0 63.0 23.0 83.0 43.0 103.0 8.0 68.0 28.0 88.0 48.0 108.0 13.0 73.0 33.0 93.0 53.0 113.0 18.0 78.0 38.0 98.0 58.0 118.0 4.0 64.0 24.0 84.0 44.0 104.0 9.0 69.0 29.0 89.0 49.0 109.0 14.0 74.0 34.0 94.0 54.0 114.0 19.0 79.0 39.0 99.0 59.0 119.0
    # CHECK: ---tensor-end---
    var t_4d_view = tb[DType.float32]().row_major[4]((2, 3, 4, 5)).view(
        t_4d.ptr
    )
    arange(t_4d_view)
    print_tensor_info(t_4d_view)
    _ = t_4d


fn test_col_major():
    # CHECK-LABEL: test_tensor_builder_col_major
    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3):(1, 2))
    # CHECK: runtime_layout:  ((2, 3):(1, 2))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    print("== test_tensor_builder_col_major")
    var t = tb[DType.float32]().col_major[2, 3]().alloc()
    arange(t)
    print_tensor_info(t)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3):(1, 2))
    # CHECK: runtime_layout:  ((2, 3):(1, 2))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    var t_view = tb[DType.float32]().col_major[2, 3]().view(t.ptr)
    print_tensor_info(t_view)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((-1, -1):(1, -1))
    # CHECK: runtime_layout:  ((2, 3):(1, 2))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    var t_dynamic = tb[DType.float32]().col_major[2]((2, 3)).view(t.ptr)
    print_tensor_info(t_dynamic)
    _ = t

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3, 4, 5):(1, 2, 6, 24))
    # CHECK: runtime_layout:  ((2, 3, 4, 5):(1, 2, 6, 24))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ---tensor-end---

    var t_4d = tb[DType.float32]().col_major[2, 3, 4, 5]().alloc()
    arange(t_4d)
    print_tensor_info(t_4d)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((-1, -1, -1, -1):(1, -1, -1, -1))
    # CHECK: runtime_layout:  ((2, 3, 4, 5):(1, 2, 6, 24))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0 56.0 57.0 58.0 59.0 60.0 61.0 62.0 63.0 64.0 65.0 66.0 67.0 68.0 69.0 70.0 71.0 72.0 73.0 74.0 75.0 76.0 77.0 78.0 79.0 80.0 81.0 82.0 83.0 84.0 85.0 86.0 87.0 88.0 89.0 90.0 91.0 92.0 93.0 94.0 95.0 96.0 97.0 98.0 99.0 100.0 101.0 102.0 103.0 104.0 105.0 106.0 107.0 108.0 109.0 110.0 111.0 112.0 113.0 114.0 115.0 116.0 117.0 118.0 119.0
    # CHECK: ---tensor-end---
    var t_4d_view = tb[DType.float32]().col_major[4]((2, 3, 4, 5)).view(
        t_4d.ptr
    )
    arange(t_4d_view)
    print_tensor_info(t_4d_view)
    _ = t_4d


fn test_layout():
    # CHECK-LABEL: test_tensor_builder_layout
    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3):(1, 2))
    # CHECK: runtime_layout:  ((2, 3):(1, 2))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    print("== test_tensor_builder_layout")
    var t = tb[DType.float32]().layout[2, (2, 3), (1, 2)]().alloc()
    arange(t)
    print_tensor_info(t)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((2, 3):(1, 2))
    # CHECK: runtime_layout:  ((2, 3):(1, 2))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    var t_view = tb[DType.float32]().layout[2, (2, 3), (1, 2)]().view(t.ptr)
    print_tensor_info(t_view)

    # CHECK: ---tensor-begin---
    # CHECK: layout:  ((-1, -1):(-1, -1))
    # CHECK: runtime_layout:  ((2, 3):(1, 2))
    # CHECK: address_space:  0
    # CHECK: values:
    # CHECK: 0.0 1.0 2.0
    # CHECK: 3.0 4.0 5.0
    # CHECK: ---tensor-end---
    var t_dynamic = tb[DType.float32]().layout[2]((2, 3), (1, 2)).view(t.ptr)
    print_tensor_info(t_dynamic)
    _ = t


fn main():
    test_row_major()
    test_col_major()
    test_layout()
