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

from sys import simdwidthof

import builtin
from buffer import NDBuffer
from buffer.dimlist import DimList
from internal_utils import TestTensor
from memory import UnsafePointer
from nn.image import Image2DLayout, ImageData, ImageShape
from nn.pool import PoolMethod, avg_pool, max_pool, pool_shape_impl

from utils.index import IndexList


fn fill_tensor(tensor: UnsafePointer[Float32], num_elements: Int):
    for j in range(num_elements):
        tensor[j] = Float32(j)


fn fill_tensor(tensor: UnsafePointer[Float32], num_elements: Int, val: Float32):
    for j in range(num_elements):
        tensor[j] = val


fn print_buffer[rank: Int](buf: NDBuffer[DType.float32, 4]):
    var s: Int = 1
    for i in range(buf.get_rank()):
        s *= buf.dim(i)

    for j in range(s):
        builtin.io._printf["%.4f\n"](buf.flatten()[j].cast[DType.float64]())


fn pool[count_boundary: Bool = False](pool_method: PoolMethod):
    alias in_shape = DimList(2, 5, 7, 2)
    alias out_shape = DimList(2, 2, 2, 2)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](0, 0, 0, 0)
    var filter = List[Int32](3, 2)
    var stride = List[Int32](2, 3)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    alias simd_width = simdwidthof[DType.float32]()

    if pool_method == PoolMethod.MAX:
        max_pool[int_type = DType.int32](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            output_tensor.ndbuffer,
        )
    else:
        avg_pool[int_type = DType.int32, count_boundary=count_boundary](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            output_tensor.ndbuffer,
        )

    print_buffer[4](output_tensor.ndbuffer)
    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor


# CHECK-LABEL: test_max_pool_2d
fn test_max_pool_2d():
    print("== test_max_pool_2d")

    # output should have form
    # ([[[[ 30.,  31.],
    #    [ 36.,  37.]],
    #   [[ 58.,  59.],
    #    [ 64.,  65.]]],
    #  [[[ 100.,  101.],
    #    [ 106., 107.]],
    #   [[128., 129.],
    #    [134., 135.]]]])

    # CHECK: 30.0000
    # CHECK: 31.0000
    # CHECK: 36.0000
    # CHECK: 37.0000
    # CHECK: 58.0000
    # CHECK: 59.0000
    # CHECK: 64.0000
    # CHECK: 65.0000
    # CHECK: 100.0000
    # CHECK: 101.0000
    # CHECK: 106.0000
    # CHECK: 107.0000
    # CHECK: 128.0000
    # CHECK: 129.0000
    # CHECK: 134.0000
    # CHECK: 135.0000
    pool(PoolMethod.MAX)


# CHECK-LABEL: test_avg_pool_2d
fn test_avg_pool_2d():
    print("== test_avg_pool_2d")

    # output should have form
    # ([[[[  15.5,  16.0],
    #    [ 21.0,  22.0]],
    #   [[ 43.0,  44.0],
    #    [ 49.0,  50.0]]],
    #  [[[ 85.0,  86.0],
    #    [ 91.0,  92.0]],
    #   [[113.0, 114.0],
    #    [119.0, 120.0]]]])

    # CHECK: 15.0000
    # CHECK: 16.0000
    # CHECK: 21.0000
    # CHECK: 22.0000
    # CHECK: 43.0000
    # CHECK: 44.0000
    # CHECK: 49.0000
    # CHECK: 50.0000
    # CHECK: 85.0000
    # CHECK: 86.0000
    # CHECK: 91.0000
    # CHECK: 92.0000
    # CHECK: 113.0000
    # CHECK: 114.0000
    # CHECK: 119.0000
    # CHECK: 120.0000
    pool(PoolMethod.AVG)


fn test_avg_pool_2d_with_padding[count_boundary: Bool = False]():
    print("== test_avg_pool_2d_count_boundary:", count_boundary)

    alias in_shape = DimList(1, 7, 7, 1)
    alias out_shape = DimList(1, 7, 7, 1)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](1, 1, 1, 1)
    var filter = List[Int32](3, 3)
    var stride = List[Int32](1, 1)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    alias simd_width = simdwidthof[DType.float32]()

    avg_pool[int_type = DType.int32, count_boundary=count_boundary](
        input_tensor.ndbuffer,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
        output_tensor.ndbuffer,
    )

    print_buffer[4](output_tensor.ndbuffer)
    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor


fn pool_ceil_test[
    count_boundary: Bool = False, ceil_mode: Bool = True
](pool_method: PoolMethod) raises:
    alias in_shape = DimList(1, 4, 4, 1)
    alias out_shape = DimList(1, 2, 2, 1)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](0, 0, 0, 0)
    var filter = List[Int32](3, 3)
    var stride = List[Int32](2, 2)
    var dilation = List[Int32](1, 1)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    alias simd_width = simdwidthof[DType.float32]()

    var output_shape = pool_shape_impl[
        4,
        DType.float32,
        DType.int32,
        DType.int32,
        DType.int32,
        DType.int32,
        True,
        ceil_mode,
    ](
        input_tensor.ndbuffer,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
    )

    if pool_method == PoolMethod.MAX:
        max_pool[int_type = DType.int32](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            output_tensor.ndbuffer,
            ceil_mode,
        )
    else:
        avg_pool[int_type = DType.int32, count_boundary=count_boundary](
            input_tensor.ndbuffer,
            filter_tensor.ndbuffer,
            stride_tensor.ndbuffer,
            dilation_tensor.ndbuffer,
            paddings_tensor.ndbuffer,
            output_tensor.ndbuffer,
            ceil_mode,
        )

    print_buffer[4](output_tensor.ndbuffer)
    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor


fn test_maxpool_2d_ceil() raises:
    print("== test_max_pool_2d_ceil")
    pool_ceil_test(PoolMethod.MAX)


fn test_average_pool_2d_ceil_excludeBound() raises:
    print("== test_average_pool_2d_ceil_excludeBound")
    pool_ceil_test(PoolMethod.AVG)


fn test_average_pool_2d_ceil_includeBound() raises:
    print("== test_average_pool_2d_ceil_includeBound")
    pool_ceil_test[True, True](PoolMethod.AVG)


fn test_max_pool_pad_dilation_2d():
    print("== test_max_pool_pad_dilation_2d")

    alias in_shape = DimList(1, 4, 4, 1)
    alias out_shape = DimList(1, 1, 3, 1)

    var input_tensor = TestTensor[DType.float32, 4](in_shape)
    fill_tensor(input_tensor.ndbuffer.data, input_tensor.num_elements)

    var output_tensor = TestTensor[DType.float32, 4](out_shape)
    fill_tensor(output_tensor.ndbuffer.data, output_tensor.num_elements, 0)

    var paddings = List[Int32](0, 0, 2, 0)
    var filter = List[Int32](2, 2)
    var stride = List[Int32](1, 1)
    var dilation = List[Int32](3, 3)

    var paddings_tensor = TestTensor[DType.int32, 1](DimList(4), paddings)
    var filter_tensor = TestTensor[DType.int32, 1](DimList(2), filter)
    var stride_tensor = TestTensor[DType.int32, 1](DimList(2), stride)
    var dilation_tensor = TestTensor[DType.int32, 1](DimList(2), dilation)

    alias simd_width = simdwidthof[DType.float32]()

    max_pool[int_type = DType.int32](
        input_tensor.ndbuffer,
        filter_tensor.ndbuffer,
        stride_tensor.ndbuffer,
        dilation_tensor.ndbuffer,
        paddings_tensor.ndbuffer,
        output_tensor.ndbuffer,
    )

    print_buffer[4](output_tensor.ndbuffer)
    _ = input_tensor
    _ = filter_tensor
    _ = stride_tensor
    _ = dilation_tensor
    _ = paddings_tensor
    _ = output_tensor


fn main() raises:
    test_max_pool_2d()
    test_avg_pool_2d()

    # CHECK-LABEL: test_avg_pool_2d_count_boundary: True
    # CHECK: 1.7778
    # CHECK: 3.0000
    # CHECK: 3.6667
    # CHECK: 4.3333
    # CHECK: 5.0000
    # CHECK: 5.6667
    # CHECK: 4.0000
    # CHECK: 5.0000
    # CHECK: 8.0000
    # CHECK: 9.0000
    # CHECK: 10.0000
    # CHECK: 11.0000
    # CHECK: 12.0000
    # CHECK: 8.3333
    # CHECK: 9.6667
    # CHECK: 15.0000
    # CHECK: 16.0000
    # CHECK: 17.0000
    # CHECK: 18.0000
    # CHECK: 19.0000
    # CHECK: 13.0000
    # CHECK: 14.3333
    # CHECK: 22.0000
    # CHECK: 23.0000
    # CHECK: 24.0000
    # CHECK: 25.0000
    # CHECK: 26.0000
    # CHECK: 17.6667
    # CHECK: 19.0000
    # CHECK: 29.0000
    # CHECK: 30.0000
    # CHECK: 31.0000
    # CHECK: 32.0000
    # CHECK: 33.0000
    # CHECK: 22.3333
    # CHECK: 23.6667
    # CHECK: 36.0000
    # CHECK: 37.0000
    # CHECK: 38.0000
    # CHECK: 39.0000
    # CHECK: 40.0000
    # CHECK: 27.0000
    # CHECK: 17.3333
    # CHECK: 26.3333
    # CHECK: 27.0000
    # CHECK: 27.6667
    # CHECK: 28.3333
    # CHECK: 29.0000
    # CHECK: 19.5556

    test_avg_pool_2d_with_padding[True]()

    # CHECK-LABEL: test_avg_pool_2d_count_boundary: False
    # CHECK: 4.0000
    # CHECK: 4.5000
    # CHECK: 5.5000
    # CHECK: 6.5000
    # CHECK: 7.5000
    # CHECK: 8.5000
    # CHECK: 9.0000
    # CHECK: 7.5000
    # CHECK: 8.0000
    # CHECK: 9.0000
    # CHECK: 10.0000
    # CHECK: 11.0000
    # CHECK: 12.0000
    # CHECK: 12.5000
    # CHECK: 14.5000
    # CHECK: 15.0000
    # CHECK: 16.0000
    # CHECK: 17.0000
    # CHECK: 18.0000
    # CHECK: 19.0000
    # CHECK: 19.5000
    # CHECK: 21.5000
    # CHECK: 22.0000
    # CHECK: 23.0000
    # CHECK: 24.0000
    # CHECK: 25.0000
    # CHECK: 26.0000
    # CHECK: 26.5000
    # CHECK: 28.5000
    # CHECK: 29.0000
    # CHECK: 30.0000
    # CHECK: 31.0000
    # CHECK: 32.0000
    # CHECK: 33.0000
    # CHECK: 33.5000
    # CHECK: 35.5000
    # CHECK: 36.0000
    # CHECK: 37.0000
    # CHECK: 38.0000
    # CHECK: 39.0000
    # CHECK: 40.0000
    # CHECK: 40.5000
    # CHECK: 39.0000
    # CHECK: 39.5000
    # CHECK: 40.5000
    # CHECK: 41.5000
    # CHECK: 42.5000
    # CHECK: 43.5000
    # CHECK: 44.0000
    test_avg_pool_2d_with_padding[False]()

    # CHECK-LABEL: test_max_pool_2d_ceil
    # CHECK: 10.0000
    # CHECK: 11.0000
    # CHECK: 14.0000
    # CHECK: 15.0000
    test_maxpool_2d_ceil()

    # CHECK-LABEL: test_average_pool_2d_ceil_excludeBound
    # CHECK: 5.0000
    # CHECK: 6.5000
    # CHECK: 11.0000
    # CHECK: 12.5000
    test_average_pool_2d_ceil_excludeBound()

    # CHECK-LABEL: test_average_pool_2d_ceil_includeBound
    # CHECK: 5.0000
    # CHECK: 4.3333
    # CHECK: 7.3333
    # CHECK: 5.5556
    test_average_pool_2d_ceil_includeBound()

    # CHECK-LABEL: test_max_pool_pad_dilation_2d
    # CHECK: 13.0000
    # CHECK: 14.0000
    # CHECK: 15.0000
    test_max_pool_pad_dilation_2d()
