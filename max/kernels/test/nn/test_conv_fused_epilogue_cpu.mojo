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
"""Covers the fused-epilogue closures inside the CPU conv wrappers.

`conv_nhwc_direct` and `conv_transposed_cpu` each build an internal
`ConvEpilogueFn` closure around the caller's SIMD epilogue. Those
two closures are reached only through op registration, so no other Mojo test
elaborates them; `test_conv_epilogue.mojo` calls `ConvDirectNHWC` directly and
bypasses both wrappers.
"""

from std.math import isclose
from std.random import rand

from layout import Coord, TileTensor, row_major
from nn.conv.conv import ConvInfoStatic, conv_nhwc_direct
from nn.conv.conv_transpose import conv_transposed_cpu
from nn.conv.conv_utils import append_shape, extend_shape

from std.utils.index import Index, IndexList

comptime dtype = DType.float32
comptime SCALE = Scalar[dtype](3.0)


def test_conv_nhwc_direct_fused() raises:
    print("== test_conv_nhwc_direct_fused")

    comptime rank = 2
    var N = 1
    var C = 2
    var F = 4
    var input_dims = Index(5, 5)
    var filter_dims = Index(3, 3)
    var stride = Index(1, 1)
    var dilation = Index(1, 1)

    var output_dims = IndexList[rank](1)
    comptime for i in range(rank):
        output_dims[i] = input_dims[i] - filter_dims[i] + 1

    var input_size = N * input_dims[0] * input_dims[1] * C
    var input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    rand(input_ptr)

    var filter_size = filter_dims[0] * filter_dims[1] * C * F
    var filter_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    rand(filter_ptr)

    var output_size = N * output_dims[0] * output_dims[1] * F
    var plain_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var fused_ptr = List(length=output_size, fill=Scalar[dtype](0))

    var input_tt = TileTensor(
        input_ptr, row_major(Coord(extend_shape(input_dims, N, C)))
    )
    var filter_tt = TileTensor(
        filter_ptr, row_major(Coord(extend_shape(filter_dims, C, F)))
    )
    var plain_tt = TileTensor(
        plain_ptr, row_major(Coord(extend_shape(output_dims, N, F)))
    )
    var fused_tt = TileTensor(
        fused_ptr, row_major(Coord(extend_shape(output_dims, N, F)))
    )

    comptime conv_attr = ConvInfoStatic[rank]()

    var out_ptr = fused_tt.ptr
    var col_stride = F
    var row_stride = output_dims[1] * F
    var plane_stride = output_dims[0] * row_stride

    # Scale each computed element, so a coordinate the wrapper's internal
    # closure mis-translates lands on the wrong element and fails the compare.
    @inline(.always)
    def scale_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]) {
        var out_ptr, var col_stride, var row_stride, var plane_stride
    }:
        comptime assert coords.rank == 4
        var flat = (
            Int(coords[0]) * plane_stride
            + Int(coords[1]) * row_stride
            + Int(coords[2]) * col_stride
            + Int(coords[3])
        )
        (out_ptr.unsafe_offset(flat)).unsafe_store(val.cast[dtype]() * SCALE)

    @inline(.always)
    def null_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]):
        pass

    conv_nhwc_direct[
        dtype,
        dtype,
        dtype,
        False,
        conv_attr,
        False,
    ](
        input_tt,
        filter_tt,
        plain_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        1,
        null_epilogue,
    )

    conv_nhwc_direct[
        dtype,
        dtype,
        dtype,
        False,
        conv_attr,
        True,
    ](
        input_tt,
        filter_tt,
        fused_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        1,
        scale_epilogue,
    )

    for i in range(output_size):
        if not isclose(fused_ptr[i], plain_ptr[i] * SCALE):
            print("FAIL at", i, fused_ptr[i], plain_ptr[i] * SCALE)
            raise Error("conv_nhwc_direct fused epilogue mismatch")
    print("PASS")


def test_conv_transposed_cpu_fused() raises:
    print("== test_conv_transposed_cpu_fused")

    comptime rank = 2
    var N = 1
    var C = 2
    var F = 3
    var input_dims = Index(4, 4)
    var filter_dims = Index(3, 3)
    var stride = Index(1, 1)
    var dilation = Index(1, 1)

    var output_dims = IndexList[rank](1)
    comptime for i in range(rank):
        output_dims[i] = (
            (input_dims[i] - 1) * stride[i]
            + dilation[i] * (filter_dims[i] - 1)
            + 1
        )

    var input_size = N * input_dims[0] * input_dims[1] * C
    var input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    rand(input_ptr)

    var filter_size = filter_dims[0] * filter_dims[1] * F * C
    var filter_ptr = List(length=filter_size, fill=Scalar[dtype](0))
    rand(filter_ptr)

    var output_size = N * output_dims[0] * output_dims[1] * F
    var plain_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var fused_ptr = List(length=output_size, fill=Scalar[dtype](0))

    var input_tt = TileTensor(
        input_ptr, row_major(Coord(extend_shape(input_dims, N, C)))
    )
    var filter_tt = TileTensor(
        filter_ptr, row_major(Coord(append_shape(filter_dims, F, C)))
    )
    var plain_tt = TileTensor(
        plain_ptr, row_major(Coord(extend_shape(output_dims, N, F)))
    )
    var fused_tt = TileTensor(
        fused_ptr, row_major(Coord(extend_shape(output_dims, N, F)))
    )

    var out_ptr = fused_tt.ptr
    var col_stride = F
    var row_stride = output_dims[1] * F
    var plane_stride = output_dims[0] * row_stride

    @inline(.always)
    def scale_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]) {
        var out_ptr, var col_stride, var row_stride, var plane_stride
    }:
        comptime assert coords.rank == 4
        var flat = (
            Int(coords[0]) * plane_stride
            + Int(coords[1]) * row_stride
            + Int(coords[2]) * col_stride
            + Int(coords[3])
        )
        (out_ptr.unsafe_offset(flat)).unsafe_store(val.cast[dtype]() * SCALE)

    @inline(.always)
    def null_epilogue[
        _dtype: DType, _width: SIMDLength, _alignment: Int = 1
    ](coords: Coord, val: SIMD[_dtype, _width]):
        pass

    conv_transposed_cpu[False, False, False](
        plain_tt,
        input_tt,
        filter_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        null_epilogue,
    )

    conv_transposed_cpu[False, False, True](
        fused_tt,
        input_tt,
        filter_tt,
        stride,
        dilation,
        Index(0, 0),
        Index(0, 0),
        Index(0, 0),
        scale_epilogue,
    )

    for i in range(output_size):
        if not isclose(fused_ptr[i], plain_ptr[i] * SCALE):
            print("FAIL at", i, fused_ptr[i], plain_ptr[i] * SCALE)
            raise Error("conv_transposed_cpu fused epilogue mismatch")
    print("PASS")


def main() raises:
    test_conv_nhwc_direct_fused()
    test_conv_transposed_cpu_fused()
