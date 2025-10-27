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

from math import ceildiv, isclose
from random import rand
from sys.info import simd_width_of

from algorithm.functional import vectorize
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from layout.int_tuple import fill_like
from nn.conv_transpose import (
    ConvTransposedPacked,
    conv_transpose_naive,
    conv_transpose_shape,
    pack_filter,
    pack_filter_shape,
)
from nn.conv_utils import (
    ConvInfoStatic,
    ConvShape,
    append_shape,
    extend_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from testing import assert_equal, assert_raises, TestSuite

from utils.index import Index, IndexList

alias simd_size: Int = simd_width_of[DType.float32]()
alias dtype = DType.float32


@always_inline
fn extend_shape_5d[
    rank: Int
](in_shape: IndexList[rank], first: Int, last: Int) -> IndexList[5]:
    var out_shape = IndexList[5](1)
    out_shape[0] = first
    out_shape[4] = last

    @parameter
    if rank == 1:
        out_shape[3] = in_shape[0]
    elif rank == 2:
        out_shape[2] = in_shape[0]
        out_shape[3] = in_shape[1]
    elif rank == 3:
        out_shape[1] = in_shape[0]
        out_shape[2] = in_shape[1]
        out_shape[3] = in_shape[2]

    return out_shape


@always_inline
fn extend_shape_3d[rank: Int](in_shape: IndexList[rank]) -> IndexList[3]:
    var out_shape = IndexList[3](1)

    @parameter
    for i in range(rank):
        out_shape[2 - i] = in_shape[rank - i - 1]

    return out_shape


@always_inline
fn append_shape_5d[
    rank: Int
](in_shape: IndexList[rank], last2nd: Int, last: Int) -> IndexList[5]:
    var out_shape = IndexList[5](1)
    out_shape[3] = last2nd
    out_shape[4] = last

    @parameter
    if rank == 1:
        out_shape[2] = in_shape[0]
    elif rank == 2:
        out_shape[1] = in_shape[0]
        out_shape[2] = in_shape[1]
    elif rank == 3:
        out_shape[0] = in_shape[0]
        out_shape[1] = in_shape[1]
        out_shape[2] = in_shape[2]

    return out_shape


fn test_conv_transposed[
    dtype: DType, rank: Int
](
    N: Int,
    input_dims: IndexList[rank],
    C: Int,
    filter_dims: IndexList[rank],
    F: Int,
    stride: IndexList[rank],
    dilation: IndexList[rank],
    pad: IndexList[2 * rank],
    num_groups: Int,
) raises:
    print("test_conv_transposed")

    var output_dims = IndexList[rank](1)

    @parameter
    for i in range(rank):
        output_dims[i] = (
            (input_dims[i] - 1) * stride[i]
            - pad[2 * i]
            - pad[2 * i + 1]
            + dilation[i] * (filter_dims[i] - 1)
            + 1
        )

    var pad_d = IndexList[2](0)
    var pad_h = IndexList[2](0)
    var pad_w = IndexList[2](0)

    @parameter
    if rank == 1:
        pad_w = Index(pad[0], pad[1])
    elif rank == 2:
        pad_h = Index(pad[0], pad[1])
        pad_w = Index(pad[2], pad[3])
    elif rank == 3:
        pad_d = Index(pad[0], pad[1])
        pad_h = Index(pad[2], pad[3])
        pad_w = Index(pad[4], pad[5])

    var conv_shape = ConvShape[rank](
        n=N,
        input_dims=input_dims,
        output_dims=output_dims,
        filter_dims=filter_dims,
        c=C,
        f=F,
        stride=stride,
        dilation=dilation,
        pad_d=pad_d,
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )

    var C_per_group = C // num_groups

    var input_size = N * conv_shape.input_image_flat_size() * C
    var input_ptr = UnsafePointer[Scalar[dtype]].alloc(input_size)
    rand(input_ptr, input_size)

    var filter_size = conv_shape.filter_window_flat_size() * C_per_group * F
    var filter_ptr = UnsafePointer[Scalar[dtype]].alloc(filter_size)
    rand(filter_ptr, filter_size)

    var output_size = N * conv_shape.output_image_flat_size() * F
    var output_ptr = UnsafePointer[Scalar[dtype]].alloc(output_size)
    var output_ref_ptr = UnsafePointer[Scalar[dtype]].alloc(output_size)

    # Find the tile size used in packing.
    alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    # Rounded C and F size for pre-packed filter.
    # alias micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    # var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size

    # Input buffer.
    alias input_layout = Layout.row_major[rank + 2]()
    var input_shape = extend_shape(input_dims, N, C)
    var input = LayoutTensor[dtype, input_layout](
        input_ptr,
        RuntimeLayout[input_layout].row_major(input_shape),
    )
    var input_ref = LayoutTensor[dtype, Layout.row_major[5]()](
        input_ptr,
        RuntimeLayout[Layout.row_major[5]()].row_major(
            extend_shape_5d(input_dims, N, C)
        ),
    )

    # Filter buffer.
    var filter_shape = append_shape(filter_dims, F, C_per_group)
    var filter = LayoutTensor[dtype, Layout.row_major[rank + 2]()](
        filter_ptr,
        RuntimeLayout[Layout.row_major[rank + 2]()].row_major(filter_shape),
    )
    var filter_ref = LayoutTensor[dtype, Layout.row_major[5]()](
        filter_ptr,
        RuntimeLayout[Layout.row_major[5]()].row_major(
            append_shape_5d(filter_dims, F, C_per_group)
        ),
    )

    var packed_filter_shape = pack_filter_shape(filter, num_groups)
    var packed_filter_ptr = UnsafePointer[Scalar[dtype]].alloc(
        packed_filter_shape.flattened_length()
    )
    var packed_filter = LayoutTensor[dtype, Layout.row_major[rank + 3]()](
        packed_filter_ptr,
        RuntimeLayout[Layout.row_major[rank + 3]()].row_major(
            rebind[IndexList[rank + 3]](packed_filter_shape)
        ),
    )

    var output_shape = extend_shape(output_dims, N, F)
    var output = LayoutTensor[dtype, Layout.row_major[rank + 2]()](
        output_ptr,
        RuntimeLayout[Layout.row_major[rank + 2]()].row_major(output_shape),
    )
    var output_ref = LayoutTensor[dtype, Layout.row_major[5]()](
        output_ref_ptr,
        RuntimeLayout[Layout.row_major[5]()].row_major(
            extend_shape_5d(output_dims, N, F)
        ),
    )

    # Bias for epilogue
    var bias_ptr = UnsafePointer[Scalar[dtype]].alloc(F)
    rand(bias_ptr, F)

    pack_filter(filter, packed_filter, num_groups)

    # Reference.
    conv_transpose_naive[dtype](
        output_ref,
        input_ref,
        filter_ref,
        extend_shape_3d[rank](stride),
        extend_shape_3d[rank](dilation),
        pad_d,
        pad_h,
        pad_w,
    )

    # Add bias and activatiion separately.
    var output_image_size = output_dims.flattened_length()
    for n in range(N):
        for i in range(output_image_size):
            var output_ref_ptr = output_ref.ptr + F * (
                i + output_image_size * n
            )

            @always_inline
            @__copy_capture(output_ref_ptr, bias_ptr)
            @parameter
            fn body0[width: Int](offset: Int):
                output_ref_ptr.store(
                    offset,
                    10.0
                    * (
                        output_ref_ptr.load[width=width](offset)
                        + bias_ptr.load[width=width](offset)
                    ),
                )

            vectorize[body0, simd_size](F)

    # Test.
    alias conv_attr = ConvInfoStatic[input_layout.rank() - 2]()

    # Test epilogue
    @always_inline
    @__copy_capture(output, bias_ptr)
    @parameter
    fn epilogue[_rank: Int](coords: IndexList[_rank], f_size: Int):
        @always_inline
        @parameter
        fn body1[width: Int](idx: Int):
            var curr_coords = rebind[IndexList[rank + 2]](coords)
            curr_coords[rank + 1] += idx

            var output_idx = output.runtime_layout(
                RuntimeTuple[fill_like(output.layout.shape, UNKNOWN_VALUE)](
                    curr_coords
                )
            )

            var vec = output.ptr.load[width=width](output_idx)

            output.ptr.store(
                output_idx,
                10.0
                * (vec + bias_ptr.load[width=width](curr_coords[rank + 1])),
            )

        vectorize[body1, simd_size](f_size)

    ConvTransposedPacked[
        _,  # input origin
        _,  # filter origin,
        _,  # output origin,
        input_layout,  # input layout
        Layout.row_major[rank + 3](),  # filter layout
        Layout.row_major[rank + 2](),  # output layout
        dtype,  # input
        dtype,  # filter
        dtype,  # output
        conv_attr,
        epilogue,
    ].run(
        output,
        input,
        packed_filter,
        rebind[ConvShape[input_layout.rank() - 2]](conv_shape),
    )

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()
    bias_ptr.free()

    # Check results, return on the first failed comparison.
    for i in range(output_size):
        if not all(
            isclose(
                output_ref.ptr.load[width=1](i),
                output.ptr.load[width=1](i),
                atol=1e-4,  # absolute error tolerance
                rtol=1e-4,  # relative error tolerance
            )
        ):
            print("Input shape: ", input_shape)
            print("filter shape: ", filter_shape)
            print("num groups", num_groups)
            print("flat output index:", i)
            print("Golden value: ", output_ref.ptr.load[width=1](i))
            print("Actual value: ", output.ptr.load[width=1](i))
            output_ptr.free()
            output_ref_ptr.free()
            return

    output_ptr.free()
    output_ref_ptr.free()

    # CHECK: Succeed
    print("Succeed")


fn test_conv_transpose_shape_basic() raises:
    """Test conv_transpose_shape function with basic cases."""
    # Test 4D: Basic 2D conv transpose (N=1, H=3, W=3, C=1) x (R=3, S=3, F=2, C=1)
    # With stride=1, dilation=1, no padding
    # Expected output: (1, 5, 5, 2)
    var input_ptr = UnsafePointer[Scalar[DType.float32]].alloc(9)
    var kernel_ptr = UnsafePointer[Scalar[DType.float32]].alloc(18)
    var strides_ptr = UnsafePointer[Scalar[DType.int32]].alloc(2)
    var dilations_ptr = UnsafePointer[Scalar[DType.int32]].alloc(2)
    var pads_ptr = UnsafePointer[Scalar[DType.int32]].alloc(4)
    var output_pads_ptr = UnsafePointer[Scalar[DType.int32]].alloc(2)

    strides_ptr[0] = 1
    strides_ptr[1] = 1
    dilations_ptr[0] = 1
    dilations_ptr[1] = 1
    for i in range(4):
        pads_ptr[i] = 0
    output_pads_ptr[0] = 0
    output_pads_ptr[1] = 0

    var input = LayoutTensor[DType.float32, Layout.row_major[4]()](
        input_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(Index(1, 3, 3, 1)),
    )
    var kernel = LayoutTensor[DType.float32, Layout.row_major[4]()](
        kernel_ptr,
        RuntimeLayout[Layout.row_major[4]()].row_major(Index(3, 3, 2, 1)),
    )
    var strides = LayoutTensor[DType.int32, Layout.row_major[1]()](
        strides_ptr, RuntimeLayout[Layout.row_major[1]()].row_major(Index(2))
    )
    var dilations = LayoutTensor[DType.int32, Layout.row_major[1]()](
        dilations_ptr, RuntimeLayout[Layout.row_major[1]()].row_major(Index(2))
    )
    var pads = LayoutTensor[DType.int32, Layout.row_major[1]()](
        pads_ptr, RuntimeLayout[Layout.row_major[1]()].row_major(Index(4))
    )
    var output_pads = LayoutTensor[DType.int32, Layout.row_major[1]()](
        output_pads_ptr,
        RuntimeLayout[Layout.row_major[1]()].row_major(Index(2)),
    )

    var shape = conv_transpose_shape[
        DType.float32, DType.int32, DType.int32, DType.int32, DType.int32, False
    ](input, kernel, strides, dilations, pads, output_pads)

    assert_equal(shape[0], 1)
    assert_equal(shape[1], 5)
    assert_equal(shape[2], 5)
    assert_equal(shape[3], 2)

    input_ptr.free()
    kernel_ptr.free()
    strides_ptr.free()
    dilations_ptr.free()
    pads_ptr.free()
    output_pads_ptr.free()


fn test_2d_stride_3_2_pad_1_1_2_2() raises:
    test_conv_transposed[DType.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(3, 3),
        2,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1, 2, 2),  # pad h, w
        1,  # num_groups
    )


fn test_2d_basic_no_pad() raises:
    test_conv_transposed[DType.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(3, 3),
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad h, w
        1,  # num_groups
    )


fn test_2d_dilation_2_2() raises:
    test_conv_transposed[DType.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(3, 3),
        1,  # F
        Index(1, 1),  # stride
        Index(2, 2),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )


fn test_2d_stride_3_2_kernel_2_2() raises:
    test_conv_transposed[DType.float32, 2](
        1,  # N
        Index(3, 3),
        1,  # C
        Index(2, 2),
        2,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0, 0, 0),  # pad_h, pad_w
        1,  # num_groups
    )


fn test_3d_stride_1_3_2() raises:
    test_conv_transposed[DType.float32, 3](
        1,  # N
        Index(2, 3, 3),
        1,  # C
        Index(2, 2, 2),
        2,  # F
        Index(1, 3, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](0),  # pad
        1,  # num_groups
    )


fn test_3d_stride_2_1_2_dilation_1_1_2() raises:
    test_conv_transposed[DType.float32, 3](
        1,  # N
        Index(3, 4, 7),
        1,  # C
        Index(3, 2, 2),
        2,  # F
        Index(2, 1, 2),  # stride
        Index(1, 1, 2),  # dilation
        IndexList[6](0),  # pad
        1,  # num_groups
    )


fn test_3d_with_padding() raises:
    test_conv_transposed[DType.float32, 3](
        1,  # N
        Index(4, 3, 3),
        1,  # C
        Index(1, 4, 2),
        2,  # F
        Index(1, 3, 2),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](1, 0, 2, 1, 0, 1),  # pad
        1,  # num_groups
    )


fn test_3d_complex_padding_dilation() raises:
    test_conv_transposed[DType.float32, 3](
        1,  # N
        Index(4, 5, 7),
        1,  # C
        Index(3, 2, 1),
        2,  # F
        Index(1, 3, 2),  # stride
        Index(2, 3, 1),  # dilation
        IndexList[6](2, 2, 1, 1, 1, 1),  # pad
        1,  # num_groups
    )


fn test_3d_multi_channel() raises:
    test_conv_transposed[DType.float32, 3](
        1,  # N
        Index(5, 5, 5),
        4,  # C
        Index(3, 3, 3),
        8,  # F
        Index(1, 1, 1),  # stride
        Index(1, 1, 1),  # dilation
        IndexList[6](0, 0, 0, 0, 0, 0),  # pad
        1,  # num_groups
    )


def main():
    var suite = TestSuite()

    # Test conv_transpose_shape function
    suite.test[test_conv_transpose_shape_basic]()

    # Test full conv transposed operations
    suite.test[test_2d_stride_3_2_pad_1_1_2_2]()
    suite.test[test_2d_basic_no_pad]()
    suite.test[test_2d_dilation_2_2]()
    suite.test[test_2d_stride_3_2_kernel_2_2]()
    suite.test[test_3d_stride_1_3_2]()
    suite.test[test_3d_stride_2_1_2_dilation_1_1_2]()
    suite.test[test_3d_with_padding]()
    suite.test[test_3d_complex_padding_dilation]()
    suite.test[test_3d_multi_channel]()

    suite^.run()

    # Large shapes commented out to save CI cost.

    # # StarGan shape
    # test_conv_transposed[DType.float32, 2](
    #     16,  # N
    #     Index(32, 32),
    #     256,  # C
    #     Index(4, 4),
    #     128,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1, 1, 1),  # pad_h, w
    #     1,  # num_groups
    # )

    # test_conv_transposed[DType.float32, 2](
    #     16,  # N
    #     Index(64, 64),
    #     128,  # C
    #     Index(4, 4),
    #     64,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1, 1, 1),  # pad_h, pad_w
    #     1,  # num_groups
    # )

    # # 3d Unet shapes
    # test_conv_transposed[DType.float32, 3](
    #     1,  # N
    #     Index(4, 4, 4),
    #     320,  # C
    #     Index(2, 2, 2),
    #     320,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[DType.float32, 3](
    #     1,  # N
    #     Index(8, 8, 8),
    #     320,  # C
    #     Index(2, 2, 2),
    #     256,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[DType.float32, 3](
    #     1,  # N
    #     Index(16, 16, 16),
    #     256,  # C
    #     Index(2, 2, 2),
    #     128,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[DType.float32, 3](
    #     1,  # N
    #     Index(32, 32, 32),
    #     128,  # C
    #     Index(2, 2, 2),
    #     64,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )

    # test_conv_transposed[DType.float32, 3](
    #     1,  # N
    #     Index(64, 64, 64),
    #     64,  # C
    #     Index(2, 2, 2),
    #     32,  # F
    #     Index(2, 2, 2),  # stride
    #     Index(1, 1, 1),  # dilation
    #     IndexList[6](0),  # pad
    #     1,  # num_groups
    # )
