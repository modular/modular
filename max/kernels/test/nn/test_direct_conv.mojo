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
from sys.info import num_physical_cores, simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from nn.conv import (
    ConvDirectNHWC,
    ConvInfoStatic,
    Naive2dConvolution,
    pack_conv_filter_shape,
    pack_filter,
)
from nn.conv_utils import (
    ConvShape,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)

from utils.index import Index, IndexList

alias simd_size: Int = simdwidthof[DType.float32]()
alias type = DType.float32


# CHECK-LABEL: test_direct_conv
fn test[
    type: DType, filter_packed: Bool
](
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    R: Int,
    S: Int,
    F: Int,
    stride: IndexList[2],
    dilation: IndexList[2],
    pad_h: IndexList[2],
    pad_w: IndexList[2],
    num_groups: Int,
) raises:
    print("== test_direct_conv")

    # fmt: off
    var HO = (H + pad_h[0] + pad_h[1] - dilation[0] * (R - 1) - 1) // stride[0] + 1
    var WO = (W + pad_w[0] + pad_w[1] - dilation[1] * (S - 1) - 1) // stride[1] + 1
    # fmt: on

    var conv_shape = ConvShape[2](
        n=N,
        input_dims=Index(H, W),
        output_dims=Index(HO, WO),
        filter_dims=Index(R, S),
        c=C,
        f=F,
        stride=stride,
        dilation=dilation,
        pad_d=Index(0, 0),
        pad_h=pad_h,
        pad_w=pad_w,
        num_groups=num_groups,
    )

    var input_ptr = UnsafePointer[Scalar[type]].alloc(N * H * W * C)
    var filter_ptr = UnsafePointer[Scalar[type]].alloc(R * S * C * F)
    var output_ptr = UnsafePointer[Scalar[type]].alloc(N * HO * WO * F)
    var output_ref_ptr = UnsafePointer[Scalar[type]].alloc(N * HO * WO * F)

    rand[type](input_ptr, N * H * W * C)
    rand[type](filter_ptr, R * S * C * F)

    # Find the tile size used in packing.
    alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    var num_threads = num_physical_cores()
    var num_tasks = get_conv_num_tasks(num_threads, conv_shape)
    var num_partitions = get_conv_num_partitions[
        micro_kernel_height, micro_kernel_width * simd_size
    ](num_tasks, conv_shape)

    # Rounded C and F size for pre-packed filter.
    var micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    var rounded_F = ceildiv(F, micro_kernel_f_size) * micro_kernel_f_size

    var input = NDBuffer[type, 4](input_ptr, Index(N, H, W, C))
    var filter = NDBuffer[type, 4](filter_ptr, Index(R, S, C // num_groups, F))
    var packed_filter_shape = pack_conv_filter_shape[False](filter, num_groups)
    var packed_filter_ptr = UnsafePointer[Scalar[type]].alloc(
        packed_filter_shape.flattened_length()
    )
    var packed_filter = NDBuffer[type, 5, _, DimList.create_unknown[5]()](
        packed_filter_ptr,
        packed_filter_shape,
    )
    var output = NDBuffer[type, 4](output_ptr, Index(N, HO, WO, F))
    var output_ref = NDBuffer[type, 4](output_ref_ptr, Index(N, HO, WO, F))

    @parameter
    if filter_packed:
        pack_filter(filter, packed_filter, num_groups)

    # Reference: naive conv
    Naive2dConvolution[
        type,  # Data type.
        type,
        type,
    ].run(
        output_ref_ptr,
        input_ptr,
        filter_ptr,
        Index(N, 1, HO, WO, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C // num_groups, F),
        Index(0, 0),  #  pad_d
        pad_h,
        pad_w,
        (1, stride[0], stride[1]),
        (1, dilation[0], dilation[1]),
        num_groups,
    )

    # Test direct conv
    alias conv_attr = ConvInfoStatic[2]()

    @parameter
    if filter_packed:
        ConvDirectNHWC[
            4,
            5,
            4,
            _,
            _,
            _,
            DimList.create_unknown[4](),
            DimList.create_unknown[5](),
            DimList.create_unknown[4](),
            type,
            type,
            type,
            True,
            conv_attr,
        ].run(
            output,
            input,
            packed_filter,
            conv_shape,
        )
    else:
        ConvDirectNHWC[
            4,
            4,
            4,
            _,
            _,
            _,
            DimList.create_unknown[4](),
            DimList.create_unknown[4](),
            DimList.create_unknown[4](),
            type,
            type,
            type,
            False,
            conv_attr,
        ].run(output, input, filter, conv_shape)

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr.free()

    # Check results, return on the first failed comparison.
    for n in range(N):
        for ho in range(HO):
            for wo in range(WO):
                for f in range(F):
                    if not isclose(
                        output_ref[n, ho, wo, f],
                        output[n, ho, wo, f],
                        atol=1e-4,  # absolute error tolerance
                        rtol=1e-4,  # relative error tolerance
                    ):
                        print("Input shape NHWC: ", Index(N, H, W, C))
                        print("filter shape RSCF: ", Index(R, S, C, F))
                        print("filter packed", filter_packed)
                        print("num groups", num_groups)
                        print("Test failed at index: ", Index(n, ho, wo, f))
                        print("Golden value: ", output_ref[n, ho, wo, f])
                        print("Actual value: ", output[n, ho, wo, f])
                        output_ptr.free()
                        output_ref_ptr.free()
                        return

    output_ptr.free()
    output_ref_ptr.free()

    # CHECK: Succeed
    print("Succeed")


fn main() raises:
    """It only includes shapes where F is multiple simd_size."""
    # No packing or padding.
    test[DType.float32, False](
        1,  # N
        6,  # H
        5,  # W
        1,  # C
        3,  # R
        4,  # S
        4,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        12,  # H
        12,  # W
        12,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        13,  # H
        13,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        7,  # H
        7,  # W
        32,  # C
        3,  # R
        3,  # S
        16,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        17,  # H
        17,  # W
        16,  # C
        5,  # R
        5,  # S
        32,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        5,  # N
        7,  # H
        7,  # W
        8,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # likely partition in F or both
    test[DType.float32, False](
        1,  # N
        7,  # H
        7,  # W
        7,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        7,  # H
        7,  # W
        5,  # C
        5,  # R
        5,  # S
        288,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # Pre-packed test w/o padding.

    test[DType.float32, True](
        1,  # N
        12,  # H
        12,  # W
        12,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        13,  # H
        13,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        7,  # H
        7,  # W
        32,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        17,  # H
        17,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        5,  # N
        12,  # H
        12,  # W
        8,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        7,  # H
        7,  # W
        11,  # C
        3,  # R
        3,  # S
        192,  # F
        Index(3, 3),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        7,  # H
        7,  # W
        5,  # C
        5,  # R
        5,  # S
        256,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # No packing, w/ padding, and F not multiple of simd_size.

    test[DType.float32, False](
        1,  # N
        5,  # H
        5,  # W
        3,  # C
        3,  # R
        3,  # S
        1,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        2,  # N
        12,  # H
        11,  # W
        5,  # C
        4,  # R
        3,  # S
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        8,  # H
        12,  # W
        6,  # C
        2,  # R
        5,  # S
        3,  # F
        Index(1, 3),  # stride
        Index(1, 1),  # dilation
        Index(1, 0),  # pad_h
        Index(2, 2),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        9,  # H
        7,  # W
        1,  # C
        5,  # R
        4,  # S
        3,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(2, 2),  # pad_h
        Index(2, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, False](
        1,  # N
        10,  # H
        5,  # W
        2,  # C
        4,  # R
        3,  # S
        6,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(2, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # Pre-packed, F not multiple of simd_size

    test[DType.float32, True](
        1,  # N
        5,  # H
        5,  # W
        2,  # C
        3,  # R
        3,  # S
        7,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        7,  # H
        7,  # W
        2,  # C
        3,  # R
        3,  # S
        42,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        23,  # H
        23,  # W
        17,  # C
        3,  # R
        3,  # S
        90,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        5,  # H
        11,  # W
        2,  # C
        3,  # R
        5,  # S
        7,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        7,  # H
        9,  # W
        2,  # C
        3,  # R
        3,  # S
        42,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        11,  # H
        7,  # W
        17,  # C
        3,  # R
        5,  # S
        90,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
        1,  # num_groups
    )
    # Top resnet shapes, all pre-packed w/ padding.

    test[DType.float32, True](
        1,  # N
        224,  # H
        224,  # W
        3,  # C
        7,  # R
        7,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(3, 3),  # pad_h
        Index(3, 3),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        56,  # H
        56,  # W
        64,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        56,  # H
        56,  # W
        128,  # C
        3,  # R
        3,  # S
        128,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        28,  # H
        28,  # W
        256,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        14,  # H
        14,  # W
        256,  # C
        3,  # R
        3,  # S
        256,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        14,  # H
        14,  # W
        3,  # C
        3,  # R
        3,  # S
        16,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        7,  # H
        7,  # W
        512,  # C
        3,  # R
        3,  # S
        512,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        19,  # N
        7,  # H
        7,  # W
        1,  # C
        3,  # R
        3,  # S
        16,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        13,  # N
        14,  # H
        14,  # W
        2,  # C
        3,  # R
        3,  # S
        32,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        1,  # num_groups
    )

    # MaskRCNN shapes.

    test[DType.float32, True](
        2,  # N
        19,  # H
        19,  # W
        256,  # C
        3,  # R
        3,  # S
        384,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        2,  # N
        19,  # H
        19,  # W
        288,  # C
        3,  # R
        3,  # S
        320,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        2,  # N
        19,  # H
        19,  # W
        256,  # C
        3,  # R
        3,  # S
        288,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        2,  # N
        19,  # H
        19,  # W
        256,  # C
        3,  # R
        3,  # S
        384,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        2,  # N
        19,  # H
        19,  # W
        288,  # C
        3,  # R
        3,  # S
        320,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        2,  # N
        19,  # H
        19,  # W
        256,  # C
        3,  # R
        3,  # S
        288,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        129,  # H
        129,  # W
        320,  # C
        3,  # R
        3,  # S
        384,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        129,  # H
        129,  # W
        256,  # C
        3,  # R
        3,  # S
        384,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        1025,  # H
        1025,  # W
        3,  # C
        3,  # R
        3,  # S
        32,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # grouped conv tests
    # focus on C, F, and num_groups since grouped conv is independent of spatial dims
    test[DType.float32, True](
        1,  # N
        1,  # H
        1,  # W
        2,  # C
        1,  # R
        1,  # S
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        2,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        1,  # H
        1,  # W
        25,  # C
        1,  # R
        1,  # S
        25,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        5,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        1,  # H
        1,  # W
        16,  # C
        1,  # R
        1,  # S
        4,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        2,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        1,  # H
        1,  # W
        32,  # C
        1,  # R
        1,  # S
        20,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        2,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        1,  # H
        1,  # W
        34,  # C
        1,  # R
        1,  # S
        40,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        2,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        13,  # H
        13,  # W
        16,  # C
        5,  # R
        5,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        4,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        1,  # H
        1,  # W
        2,  # C
        1,  # R
        1,  # S
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
        2,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        3,  # H
        3,  # W
        18,  # C
        3,  # R
        3,  # S
        18,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        3,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        11,  # H
        7,  # W
        33,  # C
        3,  # R
        5,  # S
        90,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
        3,  # num_groups
    )

    test[DType.float32, True](
        3,  # N
        11,  # H
        17,  # W
        36,  # C
        3,  # R
        5,  # S
        93,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
        3,  # num_groups
    )

    test[DType.float32, True](
        1,  # N
        11,  # H
        17,  # W
        36,  # C
        2,  # R
        6,  # S
        198,  # F
        Index(2, 3),  # stride
        Index(1, 1),  # dilation
        Index(1, 0),  # pad_h
        Index(3, 2),  # pad_w
        2,  # num_groups
    )

    # depthwise conv
    test[DType.float32, True](
        1,  # N
        11,  # H
        7,  # W
        33,  # C
        3,  # R
        5,  # S
        66,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
        33,  # num_groups
    )

    # 1D edge case
    test[DType.float32, True](
        2,  # N
        1,  # H
        49,  # W
        1024,  # C
        1,  # R
        128,  # S
        1024,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(64, 64),  # pad_w
        64,  # num_groups
    )
