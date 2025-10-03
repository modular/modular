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

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.conv import ConvDirectNHWC, ConvInfoStatic, pack_filter
from nn.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)

from utils.index import Index, IndexList


fn test[
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
]() raises:
    # Output Shape.
    # fmt: off
    alias HO = (H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1) // stride[0] + 1
    alias WO = (W + pad_w[0] + pad_w[1] - dilation[0] * (S - 1) - 1) // stride[1] + 1
    # fmt: on
    alias type = DType.float32
    alias simd_size = simd_width_of[type]()
    alias num_groups = 1

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

    # output from conv w/ dynamic and static shapes.
    var output_ptr_static = UnsafePointer[Scalar[type]].alloc(N * HO * WO * F)
    var output_ptr_dynamic = UnsafePointer[Scalar[type]].alloc(N * HO * WO * F)

    rand[type](input_ptr, N * H * W * C)
    rand[type](filter_ptr, R * S * C * F)

    var input = NDBuffer[type, 4, _, DimList(N, H, W, C)](input_ptr)
    var filter = NDBuffer[type, 4](filter_ptr, Index(R, S, C, F))
    var output_static = NDBuffer[type, 4, _, DimList(N, HO, WO, F)](
        output_ptr_static
    )
    var output_dynamic = NDBuffer[type, 4](
        output_ptr_dynamic, Index(N, HO, WO, F)
    )

    # Pre-packed filter for dynamic shapes.
    alias micro_kernel_width_default = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size_default = micro_kernel_width_default * simd_size
    var rounded_F_dynamic = (
        ceildiv(F, micro_kernel_f_size_default) * micro_kernel_f_size_default
    )
    var packed_filter_ptr_dynamic = UnsafePointer[Scalar[type]].alloc(
        R * S * C * rounded_F_dynamic
    )
    var packed_filter_dynamic = NDBuffer[type, 5](
        packed_filter_ptr_dynamic,
        Index(
            ceildiv(F, micro_kernel_f_size_default),
            R,
            S,
            C,
            micro_kernel_f_size_default,
        ),
    )

    pack_filter(filter, packed_filter_dynamic, num_groups)

    # Conv attributes.
    alias conv_attr_dynamic = ConvInfoStatic[2]()

    ConvDirectNHWC[
        4,
        5,
        4,
        _,
        _,
        _,
        DimList.create_unknown[4](),  # input shape
        DimList.create_unknown[5](),  # filter shape
        DimList.create_unknown[4](),  # output shape
        type,  # input type
        type,  # filter type
        type,  # output type
        True,
        conv_attr_dynamic,
    ].run(
        output_dynamic,
        rebind[NDBuffer[type, 4, input.origin]](input),
        packed_filter_dynamic,
        conv_shape,
    )

    alias conv_attr_static = ConvInfoStatic[2](
        DimList(pad_h[0], pad_w[0], pad_h[1], pad_w[1]),
        DimList(stride[0], stride[1]),
        DimList(dilation[0], dilation[1]),
        Dim(num_groups),
    )

    alias micro_kernel_shape = get_micro_kernel_shape[
        2, WO, F, conv_attr_static, simd_size
    ]()
    alias micro_kernel_f_size = micro_kernel_shape[1] * simd_size
    alias num_f_micro_tiles = ceildiv(F, micro_kernel_f_size)
    alias rounded_F_static = num_f_micro_tiles * micro_kernel_f_size
    alias packed_filter_shape = DimList(
        num_f_micro_tiles, R, S, C, micro_kernel_f_size
    )
    var packed_filter_ptr_static = UnsafePointer[Scalar[type]].alloc(
        R * S * C * rounded_F_static
    )
    var packed_filter_static = NDBuffer[type, 5, _, packed_filter_shape](
        packed_filter_ptr_static
    )

    pack_filter[simd_size, micro_kernel_f_size](
        filter,
        rebind[
            NDBuffer[
                type,
                5,
                packed_filter_static.origin,
                DimList.create_unknown[5](),
            ]
        ](packed_filter_static),
        num_groups,
    )

    ConvDirectNHWC[
        4,
        5,
        4,
        _,
        _,
        _,
        DimList(N, H, W, C),
        packed_filter_shape,
        DimList(N, HO, WO, F),
        type,  # input type
        type,  # filter type
        type,  # output type
        True,
        conv_attr_static,
    ].run(
        output_static,
        input,
        packed_filter_static,
        conv_shape,
    )

    input_ptr.free()
    filter_ptr.free()
    packed_filter_ptr_dynamic.free()
    packed_filter_ptr_static.free()

    # Check results, return on the first failed comparison.
    for n in range(N):
        for ho in range(HO):
            for wo in range(WO):
                for f in range(F):
                    if not isclose(
                        output_dynamic[n, ho, wo, f],
                        output_static[n, ho, wo, f],
                        atol=1e-4,  # absolute error tolerance
                        rtol=1e-5,  # relative error tolerance
                    ):
                        var expected = output_dynamic[n, ho, wo, f]
                        var actual = output_static[n, ho, wo, f]
                        print("Input shape NHWC: ", Index(N, H, W, C))
                        print("filter shape RSCF: ", Index(R, S, C, F))
                        print(
                            "Failed at",
                            Index(n, ho, wo, f),
                            "expected",
                            expected,
                            "actual",
                            actual,
                            "rerr",
                            abs(actual - expected) / abs(expected + 1e-10),
                        )
                        output_ptr_dynamic.free()
                        output_ptr_static.free()
                        return

    output_ptr_dynamic.free()
    output_ptr_static.free()

    # CHECK: Succeed
    print("Succeed")


fn main() raises:
    test[
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
    ]()
    test[
        1,  # N
        2,  # H
        2,  # W
        64,  # C
        3,  # R
        3,  # S
        64,  # F
        Index(2, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(1, 1),  # pad_w
    ]()

    # Each test will build a specialization of the conv kernel.
    # Disable the following tests for now to monitor build time.

    # test[
    #     1,  # N
    #     56,  # H
    #     56,  # W
    #     64,  # C
    #     3,  # R
    #     3,  # S
    #     64,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     28,  # H
    #     28,  # W
    #     128,  # C
    #     3,  # R
    #     3,  # S
    #     128,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     7,  # H
    #     7,  # W
    #     512,  # C
    #     3,  # R
    #     3,  # S
    #     512,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     224,  # H
    #     224,  # W
    #     3,  # C
    #     7,  # R
    #     7,  # S
    #     64,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(3, 3),  # pad_h
    #     Index(3, 3),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     56,  # H
    #     56,  # W
    #     128,  # C
    #     3,  # R
    #     3,  # S
    #     128,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     28,  # H
    #     28,  # W
    #     256,  # C
    #     3,  # R
    #     3,  # S
    #     256,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     1,  # N
    #     14,  # H
    #     14,  # W
    #     512,  # C
    #     3,  # R
    #     3,  # S
    #     512,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     19,  # N
    #     7,  # H
    #     7,  # W
    #     1,  # C
    #     3,  # R
    #     3,  # S
    #     16,  # F
    #     Index(1, 1),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()

    # test[
    #     13,  # N
    #     14,  # H
    #     14,  # W
    #     2,  # C
    #     3,  # R
    #     3,  # S
    #     32,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    # ]()
