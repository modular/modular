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

# Use `kgen --emit-asm %s -o %t.asm` to exam the assembly code.

from math import ceildiv
from sys.info import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.conv import ConvDirectNHWC, ConvInfoStatic
from nn.conv_utils import (
    ConvShape,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)

from utils.index import Index

alias N = 1
alias H = 14
alias W = 14
alias C = 8
alias R = 3
alias S = 3
alias F = 8
alias stride_h = 1
alias stride_w = 1
alias pad_left = 1
alias pad_right = 1
alias pad_top = 1
alias pad_bottom = 1
alias dilation_h = 1
alias dilation_w = 1
alias HO = (H + pad_left + pad_right - dilation_h * (R - 1) - 1) // stride_h + 1
alias WO = (W + pad_top + pad_bottom - dilation_w * (S - 1) - 1) // stride_w + 1
alias num_groups = 1

alias conv_attr = ConvInfoStatic[2](
    DimList(pad_bottom, pad_left, pad_top, pad_right),
    DimList(stride_h, stride_w),
    DimList(dilation_h, dilation_w),
    Dim(num_groups),
)

alias value_type = DType.float32
alias simd_size = simdwidthof[value_type]()
alias micro_kernel_shape = get_micro_kernel_shape[
    2, WO, F, conv_attr, simd_size
]()
# alias micro_kernel_width = get_direct_conv_micro_kernel_width()
alias micro_kernel_f_size = micro_kernel_shape[1] * simd_size
alias num_micro_tile = ceildiv(F, micro_kernel_f_size)


@export(ABI="C")
fn static_conv(
    output: NDBuffer[mut=True, value_type, 4, _, DimList(N, HO, WO, F)],
    input: NDBuffer[value_type, 4, _, DimList(N, H, W, C)],
    filter: NDBuffer[
        value_type,
        5,
        _,
        DimList(num_micro_tile, R, S, C, micro_kernel_f_size),
    ],
):
    var conv_shape = ConvShape[2](
        n=N,
        input_dims=Index(H, W),
        output_dims=Index(HO, WO),
        filter_dims=Index(R, S),
        c=C,
        f=F,
        stride=Index(stride_h, stride_w),
        dilation=Index(dilation_h, dilation_w),
        pad_d=Index(0, 0),
        pad_h=Index(pad_bottom, pad_top),
        pad_w=Index(pad_left, pad_right),
        num_groups=num_groups,
    )

    fn direct_null_elementwise_epilogue(
        n: Int, ho: Int, wo: Int, f_offset: Int, f_size: Int
    ):
        pass

    try:
        ConvDirectNHWC[
            4,
            5,
            4,
            _,
            _,
            _,
            DimList(N, H, W, C),
            DimList(num_micro_tile, R, S, C, micro_kernel_f_size),
            DimList(N, HO, WO, F),
            value_type,
            value_type,
            value_type,
            True,
            conv_attr,
        ].run(output, input, filter, conv_shape)
    except e:
        print(e)


# CHECK-LABEL: test_static_conv
def test_static_conv():
    print("== test_static_conv")

    var output_stack = InlineArray[Scalar[value_type], N * HO * WO * F](
        uninitialized=True
    )
    var output = NDBuffer[value_type, 4, _, DimList(N, HO, WO, F)](output_stack)
    var input_stack = InlineArray[Scalar[value_type], N * H * W * C](
        uninitialized=True
    )
    var input = NDBuffer[value_type, 4, _, DimList(N, H, W, C)](input_stack)
    var filter_stack = InlineArray[
        Scalar[value_type], num_micro_tile * R * S * C * micro_kernel_f_size
    ](uninitialized=True)
    var filter = NDBuffer[
        value_type,
        5,
        _,
        DimList(num_micro_tile, R, S, C, micro_kernel_f_size),
    ](filter_stack)

    output.fill(0.0)
    input.fill(1.0)
    filter.fill(1.0)

    static_conv(output, input, filter)

    # CHECK: 32.0
    print(output[0, 0, 0, 0])


def main():
    test_static_conv()
