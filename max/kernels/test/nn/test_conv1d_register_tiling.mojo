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

from sys import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.conv import conv1d_update_wo_tile
from nn.conv_utils import ConvShape
from testing import *

from utils.index import Index

alias type = DType.float32
alias micro_kernel_height = 2
alias micro_kernel_width = 2
alias simd_size = simdwidthof[type]()
alias micro_kernel_f_size = micro_kernel_width * simd_size

alias N = 1
alias H = 1
alias W = 14
alias C = 2 * simd_size
alias R = 1
alias S = 3
alias F = 2 * micro_kernel_f_size
alias stride_h = 1
alias stride_w = 1
alias pad_left = 1
alias pad_right = 1
alias pad_top = 0
alias pad_bottom = 0
alias dilation_h = 1
alias dilation_w = 1
# alias HO = (H + pad_top + pad_bottom - dilation_h * (R - 1) - 1) // stride_h + 1
alias HO = 1
alias WO = (W + pad_left + pad_right - dilation_w * (S - 1) - 1) // stride_w + 1
alias num_micro_tile = F // micro_kernel_f_size

alias output_shape = DimList(N, WO, F)
alias input_shape = DimList(N, W, C)
alias filter_shape = DimList(num_micro_tile, S, C, micro_kernel_f_size)


@export(ABI="C")
fn conv1d_register_tiling(
    output: UnsafePointer[Scalar[type]],
    input: UnsafePointer[Scalar[type]],
    filter: UnsafePointer[Scalar[type]],
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    wo: Int,
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
        num_groups=1,
    )

    conv1d_update_wo_tile[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        filter_packed=True,
        effected_by_padding=False,
        has_residual=False,
        last_c_tile=False,
    ](
        output,
        input,
        filter,
        True,
        c_tile_size,
        f_tile_offset,
        f_tile_size,
        conv_shape,
        0,
        wo,
    )


fn test_conv1d_register_tiling() raises:
    var output_stack = InlineArray[Scalar[type], Int(output_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, 3, _, output_shape](output_stack)
    var input_stack = InlineArray[Scalar[type], Int(input_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[type, 3, _, input_shape](input_stack)
    var filter_stack = InlineArray[Scalar[type], Int(filter_shape.product())](
        uninitialized=True
    )
    var filter = NDBuffer[type, 4, _, filter_shape](filter_stack)

    output.fill(0.0)
    input.fill(1.0)
    filter.fill(1.0)

    var c_tile_offset = 0
    var c_tile_size = C
    var f_tile_offset = F // 2
    var f_tile_size = F // 2
    var wo = 2
    var w = wo * stride_w - pad_left

    # FRSCf
    var filter_ptr = filter.data + f_tile_offset * R * S * C
    # NHWC
    var input_ptr = input.data + c_tile_offset + C * w
    var output_ptr = output.data + f_tile_offset + F * (wo)

    conv1d_register_tiling(
        output_ptr,
        input_ptr,
        filter_ptr,
        c_tile_size,
        f_tile_offset,
        f_tile_size,
        wo,
    )

    var actual = output.load[width=simd_size](Index(0, wo, f_tile_size))
    var expect = SIMD[type, simd_size](R * S * c_tile_size)
    assert_equal(expect, actual)

    actual = output.load[width=simd_size](
        Index(0, wo + micro_kernel_height - 1, f_tile_size)
    )

    assert_equal(expect, actual)

    actual = output.load[width=simd_size](
        Index(0, wo + micro_kernel_height, f_tile_size)
    )

    assert_equal(0, actual)


fn main() raises:
    test_conv1d_register_tiling()
