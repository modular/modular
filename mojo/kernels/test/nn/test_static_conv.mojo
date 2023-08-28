# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# Use `kgen --emit-asm %s -o %t.asm` to exam the assembly code.

from sys.info import simdwidthof

from Conv import ConvDirectNHWC, direct_null_elementwise_epilogue
from ConvUtils import (
    ConvShape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_width,
)
from memory.buffer import NDBuffer

from utils.index import Index
from utils.list import DimList

alias N = 1
alias H = 7
alias W = 7
alias C = 8
alias R = 3
alias S = 3
alias F = 64
alias stride_h = 1
alias stride_w = 1
alias pad_left = 0
alias pad_right = 0
alias pad_top = 0
alias pad_bottom = 0
alias dilation_h = 1
alias dilation_w = 1
alias HO = (H + pad_left + pad_right - dilation_h * (R - 1) - 1) // stride_h + 1
alias WO = (W + pad_top + pad_bottom - dilation_w * (S - 1) - 1) // stride_w + 1

alias value_type = DType.float32
alias simd_width = simdwidthof[value_type]()
alias micro_kernel_width = get_direct_conv_micro_kernel_width()
alias micro_kernel_f_size = micro_kernel_width * simd_width


@export(ABI="C")
fn static_conv(
    output: NDBuffer[4, DimList(N, HO, WO, F), value_type],
    input: NDBuffer[4, DimList(N, H, W, C), value_type],
    filter: NDBuffer[
        5,
        DimList(F // micro_kernel_f_size, R, S, C, micro_kernel_f_size),
        value_type,
    ],
):

    let conv_shape = ConvShape {
        n: N,
        h: H,
        w: W,
        c: C,
        out_h: HO,
        out_w: WO,
        f: F,
        r: R,
        s: S,
        stride: Index(stride_h, stride_w),
        dilation: Index(dilation_h, dilation_w),
        pad_h: Index(pad_bottom, pad_top),
        pad_w: Index(pad_left, pad_right),
    }

    let tile_size = get_conv_tile_shape[value_type, micro_kernel_width](
        conv_shape
    )

    let instance = ConvDirectNHWC[
        5,
        DimList(N, H, W, C),
        DimList(F // micro_kernel_f_size, R, S, C, micro_kernel_f_size),
        DimList(N, HO, WO, F),
        value_type,
        True,
        False,
    ](
        output,
        input,
        filter,
        conv_shape,
        Index(0, 0, 0, 0),
        Index(N * HO * WO, C, F, 0),
        tile_size,
        direct_null_elementwise_epilogue,
    )

    instance.direct_conv()


# CHECK-LABEL: test_static_conv
fn test_static_conv():
    print("== test_static_conv")

    let output = NDBuffer[
        4, DimList(N, HO, WO, F), value_type
    ].stack_allocation()
    let input = NDBuffer[4, DimList(N, H, W, C), value_type].stack_allocation()
    let filter = NDBuffer[
        5,
        DimList(F // micro_kernel_f_size, R, S, C, micro_kernel_f_size),
        value_type,
    ].stack_allocation()

    output.fill(0.0)
    input.fill(1.0)
    filter.fill(1.0)

    static_conv(output, input, filter)

    # CHECK: 72.0
    print(output[0, 0, 0, 0])


fn main():
    test_static_conv()
