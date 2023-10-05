# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# Use `kgen --emit-asm %s -o %t.asm` to exam the assembly code.

from sys.info import simdwidthof
from Conv import (
    ConvDirectNHWC,
    direct_null_elementwise_epilogue,
    ConvInfoStatic,
)
from ConvUtils import (
    ConvShape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)
from math import div_ceil
from memory.buffer import NDBuffer
from utils.index import Index
from utils.list import DimList

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

alias conv_attr = ConvInfoStatic(
    DimList(pad_bottom, pad_top),
    DimList(pad_left, pad_right),
    DimList(stride_h, stride_w),
    DimList(dilation_h, dilation_w),
)

alias value_type = DType.float32
alias simd_size = simdwidthof[value_type]()
alias micro_kernel_shape = get_micro_kernel_shape[WO, F, conv_attr, simd_size]()
# alias micro_kernel_width = get_direct_conv_micro_kernel_width()
alias micro_kernel_f_size = micro_kernel_shape[1] * simd_size
alias num_micro_tile = div_ceil(F, micro_kernel_f_size)


@export(ABI="C")
fn static_conv(
    output: NDBuffer[4, DimList(N, HO, WO, F), value_type],
    input: NDBuffer[4, DimList(N, H, W, C), value_type],
    filter: NDBuffer[
        5,
        DimList(num_micro_tile, R, S, C, micro_kernel_f_size),
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

    let tile_size = get_conv_tile_shape[value_type, micro_kernel_shape[1]](
        conv_shape
    )

    let instance = ConvDirectNHWC[
        5,
        DimList(N, H, W, C),
        DimList(num_micro_tile, R, S, C, micro_kernel_f_size),
        DimList(N, HO, WO, F),
        value_type,
        value_type,
        value_type,
        True,
        conv_attr,
        False,
    ](
        output,
        input,
        filter,
        conv_shape,
        Index(0, 0, 0, 0),
        Index(N, C, F, HO),
        tile_size,
        direct_null_elementwise_epilogue,
    )

    instance._n_loop()


# CHECK-LABEL: test_static_conv
fn test_static_conv():
    print("== test_static_conv")

    let output = NDBuffer[
        4, DimList(N, HO, WO, F), value_type
    ].stack_allocation()
    let input = NDBuffer[4, DimList(N, H, W, C), value_type].stack_allocation()
    let filter = NDBuffer[
        5,
        DimList(num_micro_tile, R, S, C, micro_kernel_f_size),
        value_type,
    ].stack_allocation()

    output.fill(0.0)
    input.fill(1.0)
    filter.fill(1.0)

    static_conv(output, input, filter)

    # CHECK: 32.0
    print(output[0, 0, 0, 0])


fn main():
    test_static_conv()
