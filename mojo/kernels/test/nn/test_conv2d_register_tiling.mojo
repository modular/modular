# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

# Use `kgen --emit-asm %s -o %t.asm` to exam the assembly code.

from math import div_ceil
from sys.info import simdwidthof

from Conv import conv2d_update_wo_tile
from ConvUtils import ConvShape

from memory.buffer import NDBuffer

from testing import *

from utils.index import Index
from utils.list import DimList, Dim

alias type = DType.float32
alias micro_kernel_height = 2
alias micro_kernel_width = 2
alias simd_size = simdwidthof[type]()
alias micro_kernel_f_size = micro_kernel_width * simd_size

alias N = 1
alias H = 14
alias W = 14
alias C = 2 * simd_size
alias R = 3
alias S = 3
alias F = 2 * micro_kernel_f_size
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
alias num_micro_tile = F // micro_kernel_f_size

alias output_shape = DimList(N, HO, WO, F)
alias input_shape = DimList(N, H, W, C)
alias filter_shape = DimList(num_micro_tile, R, S, C, micro_kernel_f_size)


@export(ABI="C")
fn conv2d_register_tiling(
    output: DTypePointer[type],
    input: DTypePointer[type],
    filter: DTypePointer[type],
    c_tile_size: Int,
    f_tile_offset: Int,
    f_tile_size: Int,
    howo: StaticIntTuple[2],
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
        num_groups: 1,
    }

    conv2d_update_wo_tile[
        micro_kernel_height,
        micro_kernel_width,
        simd_size,
        filter_packed=True,
        effected_by_padding=False,
        has_residual=False,
        last_c_tile=False,
        elementwise_epilogue_enabled=False,
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
        howo,
    )


def test_conv2d_register_tiling():
    print("== test_static_conv")

    let output = NDBuffer[4, output_shape, type].stack_allocation()
    let input = NDBuffer[4, input_shape, type].stack_allocation()
    let filter = NDBuffer[5, filter_shape, type].stack_allocation()

    output.fill(0.0)
    input.fill(1.0)
    filter.fill(1.0)

    let c_tile_offset = 0
    let c_tile_size = C
    let f_tile_offset = F // 2
    let f_tile_size = F // 2
    let howo = Index(3, 2)
    let hw = Index(
        howo[0] * stride_h - pad_bottom, howo[1] * stride_w - pad_left
    )

    # FRSCf
    let filter_ptr = filter.data + f_tile_offset * R * S * C
    # NHWC
    let input_ptr = input.data + c_tile_offset + C * (hw[1] + W * hw[0])
    let output_ptr = output.data + f_tile_offset + F * (howo[1] + WO * howo[0])

    conv2d_register_tiling(
        output_ptr,
        input_ptr,
        filter_ptr,
        c_tile_size,
        f_tile_offset,
        f_tile_size,
        howo,
    )

    var actual = output.simd_load[simd_size](
        Index(0, howo[0], howo[1], f_tile_size)
    )
    var expect = SIMD[type, simd_size](R * S * c_tile_size)

    assert_equal(expect, actual)

    actual = output.simd_load[simd_size](
        Index(0, howo[0], howo[1] + micro_kernel_height - 1, f_tile_size)
    )

    assert_equal(expect, actual)

    actual = output.simd_load[simd_size](
        Index(0, howo[0], howo[1] + micro_kernel_height, f_tile_size)
    )
    expect = SIMD[type, simd_size](0.0)

    assert_equal(expect, actual)


def main():
    test_conv2d_register_tiling()
