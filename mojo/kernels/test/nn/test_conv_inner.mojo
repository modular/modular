# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import simdwidthof

from Conv import ConvNHWCInnerLoopFilterPacked, ConvShape
from Matmul import GemmShape
from memory.buffer import NDBuffer

from utils.index import Index, StaticIntTuple
from utils.list import DimList

alias type = DType.float32
alias simd_size: Int = simdwidthof[DType.float32]()
alias a_row_size: Int = 5
alias pack_inner_size: Int = 4 * simd_size
alias skip_boundary_check = True
alias use_padding = False

alias col_start_idx: Int = 0
alias total_col_count: Int = 16

alias cs_n: Int = 1
alias cs_h: Int = 16
alias cs_w: Int = 16
alias cs_f: Int = 16
alias cs_out_h: Int = 14
alias cs_out_w: Int = 14
alias cs_r: Int = 3
alias cs_s: Int = 3
alias cs_c: Int = 16

alias shape_input = DimList(cs_n, cs_h, cs_w, cs_f)
alias shape_c = DimList(cs_out_h * cs_out_w, cs_f)
alias packed_shape = DimList(1, 16, pack_inner_size)

alias MAX_NUM_CHANNELS_TILE = 384


@export(ABI="C")
fn conv_inner_loop(
    c: NDBuffer[2, shape_c, type],
    input: NDBuffer[4, shape_input, type],
    b_packed: NDBuffer[3, packed_shape, type],
    global_offset: GemmShape,
    tile_n_k: StaticIntTuple[2],
    conv_shape: ConvShape,
):
    ConvNHWCInnerLoopFilterPacked[
        shape_input,
        shape_c,
        packed_shape,
        type,
        type,
        simd_size,
        a_row_size,
        pack_inner_size,
        skip_boundary_check,
        use_padding,
    ].run(
        c,
        input,
        b_packed,
        global_offset,
        tile_n_k,
        conv_shape,
        col_start_idx,
        total_col_count,
    )


# CHECK-LABEL: test_micro_kernel
fn test_micro_kernel():
    print("== test_micro_kernel")

    let cs_stride = StaticIntTuple[2](1, 1)
    let cs_dilation = StaticIntTuple[2](1, 1)
    let cs_pad_w = StaticIntTuple[2](0, 0)
    let cs_pad_h = StaticIntTuple[2](0, 0)

    let global_offset = GemmShape(0, 0, 0)
    let tile_n_k = StaticIntTuple[2](16, 16)

    let conv_shape = ConvShape {
        n: cs_n,
        h: cs_h,
        w: cs_w,
        c: cs_c,
        out_h: cs_out_h,
        out_w: cs_out_w,
        f: cs_f,
        r: cs_r,
        s: cs_s,
        stride: cs_stride,
        dilation: cs_dilation,
        pad_h: cs_pad_h,
        pad_w: cs_pad_w,
        num_groups: 1,
    }

    let c = NDBuffer[2, shape_c, type].aligned_stack_allocation[128]()
    c.fill(0)
    let input = NDBuffer[4, shape_input, type].aligned_stack_allocation[128]()
    input.fill(2)

    let b_packed = NDBuffer[3, packed_shape, type].aligned_stack_allocation[
        128
    ]()
    b_packed.fill(1)

    conv_inner_loop(c, input, b_packed, global_offset, tile_n_k, conv_shape)

    # CHECK: 32.0
    print(c[0, 0])


fn main():
    test_micro_kernel()
