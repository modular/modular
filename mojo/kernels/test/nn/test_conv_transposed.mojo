# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import abs, div_ceil, isclose, min
from random import rand, seed
from sys import external_call
from sys.info import simdwidthof

from NN.ConvTranspose import (
    conv_transpose_naive,
    ConvTransposedPacked,
    pack_filter,
    pack_filter_shape,
)
from NN.ConvUtils import (
    ConvShape,
    ConvInfoStatic,
    get_conv_num_partitions,
    get_conv_num_tasks,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_height,
    get_direct_conv_micro_kernel_width,
)
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer
from runtime.llcl import Runtime

from utils.index import Index, StaticIntTuple
from utils.list import DimList

alias simd_size: Int = simdwidthof[DType.float32]()
alias type = DType.float32


fn test[
    type: DType
](
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    R: Int,
    S: Int,
    F: Int,
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    num_groups: Int,
) raises:
    print("== test_direct_conv")

    # fmt: off
    let HO = (H - 1) * stride[0] - pad_h[0] - pad_h[1] + (R - 1) * dilation[0] + 1
    let WO = (W - 1) * stride[1] - pad_w[0] - pad_w[1] + (S - 1) * dilation[1] + 1
    # fmt: on

    let conv_shape = ConvShape[2] {
        n: N,
        input_dims: Index(H, W),
        output_dims: Index(HO, WO),
        filter_dims: Index(R, S),
        c: C,
        f: F,
        stride: stride,
        dilation: dilation,
        pad_d: Index(0, 0),
        pad_h: pad_h,
        pad_w: pad_w,
        num_groups: num_groups,
    }

    let input_ptr = DTypePointer[type].alloc(N * H * W * C)
    let filter_ptr = DTypePointer[type].alloc(R * S * C * F)
    let output_ptr = DTypePointer[type].alloc(N * HO * WO * F)
    let output_ref_ptr = DTypePointer[type].alloc(N * HO * WO * F)

    rand[type](input_ptr, N * H * W * C)
    rand[type](filter_ptr, R * S * C * F)

    # Find the tile size used in packing.
    alias micro_kernel_height = get_direct_conv_micro_kernel_height()
    alias micro_kernel_width = get_direct_conv_micro_kernel_width()

    # Rounded C and F size for pre-packed filter.
    alias micro_kernel_f_size = get_direct_conv_micro_kernel_width() * simd_size
    let rounded_F = div_ceil(F, micro_kernel_f_size) * micro_kernel_f_size

    let input = NDBuffer[type, 4](input_ptr, Index(N, H, W, C))
    let filter = NDBuffer[type, 4](filter_ptr, Index(R, S, F, C // num_groups))
    let packed_filter_shape = pack_filter_shape(filter, num_groups)
    let packed_filter_ptr = DTypePointer[type].alloc(
        packed_filter_shape.flattened_length()
    )
    let packed_filter = NDBuffer[type, 5](
        packed_filter_ptr, packed_filter_shape
    )

    let output = NDBuffer[type, 4](output_ptr, Index(N, HO, WO, F))
    let output_ref = NDBuffer[type, 4](output_ref_ptr, Index(N, HO, WO, F))

    let stride_buf = NDBuffer[DType.index, 1, DimList(2)].stack_allocation()
    stride_buf[0] = stride[0]
    stride_buf[1] = stride[1]

    let dilation_buf = NDBuffer[DType.index, 1, DimList(2)].stack_allocation()
    dilation_buf[0] = dilation[0]
    dilation_buf[1] = dilation[1]

    let padding_buf = NDBuffer[DType.index, 1, DimList(4)].stack_allocation()
    padding_buf[0] = pad_h[0]
    padding_buf[1] = pad_w[0]
    padding_buf[2] = pad_h[1]
    padding_buf[3] = pad_w[1]

    pack_filter(filter, packed_filter, num_groups)

    # Reference.
    conv_transpose_naive[4, type, DType.index, DType.index, DType.index](
        output_ref,
        input,
        filter,
        stride_buf.make_dims_unknown(),
        dilation_buf.make_dims_unknown(),
        padding_buf.make_dims_unknown(),
    )

    # Test.
    alias conv_attr = ConvInfoStatic.create_unknown[2]()

    ConvTransposedPacked[
        4,
        5,
        4,
        DimList.create_unknown[4](),
        DimList.create_unknown[5](),
        DimList.create_unknown[4](),
        type,
        type,
        type,
        conv_attr,
    ].run(output, input, packed_filter, conv_shape)

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
                        1e-4,  # absolute error tolerance
                        1e-4,  # relative error tolerance
                    ):
                        print("Input shape NHWC: ", Index(N, H, W, C))
                        print("filter shape RSCF: ", Index(R, S, C, F))
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
    test[DType.float32](
        1,  # N
        3,  # H
        3,  # W
        1,  # C
        3,  # R
        3,  # S
        2,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(1, 1),  # pad_h
        Index(2, 2),  # pad_w
        1,  # num_groups
    )

    test[DType.float32](
        1,  # N
        3,  # H
        3,  # W
        1,  # C
        3,  # R
        3,  # S
        2,  # F
        Index(1, 1),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32](
        1,  # N
        3,  # H
        3,  # W
        1,  # C
        2,  # R
        2,  # S
        1,  # F
        Index(1, 1),  # stride
        Index(2, 2),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    test[DType.float32](
        1,  # N
        3,  # H
        3,  # W
        1,  # C
        2,  # R
        2,  # S
        2,  # F
        Index(3, 2),  # stride
        Index(1, 1),  # dilation
        Index(0, 0),  # pad_h
        Index(0, 0),  # pad_w
        1,  # num_groups
    )

    # Large shapes commented out to save CI cost.

    # # StarGan shape
    # test[DType.float32](
    #     16,  # N
    #     32,  # H
    #     32,  # W
    #     256,  # C
    #     4,  # R
    #     4,  # S
    #     128,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )

    # test[DType.float32](
    #     16,  # N
    #     64,  # H
    #     64,  # W
    #     128,  # C
    #     4,  # R
    #     4,  # S
    #     64,  # F
    #     Index(2, 2),  # stride
    #     Index(1, 1),  # dilation
    #     Index(1, 1),  # pad_h
    #     Index(1, 1),  # pad_w
    #     1,  # num_groups
    # )
