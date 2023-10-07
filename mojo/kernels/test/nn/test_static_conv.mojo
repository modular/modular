# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import abs, div_ceil, min, isclose
from random import rand
from sys.info import simdwidthof
from Conv import (
    ConvDirectNHWC,
    direct_null_elementwise_epilogue,
    ConvInfoStatic,
    pack_filter,
)
from ConvUtils import (
    ConvShape,
    get_conv_tile_shape,
    get_direct_conv_micro_kernel_width,
    get_micro_kernel_shape,
)
from math import div_ceil
from memory.buffer import NDBuffer
from runtime.llcl import OwningOutputChainPtr, Runtime
from utils.index import Index, StaticIntTuple
from utils.list import DimList


fn test[
    N: Int,
    H: Int,
    W: Int,
    C: Int,
    R: Int,
    S: Int,
    F: Int,
    pad_h: StaticIntTuple[2],
    pad_w: StaticIntTuple[2],
    stride: StaticIntTuple[2],
    dilation: StaticIntTuple[2],
](rt: Runtime):
    # Skip architectures other than avx512 for now.
    # TODO: tune on other architectures and enable testing.
    @parameter
    if not has_avx512f():
        print("Succeed")
        return

    # Output Shape.
    # fmt: off
    alias HO = (H + pad_h[0] + pad_h[1] - dilation[1] * (R - 1) - 1) // stride[0] + 1
    alias WO = (W + pad_w[0] + pad_w[1] - dilation[0] * (S - 1) - 1) // stride[1] + 1
    # fmt: on
    alias type = DType.float32
    alias simd_size = simdwidthof[type]()
    alias num_groups = 1

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
        stride: stride,
        dilation: dilation,
        pad_h: pad_h,
        pad_w: pad_w,
        num_groups: num_groups,
    }

    let input_ptr = DTypePointer[type].alloc(N * H * W * C)
    let filter_ptr = DTypePointer[type].alloc(R * S * C * F)

    # output from conv w/ dynamic and static shapes.
    let output_ptr_static = DTypePointer[type].alloc(N * HO * WO * F)
    let output_ptr_dynamic = DTypePointer[type].alloc(N * HO * WO * F)

    rand[type](input_ptr, N * H * W * C)
    rand[type](filter_ptr, R * S * C * F)

    let input = NDBuffer[4, DimList(N, H, W, C), type](input_ptr)
    let filter = NDBuffer[4, DimList.create_unknown[4](), type](
        filter_ptr, Index(R, S, C, F)
    )
    let output_static = NDBuffer[4, DimList(N, HO, WO, F), type](
        output_ptr_static
    )
    let output_dynamic = NDBuffer[4, DimList.create_unknown[4](), type](
        output_ptr_dynamic, Index(N, HO, WO, F)
    )

    # Pre-packed filter for dynamic shapes.
    alias micro_kernel_width_default = get_direct_conv_micro_kernel_width()
    alias micro_kernel_f_size_default = micro_kernel_width_default * simd_size
    let rounded_F_dynamic = div_ceil(
        F, micro_kernel_f_size_default
    ) * micro_kernel_f_size_default
    let packed_filter_ptr_dynamic = DTypePointer[type].alloc(
        R * S * C * rounded_F_dynamic
    )
    let packed_filter_dynamic = NDBuffer[5, DimList.create_unknown[5](), type](
        packed_filter_ptr_dynamic,
        Index(
            div_ceil(F, micro_kernel_f_size_default),
            R,
            S,
            C,
            micro_kernel_f_size_default,
        ),
    )

    pack_filter[type](filter, packed_filter_dynamic, num_groups)

    # Conv attributes.
    alias conv_attr_dynamic = ConvInfoStatic.create_unknown()

    let chain0 = OwningOutputChainPtr(rt)
    ConvDirectNHWC[
        5,
        DimList.create_unknown[4](),  # input shape
        DimList.create_unknown[5](),  # filter shape
        DimList.create_unknown[4](),  # output shape
        type,  # input type
        type,  # filter type
        type,  # output type
        True,
        conv_attr_dynamic,
        False,
    ].run(
        output_dynamic,
        rebind[NDBuffer[4, DimList.create_unknown[4](), type]](input),
        packed_filter_dynamic,
        conv_shape,
        chain0.borrow(),
    )
    chain0.wait()

    alias conv_attr_static = ConvInfoStatic(
        DimList(pad_h[0], pad_h[1]),
        DimList(pad_w[0], pad_w[1]),
        DimList(stride[0], stride[1]),
        DimList(dilation[0], dilation[1]),
        Dim(num_groups),
    )

    alias micro_kernel_shape = get_micro_kernel_shape[
        WO, F, conv_attr_static, simd_size
    ]()
    alias micro_kernel_f_size = micro_kernel_shape[1] * simd_size
    alias num_f_micro_tiles = div_ceil(F, micro_kernel_f_size)
    alias rounded_F_static = num_f_micro_tiles * micro_kernel_f_size
    alias packed_filter_shape = DimList(
        num_f_micro_tiles, R, S, C, micro_kernel_f_size
    )
    let packed_filter_ptr_static = DTypePointer[type].alloc(
        R * S * C * rounded_F_static
    )
    let packed_filter_static = NDBuffer[5, packed_filter_shape, type](
        packed_filter_ptr_static
    )

    pack_filter[type, simd_size, micro_kernel_f_size](
        filter,
        rebind[NDBuffer[5, DimList.create_unknown[5](), type]](
            packed_filter_static
        ),
        num_groups,
    )

    let chain1 = OwningOutputChainPtr(rt)
    ConvDirectNHWC[
        5,
        DimList(N, H, W, C),
        packed_filter_shape,
        DimList(N, HO, WO, F),
        type,  # input type
        type,  # filter type
        type,  # output type
        True,
        conv_attr_static,
        False,
    ].run(
        output_static,
        input,
        packed_filter_static,
        conv_shape,
        chain1.borrow(),
    )
    chain1.wait()

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
                        1e-4,  # absolute error tolerance
                        1e-5,  # relative error tolerance
                    ):
                        let expected = output_dynamic[n, ho, wo, f]
                        let actual = output_static[n, ho, wo, f]
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


fn main():
    with Runtime() as rt:
        test[
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
        ](rt)

        # Each test will build a specialization of the conv kernel.
        # Disable the following tests for now to monitor build time.

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
        # ](rt)

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
        # ](rt)

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
        # ](rt)

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
        # ](rt)

        # test[
        #     1,  # N
        #     14,  # H
        #     14,  # W
        #     256,  # C
        #     3,  # R
        #     3,  # S
        #     256,  # F
        #     Index(1, 1),  # stride
        #     Index(1, 1),  # dilation
        #     Index(1, 1),  # pad_h
        #     Index(1, 1),  # pad_w
        # ](rt)

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
        # ](rt)

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
        # ](rt)
