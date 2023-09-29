# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory.buffer import NDBuffer
from runtime.llcl import Runtime, OutputChainPtr, OwningOutputChainPtr
from ConvTranspose import AutoPadMode, convtranspose


# CHECK-LABEL: test_convtranspose_pads
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 1.0 ,1.0 ,3.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 7.0 ,4.0 ,9.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
# CHECK: 13.0 ,7.0 ,15.0 ,
fn test_convtranspose_pads():
    print("== test_convtranspose_pads")
    alias rank = 4
    alias type = DType.float32

    let input = NDBuffer[
        rank,
        DimList(1, 1, 3, 3),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 0, 2)] = 2

    input[StaticIntTuple[rank](0, 0, 1, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 4
    input[StaticIntTuple[rank](0, 0, 1, 2)] = 5

    input[StaticIntTuple[rank](0, 0, 2, 0)] = 6
    input[StaticIntTuple[rank](0, 0, 2, 1)] = 7
    input[StaticIntTuple[rank](0, 0, 2, 2)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(1, 2, 3, 3),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 2)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 2, 7, 3),
        type,
    ].stack_allocation()

    for i in range(2):
        for j in range(7):
            for k in range(3):
                output[StaticIntTuple[rank](0, i, j, k)] = 0

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    alias group = 1
    alias input_shape = StaticIntTuple[rank](1, 1, 3, 3)
    alias output_shape = StaticIntTuple[rank](1, 2, 7, 3)
    alias kernel_shape = StaticIntTuple[rank](1, 2, 3, 3)
    alias strides = StaticIntTuple[2](3, 2)
    alias dilations = StaticIntTuple[2](1, 1)
    alias output_padding = StaticIntTuple[2](0, 0)
    alias pads = StaticIntTuple[4](1, 2, 1, 2)
    alias auto_pad = AutoPadMode.NOTSET

    @always_inline
    @parameter
    fn epilogue_fn(index: Int, update_val: SIMD[type, 1]) -> SIMD[type, 1]:
        return 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        convtranspose[
            rank,
            type,
            group,
            input_shape,
            output_shape,
            kernel_shape,
            strides,
            dilations,
            pads,
            output_padding,
            auto_pad,
            epilogue_fn,
        ](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(2):
        for j in range(7):
            for k in range(3):
                print_no_newline(output[0, i, j, k], ",")
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose
# CHECK: 0.0 ,1.0 ,3.0 ,3.0 ,2.0 ,
# CHECK: 3.0 ,8.0 ,15.0 ,12.0 ,7.0 ,
# CHECK: 9.0 ,21.0 ,36.0 ,27.0 ,15.0 ,
# CHECK: 9.0 ,20.0 ,33.0 ,24.0 ,13.0 ,
# CHECK: 6.0 ,13.0 ,21.0 ,15.0 ,8.0 ,
# CHECK: 0.0 ,1.0 ,3.0 ,3.0 ,2.0 ,
# CHECK: 3.0 ,8.0 ,15.0 ,12.0 ,7.0 ,
# CHECK: 9.0 ,21.0 ,36.0 ,27.0 ,15.0 ,
# CHECK: 9.0 ,20.0 ,33.0 ,24.0 ,13.0 ,
# CHECK: 6.0 ,13.0 ,21.0 ,15.0 ,8.0 ,
fn test_convtranspose():
    print("== test_convtranspose")
    alias rank = 4
    alias type = DType.float32

    let input = NDBuffer[
        rank,
        DimList(1, 1, 3, 3),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 0, 2)] = 2

    input[StaticIntTuple[rank](0, 0, 1, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 4
    input[StaticIntTuple[rank](0, 0, 1, 2)] = 5

    input[StaticIntTuple[rank](0, 0, 2, 0)] = 6
    input[StaticIntTuple[rank](0, 0, 2, 1)] = 7
    input[StaticIntTuple[rank](0, 0, 2, 2)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(1, 2, 3, 3),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 2)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 2, 5, 5),
        type,
    ].stack_allocation()

    for i in range(1):
        for j in range(2):
            for k in range(5):
                for l in range(5):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    alias group = 1
    alias input_shape = StaticIntTuple[rank](1, 1, 3, 3)
    alias output_shape = StaticIntTuple[rank](1, 2, 5, 5)
    alias kernel_shape = StaticIntTuple[rank](1, 2, 3, 3)
    alias strides = StaticIntTuple[2](1, 1)
    alias dilations = StaticIntTuple[2](1, 1)
    alias output_padding = StaticIntTuple[2](0, 0)
    alias pads = StaticIntTuple[4](0, 0, 0, 0)
    alias auto_pad = AutoPadMode.NOTSET

    @always_inline
    @parameter
    fn epilogue_fn(index: Int, update_val: SIMD[type, 1]) -> SIMD[type, 1]:
        return 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        convtranspose[
            rank,
            type,
            group,
            input_shape,
            output_shape,
            kernel_shape,
            strides,
            dilations,
            pads,
            output_padding,
            auto_pad,
            epilogue_fn,
        ](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for j in range(2):
            for k in range(5):
                for l in range(5):
                    print_no_newline(output[i, j, k, l], ",")
                print()
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose_dilation
# CHECK: 21.0 ,56.0 ,13.0 ,16.0 ,2.0 ,
# CHECK: 63.0 ,35.0 ,67.0 ,10.0 ,14.0 ,
# CHECK: 24.0 ,22.0 ,76.0 ,76.0 ,21.0 ,
# CHECK: 9.0 ,5.0 ,88.0 ,45.0 ,63.0 ,
# CHECK: 3.0 ,2.0 ,33.0 ,18.0 ,54.0 ,
fn test_convtranspose_dilation():
    print("== test_convtranspose_dilation")
    alias rank = 4
    alias type = DType.float32

    let input = NDBuffer[
        rank,
        DimList(1, 1, 3, 3),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 8
    input[StaticIntTuple[rank](0, 0, 0, 2)] = 1

    input[StaticIntTuple[rank](0, 0, 1, 0)] = 9
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 5
    input[StaticIntTuple[rank](0, 0, 1, 2)] = 7

    input[StaticIntTuple[rank](0, 0, 2, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 2, 1)] = 2
    input[StaticIntTuple[rank](0, 0, 2, 2)] = 6

    let kernel = NDBuffer[
        rank,
        DimList(1, 1, 2, 2),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 7
    kernel[StaticIntTuple[rank](0, 0, 0, 1)] = 2

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 1)] = 9

    let output = NDBuffer[
        rank,
        DimList(1, 1, 5, 5),
        type,
    ].stack_allocation()

    for i in range(1):
        for j in range(1):
            for k in range(5):
                for l in range(5):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    alias group = 1
    alias input_shape = StaticIntTuple[rank](1, 1, 3, 3)
    alias output_shape = StaticIntTuple[rank](1, 1, 5, 5)
    alias kernel_shape = StaticIntTuple[rank](1, 1, 2, 2)
    alias strides = StaticIntTuple[2](1, 1)
    alias dilations = StaticIntTuple[2](2, 2)
    alias output_padding = StaticIntTuple[2](0, 0)
    alias pads = StaticIntTuple[4](0, 0, 0, 0)
    alias auto_pad = AutoPadMode.NOTSET

    @always_inline
    @parameter
    fn epilogue_fn(index: Int, update_val: SIMD[type, 1]) -> SIMD[type, 1]:
        return 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        convtranspose[
            rank,
            type,
            group,
            input_shape,
            output_shape,
            kernel_shape,
            strides,
            dilations,
            pads,
            output_padding,
            auto_pad,
            epilogue_fn,
        ](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for j in range(1):
            for k in range(5):
                for l in range(5):
                    print_no_newline(output[i, j, k, l], ",")
                print()
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose_attributes
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,1.0 ,1.0 ,3.0 ,2.0 ,2.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 3.0 ,3.0 ,7.0 ,4.0 ,9.0 ,5.0 ,5.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 6.0 ,6.0 ,13.0 ,7.0 ,15.0 ,8.0 ,8.0 ,0.0 ,
# CHECK: 0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,0.0 ,
fn test_convtranspose_attributes():
    print("== test_convtranspose_attributes")
    alias rank = 4
    alias type = DType.float32

    let input = NDBuffer[
        rank,
        DimList(1, 1, 3, 3),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 0, 2)] = 2

    input[StaticIntTuple[rank](0, 0, 1, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 4
    input[StaticIntTuple[rank](0, 0, 1, 2)] = 5

    input[StaticIntTuple[rank](0, 0, 2, 0)] = 6
    input[StaticIntTuple[rank](0, 0, 2, 1)] = 7
    input[StaticIntTuple[rank](0, 0, 2, 2)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(1, 2, 3, 3),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 2)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 2, 10, 8),
        type,
    ].stack_allocation()

    for i in range(1):
        for j in range(2):
            for k in range(10):
                for l in range(8):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    alias group = 1
    alias input_shape = StaticIntTuple[rank](1, 1, 3, 3)
    alias output_shape = StaticIntTuple[rank](1, 2, 10, 8)
    alias kernel_shape = StaticIntTuple[rank](1, 2, 3, 3)
    alias strides = StaticIntTuple[2](3, 2)
    alias dilations = StaticIntTuple[2](1, 1)
    alias output_padding = StaticIntTuple[2](1, 1)
    alias pads = StaticIntTuple[4](0, 0, 0, 0)
    alias auto_pad = AutoPadMode.NOTSET

    @always_inline
    @parameter
    fn epilogue_fn(index: Int, update_val: SIMD[type, 1]) -> SIMD[type, 1]:
        return 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        convtranspose[
            rank,
            type,
            group,
            input_shape,
            output_shape,
            kernel_shape,
            strides,
            dilations,
            pads,
            output_padding,
            auto_pad,
            epilogue_fn,
        ](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for j in range(2):
            for k in range(10):
                for l in range(8):
                    print_no_newline(output[i, j, k, l], ",")
                print()
            print()
        print()
    print()


# CHECK-LABEL: test_convtranspose_bias
# CHECK: 2.0 ,3.0 ,5.0 ,5.0 ,4.0 ,
# CHECK: 5.0 ,10.0 ,17.0 ,14.0 ,9.0 ,
# CHECK: 11.0 ,23.0 ,38.0 ,29.0 ,17.0 ,
# CHECK: 11.0 ,22.0 ,35.0 ,26.0 ,15.0 ,
# CHECK: 8.0 ,15.0 ,23.0 ,17.0 ,10.0 ,
# CHECK: 3.0 ,4.0 ,6.0 ,6.0 ,5.0 ,
# CHECK: 6.0 ,11.0 ,18.0 ,15.0 ,10.0 ,
# CHECK: 12.0 ,24.0 ,39.0 ,30.0 ,18.0 ,
# CHECK: 12.0 ,23.0 ,36.0 ,27.0 ,16.0 ,
# CHECK: 9.0 ,16.0 ,24.0 ,18.0 ,11.0 ,
fn test_convtranspose_bias():
    print("== test_convtranspose_bias")
    alias rank = 4
    alias type = DType.float32

    let input = NDBuffer[
        rank,
        DimList(1, 1, 3, 3),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    input[StaticIntTuple[rank](0, 0, 0, 2)] = 2

    input[StaticIntTuple[rank](0, 0, 1, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 1, 1)] = 4
    input[StaticIntTuple[rank](0, 0, 1, 2)] = 5

    input[StaticIntTuple[rank](0, 0, 2, 0)] = 6
    input[StaticIntTuple[rank](0, 0, 2, 1)] = 7
    input[StaticIntTuple[rank](0, 0, 2, 2)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(1, 2, 3, 3),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 0, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 0, 2, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 2)] = 1

    kernel[StaticIntTuple[rank](0, 1, 2, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 1)] = 1
    kernel[StaticIntTuple[rank](0, 1, 2, 2)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 2, 5, 5),
        type,
    ].stack_allocation()

    for i in range(1):
        for j in range(2):
            for k in range(5):
                for l in range(5):
                    output[StaticIntTuple[rank](i, j, k, l)] = 0

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    bias[StaticIntTuple[1](0)] = 2
    bias[StaticIntTuple[1](1)] = 3

    alias group = 1
    alias input_shape = StaticIntTuple[rank](1, 1, 3, 3)
    alias output_shape = StaticIntTuple[rank](1, 2, 5, 5)
    alias kernel_shape = StaticIntTuple[rank](1, 2, 3, 3)
    alias strides = StaticIntTuple[2](1, 1)
    alias dilations = StaticIntTuple[2](1, 1)
    alias output_padding = StaticIntTuple[2](0, 0)
    alias pads = StaticIntTuple[4](0, 0, 0, 0)
    alias auto_pad = AutoPadMode.NOTSET

    @always_inline
    @parameter
    fn epilogue_fn(index: Int, update_val: SIMD[type, 1]) -> SIMD[type, 1]:
        return bias[index]

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        convtranspose[
            rank,
            type,
            group,
            input_shape,
            output_shape,
            kernel_shape,
            strides,
            dilations,
            pads,
            output_padding,
            auto_pad,
            epilogue_fn,
        ](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for j in range(2):
            for k in range(5):
                for l in range(5):
                    print_no_newline(output[i, j, k, l], ",")
                print()
            print()
        print()
    print()


fn main():
    test_convtranspose_pads()
    test_convtranspose()
    test_convtranspose_dilation()
    test_convtranspose_attributes()
    test_convtranspose_bias()
