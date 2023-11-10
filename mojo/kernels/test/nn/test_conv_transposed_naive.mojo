# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from ConvTranspose import conv_transpose
from memory.buffer import NDBuffer
from runtime.llcl import OutputChainPtr, OwningOutputChainPtr, Runtime


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
        DimList(1, 3, 3, 1),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    input[StaticIntTuple[rank](0, 0, 2, 0)] = 2

    input[StaticIntTuple[rank](0, 1, 0, 0)] = 3
    input[StaticIntTuple[rank](0, 1, 1, 0)] = 4
    input[StaticIntTuple[rank](0, 1, 2, 0)] = 5

    input[StaticIntTuple[rank](0, 2, 0, 0)] = 6
    input[StaticIntTuple[rank](0, 2, 1, 0)] = 7
    input[StaticIntTuple[rank](0, 2, 2, 0)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(3, 3, 2, 1),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](1, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](2, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](2, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](2, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 2, 1, 0)] = 1

    kernel[StaticIntTuple[rank](1, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](1, 2, 1, 0)] = 1

    kernel[StaticIntTuple[rank](2, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](2, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](2, 2, 1, 0)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 7, 3, 2),
        type,
    ].stack_allocation()

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    let strides = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    strides[0] = 3
    strides[1] = 2

    let dilations = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    dilations[0] = 1
    dilations[1] = 1

    let output_padding = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    output_padding[0] = 0
    output_padding[1] = 0

    let pads = NDBuffer[
        1,
        DimList(4),
        DType.index,
    ].stack_allocation()

    pads[0] = 1
    pads[1] = 2
    pads[2] = 1
    pads[3] = 2

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        conv_transpose[rank, type, DType.index, DType.index, DType.index](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            strides.make_dims_unknown(),
            dilations.make_dims_unknown(),
            pads.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for k in range(2):
        for i in range(7):
            for j in range(3):
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
        DimList(1, 3, 3, 1),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    input[StaticIntTuple[rank](0, 0, 2, 0)] = 2

    input[StaticIntTuple[rank](0, 1, 0, 0)] = 3
    input[StaticIntTuple[rank](0, 1, 1, 0)] = 4
    input[StaticIntTuple[rank](0, 1, 2, 0)] = 5

    input[StaticIntTuple[rank](0, 2, 0, 0)] = 6
    input[StaticIntTuple[rank](0, 2, 1, 0)] = 7
    input[StaticIntTuple[rank](0, 2, 2, 0)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(3, 3, 2, 1),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](1, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](2, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](2, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](2, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 2, 1, 0)] = 1

    kernel[StaticIntTuple[rank](1, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](1, 2, 1, 0)] = 1

    kernel[StaticIntTuple[rank](2, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](2, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](2, 2, 1, 0)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 5, 5, 2),
        type,
    ].stack_allocation()

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    let strides = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    strides[0] = 1
    strides[1] = 1

    let dilations = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    dilations[0] = 1
    dilations[1] = 1

    let output_padding = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    output_padding[0] = 0
    output_padding[1] = 0

    let pads = NDBuffer[
        1,
        DimList(4),
        DType.index,
    ].stack_allocation()

    pads[0] = 0
    pads[1] = 0
    pads[2] = 0
    pads[3] = 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        conv_transpose[rank, type, DType.index, DType.index, DType.index](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            strides.make_dims_unknown(),
            dilations.make_dims_unknown(),
            pads.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for l in range(2):
            for j in range(5):
                for k in range(5):
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
        DimList(1, 3, 3, 1),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 3
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 8
    input[StaticIntTuple[rank](0, 0, 2, 0)] = 1

    input[StaticIntTuple[rank](0, 1, 0, 0)] = 9
    input[StaticIntTuple[rank](0, 1, 1, 0)] = 5
    input[StaticIntTuple[rank](0, 1, 2, 0)] = 7

    input[StaticIntTuple[rank](0, 2, 0, 0)] = 3
    input[StaticIntTuple[rank](0, 2, 1, 0)] = 2
    input[StaticIntTuple[rank](0, 2, 2, 0)] = 6

    let kernel = NDBuffer[
        rank,
        DimList(2, 2, 1, 1),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 7
    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 2

    kernel[StaticIntTuple[rank](1, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 0, 0)] = 9

    let output = NDBuffer[
        rank,
        DimList(1, 5, 5, 1),
        type,
    ].stack_allocation()

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    let strides = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    strides[0] = 1
    strides[1] = 1

    let dilations = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    dilations[0] = 2
    dilations[1] = 2

    let output_padding = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    output_padding[0] = 0
    output_padding[1] = 0

    let pads = NDBuffer[
        1,
        DimList(4),
        DType.index,
    ].stack_allocation()

    pads[0] = 0
    pads[1] = 0
    pads[2] = 0
    pads[3] = 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        conv_transpose[rank, type, DType.index, DType.index, DType.index](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            strides.make_dims_unknown(),
            dilations.make_dims_unknown(),
            pads.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for l in range(1):
            for j in range(5):
                for k in range(5):
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
        DimList(1, 3, 3, 1),
        type,
    ].stack_allocation()

    input[StaticIntTuple[rank](0, 0, 0, 0)] = 0
    input[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    input[StaticIntTuple[rank](0, 0, 2, 0)] = 2

    input[StaticIntTuple[rank](0, 1, 0, 0)] = 3
    input[StaticIntTuple[rank](0, 1, 1, 0)] = 4
    input[StaticIntTuple[rank](0, 1, 2, 0)] = 5

    input[StaticIntTuple[rank](0, 2, 0, 0)] = 6
    input[StaticIntTuple[rank](0, 2, 1, 0)] = 7
    input[StaticIntTuple[rank](0, 2, 2, 0)] = 8

    let kernel = NDBuffer[
        rank,
        DimList(3, 3, 2, 1),
        type,
    ].stack_allocation()

    kernel[StaticIntTuple[rank](0, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](0, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](1, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](1, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](2, 0, 0, 0)] = 1
    kernel[StaticIntTuple[rank](2, 1, 0, 0)] = 1
    kernel[StaticIntTuple[rank](2, 2, 0, 0)] = 1

    kernel[StaticIntTuple[rank](0, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](0, 2, 1, 0)] = 1

    kernel[StaticIntTuple[rank](1, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](1, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](1, 2, 1, 0)] = 1

    kernel[StaticIntTuple[rank](2, 0, 1, 0)] = 1
    kernel[StaticIntTuple[rank](2, 1, 1, 0)] = 1
    kernel[StaticIntTuple[rank](2, 2, 1, 0)] = 1

    let output = NDBuffer[
        rank,
        DimList(1, 10, 8, 2),
        type,
    ].stack_allocation()

    let bias = NDBuffer[
        1,
        DimList(2),
        type,
    ].stack_allocation()

    let strides = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    strides[0] = 3
    strides[1] = 2

    let dilations = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    dilations[0] = 1
    dilations[1] = 1

    let output_padding = NDBuffer[
        1,
        DimList(2),
        DType.index,
    ].stack_allocation()

    output_padding[0] = 1
    output_padding[1] = 1

    let pads = NDBuffer[
        1,
        DimList(4),
        DType.index,
    ].stack_allocation()

    pads[0] = 0
    pads[1] = 0
    pads[2] = 0
    pads[3] = 0

    with Runtime() as rt:
        let out_chain = OwningOutputChainPtr(rt)
        conv_transpose[rank, type, DType.index, DType.index, DType.index](
            output.make_dims_unknown(),
            input.make_dims_unknown(),
            kernel.make_dims_unknown(),
            strides.make_dims_unknown(),
            dilations.make_dims_unknown(),
            pads.make_dims_unknown(),
            out_chain.borrow(),
        )
        out_chain.wait()

    print()
    for i in range(1):
        for l in range(2):
            for j in range(10):
                for k in range(8):
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
