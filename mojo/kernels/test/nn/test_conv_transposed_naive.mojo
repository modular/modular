# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.conv_transpose import conv_transpose_naive

from utils.index import Index


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
    alias type = DType.float32

    alias input_shape = DimList(1, 1, 3, 3, 1)
    var input_stack = InlineArray[Scalar[type], Int(input_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[type, 5, _, input_shape](input_stack)
    for i in range(9):
        input.data[i] = i

    alias filter_shape = DimList(1, 3, 3, 2, 1)
    var filter_stack = InlineArray[Scalar[type], Int(filter_shape.product())](
        uninitialized=True
    )
    var filter = NDBuffer[type, 5, _, filter_shape](filter_stack)
    filter.fill(1.0)

    alias output_shape = DimList(1, 1, 7, 3, 2)
    var output_stack = InlineArray[Scalar[type], Int(output_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, 5, _, output_shape](output_stack)

    var stride = Index(1, 3, 2)
    var dilation = Index(1, 1, 1)
    var pad_d = Index(0, 0)
    var pad_h = Index(1, 1)
    var pad_w = Index(2, 2)

    conv_transpose_naive[type](
        output.make_dims_unknown(),
        input.make_dims_unknown(),
        filter.make_dims_unknown(),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for k in range(2):
        for i in range(7):
            for j in range(3):
                print(output[0, 0, i, j, k], ",", end="")
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
    alias type = DType.float32

    alias input_shape = DimList(1, 1, 3, 3, 1)
    var input_stack = InlineArray[Scalar[type], Int(input_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[type, 5, _, input_shape](input_stack)
    for i in range(9):
        input.data[i] = i

    alias filter_shape = DimList(1, 3, 3, 2, 1)
    var filter_stack = InlineArray[Scalar[type], Int(filter_shape.product())](
        uninitialized=True
    )
    var filter = NDBuffer[type, 5, _, DimList(1, 3, 3, 2, 1)](filter_stack)
    filter.fill(1.0)

    alias output_shape = DimList(1, 1, 5, 5, 2)
    var output_stack = InlineArray[Scalar[type], Int(output_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, 5, _, output_shape](output_stack)

    var stride = Index(1, 1, 1)
    var dilation = Index(1, 1, 1)
    var pad_d = Index(0, 0)
    var pad_h = Index(0, 0)
    var pad_w = Index(0, 0)

    conv_transpose_naive[type](
        output.make_dims_unknown(),
        input.make_dims_unknown(),
        filter.make_dims_unknown(),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for l in range(2):
        for j in range(5):
            for k in range(5):
                print(output[0, 0, j, k, l], ",", end="")
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
    alias type = DType.float32

    alias input_shape = DimList(1, 1, 3, 3, 1)
    var input_stack = InlineArray[Scalar[type], Int(input_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[type, 5, _, input_shape](input_stack)
    input.data[0] = 3
    input.data[1] = 8
    input.data[2] = 1
    input.data[3] = 9
    input.data[4] = 5
    input.data[5] = 7
    input.data[6] = 3
    input.data[7] = 2
    input.data[8] = 6

    alias filter_shape = DimList(1, 2, 2, 1, 1)
    var filter_stack = InlineArray[Scalar[type], Int(filter_shape.product())](
        uninitialized=True
    )
    var filter = NDBuffer[type, 5, _, filter_shape](filter_stack)
    filter.data[0] = 7
    filter.data[1] = 2
    filter.data[2] = 1
    filter.data[3] = 9

    alias output_shape = DimList(1, 1, 5, 5, 1)
    var output_stack = InlineArray[Scalar[type], Int(output_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, 5, _, output_shape](output_stack)
    var stride = Index(1, 1, 1)
    var dilation = Index(1, 2, 2)
    var pad_d = Index(0, 0)
    var pad_h = Index(0, 0)
    var pad_w = Index(0, 0)

    conv_transpose_naive[type](
        output.make_dims_unknown(),
        input.make_dims_unknown(),
        filter.make_dims_unknown(),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for l in range(1):
        for j in range(5):
            for k in range(5):
                print(output[0, 0, j, k, l], ",", end="")
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
    alias type = DType.float32

    alias input_shape = DimList(1, 1, 3, 3, 1)
    var input_stack = InlineArray[Scalar[type], Int(input_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[type, 5, _, input_shape](input_stack)
    for i in range(9):
        input.data[i] = i

    alias filter_shape = DimList(1, 3, 3, 2, 1)
    var filter_stack = InlineArray[Scalar[type], Int(filter_shape.product())](
        uninitialized=True
    )
    var filter = NDBuffer[type, 5, _, filter_shape](filter_stack)
    filter.fill(1.0)

    alias output_shape = DimList(1, 1, 10, 8, 2)
    var output_stack = InlineArray[Scalar[type], Int(output_shape.product())](
        uninitialized=True
    )
    var output = NDBuffer[type, 5, _, output_shape](output_stack)

    var stride = Index(1, 3, 2)
    var dilation = Index(1, 1, 1)
    var pad_d = Index(0, 0)
    var pad_h = Index(0, 0)
    var pad_w = Index(0, 0)

    conv_transpose_naive[type](
        output.make_dims_unknown(),
        input.make_dims_unknown(),
        filter.make_dims_unknown(),
        stride,
        dilation,
        pad_d,
        pad_h,
        pad_w,
    )

    print()
    for l in range(2):
        for j in range(10):
            for k in range(8):
                print(output[0, 0, j, k, l], ",", end="")
            print()
        print()
    print()


fn main():
    test_convtranspose_pads()
    test_convtranspose()
    test_convtranspose_dilation()
    test_convtranspose_attributes()
