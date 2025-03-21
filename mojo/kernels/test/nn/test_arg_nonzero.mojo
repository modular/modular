# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.arg_nonzero import arg_nonzero, arg_nonzero_shape
from testing import assert_equal

from utils import IndexList


# CHECK-LABEL: test_where_size
def test_where_size():
    print("== test_where_size")
    alias rank = 3
    alias values_shape = DimList(3, 2, 1)
    var values_stack = InlineArray[Float32, Int(values_shape.product())](
        uninitialized=True
    )
    var values = NDBuffer[DType.float32, rank, _, values_shape](values_stack)

    values[IndexList[rank](0, 0, 0)] = 1.0
    values[IndexList[rank](0, 1, 0)] = 2.0
    values[IndexList[rank](1, 0, 0)] = 0.0
    values[IndexList[rank](1, 1, 0)] = 0.0
    values[IndexList[rank](2, 0, 0)] = 0.0
    values[IndexList[rank](2, 1, 0)] = -3.0

    var output_shape = arg_nonzero_shape[DType.float32, rank, True](
        values.make_dims_unknown()
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where_size_bool
def test_where_size_bool():
    print("== test_where_size_bool")
    alias rank = 3
    alias values_shape = DimList(3, 2, 1)
    var values_stack = InlineArray[
        Scalar[DType.bool], Int(values_shape.product())
    ](uninitialized=True)
    var values = NDBuffer[DType.bool, rank, _, values_shape](values_stack)

    values[IndexList[rank](0, 0, 0)] = True
    values[IndexList[rank](0, 1, 0)] = True
    values[IndexList[rank](1, 0, 0)] = False
    values[IndexList[rank](1, 1, 0)] = False
    values[IndexList[rank](2, 0, 0)] = False
    values[IndexList[rank](2, 1, 0)] = True

    var output_shape = arg_nonzero_shape[DType.bool, rank, True](
        values.make_dims_unknown()
    )

    assert_equal(output_shape[0], 3)
    assert_equal(output_shape[1], 3)


# CHECK-LABEL: test_where
def test_where():
    print("== test_where")
    alias rank = 3
    alias values_shape = DimList(3, 2, 1)
    var values_stack = InlineArray[Float32, Int(values_shape.product())](
        uninitialized=True
    )
    var values = NDBuffer[DType.float32, rank, _, values_shape](values_stack)

    values[IndexList[rank](0, 0, 0)] = 1.0
    values[IndexList[rank](0, 1, 0)] = 2.0
    values[IndexList[rank](1, 0, 0)] = 0.0
    values[IndexList[rank](1, 1, 0)] = 0.0
    values[IndexList[rank](2, 0, 0)] = 0.0
    values[IndexList[rank](2, 1, 0)] = -3.0

    var computed_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var computed_outputs = NDBuffer[DType.index, 2, _, DimList(3, 3)](
        computed_stack
    )

    var golden_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var golden_outputs = NDBuffer[
        DType.index,
        2,
        _,
        DimList(3, 3),
    ](golden_stack)

    golden_outputs[IndexList[2](0, 0)] = 0
    golden_outputs[IndexList[2](0, 1)] = 0
    golden_outputs[IndexList[2](0, 2)] = 0
    golden_outputs[IndexList[2](1, 0)] = 0
    golden_outputs[IndexList[2](1, 1)] = 1
    golden_outputs[IndexList[2](1, 2)] = 0
    golden_outputs[IndexList[2](2, 0)] = 2
    golden_outputs[IndexList[2](2, 1)] = 1
    golden_outputs[IndexList[2](2, 2)] = 0

    arg_nonzero(
        values.make_dims_unknown(), computed_outputs.make_dims_unknown()
    )

    for i in range(3):
        for j in range(3):
            assert_equal(computed_outputs[i, j], golden_outputs[i, j])


# CHECK-LABEL: test_where_1d
def test_where_1d():
    print("== test_where_1d")
    alias num_elements = 12
    alias num_indices = 6

    var values_stack = InlineArray[Float32, num_elements](uninitialized=True)
    var values = NDBuffer[
        DType.float32,
        1,
        _,
        DimList(num_elements),
    ](values_stack)

    values[0] = 0.0
    values[1] = 1.0
    values[2] = 0.0
    values[3] = 1.0
    values[4] = 0.0
    values[5] = 1.0
    values[6] = 0.0
    values[7] = 1.0
    values[8] = 0.0
    values[9] = 1.0
    values[10] = 0.0
    values[11] = 1.0

    var computed_stack = InlineArray[Scalar[DType.index], num_indices](
        uninitialized=True
    )
    var computed_outputs = NDBuffer[
        DType.index,
        2,
        _,
        DimList(num_indices, 1),
    ](computed_stack)

    var golden_stack = InlineArray[Scalar[DType.index], num_indices](
        uninitialized=True
    )
    var golden_outputs = NDBuffer[
        DType.index,
        1,
        _,
        DimList(num_indices),
    ](golden_stack)

    golden_outputs[0] = 1
    golden_outputs[1] = 3
    golden_outputs[2] = 5
    golden_outputs[3] = 7
    golden_outputs[4] = 9
    golden_outputs[5] = 11

    arg_nonzero(
        values.make_dims_unknown(), computed_outputs.make_dims_unknown()
    )

    for i in range(num_indices):
        assert_equal(computed_outputs[i, 0], golden_outputs[i])


# CHECK-LABEL: test_where_bool
def test_where_bool():
    print("== test_where_bool")
    alias rank = 3
    alias values_shape = DimList(3, 2, 1)
    var values_stack = InlineArray[
        Scalar[DType.bool], Int(values_shape.product())
    ](uninitialized=True)
    var values = NDBuffer[DType.bool, rank, _, values_shape](values_stack)

    values[IndexList[rank](0, 0, 0)] = True
    values[IndexList[rank](0, 1, 0)] = True
    values[IndexList[rank](1, 0, 0)] = False
    values[IndexList[rank](1, 1, 0)] = False
    values[IndexList[rank](2, 0, 0)] = False
    values[IndexList[rank](2, 1, 0)] = True

    var computed_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var computed_outputs = NDBuffer[DType.index, 2, _, DimList(3, 3)](
        computed_stack
    )

    var golden_stack = InlineArray[Scalar[DType.index], 9](uninitialized=True)
    var golden_outputs = NDBuffer[DType.index, 2, _, DimList(3, 3)](
        golden_stack
    )

    golden_outputs[IndexList[2](0, 0)] = 0
    golden_outputs[IndexList[2](0, 1)] = 0
    golden_outputs[IndexList[2](0, 2)] = 0
    golden_outputs[IndexList[2](1, 0)] = 0
    golden_outputs[IndexList[2](1, 1)] = 1
    golden_outputs[IndexList[2](1, 2)] = 0
    golden_outputs[IndexList[2](2, 0)] = 2
    golden_outputs[IndexList[2](2, 1)] = 1
    golden_outputs[IndexList[2](2, 2)] = 0

    arg_nonzero(
        values.make_dims_unknown(), computed_outputs.make_dims_unknown()
    )

    for i in range(3):
        for j in range(3):
            assert_equal(computed_outputs[i, j], golden_outputs[i, j])


def main():
    test_where_size()
    test_where_size_bool()
    test_where()
    test_where_1d()
    test_where_bool()
