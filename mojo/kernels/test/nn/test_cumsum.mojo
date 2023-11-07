# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota

from Cumsum import cumsum
from memory.buffer import NDBuffer


# CHECK-LABEL: test_cumsum_1d
# CHECK: 1.0 ,3.0 ,6.0 ,10.0 ,15.0 ,
fn test_cumsum_1d():
    print("== test_cumsum_1d")
    alias exclusive = False
    alias reverse = False
    let axis = 0

    var matrix_data = DTypePointer[DType.float64].alloc(5)
    let matrix = NDBuffer[1, DimList(5), DType.float64](matrix_data, DimList(5))

    iota[DType.float64](matrix_data, 5, 1)

    let cumsum_matrix = NDBuffer[
        1, DimList(5), DType.float64
    ].stack_allocation()

    cumsum[1, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(5):
        print_no_newline(cumsum_matrix[i], ",")
    print()

    matrix_data.free()


# CHECK-LABEL: test_cumsum_1d_exclusive
# CHECK: 0.0 ,1.0 ,3.0 ,6.0 ,10.0 ,
fn test_cumsum_1d_exclusive():
    print("== test_cumsum_1d_exclusive")
    alias exclusive = True
    alias reverse = False
    let axis = 0

    var matrix_data = DTypePointer[DType.float64].alloc(5)
    let matrix = NDBuffer[1, DimList(5), DType.float64](matrix_data, DimList(5))

    iota[DType.float64](matrix_data, 5, 1)

    let cumsum_matrix = NDBuffer[
        1, DimList(5), DType.float64
    ].stack_allocation()

    cumsum[1, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(5):
        print_no_newline(cumsum_matrix[i], ",")
    print()

    matrix_data.free()


# CHECK-LABEL: test_cumsum_1d_reverse
# CHECK: 15.0 ,14.0 ,12.0 ,9.0 ,5.0 ,
fn test_cumsum_1d_reverse():
    print("== test_cumsum_1d_reverse")
    alias exclusive = False
    alias reverse = True
    let axis = 0

    var matrix_data = DTypePointer[DType.float64].alloc(5)
    let matrix = NDBuffer[1, DimList(5), DType.float64](matrix_data, DimList(5))

    iota[DType.float64](matrix_data, 5, 1)

    let cumsum_matrix = NDBuffer[
        1, DimList(5), DType.float64
    ].stack_allocation()

    cumsum[1, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(5):
        print_no_newline(cumsum_matrix[i], ",")
    print()

    matrix_data.free()


# CHECK-LABEL: test_cumsum_1d_reverse_exclusive
# CHECK: 14.0 ,12.0 ,9.0 ,5.0 ,0.0 ,
fn test_cumsum_1d_reverse_exclusive():
    print("== test_cumsum_1d_reverse_exclusive")
    alias exclusive = True
    alias reverse = True
    alias axis = 0

    var matrix_data = DTypePointer[DType.float64].alloc(5)
    let matrix = NDBuffer[1, DimList(5), DType.float64](matrix_data, DimList(5))

    iota[DType.float64](matrix_data, 5, 1)

    let cumsum_matrix = NDBuffer[
        1, DimList(5), DType.float64
    ].stack_allocation()

    cumsum[1, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(5):
        print_no_newline(cumsum_matrix[i], ",")
    print()

    matrix_data.free()


# CHECK-LABEL: test_cumsum_2d_axis_0
# CHECK: 1.0 ,2.0 ,3.0 ,5.0 ,7.0 ,9.0 ,
fn test_cumsum_2d_axis_0():
    print("== test_cumsum_2d_axis_0")
    alias exclusive = False
    alias reverse = False
    let axis = 0

    var matrix_data = DTypePointer[DType.float64].alloc(6)
    let matrix = NDBuffer[
        2,
        DimList(2, 3),
        DType.float64,
    ](matrix_data, DimList(2, 3))

    iota[DType.float64](matrix_data, 6, 1)

    let cumsum_matrix = NDBuffer[
        2,
        DimList(2, 3),
        DType.float64,
    ].stack_allocation()

    cumsum[2, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(2):
        for j in range(3):
            print_no_newline(cumsum_matrix[i, j], ",")
    print()

    matrix_data.free()


# CHECK-LABEL: test_cumsum_2d_axis_1
# CHECK: 1.0 ,3.0 ,6.0 ,4.0 ,9.0 ,15.0 ,
fn test_cumsum_2d_axis_1():
    print("== test_cumsum_2d_axis_1")
    alias exclusive = False
    alias reverse = False
    let axis = 1

    var matrix_data = DTypePointer[DType.float64].alloc(6)
    let matrix = NDBuffer[
        2,
        DimList(2, 3),
        DType.float64,
    ](matrix_data, DimList(2, 3))

    iota[DType.float64](matrix_data, 6, 1)

    let cumsum_matrix = NDBuffer[
        2,
        DimList(2, 3),
        DType.float64,
    ].stack_allocation()

    cumsum[2, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(2):
        for j in range(3):
            print_no_newline(cumsum_matrix[i, j], ",")
    print()

    matrix_data.free()


# CHECK-LABEL: test_cumsum_2d_negative_axis
# CHECK: 1.0 ,3.0 ,6.0 ,4.0 ,9.0 ,15.0 ,
fn test_cumsum_2d_negative_axis():
    print("== test_cumsum_2d_negative_axis")
    alias exclusive = False
    alias reverse = False
    let axis = -1

    var matrix_data = DTypePointer[DType.float64].alloc(6)
    let matrix = NDBuffer[
        2,
        DimList(2, 3),
        DType.float64,
    ](matrix_data, DimList(2, 3))

    iota[DType.float64](matrix_data, 6, 1)

    let cumsum_matrix = NDBuffer[
        2,
        DimList(2, 3),
        DType.float64,
    ].stack_allocation()

    cumsum[2, DType.float64, exclusive, reverse](
        cumsum_matrix.make_dims_unknown(), matrix.make_dims_unknown(), axis
    )

    for i in range(2):
        for j in range(3):
            print_no_newline(cumsum_matrix[i, j], ",")
    print()

    matrix_data.free()


fn main():
    test_cumsum_1d()
    test_cumsum_1d_exclusive()
    test_cumsum_1d_reverse()
    test_cumsum_1d_reverse_exclusive()
    test_cumsum_2d_axis_0()
    test_cumsum_2d_axis_1()
    test_cumsum_2d_negative_axis()
