# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo %s | FileCheck %s

from kernel_utils.int_tuple import IntTuple
from kernel_utils.layout import (
    Layout,
    LayoutList,
    coalesce,
    complement,
    composition,
    logical_divide,
    logical_product,
    print_layout,
    zipped_divide,
)


# CHECK-LABEL: test_layout_basic
# CHECK: Layout((2, (3, 4)):(1, (2, 6)))
fn test_layout_basic():
    print("== test_layout_basic")
    var shape = IntTuple(2, IntTuple(3, IntTuple(4)))
    var stride = IntTuple(1, IntTuple(2, IntTuple(6)))
    var layout = Layout(shape, stride)
    print(layout)


# CHECK-LABEL: test_coalesce
# CHECK: Layout((2, 6):(1, 2))
fn test_coalesce():
    print("== test_coalesce")
    var layout = Layout(
        IntTuple(2, IntTuple(1, 6)), IntTuple(1, IntTuple(6, 2))
    )
    print(coalesce(layout))


# CHECK-LABEL: test_composition
# CHECK: Layout((5, 4):(8, 2))
# CHECK: Layout((5, 8):(16, 80))
fn test_composition():
    print("== test_composition")
    print(composition(Layout(20, 2), Layout(IntTuple(5, 4), IntTuple(4, 1))))
    print(
        composition(
            Layout(IntTuple(10, 2), IntTuple(16, 4)),
            Layout(IntTuple(5, 4), IntTuple(1, 5)),
        )
    )


# CHECK-LABEL: test_complement
# CHECK: Layout(6:4)
# CHECK: Layout(4:1)
# CHECK: Layout(1:0)
# CHECK: Layout((2, 3):(1, 8))
# CHECK: Layout(3:2)
# CHECK: Layout((3, 2):(2, 12))
fn test_complement():
    print("== test_complement")
    print(complement(Layout(4, 1), 24))
    print(complement(Layout(6, 4), 24))
    print(complement(Layout(IntTuple(4, 6), IntTuple(1, 4)), 24))
    print(complement(Layout(4, 2), 24))
    print(complement(Layout(IntTuple(2, 4), IntTuple(1, 6)), 24))
    print(complement(Layout(IntTuple(2, 2), IntTuple(1, 6)), 24))


# CHECK-LABEL: test_logcial_divide
# CHECK: Layout(((2, 2), (2, 3)):((4, 1), (2, 8)))
# CHECK: Layout(((3, 3), (2, 4, (2, 2))):((177, 59), (13, 2, (26, 1))))
fn test_logcial_divide():
    print("== test_logcial_divide")
    print(
        logical_divide(
            Layout(IntTuple(4, 2, 3), IntTuple(2, 1, 8)), Layout(4, 2)
        )
    )
    print(
        logical_divide(
            Layout(IntTuple(9, IntTuple(4, 8)), IntTuple(59, IntTuple(13, 1))),
            LayoutList(Layout(3, 3), Layout(IntTuple(2, 4), IntTuple(1, 8))),
        )
    )


# CHECK-LABEL: test_logical_product
# CHECK: Layout((2, 2, (2, 3)):(4, 1, (2, 8)))
# CHECK: Layout(((2, 3), (5, 4)):((5, 10), (1, 30)))
fn test_logical_product():
    print("== test_logical_product")
    print(logical_product(Layout(IntTuple(2, 2), IntTuple(4, 1)), Layout(6, 1)))
    print(
        logical_product(
            Layout(IntTuple(2, 5), IntTuple(5, 1)),
            LayoutList(Layout(3, 5), Layout(4, 6)),
        )
    )


# CHECK-LABEL: test_print_layout
# CHECK: Layout((2, 2):(1, 2))
# CHECK:       0   1
# CHECK:     +---+---+
# CHECK:  0  | 0 | 2 |
# CHECK:     +---+---+
# CHECK:  1  | 1 | 3 |
# CHECK:     +---+---+
# CHECK: Layout(((2, 2), (2, 2)):((2, 8), (1, 4)))
# CHECK:        0    1    2    3
# CHECK:     +----+----+----+----+
# CHECK:  0  |  0 |  1 |  4 |  5 |
# CHECK:     +----+----+----+----+
# CHECK:  1  |  2 |  3 |  6 |  7 |
# CHECK:     +----+----+----+----+
# CHECK:  2  |  8 |  9 | 12 | 13 |
# CHECK:     +----+----+----+----+
# CHECK:  3  | 10 | 11 | 14 | 15 |
# CHECK:     +----+----+----+----+
fn test_print_layout():
    print("== test_print_layout")
    print_layout(Layout(IntTuple(2, 2), IntTuple(1, 2)))
    print_layout(
        Layout(
            IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
            IntTuple(IntTuple(2, 8), IntTuple(1, 4)),
        )
    )


# CHECK-LABEL: test_zipped_divide
fn test_zipped_divide():
    print("== test_zipped_divide")
    # CHECK: Layout((2, (2, 4)):(4, (8, 1)))
    var layout_4x4_row_major = Layout(IntTuple(4, 4), IntTuple(4, 1))
    print(zipped_divide(layout_4x4_row_major, Layout(2, 1)))
    # CHECK: Layout(((2, 2), (2, 2)):((4, 1), (8, 2)))
    print(
        zipped_divide(
            layout_4x4_row_major,
            LayoutList(Layout(2, 1), Layout(2, 1)),
        )
    )


fn main():
    test_layout_basic()
    test_coalesce()
    test_composition()
    test_complement()
    test_logcial_divide()
    test_logical_product()
    test_print_layout()
    test_zipped_divide()
