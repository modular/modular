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
    coalesce,
    composition,
    complement,
    logical_divide,
    logical_product,
    zipped_divide,
    print_layout,
)


# CHECK-LABEL: test_layout_basic
# CHECK: Layout((2, (3, 4)):(1, (2, 6)))
fn test_layout_basic():
    print("== test_layout_basic")
    let shape = IntTuple(2, IntTuple(3, IntTuple(4)))
    let stride = IntTuple(1, IntTuple(2, IntTuple(6)))
    let layout = Layout(shape, stride)
    print(layout)


# CHECK-LABEL: test_coalesce
# CHECK: Layout((2, 6):(1, 2))
fn test_coalesce():
    print("== test_coalesce")
    let layout = Layout(
        IntTuple(2, IntTuple(1, 6)), IntTuple(1, IntTuple(6, 2))
    )
    print(coalesce(layout))


# CHECK-LABEL: test_composition
# CHECK: Layout((5, 4):(8, 2))
# CHECK: Layout((5, 8):(16, 80))
fn test_composition() raises:
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
fn test_complement() raises:
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
fn test_logcial_divide() raises:
    print("== test_logcial_divide")
    print(
        logical_divide(
            Layout(IntTuple(4, 2, 3), IntTuple(2, 1, 8)), Layout(4, 2)
        )
    )
    var tiler = DynamicVector[Layout]()
    tiler.append(Layout(3, 3))
    tiler.append(Layout(IntTuple(2, 4), IntTuple(1, 8)))
    print(
        logical_divide(
            Layout(IntTuple(9, IntTuple(4, 8)), IntTuple(59, IntTuple(13, 1))),
            tiler,
        )
    )


# CHECK-LABEL: test_logical_product
# CHECK: Layout((2, 2, (2, 3)):(4, 1, (2, 8)))
# CHECK: Layout(((2, 3), (5, 4)):((5, 10), (1, 30)))
fn test_logical_product() raises:
    print("== test_logical_product")
    print(logical_product(Layout(IntTuple(2, 2), IntTuple(4, 1)), Layout(6, 1)))
    var tiler = DynamicVector[Layout]()
    tiler.append(Layout(3, 5))
    tiler.append(Layout(4, 6))
    print(logical_product(Layout(IntTuple(2, 5), IntTuple(5, 1)), tiler))


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
fn test_print_layout() raises:
    print("== test_print_layout")
    print_layout(Layout(IntTuple(2, 2), IntTuple(1, 2)))
    print_layout(
        Layout(
            IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
            IntTuple(IntTuple(2, 8), IntTuple(1, 4)),
        )
    )


# CHECK-LABEL: test_zipped_divide
fn test_zipped_divide() raises:
    print("== test_zipped_divide")
    # CHECK: Layout((2, (2, 4)):(4, (8, 1)))
    let layout_4x4_row_major = Layout(IntTuple(4, 4), IntTuple(4, 1))
    let tile_layout = Layout(2, 1)
    print(zipped_divide(layout_4x4_row_major, tile_layout))
    var tiler = DynamicVector[Layout]()
    tiler.append(tile_layout)
    tiler.append(tile_layout)
    # CHECK: Layout(((2, 2), (2, 2)):((4, 1), (8, 2)))
    print(
        zipped_divide(
            layout_4x4_row_major,
            tiler,
        )
    )


fn main() raises:
    test_layout_basic()
    test_coalesce()
    test_composition()
    test_complement()
    test_logcial_divide()
    test_logical_product()
    test_print_layout()
    test_zipped_divide()
