# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from layout import *
from layout.layout import (
    Layout,
    MakeLayoutList,
    coalesce,
    complement,
    composition,
    cosize,
    logical_divide,
    logical_product,
    print_layout,
    size,
    zipped_divide,
)
from testing import assert_equal, assert_not_equal


# CHECK-LABEL: test_layout_basic
fn test_layout_basic() raises:
    print("== test_layout_basic")
    alias shape = IntTuple(2, IntTuple(3, IntTuple(4)))
    alias stride = IntTuple(1, IntTuple(2, IntTuple(6)))
    alias layout = Layout(shape, stride)
    assert_equal(
        layout, Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(1, IntTuple(2, 6)))
    )
    assert_equal(Layout.row_major(2, 3), Layout(IntTuple(2, 3), IntTuple(3, 1)))
    assert_equal(Layout.col_major(2, 3), Layout(IntTuple(2, 3), IntTuple(1, 2)))


fn validate_coalesce[layout: Layout]() raises:
    alias layoutR = coalesce(layout)

    # print(layout, "=> ", layoutR)

    assert_equal(size(layoutR), size(layout))

    for i in range(size(layout)):
        assert_equal(layoutR(i), layout(i))


# CHECK-LABEL: test_coalesce
fn test_coalesce() raises:
    print("== test_coalesce")

    validate_coalesce[
        Layout(IntTuple(2, IntTuple(1, 6)), IntTuple(1, IntTuple(6, 2)))
    ]()

    validate_coalesce[Layout(1, 0)]()

    validate_coalesce[Layout(1, 0)]()

    validate_coalesce[Layout(IntTuple(2, 4))]()

    validate_coalesce[Layout(IntTuple(2, 4, 6))]()

    validate_coalesce[Layout(IntTuple(2, 4, 6), IntTuple(1, 6, 2))]()

    validate_coalesce[Layout(IntTuple(2, 1, 6), IntTuple(1, 7, 2))]()

    validate_coalesce[Layout(IntTuple(2, 1, 6), IntTuple(4, 7, 8))]()

    validate_coalesce[Layout(IntTuple(2, IntTuple(4, 6)))]()

    validate_coalesce[Layout(IntTuple(2, 4), IntTuple(4, 1))]()

    validate_coalesce[Layout(IntTuple(2, 4, 6), IntTuple(24, 6, 1))]()

    validate_coalesce[Layout(IntTuple(2, 1, 3), IntTuple(2, 4, 4))]()

    validate_coalesce[
        Layout(
            IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
            IntTuple(IntTuple(1, 4), IntTuple(8, 32)),
        )
    ]()


fn validate_composition[layoutA: Layout, layoutB: Layout]() raises:
    alias layoutR = composition(layoutA, layoutB)

    # print(layoutA, "o", layoutB, "=>", layoutR)

    # True post-condition: Every coordinate c of layoutB with L1D(c) < size(layoutR) is a coordinate of layoutR.

    # Test that R(c) = A(B(c)) for all coordinates c in layoutR
    for i in range(size(layoutR)):
        assert_equal(layoutR(i), layoutA(layoutB(i)))


# CHECK-LABEL: test_composition
fn test_composition() raises:
    print("== test_composition")

    validate_composition[Layout(1, 0), Layout(1, 0)]()

    validate_composition[Layout(1, 0), Layout(1, 1)]()

    validate_composition[Layout(1, 1), Layout(1, 0)]()

    validate_composition[Layout(1, 1), Layout(1, 1)]()

    validate_composition[Layout(IntTuple(4)), Layout(IntTuple(4))]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(2)), Layout(IntTuple(4))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(4), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(0)), Layout(IntTuple(4))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(4), IntTuple(0))
    ]()

    validate_composition[
        Layout(IntTuple(1), IntTuple(0)), Layout(IntTuple(4))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(1), IntTuple(0))
    ]()

    validate_composition[Layout(IntTuple(4)), Layout(IntTuple(2))]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(2)), Layout(IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4)), Layout(IntTuple(2), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4), IntTuple(2)), Layout(IntTuple(2), IntTuple(2))
    ]()

    validate_composition[Layout(IntTuple(12)), Layout(IntTuple(4, 3))]()

    validate_composition[
        Layout(IntTuple(12), IntTuple(2)), Layout(IntTuple(4, 3))
    ]()

    validate_composition[
        Layout(IntTuple(12)), Layout(IntTuple(4, 3), IntTuple(3, 1))
    ]()

    validate_composition[
        Layout(IntTuple(12), IntTuple(2)),
        Layout(IntTuple(4, 3), IntTuple(3, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(12)), Layout(IntTuple(2, 3), IntTuple(2, 4))
    ]()

    validate_composition[Layout(IntTuple(4, 3)), Layout(IntTuple(4, 3))]()

    validate_composition[Layout(IntTuple(4, 3)), Layout(IntTuple(12))]()

    validate_composition[
        Layout(IntTuple(4, 3)), Layout(IntTuple(6), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3)), Layout(IntTuple(6, 2), IntTuple(2, 1))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)), Layout(IntTuple(4, 3))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)), Layout(IntTuple(12))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)), Layout(IntTuple(6), IntTuple(2))
    ]()

    validate_composition[
        Layout(IntTuple(4, 3), IntTuple(3, 1)),
        Layout(IntTuple(6, 2), IntTuple(2, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(8, 8)),
        Layout(
            IntTuple(IntTuple(2, 2, 2), IntTuple(2, 2, 2)),
            IntTuple(IntTuple(1, 16, 4), IntTuple(8, 2, 32)),
        ),
    ]()

    validate_composition[
        Layout(IntTuple(8, 8), IntTuple(8, 1)),
        Layout(
            IntTuple(IntTuple(2, 2, 2), IntTuple(2, 2, 2)),
            IntTuple(IntTuple(1, 16, 4), IntTuple(8, 2, 32)),
        ),
    ]()

    validate_composition[
        Layout(
            IntTuple(IntTuple(2, 2, 2), IntTuple(2, 2, 2)),
            IntTuple(IntTuple(1, 16, 4), IntTuple(8, 2, 32)),
        ),
        Layout(8, 4),
    ]()

    validate_composition[
        Layout(IntTuple(IntTuple(4, 2)), IntTuple(IntTuple(1, 16))),
        Layout(IntTuple(4, 2), IntTuple(2, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(2, 2), IntTuple(2, 1)),
        Layout(IntTuple(2, 2), IntTuple(2, 1)),
    ]()

    validate_composition[
        Layout(IntTuple(4, 8, 2)), Layout(IntTuple(2, 2, 2), IntTuple(2, 8, 1))
    ]()

    validate_composition[
        Layout(IntTuple(4, 8, 2), IntTuple(2, 8, 1)),
        Layout(IntTuple(2, 2, 2), IntTuple(1, 8, 2)),
    ]()

    validate_composition[
        Layout(IntTuple(4, 8, 2), IntTuple(2, 8, 1)),
        Layout(IntTuple(4, 2, 2), IntTuple(2, 8, 1)),
    ]()


fn validate_complement[layout: Layout]() raises:
    alias layoutR = complement(layout)

    # print(layout, " => ", layoutR)

    # Post-condition: test disjointness of the codomains
    for a in range(size(layout)):
        for b in range(size(layoutR)):
            assert_equal(
                (layout(a) != layoutR(b))
                or (layout(a) == 0 and layoutR(b) == 0),
                True,
            )


# CHECK-LABEL: test_complement
fn test_complement() raises:
    print("== test_complement")
    alias c0 = complement(Layout(4, 1), 24)
    assert_equal(c0, "(6:4)")
    assert_equal(complement(Layout(6, 4), 24), "(4:1)")
    assert_equal(
        complement(Layout(IntTuple(4, 6), IntTuple(1, 4)), 24), "(1:0)"
    )
    assert_equal(complement(Layout(4, 2), 24), "((2, 3):(1, 8))")
    assert_equal(
        complement(Layout(IntTuple(2, 4), IntTuple(1, 6)), 24), "(3:2)"
    )
    assert_equal(
        complement(Layout(IntTuple(2, 2), IntTuple(1, 6)), 24),
        "((3, 2):(2, 12))",
    )

    var test = Layout(1, 0)
    validate_complement[Layout(1, 0)]()

    validate_complement[Layout(1, 1)]()

    test = Layout(4, 0)
    validate_complement[Layout(4, 0)]()

    validate_complement[Layout(IntTuple(2, 4), IntTuple(1, 2))]()

    validate_complement[Layout(IntTuple(2, 3), IntTuple(1, 2))]()

    validate_complement[Layout(IntTuple(2, 4), IntTuple(1, 4))]()

    validate_complement[Layout(IntTuple(2, 4, 8), IntTuple(8, 1, 64))]()

    validate_complement[
        Layout(
            IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
            IntTuple(IntTuple(1, 4), IntTuple(8, 32)),
        )
    ]()

    validate_complement[
        Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(3, IntTuple(1, 6)))
    ]()

    validate_complement[Layout(IntTuple(4, 6), IntTuple(1, 6))]()

    validate_complement[Layout(IntTuple(4, 10), IntTuple(1, 10))]()


# CHECK-LABEL: test_logcial_divide
fn test_logcial_divide() raises:
    print("== test_logcial_divide")
    var ld0 = logical_divide(
        Layout(IntTuple(4, 2, 3), IntTuple(2, 1, 8)), Layout(4, 2)
    )
    assert_equal(ld0, "(((2, 2), (2, 3)):((4, 1), (2, 8)))")
    assert_equal(
        logical_divide(
            Layout(IntTuple(9, IntTuple(4, 8)), IntTuple(59, IntTuple(13, 1))),
            MakeLayoutList(
                Layout(3, 3), Layout(IntTuple(2, 4), IntTuple(1, 8))
            ),
        ),
        "(((3, 3), ((2, 4), (2, 2))):((177, 59), ((13, 2), (26, 1))))",
    )


# CHECK-LABEL: test_logical_product
fn test_logical_product() raises:
    print("== test_logical_product")
    var lp0 = logical_product(
        Layout(IntTuple(2, 2), IntTuple(4, 1)), Layout(6, 1)
    )
    assert_equal(lp0, "(((2, 2), (2, 3)):((4, 1), (2, 8)))")
    assert_equal(
        logical_product(
            Layout(IntTuple(2, 5), IntTuple(5, 1)),
            MakeLayoutList(Layout(3, 5), Layout(4, 6)),
        ),
        "(((2, 3), (5, 4)):((5, 10), (1, 30)))",
    )


# CHECK-LABEL: test_print_layout
# CHECK: ((2, 2):(1, 2))
# CHECK:       0   1
# CHECK:     +---+---+
# CHECK:  0  | 0 | 2 |
# CHECK:     +---+---+
# CHECK:  1  | 1 | 3 |
# CHECK:     +---+---+
# CHECK: (((2, 2), (2, 2)):((2, 8), (1, 4)))
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
    alias l0 = Layout(IntTuple(2, 2), IntTuple(1, 2))
    alias l1 = Layout(
        IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
        IntTuple(IntTuple(2, 8), IntTuple(1, 4)),
    )
    print_layout(l0)
    print_layout(l1)


# CHECK-LABEL: test_zipped_divide
fn test_zipped_divide() raises:
    print("== test_zipped_divide")
    alias layout_4x4_row_major = Layout(IntTuple(4, 4), IntTuple(4, 1))
    assert_equal(
        zipped_divide(layout_4x4_row_major, Layout(2, 1)),
        "((2, (2, 4)):(4, (8, 1)))",
    )
    var zd0 = zipped_divide(
        layout_4x4_row_major,
        MakeLayoutList(Layout(2, 1), Layout(2, 1)),
    )
    assert_equal(zd0, "(((2, 2), (2, 2)):((4, 1), (8, 2)))")


def main():
    test_layout_basic()
    test_coalesce()
    test_composition()
    test_complement()
    test_logcial_divide()
    test_logical_product()
    test_print_layout()
    test_zipped_divide()
