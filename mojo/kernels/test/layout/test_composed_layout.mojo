# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s

from layout import *
from layout.layout import Layout
from layout.swizzle import ComposedLayout, SwizzleEx
from testing import assert_equal, assert_not_equal


fn test_composed_layout() raises:
    print("== test_composed_layout")

    alias layout_a = Layout(IntTuple(6, 2), IntTuple(8, 2))
    alias layout_b = Layout(IntTuple(4, 3), IntTuple(3, 1))

    alias comp_layout = ComposedLayout[Layout, Layout, 0](layout_b, layout_a)

    assert_equal(comp_layout(0), 0)
    assert_equal(comp_layout(1), 24)
    assert_equal(comp_layout(2), 2)
    assert_equal(comp_layout(3), 26)
    assert_equal(comp_layout(4), 8)
    assert_equal(comp_layout(5), 32)
    assert_equal(comp_layout(6), 10)
    assert_equal(comp_layout(7), 34)
    assert_equal(comp_layout(8), 16)
    assert_equal(comp_layout(9), 40)
    assert_equal(comp_layout(10), 18)
    assert_equal(comp_layout(11), 42)


fn test_composed_layout_swizzle() raises:
    print("== test_composed_layout_swizzle")

    var swizzle = SwizzleEx(1, 0, 2)
    var layout = Layout(IntTuple(6, 2), IntTuple(8, 2))

    var comp_layout = ComposedLayout[Layout, SwizzleEx, 0](layout, swizzle)

    assert_equal(comp_layout(0), 0)
    assert_equal(comp_layout(1), 8)
    assert_equal(comp_layout(2), 16)
    assert_equal(comp_layout(3), 24)
    assert_equal(comp_layout(4), 32)
    assert_equal(comp_layout(5), 40)


fn test_composed_layout_swizzle_rt() raises:
    print("== test_composed_layout_swizzle_rt")

    var swizzle = SwizzleEx(1, 0, 2)
    var layout = Layout(IntTuple(6, 2), IntTuple(8, 2))

    var comp_layout = ComposedLayout[Layout, SwizzleEx, 0](layout, swizzle)

    assert_equal(comp_layout(0), 0)
    assert_equal(comp_layout(1), 8)
    assert_equal(comp_layout(2), 16)
    assert_equal(comp_layout(3), 24)
    assert_equal(comp_layout(4), 32)
    assert_equal(comp_layout(5), 40)


fn main():
    try:
        test_composed_layout()
        test_composed_layout_swizzle()
        test_composed_layout_swizzle_rt()
    except e:
        print("Error => ", e)
