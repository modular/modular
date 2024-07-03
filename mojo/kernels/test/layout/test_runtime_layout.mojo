# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.runtime_layout import RuntimeLayout, Layout, RuntimeTuple, IntTuple

from testing import assert_equal


# CHECK-LABEL: test_runtime_layout_const
def test_runtime_layout_const():
    print("== test_runtime_layout_const")

    alias shape = IntTuple(-1, 8)
    alias stride = IntTuple(8, 1)

    alias layout = Layout(shape, stride)

    var shape_runtime = RuntimeTuple[layout.shape](16, 8)
    var stride_runtime = RuntimeTuple[layout.stride]()

    var layout_r = RuntimeLayout[layout](shape_runtime, stride_runtime)

    assert_equal(str(layout_r.layout), "((-1, 8):(8, 1))")
    assert_equal(str(layout_r), "((16, 8):(8, 1))")


def main():
    test_runtime_layout_const()
