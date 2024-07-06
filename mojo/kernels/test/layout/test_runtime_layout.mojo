# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.runtime_layout import RuntimeLayout, Layout, RuntimeTuple, IntTuple

from layout.layout import print_layout, crd2idx

from layout.int_tuple import UNKNOWN_VALUE

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


# CHECK-LABEL: test_static_and_dynamic_size
def test_static_and_dynamic_size():
    print("== test_static_and_dynamic_size")
    alias d_layout = Layout(IntTuple(UNKNOWN_VALUE, 4), IntTuple(4, 1))
    var layout = RuntimeLayout[d_layout](
        RuntimeTuple[d_layout.shape](4, 8),
        RuntimeTuple[d_layout.stride](4, 8),
    )
    assert_equal(layout.size(), 32)


# CHECK-LABEL: test_tiled_layout_indexing
def test_tiled_layout_indexing():
    print("== test_tiled_layout_indexing")

    alias shape = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    alias stride = IntTuple(IntTuple(1, 8), IntTuple(2, 4))

    alias d_tuple = IntTuple(IntTuple(-1, -1), IntTuple(-1, -1))
    alias d_layout = Layout(d_tuple, d_tuple)

    var layout = RuntimeLayout[d_layout](
        RuntimeTuple[d_layout.shape](2, 2, 2, 2),
        RuntimeTuple[d_layout.stride](1, 8, 2, 4),
    )

    for ii in range(2):
        for i in range(2):
            for jj in range(2):
                for j in range(2):
                    assert_equal(
                        crd2idx(
                            IntTuple(IntTuple(ii, i), IntTuple(jj, j)),
                            shape,
                            stride,
                        ),
                        layout(RuntimeTuple[d_tuple](ii, i, jj, j)),
                    )


# CHECK-LABEL: test_tiled_layout_indexing
def test_tiled_layout_indexing_linear_idx():
    print("== test_tiled_layout_indexing_linear_idx")

    alias shape = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    alias stride = IntTuple(IntTuple(1, 8), IntTuple(2, 4))

    alias d_tuple = IntTuple(IntTuple(-1, -1), IntTuple(-1, -1))
    alias d_layout = Layout(d_tuple, d_tuple)

    var layout = RuntimeLayout[d_layout](
        RuntimeTuple[d_layout.shape](2, 2, 2, 2),
        RuntimeTuple[d_layout.stride](1, 8, 2, 4),
    )

    for i in range(16):
        assert_equal(
            crd2idx(
                i,
                shape,
                stride,
            ),
            layout(i),
        )


def main():
    test_runtime_layout_const()
    test_static_and_dynamic_size()
    test_tiled_layout_indexing()
    test_tiled_layout_indexing_linear_idx()
