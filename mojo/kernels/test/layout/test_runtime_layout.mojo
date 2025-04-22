# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.int_tuple import UNKNOWN_VALUE
from layout.layout import coalesce as coalesce_layout
from layout.layout import crd2idx, print_layout
from layout.runtime_layout import (
    IntTuple,
    Layout,
    RuntimeLayout,
    RuntimeTuple,
    coalesce,
    make_layout,
)
from testing import assert_equal


# CHECK-LABEL: test_runtime_layout_const
def test_runtime_layout_const():
    print("== test_runtime_layout_const")

    alias shape = IntTuple(UNKNOWN_VALUE, 8)
    alias stride = IntTuple(8, 1)

    alias layout = Layout(shape, stride)

    var shape_runtime = RuntimeTuple[layout.shape, element_type = DType.uint32](
        16, 8
    )
    var stride_runtime = RuntimeTuple[
        layout.stride, element_type = DType.uint32
    ]()

    var layout_r = RuntimeLayout[
        layout, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](shape_runtime, stride_runtime)

    assert_equal(String(layout_r.layout), "((-1, 8):(8, 1))")
    assert_equal(String(layout_r), "((16, 8):(8, 1))")


# CHECK-LABEL: test_static_and_dynamic_size
def test_static_and_dynamic_size():
    print("== test_static_and_dynamic_size")
    alias d_layout = Layout(IntTuple(UNKNOWN_VALUE, 4), IntTuple(4, 1))
    var layout = RuntimeLayout[
        d_layout, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[d_layout.shape, element_type = DType.uint32](4, 8),
        RuntimeTuple[d_layout.stride, element_type = DType.uint32](4, 8),
    )
    assert_equal(layout.size(), 32)


# CHECK-LABEL: test_tiled_layout_indexing
def test_tiled_layout_indexing():
    print("== test_tiled_layout_indexing")

    alias shape = IntTuple(IntTuple(2, 2), IntTuple(2, 2))
    alias stride = IntTuple(IntTuple(1, 8), IntTuple(2, 4))

    alias d_tuple = IntTuple(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
    )
    alias d_layout = Layout(d_tuple, d_tuple)

    var layout = RuntimeLayout[
        d_layout, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[d_layout.shape, element_type = DType.uint32](2, 2, 2, 2),
        RuntimeTuple[d_layout.stride, element_type = DType.uint32](1, 8, 2, 4),
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

    alias d_tuple = IntTuple(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE),
    )
    alias d_layout = Layout(d_tuple, d_tuple)

    var layout = RuntimeLayout[
        d_layout, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[d_layout.shape, element_type = DType.uint32](2, 2, 2, 2),
        RuntimeTuple[d_layout.stride, element_type = DType.uint32](1, 8, 2, 4),
    )

    for i in range(16):
        assert_equal(
            crd2idx(
                i,
                shape,
                stride,
            ),
            layout(RuntimeTuple[IntTuple(UNKNOWN_VALUE)](i)),
        )


# CHECK-LABEL: test_sublayout_indexing
def test_sublayout_indexing():
    print("== test_sublayout_indexing")
    alias layout_t = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    alias layout = RuntimeLayout[
        layout_t, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type = DType.uint32](8, 4),
        RuntimeTuple[layout_t.stride, element_type = DType.uint32](4, 1),
    )
    assert_equal(String(layout.sublayout[0]()), "(8:4)")
    assert_equal(String(layout.sublayout[1]()), "(4:1)")


# CHECK-LABEL: test_coalesce
def test_coalesce():
    print("== test_coalesce")
    alias layout_t = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var layout = RuntimeLayout[
        layout_t, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type = DType.uint32](8, 1),
        RuntimeTuple[layout_t.stride, element_type = DType.uint32](1, 1),
    )
    assert_equal(String(coalesce(layout)), "((8, 1):(1, 1))")
    assert_equal(String(coalesce_layout(layout_t)), "((-1, -1):(-1, 1))")

    alias layout_t_2 = Layout(
        IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 8, 1),
        IntTuple(UNKNOWN_VALUE, 8, 1, 1),
    )
    var layout_2 = RuntimeLayout[
        layout_t_2, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[layout_t_2.shape, element_type = DType.uint32](
            32, 16, 8, 1
        ),
        RuntimeTuple[layout_t_2.stride, element_type = DType.uint32](
            16, 8, 1, 1
        ),
    )

    assert_equal(String(coalesce(layout_2)), "((32, 16, 8):(16, 8, 1))")
    assert_equal(
        String(coalesce_layout(layout_t_2)),
        "((-1, -1, 8):(-1, 8, 1))",
    )


# CHECK-LABEL: test_make_layout
def test_make_layout():
    print("== test_make_layout")
    alias layout_t = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    var l_a = RuntimeLayout[
        layout_t, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type = DType.uint32](2, 2),
        RuntimeTuple[layout_t.stride, element_type = DType.uint32](2, 1),
    )
    var l_b = RuntimeLayout[
        layout_t, element_type = DType.uint32, linear_idx_type = DType.uint32
    ](
        RuntimeTuple[layout_t.shape, element_type = DType.uint32](4, 4),
        RuntimeTuple[layout_t.stride, element_type = DType.uint32](4, 1),
    )
    assert_equal(
        String(make_layout(l_a, l_b)), "(((2, 2), (4, 4)):((2, 1), (4, 1)))"
    )


def main():
    test_runtime_layout_const()
    test_static_and_dynamic_size()
    test_tiled_layout_indexing()
    test_tiled_layout_indexing_linear_idx()
    test_sublayout_indexing()
    test_coalesce()
    test_make_layout()
