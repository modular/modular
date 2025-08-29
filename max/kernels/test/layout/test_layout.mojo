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

from layout import *
from layout.layout import (
    UNKNOWN_VALUE,
    Layout,
    MakeLayoutList,
    blocked_product,
    coalesce,
    complement,
    composition,
    expand_modes_alike,
    format_layout,
    is_row_major,
    logical_divide,
    logical_product,
    print_layout,
    right_inverse,
    size,
    sublayout,
    tile_to_shape,
    upcast,
    zipped_divide,
)
from testing import assert_equal
from utils import IndexList
from buffer import DimList, Dim


# CHECK-LABEL: test_layout_basic
fn test_layout_basic() raises:
    print("== test_layout_basic")

    # Basic constructor
    alias shape = IntTuple(2, IntTuple(3, IntTuple(4)))
    alias stride = IntTuple(1, IntTuple(2, IntTuple(6)))
    alias layout = Layout(shape, stride)
    assert_equal(
        layout, Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(1, IntTuple(2, 6)))
    )
    assert_equal(
        layout.make_shape_unknown[axis=0](),
        Layout(
            IntTuple(UNKNOWN_VALUE, IntTuple(3, 4)), IntTuple(1, IntTuple(2, 6))
        ),
    )
    assert_equal(
        layout.make_shape_unknown[axis=1](),
        Layout(
            IntTuple(2, IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE)),
            IntTuple(1, IntTuple(2, 6)),
        ),
    )

    # Row major variadic input
    assert_equal(Layout.row_major(2, 3), Layout(IntTuple(2, 3), IntTuple(3, 1)))
    # Row major tuple input
    assert_equal(
        Layout.row_major(IntTuple(2, 3)), Layout(IntTuple(2, 3), IntTuple(3, 1))
    )
    assert_equal(
        Layout.row_major(IntTuple(2, IntTuple(3, 4))),
        Layout(IntTuple(2, IntTuple(3, 4)), IntTuple(12, IntTuple(4, 1))),
    )

    assert_equal(Layout.col_major(2, 3), Layout(IntTuple(2, 3), IntTuple(1, 2)))

    # Check if layout is row_major
    assert_equal(is_row_major[3](Layout.row_major(3, 2, 3)), True)
    assert_equal(is_row_major[2](Layout.col_major(3, 3)), False)

    # test dynamic layout
    assert_equal(
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE),
        Layout.row_major[3](),
    )
    assert_equal(
        Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE),
        Layout.row_major[2](),
    )

    # test index list construction
    assert_equal(
        Layout.row_major(IndexList[2](2, 3)),
        Layout.row_major(2, 3),
    )

    # testing col major
    var dl = DimList(3, 64, 128)

    assert_equal(
        Layout.col_major[3](dl),
        Layout(IntTuple(3, 64, 128), IntTuple(1, 3, 192)),
    )

    assert_equal(
        Layout.col_major[3](DimList(Dim(), 64, 128)),
        Layout(
            IntTuple(UNKNOWN_VALUE, 64, 128),
            IntTuple(1, UNKNOWN_VALUE, UNKNOWN_VALUE),
        ),
    )

    var idx_list = IndexList[3](32, 8, 16)
    assert_equal(
        Layout.col_major[3](idx_list),
        Layout(IntTuple(32, 8, 16), IntTuple(1, 32, 256)),
    )

    assert_equal(
        Layout.col_major[3](IndexList[3](UNKNOWN_VALUE, 8, 16)),
        Layout(
            IntTuple(UNKNOWN_VALUE, 8, 16),
            IntTuple(1, UNKNOWN_VALUE, UNKNOWN_VALUE),
        ),
    )


fn test_unknowns() raises:
    print("== test_unknowns")
    alias shape = IntTuple(2, IntTuple(UNKNOWN_VALUE, 4))
    alias stride = IntTuple(1, IntTuple(2, 6))
    alias layout = Layout(shape, stride)
    assert_equal(layout.shape.all_known(), False)
    assert_equal(layout.stride.all_known(), True)
    assert_equal(layout.all_dims_known(), False)


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

    # Validate keeping rank
    # CHECK: (16:4)
    print(coalesce(Layout(IntTuple(2, 8), IntTuple(4, 8))))
    # CHECK: ((2, 8):(4, 8))
    print(coalesce(Layout(IntTuple(2, 8), IntTuple(4, 8)), keep_rank=True))


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


# CHECK-LABEL: test_by_mode_composition
fn test_by_mode_composition() raises:
    print("== test_by_mode_composition")

    # The correctness here is built on top of default composition, which has
    # been tested extensively above. Keep simple tests only.

    alias layout0 = Layout.row_major(8, 4)
    alias tiler = MakeLayoutList(Layout(4, 1), Layout(2, 1))
    assert_equal(
        composition(layout0, tiler), Layout(IntTuple(4, 2), IntTuple(4, 1))
    )

    alias layout1 = Layout.row_major(IntTuple(IntTuple(8, 6), 4, 2))
    assert_equal(
        composition(layout1, tiler),
        Layout(IntTuple(4, 2, 2), IntTuple(48, 2, 1)),
    )


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
    assert_equal(String(c0), "(6:4)")
    assert_equal(String(complement(Layout(6, 4), 24)), "(4:1)")
    assert_equal(
        String(complement(Layout(IntTuple(4, 6), IntTuple(1, 4)), 24)), "(1:0)"
    )
    assert_equal(String(complement(Layout(4, 2), 24)), "((2, 3):(1, 8))")
    assert_equal(
        String(complement(Layout(IntTuple(2, 4), IntTuple(1, 6)), 24)), "(3:2)"
    )
    assert_equal(
        String(complement(Layout(IntTuple(2, 2), IntTuple(1, 6)), 24)),
        "((3, 2):(2, 12))",
    )

    validate_complement[Layout(1, 0)]()

    validate_complement[Layout(1, 1)]()

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
    assert_equal(String(ld0), "(((2, 2), (2, 3)):((4, 1), (2, 8)))")
    assert_equal(
        String(
            logical_divide(
                Layout(
                    IntTuple(9, IntTuple(4, 8)), IntTuple(59, IntTuple(13, 1))
                ),
                MakeLayoutList(
                    Layout(3, 3), Layout(IntTuple(2, 4), IntTuple(1, 8))
                ),
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
    assert_equal(String(lp0), "(((2, 2), (2, 3)):((4, 1), (2, 8)))")
    assert_equal(
        String(
            logical_product(
                Layout(IntTuple(2, 5), IntTuple(5, 1)),
                MakeLayoutList(Layout(3, 5), Layout(4, 6)),
            )
        ),
        "(((2, 3), (5, 4)):((5, 10), (1, 30)))",
    )


# CHECK-LABEL: test_blocked_product
fn test_blocked_product() raises:
    print("== test_blocked_product")
    var bp0 = blocked_product(
        Layout(IntTuple(2, 5), IntTuple(5, 1)),
        Layout(IntTuple(3, 4), IntTuple(1, 3)),
    )
    assert_equal(String(bp0), "(((2, 3), (5, 4)):((5, 10), (1, 30)))")
    var cm_M = 8
    var cm_K = 8
    core_matrix = Layout.row_major(cm_M, cm_K)
    var t_M = 2
    var t_K = 3
    var bp1 = blocked_product(core_matrix, Layout.col_major(t_M, t_K))
    # ((cm_M,         t_M), (cm_K,               t_K)):
    # ((cm_K, cm_M * cm_K), (1,    t_M * cm_M * cm_K))
    reference_bp1 = Layout(
        IntTuple(IntTuple(cm_M, t_M), IntTuple(cm_K, t_K)),
        IntTuple(IntTuple(cm_K, cm_M * cm_K), IntTuple(1, t_M * cm_M * cm_K)),
    )
    assert_equal(bp1, reference_bp1)

    alias bp2 = blocked_product(
        Layout.row_major(128, 8), Layout.row_major(1, 4)
    )
    assert_equal(String(bp2), "(((128, 1), (8, 4)):((8, 0), (1, 1024)))")

    alias bp3 = blocked_product(
        Layout.row_major(128, 8),
        Layout.row_major(1, 4),
        coalesce_output=True,
    )
    assert_equal(String(bp3), "((128, (8, 4)):(8, (1, 1024)))")


fn test_tile_to_shape() raises:
    print("== test_tile_to_shape")
    var a = Layout(IntTuple(2, 5), IntTuple(5, 1))
    var b = tile_to_shape(a, IntTuple(6, 20))
    assert_equal(String(b), "(((2, 3), (5, 4)):((5, 10), (1, 30)))")
    var b2 = tile_to_shape(a, IntTuple(6, 20), IntTuple(1, 0))
    assert_equal(String(b2), "(((2, 3), (5, 4)):((5, 40), (1, 10)))")


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


fn test_format_layout_grid() raises:
    var expected = """\
       0    1    2    3
    +----+----+----+----+
 0  |  0 |  2 |  4 |  6 |
    +----+----+----+----+
 1  |  1 |  3 |  5 |  7 |
    +----+----+----+----+
 2  |  2 |  4 |  6 |  8 |
    +----+----+----+----+
 3  |  3 |  5 |  7 |  9 |
    +----+----+----+----+
"""

    var output = String()

    format_layout(
        Layout(IntTuple(4, 4), IntTuple(1, 2)),
        output,
    )

    assert_equal(output, expected)


# CHECK-LABEL: test_zipped_divide
fn test_zipped_divide() raises:
    print("== test_zipped_divide")
    alias layout_4x4_row_major = Layout.row_major(4, 4)
    assert_equal(
        String(zipped_divide(layout_4x4_row_major, Layout(2, 1))),
        "((2, (2, 4)):(4, (8, 1)))",
    )
    var zd0 = zipped_divide(
        layout_4x4_row_major,
        MakeLayoutList(Layout(2, 1), Layout(2, 1)),
    )
    assert_equal(String(zd0), "(((2, 2), (2, 2)):((4, 1), (8, 2)))")

    # Resemble the case for distributing a tile over warp group.
    tile_layout = Layout(
        IntTuple(IntTuple(16, 64), 4), IntTuple(IntTuple(128, 2), 2048)
    )
    thread_layout = Layout(
        IntTuple(IntTuple(8, 4), 4), IntTuple(IntTuple(4, 1), 32)
    )
    assert_equal(
        String(zipped_divide(tile_layout, thread_layout)),
        "((((8, 4), 4), ((2, 16), 1)):(((128, 2), 2048), ((1024, 8), 0)))",
    )

    # Swizzle for cuda core matmul.
    tile_layout = Layout(IntTuple(16, 8), IntTuple(32, 4))
    thread_layout = Layout(
        IntTuple(IntTuple(2, 2), 8), IntTuple(IntTuple(1, 16), 2)
    )
    assert_equal(
        String(zipped_divide(tile_layout, thread_layout)),
        "((((2, 2), 8), (4, 1)):(((32, 64), 4), (128, 0)))",
    )


# CHECK-LABEL: test_sublayout
def test_sublayout():
    print("== test_sublayout")
    alias layout_2x3x4 = Layout(IntTuple(2, 3, 4), IntTuple(12, 4, 1))
    assert_equal(String(sublayout(layout_2x3x4, 0, 2)), "((2, 4):(12, 1))")
    alias layout_2x3x4_rank_2 = Layout(
        IntTuple(IntTuple(2, 3), 2, 4), IntTuple(IntTuple(12, 4), 4, 1)
    )
    assert_equal(
        String(sublayout(layout_2x3x4_rank_2, 0, 1)),
        "(((2, 3), 2):((12, 4), 4))",
    )


# CHECK-LABEL: test_crd2idx
def test_crd2idx():
    print("== test_crd2idx")
    alias l_4x4_row_major = Layout.row_major(4, 4)
    alias l_4x4_col_major = Layout.col_major(4, 4)
    # CHECK: 0 (0, 0) (0, 0)
    # CHECK: 1 (0, 1) (1, 0)
    # CHECK: 2 (0, 2) (2, 0)
    # CHECK: 3 (0, 3) (3, 0)
    # CHECK: 4 (1, 0) (0, 1)
    # CHECK: 5 (1, 1) (1, 1)
    # CHECK: 6 (1, 2) (2, 1)
    # CHECK: 7 (1, 3) (3, 1)
    # CHECK: 8 (2, 0) (0, 2)
    # CHECK: 9 (2, 1) (1, 2)
    # CHECK: 10 (2, 2) (2, 2)
    # CHECK: 11 (2, 3) (3, 2)
    # CHECK: 12 (3, 0) (0, 3)
    # CHECK: 13 (3, 1) (1, 3)
    # CHECK: 14 (3, 2) (2, 3)
    # CHECK: 15 (3, 3) (3, 3)
    for i in range(16):
        print(i, l_4x4_row_major.idx2crd(i), l_4x4_col_major.idx2crd(i))


# CHECK-LABEL: test_expand_modes_alike
def test_expand_modes_alike():
    print("== test_expand_modes_alike")
    alias layout_0 = Layout(
        IntTuple(IntTuple(3, IntTuple(5, 2)), 4),
        IntTuple(IntTuple(1, IntTuple(24, 12)), 3),
    )
    alias layout_1 = Layout(
        IntTuple(30, IntTuple(2, 2)), IntTuple(2, IntTuple(60, 1))
    )
    alias ema0 = expand_modes_alike(layout_0, layout_1)
    # CHECK: (((3, (5, 2)), (2, 2)):((1, (24, 12)), (3, 6)))
    print(ema0[0])
    # CHECK: (((3, (5, 2)), (2, 2)):((2, (6, 30)), (60, 1)))
    print(ema0[1])

    alias layout_2 = Layout(
        IntTuple(IntTuple(3, IntTuple(IntTuple(IntTuple(7, 11), 5), 2)), 4),
        IntTuple(
            IntTuple(1, IntTuple(IntTuple(IntTuple(120, 840), 24), 12)), 3
        ),
    )
    alias layout_3 = Layout(IntTuple(2310, IntTuple(2, 2)))
    alias ema1 = expand_modes_alike(layout_2, layout_3)
    # CHECK: (((3, (((7, 11), 5), 2)), (2, 2)):((1, (((120, 840), 24), 12)), (3, 6)))
    print(ema1[0])
    # CHECK: (((3, (((7, 11), 5), 2)), (2, 2)):((1, (((3, 21), 231), 1155)), (2310, 4620)))
    print(ema1[1])

    alias ema2 = expand_modes_alike(
        Layout(IntTuple(2, 2), IntTuple(2, 1)), Layout(4)
    )
    # CHECK: ((2, 2):(2, 1))
    print(ema2[0])
    # CHECK: ((2, 2):(1, 2))
    print(ema2[1])

    alias ema3 = expand_modes_alike(
        Layout(IntTuple(3, 4), IntTuple(2, 6)), Layout(12)
    )
    # CHECK: ((3, 4):(2, 6))
    print(ema3[0])
    # CHECK: ((3, 4):(1, 3))
    print(ema3[1])


fn test_upcast() raises:
    print("== test_upcast")
    alias scatter = Layout(IntTuple(4, 3), IntTuple(2, 4))
    alias up2 = upcast(scatter, 2)
    assert_equal(String(up2), "((4, 3):(1, 2))")
    alias up4 = upcast(scatter, 4)
    alias up22 = upcast(up2, 2)
    assert_equal(up4, up22)
    assert_equal(String(up4), "((2, 3):(1, 1))")
    alias scatter2 = Layout(IntTuple(8, 1024), IntTuple(1024, 1))
    alias up16 = upcast(scatter2, 16)
    assert_equal(String(up16), "((8, 64):(64, 1))")


fn validate_right_inverse[layout: Layout]() raises:
    alias rinv_layout = right_inverse(layout)
    for i in range(layout.size()):
        assert_equal(i, layout(rinv_layout(i)))


fn test_right_inverse() raises:
    validate_right_inverse[
        Layout(
            IntTuple(2, IntTuple(3, IntTuple(4))),
            IntTuple(1, IntTuple(2, IntTuple(6))),
        )
    ]()
    validate_right_inverse[Layout.row_major(8, 4)]()
    validate_right_inverse[Layout.col_major(8, 4)]()
    validate_right_inverse[
        Layout(
            IntTuple(IntTuple(3, IntTuple(IntTuple(IntTuple(7, 11), 5), 2)), 4),
            IntTuple(
                IntTuple(1, IntTuple(IntTuple(IntTuple(120, 840), 24), 12)), 3
            ),
        )
    ]()
    validate_right_inverse[
        Layout(
            IntTuple(IntTuple(3, IntTuple(5, 2)), 4),
            IntTuple(IntTuple(1, IntTuple(24, 12)), 3),
        )
    ]()


# CHECK-LABEL: test_transpose
fn test_transpose() raises:
    print("== test_transpose")

    # Test 2D transpose - row-major to column-major
    alias row_major = Layout.row_major(3, 4)
    alias transposed = row_major.transpose()
    assert_equal(transposed, Layout.col_major(4, 3))

    # Test column-major to row-major
    alias col_major = Layout.col_major(3, 4)
    alias trans_col = col_major.transpose()
    assert_equal(trans_col, Layout.row_major(4, 3))

    # Test custom 2D strides
    alias custom = Layout(IntTuple(3, 5), IntTuple(7, 2))
    alias trans_custom = custom.transpose()
    assert_equal(trans_custom.shape, IntTuple(5, 3))
    assert_equal(trans_custom.stride, IntTuple(2, 7))

    # Test 4D layout
    alias layout4d = Layout.row_major(2, 3, 4, 5)
    alias trans4d = layout4d.transpose()
    assert_equal(trans4d.shape, IntTuple(5, 4, 3, 2))
    assert_equal(trans4d.stride, IntTuple(1, 5, 20, 60))

    # Test nested layout - only top level transposed
    alias nested = Layout(
        IntTuple(IntTuple(2, 3), 4), IntTuple(IntTuple(12, 4), 1)
    )
    alias trans_nested = nested.transpose()
    assert_equal(trans_nested.shape, IntTuple(4, IntTuple(2, 3)))
    assert_equal(trans_nested.stride, IntTuple(1, IntTuple(12, 4)))

    # Test 3-level nested layout
    alias deep_nested = Layout(
        IntTuple(IntTuple(2, 3), IntTuple(4, 5), 6),
        IntTuple(IntTuple(30, 10), IntTuple(6, 1), 120),
    )
    alias trans_deep = deep_nested.transpose()
    assert_equal(trans_deep.shape, IntTuple(6, IntTuple(4, 5), IntTuple(2, 3)))
    assert_equal(
        trans_deep.stride, IntTuple(120, IntTuple(6, 1), IntTuple(30, 10))
    )

    # Test 1D layout (should be unchanged)
    alias layout1d = Layout(IntTuple(10), IntTuple(1))
    alias trans1d = layout1d.transpose()
    assert_equal(trans1d, layout1d)

    # Test layout with zero strides
    alias zero_stride = Layout(IntTuple(3, 4), IntTuple(0, 1))
    alias trans_zero = zero_stride.transpose()
    assert_equal(trans_zero.shape, IntTuple(4, 3))
    assert_equal(trans_zero.stride, IntTuple(1, 0))

    # Test layout with unknown values
    alias unknown_shape = Layout(IntTuple(UNKNOWN_VALUE, 4), IntTuple(4, 1))
    alias trans_unknown = unknown_shape.transpose()
    assert_equal(trans_unknown.shape, IntTuple(4, UNKNOWN_VALUE))
    assert_equal(trans_unknown.stride, IntTuple(1, 4))

    # Test empty layout
    alias empty = Layout()
    alias trans_empty = empty.transpose()
    assert_equal(trans_empty.shape, IntTuple())
    assert_equal(trans_empty.stride, IntTuple())

    # Test double transpose (idempotence)
    assert_equal(transposed.transpose(), row_major)
    assert_equal(trans4d.transpose(), layout4d)
    assert_equal(trans_nested.transpose(), nested)

    # Test memory mapping preservation for 2D
    @parameter
    for i in range(3):

        @parameter
        for j in range(4):
            alias original_idx = row_major(IntTuple(i, j))
            alias transposed_idx = transposed(IntTuple(j, i))
            assert_equal(original_idx, transposed_idx)

    # Test size preservation
    assert_equal(row_major.size(), transposed.size())
    assert_equal(layout4d.size(), trans4d.size())
    assert_equal(nested.size(), trans_nested.size())

    # Test cosize preservation for compact layouts
    assert_equal(row_major.cosize(), transposed.cosize())
    assert_equal(col_major.cosize(), trans_col.cosize())


def main():
    test_layout_basic()
    test_unknowns()
    test_coalesce()
    test_composition()
    test_by_mode_composition()
    test_complement()
    test_logcial_divide()
    test_logical_product()
    test_blocked_product()
    test_tile_to_shape()
    test_print_layout()
    test_format_layout_grid()
    test_zipped_divide()
    test_sublayout()
    test_crd2idx()
    test_expand_modes_alike()
    test_upcast()
    test_right_inverse()
    test_transpose()
