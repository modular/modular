# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from layout import Idx, row_major, col_major
from layout.tile_layout import (
    Layout,
    CoalesceLayout,
    blocked_product,
    coalesce,
)
from std.testing import assert_equal, TestSuite


# ===----------------------------------------------------------------------=== #
# coalesce tests
# ===----------------------------------------------------------------------=== #


def test_coalesce_row_major_2d() raises:
    """Row-major (2, 4) strides (4, 1) does not coalesce.

    Pairs: (2, 4), (4, 1). Check: 2*4=8 != 1. No merge.
    Coalesce processes dimensions left to right; for row-major the
    stride-1 dim is last, so consecutive pairs are not contiguous.
    """
    var layout = row_major[2, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 4)
    assert_equal(c.shape[1]().value(), 4)
    assert_equal(c.stride[1]().value(), 1)


def test_coalesce_row_major_3d() raises:
    """Row-major (2, 3, 4) strides (12, 4, 1) does not coalesce.

    Pairs: (2, 12), (3, 4), (4, 1). No consecutive pair is contiguous.
    """
    var layout = row_major[2, 3, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 12)
    assert_equal(c.shape[1]().value(), 3)
    assert_equal(c.stride[1]().value(), 4)
    assert_equal(c.shape[2]().value(), 4)
    assert_equal(c.stride[2]().value(), 1)


def test_coalesce_col_major_2d() raises:
    """Col-major (3, 4) strides (1, 3) -> coalesced to (12,) stride (1,).

    Flat pairs: (3, 1), (4, 3). Check: 3*1 == 3? Yes -> merge to (12, 1).
    """
    var layout = col_major[3, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 12)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_col_major_3d() raises:
    """Col-major (2, 3, 4) strides (1, 2, 6) -> coalesced to (24,) stride (1,).

    Flat pairs: (2, 1), (3, 2), (4, 6).
    (2, 1): prev
    (3, 2): 2*1=2 == 2 -> merge: (6, 1)
    (4, 6): 6*1=6 == 6 -> merge: (24, 1)
    """
    var layout = col_major[2, 3, 4]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 24)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_already_1d() raises:
    """A 1D layout stays 1D after coalescing."""
    var layout = row_major[8]()
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 8)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_non_contiguous() raises:
    """Non-contiguous layout remains unchanged.

    Shape (2, 2), stride (4, 1) -> 2*4=8 != 1, so no merge.
    """
    var layout = Layout(
        shape=(Idx[2](), Idx[2]()),
        stride=(Idx[4](), Idx[1]()),
    )
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 4)
    assert_equal(c.shape[1]().value(), 2)
    assert_equal(c.stride[1]().value(), 1)


def test_coalesce_with_shape_1_dims() raises:
    """Shape-1 dimensions are removed.

    Shape (1, 4, 1), stride (8, 1, 2) -> skip shape-1 dims -> (4, 1).
    """
    var layout = Layout(
        shape=(Idx[1](), Idx[4](), Idx[1]()),
        stride=(Idx[8](), Idx[1](), Idx[2]()),
    )
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 4)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_nested_blocked_product() raises:
    """Coalesce a nested layout from blocked_product.

    block = row_major[4]() (shape (4,), stride (1,))
    tiler = row_major[3]() (shape (3,), stride (1,))
    blocked = blocked_product(block, tiler)
    shape: ((4,), (3,)), stride: ((1,), (4,))
    Flat: (4, 1), (3, 4). 4*1 == 4 -> merge to (12, 1).
    """
    var block = row_major[4]()
    var tiler = row_major[3]()
    var blocked = blocked_product(block, tiler)
    var c = coalesce(blocked)
    assert_equal(c.shape[0]().value(), 12)
    assert_equal(c.stride[0]().value(), 1)


def test_coalesce_type_level() raises:
    """CoalesceLayout can be used at the type level."""
    comptime CM = type_of(col_major[3, 4]())
    comptime C = CoalesceLayout[CM]

    comptime assert C.static_shape[0] == 12
    comptime assert C.static_stride[0] == 1

    var layout = C()
    assert_equal(layout.shape[0]().value(), 12)
    assert_equal(layout.stride[0]().value(), 1)


def test_coalesce_partial_merge() raises:
    """Coalesce merges only contiguous dimensions.

    Shape (2, 4, 3), stride (16, 1, 4).
    Flat: (2, 16), (4, 1), (3, 4).
    (2, 16): prev
    (4, 1): 2*16=32 != 1, new dim -> (2, 16), (4, 1)
    (3, 4): 4*1=4 == 4, merge -> (2, 16), (12, 1)
    """
    var layout = Layout(
        shape=(Idx[2](), Idx[4](), Idx[3]()),
        stride=(Idx[16](), Idx[1](), Idx[4]()),
    )
    var c = coalesce(layout)
    assert_equal(c.shape[0]().value(), 2)
    assert_equal(c.stride[0]().value(), 16)
    assert_equal(c.shape[1]().value(), 12)
    assert_equal(c.stride[1]().value(), 1)


# ===----------------------------------------------------------------------=== #
# blocked_product with coalesce_output tests
# ===----------------------------------------------------------------------=== #


def test_blocked_product_coalesce_output_1d() raises:
    var block = row_major[4]()
    var tiler = row_major[3]()
    var result = blocked_product[coalesce_output=True](block, tiler)
    assert_equal(result.shape[0]().value(), 12)
    assert_equal(result.stride[0]().value(), 1)


def test_blocked_product_coalesce_output_no_coalesce() raises:
    """Test blocked_product with coalesce_output=True when no modes coalesce.

    Same as the existing test_coalesced_blocked_product_no_coalesce but
    exercising the function overload instead of the type alias.
    """
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var result = blocked_product[coalesce_output=True](block, tiler)

    # Mode 0: nested shape (2, 2), stride (2, 12)
    assert_equal(result.shape[0]()[0].value(), 2)
    assert_equal(result.shape[0]()[1].value(), 2)
    assert_equal(result.stride[0]()[0].value(), 2)
    assert_equal(result.stride[0]()[1].value(), 12)

    # Mode 1: nested shape (2, 3), stride (1, 4)
    assert_equal(result.shape[1]()[0].value(), 2)
    assert_equal(result.shape[1]()[1].value(), 3)
    assert_equal(result.stride[1]()[0].value(), 1)
    assert_equal(result.stride[1]()[1].value(), 4)


def test_blocked_product_coalesce_output_false() raises:
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var result = blocked_product[coalesce_output=False](block, tiler)

    # Should be nested, same as blocked_product(block, tiler)
    assert_equal(result.shape[0]()[0].value(), 2)
    assert_equal(result.shape[0]()[1].value(), 2)
    assert_equal(result.shape[1]()[0].value(), 2)
    assert_equal(result.shape[1]()[1].value(), 3)
    assert_equal(result.stride[0]()[0].value(), 2)
    assert_equal(result.stride[0]()[1].value(), 12)
    assert_equal(result.stride[1]()[0].value(), 1)
    assert_equal(result.stride[1]()[1].value(), 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
