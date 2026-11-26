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
"""Pins the static strides a `mogg.tensor.create.collapse` view reports.

A collapse view merges consecutive groups of its input's dims; each merged
dim takes the stride of its group's innermost dim. The end-to-end numerics are
covered by the graph compiler's integration tests.
"""

from builtin_primitives.primitives import (
    _mogg_collapse_group_start,
    _mogg_collapse_view_strides,
)
from layout import ComptimeInt, Coord, CoordLike, Idx
from std.testing import TestSuite, assert_equal


def test_group_start() raises:
    comptime groups = Coord(Idx[2], Idx[1], Idx[3])
    assert_equal(_mogg_collapse_group_start[groups, 0](), 0)
    assert_equal(_mogg_collapse_group_start[groups, 1](), 2)
    assert_equal(_mogg_collapse_group_start[groups, 2](), 3)


def test_full_collapse_of_row_major() raises:
    # [4, 6] row-major -> [24]: the merged dim steps by 1.
    var strides = _mogg_collapse_view_strides[
        1,
        Coord(Idx[2]),
        TypeList.of[Trait=CoordLike, ComptimeInt[6], ComptimeInt[1]](),
    ]()
    assert_equal(strides[0], 1)


def test_partial_collapse_keeps_the_innermost_stride() raises:
    # A broadcast [B, S, D] view with strides (0, 0, 1) -> [B*S, D].
    var strides = _mogg_collapse_view_strides[
        2,
        Coord(Idx[2], Idx[1]),
        TypeList.of[
            Trait=CoordLike, ComptimeInt[0], ComptimeInt[0], ComptimeInt[1]
        ](),
    ]()
    assert_equal(strides[0], 0)
    assert_equal(strides[1], 1)


def test_unknown_strides_stay_unknown() raises:
    var strides = _mogg_collapse_view_strides[
        2,
        Coord(Idx[1], Idx[2]),
        TypeList.of[Trait=CoordLike, Int, Int, ComptimeInt[1]](),
    ]()
    assert_equal(strides[0], -1)
    assert_equal(strides[1], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
