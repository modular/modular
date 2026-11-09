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
"""Pins the `mogg.index.reshape` store-index transform primitive.

It maps a point in `from` to the point in `to` with the same row-major linear
offset. One always-inlined primitive serves both static and dynamic reshapes:
each shape Coord carries a dim's static-vs-dynamic split in its element types
and its runtime value in the component itself, so a static dim's extent folds
to a constant and a dynamic dim's is read at runtime. When the static dims
fully determine the transform, no runtime extent is read, so the same
primitive costs what a dedicated static one would.
"""

from builtin_primitives.primitives import mogg_index_reshape
from layout import Coord, Idx
from std.utils import IndexList
from std.testing import TestSuite, assert_equal


def test_static_collapse() raises:
    # A fully static collapse [2, 3] -> [6]: every extent folds to a constant.
    comptime from_shape = Coord(Idx[2], Idx[3])
    comptime to_shape = Coord(Idx[6])
    for i in range(2):
        for j in range(3):
            var got = mogg_index_reshape[to_rank=1](
                IndexList[2](i, j), from_shape, to_shape
            )
            assert_equal(got[0], i * 3 + j)


def test_inner_dynamic_relinearizes() raises:
    # [2, N] -> [N, 2] relinearizes across the dynamic inner dim `N=3`: the
    # ravel of `from` needs N for dim 0's stride, so the runtime value is
    # read from the Coord's own component.
    comptime from_shape = Coord(Idx[2], Int(3))
    comptime to_shape = Coord(Int(3), Idx[2])
    for i in range(2):
        for j in range(3):
            var linear = i * 3 + j
            var got = mogg_index_reshape[to_rank=2](
                IndexList[2](i, j), from_shape, to_shape
            )
            assert_equal(got[0], linear // 2)
            assert_equal(got[1], linear % 2)


def test_inner_dynamic_with_dynamic_batch() raises:
    # Outermost (batch) dim dynamic AND an inner collapse: [M, 2, 2] -> [M, 4],
    # M=2 at runtime. The batch dim rides through untouched (its extent is
    # never read); the inner pair collapses.
    comptime from_shape = Coord(Int(2), Idx[2], Idx[2])
    comptime to_shape = Coord(Int(2), Idx[4])
    for b in range(2):
        for i in range(2):
            for j in range(2):
                var got = mogg_index_reshape[to_rank=2](
                    IndexList[3](b, i, j), from_shape, to_shape
                )
                assert_equal(got[0], b)
                assert_equal(got[1], i * 2 + j)


def test_unit_insert_single_dynamic() raises:
    # Add unit dims around a run whose real dims include ONE dynamic (batch)
    # dim: [M, 2, 2] -> [M, 1, 2, 2, 1], M=2. This is the fast path -- a pure
    # positional shuffle, no extents read -- and it must hold even though M is
    # dynamic, because exactly one non-1 dim is dynamic.
    comptime from_shape = Coord(Int(2), Idx[2], Idx[2])
    comptime to_shape = Coord(Int(2), Idx[1], Idx[2], Idx[2], Idx[1])
    for b in range(2):
        for i in range(2):
            for j in range(2):
                var got = mogg_index_reshape[to_rank=5](
                    IndexList[3](b, i, j), from_shape, to_shape
                )
                assert_equal(got[0], b)
                assert_equal(got[1], 0)
                assert_equal(got[2], i)
                assert_equal(got[3], j)
                assert_equal(got[4], 0)


def test_multi_dynamic_reorder_is_not_a_noop() raises:
    # [A, B] -> [B, A], both dynamic (A=2, B=3). Dropping 1s leaves identical
    # (all-dynamic) type sequences, but this is a genuine relinearizing
    # reshape, not a unit shuffle. The >1-dynamic guard must keep it off the
    # positional-remap fast path so it relinearizes correctly.
    comptime from_shape = Coord(Int(2), Int(3))
    comptime to_shape = Coord(Int(3), Int(2))
    for i in range(2):
        for j in range(3):
            var linear = i * 3 + j
            var got = mogg_index_reshape[to_rank=2](
                IndexList[2](i, j), from_shape, to_shape
            )
            assert_equal(got[0], linear // 2)
            assert_equal(got[1], linear % 2)


def test_multi_dynamic_unit_insert_is_correct() raises:
    # [A, B] -> [1, A, B], both real dims dynamic (A=2, B=3). Two dynamic
    # non-1 dims, so this takes the general (relinearizing) path rather than
    # the fast shuffle -- it must still be a correct unit insert.
    comptime from_shape = Coord(Int(2), Int(3))
    comptime to_shape = Coord(Idx[1], Int(2), Int(3))
    for i in range(2):
        for j in range(3):
            var got = mogg_index_reshape[to_rank=3](
                IndexList[2](i, j), from_shape, to_shape
            )
            assert_equal(got[0], 0)
            assert_equal(got[1], i)
            assert_equal(got[2], j)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
