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

from kv_cache.context_parallel import ContextParallelLayout
from std.testing import assert_equal, assert_true


def _check[size: Int, interleave: Int]() raises:
    """Checks every map against a brute-force walk over positions."""
    comptime span = size * interleave
    for length in range(0, 5 * span + 3):
        var bound = ContextParallelLayout[size, interleave].max_local_len(
            length
        )
        var total = 0
        for rank in range(size):
            var layout = ContextParallelLayout[size, interleave](rank)
            var expected_slot = 0
            for pos in range(length):
                if layout.owner(pos) != rank:
                    assert_true(not layout.owns(pos))
                    continue
                assert_true(layout.owns(pos))
                # Owned positions fill the local slots densely, in order.
                assert_equal(layout.local_slot(pos), expected_slot)
                assert_equal(layout.global_pos(expected_slot), pos)
                expected_slot += 1
            assert_equal(layout.local_len(length), expected_slot)
            assert_true(expected_slot <= bound)
            total += expected_slot
        assert_equal(total, length)


def test_disabled_is_identity() raises:
    var layout = ContextParallelLayout[1, 1](0)
    for pos in range(300):
        assert_true(layout.owns(pos))
        assert_equal(layout.local_slot(pos), pos)
        assert_equal(layout.global_pos(pos), pos)
        assert_equal(layout.local_len(pos), pos)


def test_page_interleave_keeps_pages_whole() raises:
    # Page 2 of a 4-rank, 128-token interleave lives whole on rank 2, slot 0..127.
    var layout = ContextParallelLayout[4, 128](2)
    for pos in range(256, 384):
        assert_equal(layout.owner(pos), 2)
        assert_equal(layout.local_slot(pos), pos - 256)
    # The next page that rank 2 owns is page 6, starting at local slot 128.
    assert_equal(layout.local_slot(6 * 128), 128)


def main() raises:
    test_disabled_is_identity()
    test_page_interleave_keeps_pages_whole()
    _check[2, 1]()
    _check[8, 1]()
    _check[2, 4]()
    _check[3, 5]()
    _check[8, 16]()
