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


def _expect_local_lens[
    size: Int
](length: Int, expected: List[Int], bound: Int) raises:
    comptime Layout = ContextParallelLayout[size, 128]
    assert_equal(Layout.max_local_len(length), bound)
    for rank in range(size):
        assert_equal(
            Layout(rank).local_len(length),
            expected[rank],
            msg=String(t"size {size} length {length} rank {rank}"),
        )


def test_page_interleave_local_lens() raises:
    # Lengths around page boundaries and a ragged tail past a full lap, so
    # ranks hold uneven shares and some hold nothing.
    _expect_local_lens[4](1, [1, 0, 0, 0], 128)
    _expect_local_lens[4](127, [127, 0, 0, 0], 128)
    _expect_local_lens[4](128, [128, 0, 0, 0], 128)
    _expect_local_lens[4](129, [128, 1, 0, 0], 128)
    _expect_local_lens[4](1000, [256, 256, 256, 232], 256)
    _expect_local_lens[4](2051, [515, 512, 512, 512], 640)
    _expect_local_lens[8](1, [1, 0, 0, 0, 0, 0, 0, 0], 128)
    _expect_local_lens[8](129, [128, 1, 0, 0, 0, 0, 0, 0], 128)
    _expect_local_lens[8](1000, [128, 128, 128, 128, 128, 128, 128, 104], 128)
    _expect_local_lens[8](2051, [259, 256, 256, 256, 256, 256, 256, 256], 384)
    # Position 2050 wraps to rank 0's third page of an 8-rank lap.
    var rank0 = ContextParallelLayout[8, 128](0)
    assert_equal(rank0.owner(2050), 0)
    assert_equal(rank0.local_slot(2050), 258)
    assert_equal(rank0.global_pos(258), 2050)
    var rank7 = ContextParallelLayout[8, 128](7)
    assert_equal(rank7.owner(1023), 7)
    assert_equal(rank7.local_slot(1023), 127)
    assert_equal(rank7.global_pos(128), 1920)


def main() raises:
    test_disabled_is_identity()
    test_page_interleave_keeps_pages_whole()
    test_page_interleave_local_lens()
    _check[2, 1]()
    _check[8, 1]()
    _check[2, 4]()
    _check[3, 5]()
    _check[8, 16]()
    _check[4, 128]()
    _check[8, 128]()
