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
"""On CPU, `elementwise` and the split-axis row-wise reduction split their
work so every vector starts on a multiple of the SIMD width, as on GPU: a
fused concat relies on a vector never straddling a boundary that is a
multiple of the width.
"""

from algorithm.cpu.rowwise import _splitk_stripe
from layout import Coord
from max.algorithm.backend.cpu.elementwise import _aligned_chunk_size
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.testing import TestSuite, assert_equal, assert_true


def _check_tiling(
    size: Int, starts: List[Int], ends: List[Int], width: Int
) raises:
    """Checks the ranges cover `[0, size)` once each and start on the width."""
    var visits = List[Int](length=size, fill=0)
    for r in range(len(starts)):
        assert_true(starts[r] <= ends[r] and ends[r] <= size)
        if starts[r] < ends[r]:
            assert_equal(starts[r] % width, 0)
        for i in range(starts[r], ends[r]):
            visits[i] += 1
    for i in range(size):
        assert_equal(visits[i], 1)


def test_chunks_tile_on_the_width() raises:
    # Controlled worker counts, including more workers than vectors.
    for width in [1, 2, 4, 8, 16]:
        for size in [1, 7, 8, 63, 100, 1023, 4097, 100_003]:
            for workers in [1, 2, 3, 5, 7, 16, 64]:
                var chunk = _aligned_chunk_size(size, workers, width)
                var starts = List[Int]()
                var ends = List[Int]()
                for w in range(workers):
                    starts.append(min(w * chunk, size))
                    ends.append(min((w + 1) * chunk, size))
                _check_tiling(size, starts, ends, width)


def test_splitk_stripes_tile_on_the_width() raises:
    # Controlled split counts, including more splits than vectors.
    for width in [1, 2, 4, 8, 16]:
        for size in [1, 7, 8, 63, 100, 1023, 262_147, 300_001]:
            for splits in [1, 2, 3, 5, 7, 16, 64]:
                var starts = List[Int]()
                var ends = List[Int]()
                for s in range(splits):
                    var stripe = _splitk_stripe(size, splits, s, width)
                    starts.append(stripe[0])
                    ends.append(stripe[1])
                _check_tiling(size, starts, ends, width)


def test_elementwise_visits_each_element_once() raises:
    # Several 32768-element grains, and a length no worker count divides.
    # Each lane counts its element atomically, so an overlap between workers
    # shows as a count of 2 rather than a race.
    comptime width = 8
    var n = 100_003
    var visits = List[Int64](length=n, fill=0)
    var at = visits.unsafe_ptr()
    var misaligned = List[Int64](length=1, fill=0)
    var bad = misaligned.unsafe_ptr()

    def body[w: Int, alignment: Int = 1](idx: Coord) {var at, var bad}:
        var i = Int(idx[0])
        if w > 1 and i % width != 0:
            _ = Atomic[Int64].fetch_add(bad, 1)
        for j in range(w):
            _ = Atomic[Int64].fetch_add(at.unsafe_offset(i + j), 1)

    with DeviceContext.CPU(api="cpu") as ctx:
        elementwise[simd_width=width](body, Coord(n), ctx)

    for i in range(n):
        assert_equal(visits[i], 1)
    assert_equal(misaligned[0], 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
