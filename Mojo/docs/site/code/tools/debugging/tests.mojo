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
# tests.mojo
# Tests for debugging.mdx example programs.
#
# Not tested here (tool behavior; see test_debugging.sh):
#   - Print values, crash reports, breakpoints, debugger sessions, VS Code,
#     limitations, and troubleshooting. These depend on stderr, signals,
#     and LLDB, which a bazel test binary can't observe.
from std.testing import assert_equal, assert_raises, assert_true


@fieldwise_init
struct Histogram(Movable):
    var bins: List[Int]
    var samples: Int


def bin_index(value: UInt8, bin_count: Int) -> Int:
    return Int(value) * bin_count // 256


def build_histogram(pixels: List[UInt8], bin_count: Int) raises -> Histogram:
    if bin_count <= 0:
        raise Error("bin count must be positive")
    var bins = List[Int](length=bin_count, fill=0)
    for p in pixels:
        bins[bin_index(p, bin_count)] += 1
    return Histogram(bins^, len(pixels))


def checked_divide(total: Int, count: Int) -> Int:
    return total // count


# --- Example program (histogram.mojo) ---


def test_histogram_bins() raises:
    var pixels: List[UInt8] = [12, 200, 37, 255, 128, 90, 64, 241]
    var hist = build_histogram(pixels, 4)
    assert_equal(hist.samples, 8)
    assert_equal(len(hist.bins), 4)
    assert_equal(hist.bins[0], 2)
    assert_equal(hist.bins[1], 2)
    assert_equal(hist.bins[2], 1)
    assert_equal(hist.bins[3], 3)


def test_bin_index_stays_in_range() raises:
    # BREAK_ON_BAD_BIN guards `index >= bin_count`, which correct input
    # never reaches.
    for value in range(256):
        for bin_count in range(1, 9):
            assert_true(bin_index(UInt8(value), bin_count) < bin_count)


def test_zero_bins_raises() raises:
    var pixels: List[UInt8] = [12, 200]
    with assert_raises(contains="bin count must be positive"):
        _ = build_histogram(pixels, 0)


# --- Read a crash report (crash.mojo) ---


def test_checked_divide_nonzero() raises:
    assert_equal(checked_divide(784, 8), 98)


def main() raises:
    test_histogram_bins()
    test_bin_index_stays_in_range()
    test_zero_bins_raises()
    test_checked_divide_nonzero()
    print("debugging tests passed")
