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
"""NaN behaviour of the `ArgMax`/`ArgMin` reduction monoids.

These monoids back the `mo.reduce.arg_max` / `mo.reduce.arg_min` graph ops
(`algorithm.reductions.reduce_argmax` emits `acc_indices` straight into the
output tensor), so an out-of-range index here reaches consumers as real tensor
data. A downstream `gather` on such an index is an out-of-bounds device access
that kills the process, so the index must always land in `[0, axis_size)`.

The contract mirrors the one `nn/argmaxmin_gpu` already documents and tests:
a NaN never compares greater, so it is skipped, and a row with no valid
candidate reports index 0.
"""

from algorithm.reduce_op import ArgMax, ArgMin
from std.testing import TestSuite, assert_equal
from std.utils.numerics import nan


def _argmax_index(values: List[Float32]) -> Int:
    var acc = ArgMax[DType.float32, 1]()
    for i in range(len(values)):
        acc.accumulate[DType.float32, 1](
            SIMD[DType.float32, 1](values[i]), SIMD[DType.int64, 1](i)
        )
    return Int(acc.reduce().acc_indices[0])


def _argmin_index(values: List[Float32]) -> Int:
    var acc = ArgMin[DType.float32, 1]()
    for i in range(len(values)):
        acc.accumulate[DType.float32, 1](
            SIMD[DType.float32, 1](values[i]), SIMD[DType.int64, 1](i)
        )
    return Int(acc.reduce().acc_indices[0])


def test_argmax_plain() raises:
    assert_equal(_argmax_index([1.0, 3.0, 2.0]), 1)


def test_argmax_ties_keep_lowest_index() raises:
    assert_equal(_argmax_index([2.0, 2.0, 1.0]), 0)


def test_argmax_skips_trailing_nan() raises:
    # The NaN arrives after the winner, so it must not displace it. Before the
    # fix the `le` compare took the NaN, and the `eq` in `reduce` then matched
    # no lane, emitting the `Int64.MAX` identity.
    var nan_f32 = nan[DType.float32]()
    assert_equal(_argmax_index([1.0, 2.0, nan_f32]), 1)


def test_argmax_all_nan_reports_zero() raises:
    var nan_f32 = nan[DType.float32]()
    assert_equal(_argmax_index([nan_f32, nan_f32, nan_f32]), 0)


def test_argmin_plain() raises:
    assert_equal(_argmin_index([3.0, 1.0, 2.0]), 1)


def test_argmin_skips_trailing_nan() raises:
    var nan_f32 = nan[DType.float32]()
    assert_equal(_argmin_index([2.0, 1.0, nan_f32]), 1)


def test_argmin_all_nan_reports_zero() raises:
    var nan_f32 = nan[DType.float32]()
    assert_equal(_argmin_index([nan_f32, nan_f32, nan_f32]), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
