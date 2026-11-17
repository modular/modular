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
"""Pins how `_foreach` picks its SIMD width from a loop's width limit.

The width must divide the limit: a static one caps the preferred width at its
largest power-of-two divisor at compile time, and a dynamic one picks, at
launch, the widest power-of-two step down from the preferred width that
divides it.
"""

from builtin_primitives.primitives import _foreach, pow2_divisor_of_dims
from layout import Coord, CoordLike, Idx
from max.gpu.host import DeviceContext
from std.collections import Array
from std.testing import TestSuite, assert_equal
from std.utils import IndexList


def widest_call[
    simd_width: Int, MaxSimdWidth: CoordLike
](n: Int, max_simd_width: MaxSimdWidth) raises -> Int:
    """Runs `_foreach` over `n` elements; returns the widest call it made,
    after checking the calls visited every element exactly once."""
    var storage = Array[Int, 1](fill=0)
    var widest = storage.unsafe_ptr()
    var visits = List[Int](length=n, fill=0)
    var counts = visits.unsafe_ptr()

    def body[
        width: Int, alignment: Int = 1
    ](idx: Coord) {var widest, var counts}:
        widest[unsafe_offset=0] = max(widest[unsafe_offset=0], width)
        for lane in range(width):
            counts[unsafe_offset=Int(idx[0]) + lane] += 1

    with DeviceContext.CPU(api="cpu") as ctx:
        _foreach[simd_width=simd_width](
            Coord(Scalar[DType.int](n)), body, ctx, max_simd_width
        )
    for i in range(n):
        assert_equal(visits[i], 1)
    return storage[0]


def dynamic(limit: Int) -> Scalar[DType.int]:
    return Scalar[DType.int](limit)


def test_no_limit() raises:
    var storage = Array[Int, 1](fill=0)
    var widest = storage.unsafe_ptr()

    def body[width: Int, alignment: Int = 1](idx: Coord) {var widest}:
        widest[unsafe_offset=0] = max(widest[unsafe_offset=0], width)

    with DeviceContext.CPU(api="cpu") as ctx:
        _foreach[simd_width=8](Coord(Scalar[DType.int](16)), body, ctx)
    assert_equal(storage[0], 8)


def test_static_limit_caps_at_largest_power_of_two() raises:
    # gcd(6, 10) = 2: 8-wide vectors would cross a boundary at 6.
    assert_equal(widest_call[8](16, Idx[2]), 2)
    assert_equal(widest_call[8](24, Idx[6]), 2)
    assert_equal(widest_call[4](24, Idx[24]), 4)
    assert_equal(widest_call[8](16, Idx[0]), 8)


def test_dynamic_limit_steps_down() raises:
    assert_equal(widest_call[8](16, dynamic(8)), 8)
    assert_equal(widest_call[8](16, dynamic(12)), 4)
    assert_equal(widest_call[8](16, dynamic(6)), 2)
    assert_equal(widest_call[8](16, dynamic(3)), 1)
    assert_equal(widest_call[4](16, dynamic(16)), 4)
    # 0 is the GCD of no constraining extent: any width.
    assert_equal(widest_call[4](16, dynamic(0)), 4)


def test_limit_with_a_scalar_tail() raises:
    # 21 isn't a multiple of the width: the last elements run at width 1.
    assert_equal(widest_call[8](21, Idx[4]), 4)
    assert_equal(widest_call[8](21, dynamic(4)), 4)
    assert_equal(widest_call[8](21, dynamic(0)), 8)


def test_pow2_divisor_of_dims() raises:
    assert_equal(pow2_divisor_of_dims(IndexList[3](6, 10, 4)), 2)
    assert_equal(pow2_divisor_of_dims(IndexList[2](0, 12)), 4)
    assert_equal(pow2_divisor_of_dims(IndexList[2](24, 40)), 8)
    assert_equal(pow2_divisor_of_dims(IndexList[2](0, 0)), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
