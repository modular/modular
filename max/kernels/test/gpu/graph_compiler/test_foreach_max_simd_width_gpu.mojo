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
"""Pins how `_foreach` picks its SIMD width from a loop's width limit on GPU.

The same rule as on CPU: a static limit caps the preferred width at its
largest power-of-two divisor, and a dynamic one picks the widest power-of-two
step down that divides it. Every element is visited exactly once, including
the scalar tail of a length that isn't a multiple of the width.
"""

from builtin_primitives.primitives import _foreach
from layout import Coord, CoordLike, Idx
from max.gpu.host import DeviceContext
from std.atomic import Atomic
from std.testing import TestSuite, assert_equal


def widest_call[
    simd_width: Int, MaxSimdWidth: CoordLike
](n: Int, max_simd_width: MaxSimdWidth) raises -> Int:
    """Runs `_foreach` over `n` elements on the GPU; returns the widest call
    it made, after checking every element was visited exactly once."""
    with DeviceContext() as ctx:
        var counts_dev = ctx.enqueue_create_buffer[DType.int32](n)
        var widest_dev = ctx.enqueue_create_buffer[DType.int32](1)
        counts_dev.enqueue_fill(0)
        widest_dev.enqueue_fill(0)
        var counts = counts_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var widest = widest_dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()

        def body[
            width: Int, alignment: Int = 1
        ](idx: Coord) {var counts, var widest}:
            _ = Atomic[Int32].max(widest, Int32(width))
            for lane in range(width):
                _ = Atomic[Int32].fetch_add(
                    counts.unsafe_offset(Int(idx[0]) + lane), 1
                )

        _foreach[simd_width=simd_width](
            Coord(Scalar[DType.int](n)), body, ctx, max_simd_width
        )
        var counts_host = ctx.enqueue_create_host_buffer[DType.int32](n)
        var widest_host = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(counts_host, counts_dev)
        ctx.enqueue_copy(widest_host, widest_dev)
        ctx.synchronize()
        for i in range(n):
            assert_equal(counts_host[i], 1)
        return Int(widest_host[0])


def dynamic(limit: Int) -> Scalar[DType.int]:
    return Scalar[DType.int](limit)


def test_no_limit() raises:
    assert_equal(widest_call[8](1024, Idx[0]), 8)
    assert_equal(widest_call[8](1024, dynamic(0)), 8)


def test_static_limit() raises:
    assert_equal(widest_call[8](1024, Idx[2]), 2)
    assert_equal(widest_call[8](1024, Idx[6]), 2)
    assert_equal(widest_call[8](1024, Idx[24]), 8)


def test_dynamic_limit() raises:
    assert_equal(widest_call[8](1024, dynamic(12)), 4)
    assert_equal(widest_call[8](1024, dynamic(6)), 2)
    assert_equal(widest_call[8](1024, dynamic(3)), 1)


def test_limit_with_a_scalar_tail() raises:
    assert_equal(widest_call[8](1021, Idx[4]), 4)
    assert_equal(widest_call[8](1021, dynamic(4)), 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
