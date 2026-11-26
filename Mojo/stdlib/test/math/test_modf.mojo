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

from std.math import modf

from std.testing import TestSuite
from std.testing import assert_almost_equal, assert_equal


def test_modf() raises:
    var i32 = modf(Int32(123))
    assert_equal(i32, (Int32(123), Int32(0)))

    var f32 = modf(Float32(123.5))
    assert_almost_equal(f32[0], 123)
    assert_almost_equal(f32[1], 0.5)

    var f64 = modf(Float64(123.5))
    assert_almost_equal(f64[0], 123)
    assert_almost_equal(f64[1], 0.5)

    f64 = modf(Float64(0))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0)

    f64 = modf(Float64(0.5))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0.5)

    f64 = modf(Float64(-0.5))
    assert_almost_equal(f64[0], -0)
    assert_almost_equal(f64[1], -0.5)

    f64 = modf(Float64(-1.5))
    assert_almost_equal(f64[0], -1)
    assert_almost_equal(f64[1], -0.5)


def test_modf_comptime_large_values() raises:
    # Values with no fractional part (exponent >= mantissa width) must also
    # work at compile time, not just at run time.
    comptime f64_big = modf(Float64(1e20))
    assert_equal(f64_big[0], 1e20)
    assert_equal(f64_big[1], 0)

    comptime f64_neg = modf(Float64(-1e20))
    assert_equal(f64_neg[0], -1e20)
    assert_equal(f64_neg[1], 0)

    comptime f64_2p53 = modf(Float64(9007199254740992.0))
    assert_equal(f64_2p53[0], 9007199254740992.0)
    assert_equal(f64_2p53[1], 0)

    comptime f32_big = modf(Float32(1e10))
    assert_equal(f32_big[0], 1e10)
    assert_equal(f32_big[1], 0)

    # `**` checks whether the exponent is integral through the same helper.
    comptime p = Float64(1.0) ** Float64(1e20)
    assert_equal(p, 1.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
