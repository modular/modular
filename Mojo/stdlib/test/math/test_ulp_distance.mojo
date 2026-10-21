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

from std.math import inf
from test_utils.math import ulp_distance
from std.testing import assert_equal, assert_true, TestSuite


def test_ulp_distance() raises:
    var a: Float64 = 1.0
    var b: Float64 = 1.0
    var ulp = ulp_distance(a, b)
    assert_true(ulp == 0)

    a = 1.0
    b = 1.0000000000000002
    ulp = ulp_distance(a, b)
    assert_true(ulp == 1)

    a = 1.0
    b = 1.0000000000000022
    ulp = ulp_distance(a, b)
    assert_true(ulp == 10)

    var a_f32 = Float32(1.0)
    var b_f32 = Float32(1.0000001)
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 1)

    a_f32 = Float32(-1.0)
    b_f32 = Float32(-1.0000001)
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 1)

    a_f32 = 1.0
    b_f32 = 1.0000014
    ulp = ulp_distance(a_f32, b_f32)
    assert_true(ulp == 12)

    var a_bf16 = UInt16(0x3F80)
    var b_bf16 = UInt16(0x3F8D)
    ulp = ulp_distance(a_bf16, b_bf16)
    assert_true(ulp == 13)


def _check_class_and_sign_safe[dtype: DType]() raises:
    comptime S = Scalar[dtype]
    var one = S(1.0)
    var infinity = inf[dtype]()
    var max_finite = S(from_bits=infinity._to_bits_signed() - 1)
    var qnan = S(from_bits=infinity._to_bits_signed() + 1)
    var snan = -qnan

    # Any two NaNs agree, whatever the sign or payload.
    assert_equal(ulp_distance(qnan, qnan), 0)
    assert_equal(ulp_distance(qnan, snan), 0)
    # NaN against anything else, infinity against finite, and sign
    # mismatches are maximally far apart.
    for other in [one, -one, S(0.0), infinity, -infinity]:
        assert_equal(ulp_distance(qnan, other), Int.MAX)
        assert_equal(ulp_distance(other, qnan), Int.MAX)
    assert_equal(ulp_distance(infinity, max_finite), Int.MAX)
    assert_equal(ulp_distance(max_finite, infinity), Int.MAX)
    assert_equal(ulp_distance(infinity, infinity), 0)
    assert_equal(ulp_distance(infinity, -infinity), Int.MAX)
    assert_equal(ulp_distance(S(-4.0), one), Int.MAX)
    assert_equal(ulp_distance(one, S(-4.0)), Int.MAX)
    assert_equal(ulp_distance(S(0.0), S(-0.0)), Int.MAX)
    assert_equal(ulp_distance(S(-0.0), S(-0.0)), 0)


def test_ulp_distance_class_and_sign_safe() raises:
    _check_class_and_sign_safe[.float16]()
    _check_class_and_sign_safe[.bfloat16]()
    _check_class_and_sign_safe[.float32]()
    _check_class_and_sign_safe[.float64]()
    # The -4.0 result checked against +1.0 must never pass a tolerance.
    assert_true(ulp_distance(Float64(-4.0), Float64(1.0)) > 1000)
    assert_true(ulp_distance(Float32(-1), Float32(4)) == Int.MAX)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
