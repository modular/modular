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

from std.math import inf, nan
from std.memory import bitcast
from std.testing import assert_equal, assert_true, TestSuite

from test_expm1_gpu import _expm1_inputs, _ulp_distance


def _test_ulp_distance[dtype: DType]() raises where dtype.is_floating_point():
    comptime T = Scalar[dtype]
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    assert_equal(_ulp_distance(T(4), T(-1)), Int.MAX)
    assert_equal(_ulp_distance(T(-1), T(4)), Int.MAX)
    assert_equal(_ulp_distance(T.MAX_FINITE, -T.MAX_FINITE), Int.MAX)
    assert_equal(_ulp_distance(T.MAX_FINITE, inf[dtype]()), Int.MAX)
    assert_equal(_ulp_distance(inf[dtype](), T.MAX_FINITE), Int.MAX)
    assert_equal(_ulp_distance(-T.MAX_FINITE, -inf[dtype]()), Int.MAX)
    assert_equal(_ulp_distance(inf[dtype](), -inf[dtype]()), Int.MAX)
    assert_equal(_ulp_distance(inf[dtype](), inf[dtype]()), 0)
    assert_equal(_ulp_distance(-inf[dtype](), -inf[dtype]()), 0)
    assert_equal(_ulp_distance(T(0.0), T(-0.0)), Int.MAX)
    assert_equal(_ulp_distance(T(-0.0), T(-0.0)), 0)
    assert_equal(_ulp_distance(nan[dtype](), T(1)), Int.MAX)
    assert_equal(_ulp_distance(T(1), nan[dtype]()), Int.MAX)
    assert_equal(_ulp_distance(nan[dtype](), nan[dtype]()), 0)
    for x in [T(1), T(-1)]:
        var next_encoding = bitcast[dtype](bitcast[uint_type](x) + 1)
        assert_equal(_ulp_distance(x, next_encoding), 1)
        assert_equal(_ulp_distance(next_encoding, x), 1)


def test_ulp_distance_float32() raises:
    _test_ulp_distance[.float32]()


def test_ulp_distance_float64() raises:
    _test_ulp_distance[.float64]()


def _test_subnormal_inputs[
    dtype: DType
]() raises where dtype.is_floating_point():
    comptime uint_type = DType.uint32 if dtype == .float32 else DType.uint64
    comptime sign_bit = Scalar[uint_type](1) << Scalar[uint_type](
        31 if dtype == .float32 else 63
    )
    comptime min_normal_bits = Scalar[uint_type](1) << Scalar[uint_type](
        23 if dtype == .float32 else 52
    )
    var inputs = _expm1_inputs[dtype](Scalar[dtype](95))
    for bits in [
        Scalar[uint_type](1),
        2,
        min_normal_bits >> 1,
        min_normal_bits - 1,
        min_normal_bits,
    ]:
        for sign in [Scalar[uint_type](0), sign_bit]:
            var found = False
            for x in inputs:
                found |= bitcast[uint_type](x) == (bits | sign)
            assert_true(found, msg="Missing bit-constructed expm1 input")


def test_subnormal_inputs() raises:
    _test_subnormal_inputs[.float32]()
    _test_subnormal_inputs[.float64]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
