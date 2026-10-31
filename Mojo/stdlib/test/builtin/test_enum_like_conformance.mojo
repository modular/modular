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
"""Exercises runtime `match` on the stdlib's enum-like structs.

Each helper names every case so the match fails to compile if
`_enum_case_names` drifts from the comptime case constants.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
)

from std.builtin.variadics import ParameterList, TypeList
from std.simd import FastMathFlag
from std.testing.suite import TestResult


def _fast_math_rank(flag: FastMathFlag) -> Int:
    match flag:
        case .NONE:
            return 0
        case .NNAN:
            return 1
        case .NINF:
            return 2
        case .NSZ:
            return 3
        case .ARCP:
            return 4
        case .CONTRACT:
            return 5
        case .AFN:
            return 6
        case .REASSOC:
            return 7
        case .FAST:
            return 8


def _test_result_rank(result: TestResult) -> Int:
    match result:
        case .PASS:
            return 0
        case .FAIL:
            return 1
        case .SKIP:
            return 2


def test_fast_math_match() raises:
    var flags = List[FastMathFlag]()
    flags.append(FastMathFlag.NONE)
    flags.append(FastMathFlag.NNAN)
    flags.append(FastMathFlag.NINF)
    flags.append(FastMathFlag.NSZ)
    flags.append(FastMathFlag.ARCP)
    flags.append(FastMathFlag.CONTRACT)
    flags.append(FastMathFlag.AFN)
    flags.append(FastMathFlag.REASSOC)
    flags.append(FastMathFlag.FAST)

    for i in range(len(flags)):
        assert_equal(
            _fast_math_rank(flags[i]), i, "wrong FastMathFlag case matched"
        )


def test_result_match() raises:
    var results = List[TestResult]()
    results.append(TestResult.PASS)
    results.append(TestResult.FAIL)
    results.append(TestResult.SKIP)

    for i in range(len(results)):
        assert_equal(
            _test_result_rank(results[i]), i, "wrong TestResult case matched"
        )


__enum _Color(Equatable, Hashable, ImplicitlyCopyable):
    case red
    case green


@fieldwise_init
struct _OpenTag(EnumLike, Equatable, Hashable, ImplicitlyCopyable):
    """An open enum that gives each unnamed value its own discriminant."""

    var value: Int

    comptime A = Self(0)
    comptime B = Self(1)

    comptime _enum_is_exhaustive = False
    comptime _enum_case_names = ParameterList.of["A".value, "B".value].values
    comptime _enum_case_types = TypeList.splat[2, NoneType].values

    def _get_enum_discriminant(self) -> Int:
        if self.value < 2:
            return self.value
        return self.value + 2

    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        comptime assert False, "_OpenTag has no payload"


def test_default_eq_and_hash_without_payloads() raises:
    assert_true(_Color.red == _Color.red)
    assert_false(_Color.red == _Color.green)
    assert_equal(hash(_Color.red), hash(_Color.red))
    assert_not_equal(hash(_Color.red), hash(_Color.green))


def test_default_eq_and_hash_open_enum() raises:
    # Values outside the declared cases compare and hash by discriminant.
    assert_true(_OpenTag(5) == _OpenTag(5))
    assert_false(_OpenTag(5) == _OpenTag(6))
    assert_false(_OpenTag(5) == _OpenTag.A)
    assert_equal(hash(_OpenTag(5)), hash(_OpenTag(5)))
    assert_not_equal(hash(_OpenTag(5)), hash(_OpenTag(6)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
