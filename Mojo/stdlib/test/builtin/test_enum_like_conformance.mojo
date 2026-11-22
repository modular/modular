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
        case .none:
            return 0
        case .nnan:
            return 1
        case .ninf:
            return 2
        case .nsz:
            return 3
        case .arcp:
            return 4
        case .contract:
            return 5
        case .afn:
            return 6
        case .reassoc:
            return 7
        case .fast:
            return 8


def _test_result_rank(result: TestResult) -> Int:
    match result:
        case .pass:
            return 0
        case .fail:
            return 1
        case .skip:
            return 2


def test_fast_math_match() raises:
    var flags = List[FastMathFlag]()
    flags.append(.none)
    flags.append(.nnan)
    flags.append(.ninf)
    flags.append(.nsz)
    flags.append(.arcp)
    flags.append(.contract)
    flags.append(.afn)
    flags.append(.reassoc)
    flags.append(.fast)

    for i in range(len(flags)):
        assert_equal(
            _fast_math_rank(flags[i]), i, "wrong FastMathFlag case matched"
        )


def test_result_match() raises:
    var results = List[TestResult]()
    results.append(.pass)
    results.append(.fail)
    results.append(.skip)

    for i in range(len(results)):
        assert_equal(
            _test_result_rank(results[i]), i, "wrong TestResult case matched"
        )


def _copy_before_declaration(value: _Color) -> _Color:
    return value


__enum _Color:
    case red
    case green


def _contextual_return() -> _Color:
    return .red


__enum _Linear(not Deinitable):
    case open

    def finish(deinit self):
        pass


__enum _Conditional[n: Bool](Deinitable where n):
    case only


def _copy_value[
    T: ImplicitlyCopyable & RegisterPassable & Equatable & Hashable
](value: T) -> T:
    return value


def test_linear_value_conformances() raises:
    comptime assert not conforms_to(_Linear, Deinitable)
    comptime assert not conforms_to(_Linear, TrivialRegisterPassable)
    var value: _Linear = .open
    var copy = _copy_value(value)
    var equal = value == copy
    var equal_hashes = hash(value) == hash(copy)
    value^.finish()
    copy^.finish()
    assert_true(equal and equal_hashes)
    comptime assert conforms_to(_Conditional[True], Deinitable)
    comptime assert not conforms_to(_Conditional[False], Deinitable)
    comptime assert not conforms_to(_Conditional[True], TrivialRegisterPassable)
    comptime assert not conforms_to(
        _Conditional[False], TrivialRegisterPassable
    )


__enum _NotImplicitlyCopyable(not ImplicitlyCopyable):
    case a
    case b


__enum _NotCopyable(not Copyable):
    case a


__enum _NotMovable(not Movable):
    case a


__enum _MemoryOnly(not RegisterPassable):
    case a
    case b


__enum _NotComparable(not Equatable, not Hashable):
    case a


__enum _ConditionalHash[n: Bool](Hashable where n):
    case a


def _is_b(value: _MemoryOnly) -> Bool:
    match value:
        case .a:
            return False
        case .b:
            return True


def test_opted_out_value_conformances() raises:
    comptime assert conforms_to(_NotImplicitlyCopyable, Copyable)
    comptime assert conforms_to(_NotImplicitlyCopyable, RegisterPassable)
    comptime assert not conforms_to(_NotImplicitlyCopyable, ImplicitlyCopyable)
    comptime assert not conforms_to(
        _NotImplicitlyCopyable, TrivialRegisterPassable
    )
    var b = materialize[_NotImplicitlyCopyable.b]()
    var b_copy = b.copy()
    assert_true(b_copy == b)
    assert_false(b_copy == materialize[_NotImplicitlyCopyable.a]())

    comptime assert conforms_to(_NotCopyable, Movable)
    comptime assert conforms_to(_NotCopyable, RegisterPassable)
    comptime assert not conforms_to(_NotCopyable, Copyable)
    comptime assert not conforms_to(_NotCopyable, ImplicitlyCopyable)
    var moved = materialize[_NotCopyable.a]()
    var moved_again = moved^
    assert_true(moved_again == materialize[_NotCopyable.a]())

    comptime assert conforms_to(_NotMovable, Equatable)
    comptime assert not conforms_to(_NotMovable, Movable)
    comptime assert not conforms_to(_NotMovable, Copyable)
    comptime assert not conforms_to(_NotMovable, RegisterPassable)

    comptime assert conforms_to(_MemoryOnly, ImplicitlyCopyable)
    comptime assert not conforms_to(_MemoryOnly, RegisterPassable)
    comptime assert not conforms_to(_MemoryOnly, TrivialRegisterPassable)
    var memory_only: _MemoryOnly = .b
    var memory_only_copy = memory_only
    assert_true(memory_only == memory_only_copy)
    assert_equal(hash(memory_only), hash(memory_only_copy))
    assert_true(_is_b(memory_only_copy))

    comptime assert conforms_to(_NotComparable, TrivialRegisterPassable)
    comptime assert not conforms_to(_NotComparable, Equatable)
    comptime assert not conforms_to(_NotComparable, Hashable)

    comptime assert conforms_to(_ConditionalHash[True], Hashable)
    comptime assert not conforms_to(_ConditionalHash[False], Hashable)
    comptime assert conforms_to(_ConditionalHash[False], Equatable)
    assert_equal(hash(_ConditionalHash[True].a), hash(_ConditionalHash[True].a))


@fieldwise_init
struct _OpenTag(EnumLike, Equatable, Hashable, ImplicitlyCopyable):
    """An open enum that gives each unnamed value its own discriminant."""

    var value: Int

    comptime A = Self(0)
    comptime B = Self(1)

    comptime _enum_is_exhaustive = False
    comptime _enum_case_names = ParameterList.of["A".value, "B".value].values
    comptime _enum_case_types = TypeList.splat[2, Tuple[]].values

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
    comptime assert conforms_to(_Color, TrivialRegisterPassable)
    comptime alias = _contextual_return()
    assert_true(_copy_before_declaration(alias) == _Color.red)
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
