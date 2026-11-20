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

from std.builtin.variadics import TypeList
from std.collections import OptionalReg
from std.sys import align_of, size_of
from std.testing import TestSuite, assert_equal, assert_false, assert_true


__enum Token(ImplicitlyCopyable, Writable):
    case eof
    case word(String)
    case span(Int, Int)
    case tuple(Tuple[Int])
    case empty(Tuple[])
    case none(NoneType)


@fieldwise_init
struct Compared(Equatable, ImplicitlyCopyable):
    var value: Int

    def __eq__(self, other: Self) -> Bool:
        return True

    def __ne__(self, other: Self) -> Bool:
        return True


__enum Key(Equatable):
    case one(Compared)
    case many(Compared, Compared)


def test_payload_types() raises:
    comptime types = TypeList[Trait=AnyType, Token._enum_case_types]()
    comptime assert types[0] == Tuple[]
    comptime assert types[1] == Tuple[String]
    comptime assert types[2] == Tuple[Int, Int]
    comptime assert types[3] == Tuple[Tuple[Int]]
    comptime assert types[4] == Tuple[Tuple[]]
    comptime assert types[5] == Tuple[NoneType]
    comptime optional_types = TypeList[
        Trait=AnyType, Optional[Int]._enum_case_types
    ]()
    comptime reg_types = TypeList[
        Trait=AnyType, OptionalReg[Int]._enum_case_types
    ]()
    comptime assert (
        optional_types[0] == Tuple[] and optional_types[1] == Tuple[Int]
    )
    comptime assert reg_types[0] == Tuple[] and reg_types[1] == Tuple[Int]


def test_printing() raises:
    assert_equal(String(Token.eof), "Token.eof")
    assert_equal(String(Token.word("w")), "Token.word(w)")
    assert_equal(repr(Token.word("w")), "Token.word('w')")
    assert_equal(String(Token.span(1, 2)), "Token.span(1, 2)")
    assert_equal(repr(Token.span(1, 2)), "Token.span(Int(1), Int(2))")
    assert_equal(String(Token.tuple((3,))), "Token.tuple((3,))")
    assert_equal(String(Token.empty(())), "Token.empty(())")
    assert_equal(String(Token.none(None)), "Token.none(None)")
    assert_equal(repr(Token.none(None)), "Token.none(None)")


def test_payload_equality() raises:
    # Every payload compares as a `Tuple`, which calls `__ne__` on its elements.
    assert_false(Key.one(Compared(1)) == Key.one(Compared(2)))
    assert_false(
        Key.many(Compared(1), Compared(2)) == Key.many(Compared(1), Compared(2))
    )


def test_none_payload() raises:
    match Token.none(None):
        case .none(x):
            comptime assert type_of(x) == NoneType
            assert_equal(String(x), "None")
        case _:
            assert_true(False)
    # `Optional[NoneType](None)` builds an empty `Optional`.
    var some = Optional[NoneType](init_with=_make_none)
    match some:
        case .Some(payload):
            comptime assert type_of(payload) == NoneType
            assert_equal(String(payload), "None")
        case .None:
            assert_true(False)
    match Optional[NoneType](init_with=_make_none):
        case .Some(var payload):
            comptime assert type_of(payload) == NoneType
            assert_equal(String(payload), "None")
        case .None:
            assert_true(False)


def _make_none() -> NoneType:
    return None


def _check_layout[T: AnyType]():
    comptime assert size_of[T]() == size_of[Tuple[T]]()
    comptime assert align_of[T]() == align_of[Tuple[T]]()


def test_optional_tuple_projection() raises:
    _check_layout[Tuple[]]()
    _check_layout[Int]()
    _check_layout[String]()
    _check_layout[SIMD[DType.float32, 16]]()
    var value = Optional[Int](3)
    ref payload = value._unsafe_get_enum_payload[1]()
    payload[0] = 4
    assert_equal(value.value(), 4)
    var reg = OptionalReg[Int](5)
    reg._unsafe_get_enum_payload[1]()[0] = 6
    assert_equal(reg.value(), 6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
