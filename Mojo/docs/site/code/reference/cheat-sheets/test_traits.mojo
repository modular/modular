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
# test_traits.mojo
# Tests for the Mojo "traits" cheat-sheet card.
#
# Trait conformance is a compile-time fact, so most of this file is a
# COMPILE-FLOOR: if a claimed conformance or required-method set stops being
# true, this file stops compiling. A handful of behaviors are also asserted at
# runtime.
#
# Not tested (no portable runtime API to assert):
#   - Hasher internals; Strategy (property-based, needs an Rng)
#   - the rendered text of an else message (it appears only in a compile
#     error)
#   - var b = a^, return local^, and append() rejecting a non-Movable value
#     (compile errors)
#   - Identifiable __is__ (no portable value-type identity to assert)
#   - PathLike / ConvertibleToPython / ConvertibleFromPython (need os / Python)
#   - exact signature text of each requirement (the compile-floor covers that a
#     conforming type satisfies the trait, not the literal spelling)
from std.testing import assert_equal, assert_false, assert_true
from std.math import ceil, floor, trunc


# A struct that conforms to the value + compare + format traits the card lists.
# Implementing only the required methods (__eq__, __lt__, write_to) and letting
# the rest be provided/synthesized is itself the test: it compiles only if the
# card's required-method claims are correct.
struct Meters(Comparable, Copyable, Movable, Writable):
    var v: Int

    def __init__(out self, v: Int):
        self.v = v

    def __eq__(self, other: Self) -> Bool:
        return self.v == other.v

    def __lt__(self, rhs: Self) -> Bool:
        return self.v < rhs.v

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.v, "m")


def test_value_compare_format_floor() raises:
    # Copyable / Movable: build and store in a List
    var xs: List[Meters] = [Meters(3), Meters(1), Meters(2)]
    assert_equal(len(xs), 3)
    # Comparable: __lt__ implemented; __gt__/__le__/__ge__ provided
    assert_true(Meters(1) < Meters(2))
    assert_true(Meters(2) > Meters(1))
    assert_true(Meters(2) <= Meters(2))
    # Equatable (refined by Comparable): __eq__ implemented, __ne__ provided
    assert_true(Meters(1) == Meters(1))
    assert_true(Meters(1) != Meters(2))
    # Writable: String() calls write_to()
    assert_equal(String(Meters(5)), "5m")


@fieldwise_init
struct Seven(Floatable, Intable):  # conforms by implementing the methods
    var v: Int

    def __int__(self) -> Int:
        return self.v

    def __float__(self) -> Float64:
        return Float64(self.v)


def test_conversions() raises:
    # Boolable / Intable / Floatable on builtins
    assert_true(Bool(1))
    assert_equal(Int(Float64(3.9)), 3)  # Float64->Int truncates toward zero
    assert_equal(Float64(5), 5.0)  # Int -> Float64
    # Intable converts to any integer type; Floatable only to Float64
    var s = Seven(7)
    assert_equal(Int8(s), 7)
    assert_equal(UInt64(s), 7)
    assert_equal(Int128(s), 7)
    assert_equal(Float64(s), 7.0)


def test_sized() raises:
    assert_equal(len([1, 2, 3]), 3)
    assert_equal(len(["a", "b"]), 2)


def test_unary_math() raises:
    # Absable / Powable / Roundable
    assert_equal(abs(-5), 5)
    assert_equal(2**8, 256)
    assert_equal(round(3.6), 4.0)
    # Ceilable / Floorable / Truncable
    assert_equal(ceil(2.1), 3.0)
    assert_equal(floor(2.9), 2.0)
    assert_equal(trunc(2.9), 2.0)


def test_divmod() raises:
    # DivModable -> divmod(); the tuple is (quotient, remainder)
    var qr = divmod(7, 2)
    assert_equal(qr[0], 3)
    assert_equal(qr[1], 1)


def test_comparable_builtins() raises:
    assert_true(1 < 2)
    assert_true("a" < "b")


def test_hashable() raises:
    # Hashable must agree with Equatable: equal values hash equal
    assert_equal(hash(String("mojo")), hash(String("mojo")))


def test_iterable() raises:
    # Iterable + Iterator drive the for loop
    var total = 0
    for x in [10, 20, 30]:
        total += x
    assert_equal(total, 60)


def test_conformance_claims() raises:
    # The card's conformance claims, as a compile-time check.
    assert_true(conforms_to(String, Writable))
    assert_true(conforms_to(String, Equatable))
    assert_true(conforms_to(String, Comparable))
    assert_true(conforms_to(String, Hashable))
    assert_true(conforms_to(Int, Intable))
    assert_true(conforms_to(Int, Comparable))
    # Refinement: Comparable refines Equatable, so an ordered type is equatable.
    assert_true(conforms_to(Int, Equatable))


# Movable is needed to put a value into a container, not to name one.
struct Pinned(not Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    def unpin(deinit self) -> Int:
        return self.x


def take_pinned(var p: Pinned) -> Int:
    return p.x


struct Linear(not Deinitable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x

    def done(deinit self) -> Int:  # the only way to end a Linear value
        return self.x


def make_pinned() -> Pinned:
    return Pinned(7)  # a fresh value returns without a move


struct Undeclared:  # no conformance list at all
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


def test_movable_opt_outs() raises:
    # Movable (and Deinitable) are implicit; Copyable is not
    assert_true(conforms_to(Undeclared, AnyType))
    assert_true(conforms_to(Undeclared, Movable))
    assert_true(conforms_to(Undeclared, Deinitable))
    assert_false(conforms_to(Undeclared, Copyable))
    var moved = Undeclared(1)
    var target = moved^  # relocates to a new binding
    assert_equal(target.x, 1)
    var pins = List[Pinned]()  # a non-movable type can name a List
    assert_equal(len(pins), 0)
    var maybe = Optional[Pinned]()  # ...or an empty Optional
    assert_true(not maybe)
    var p = make_pinned()
    assert_equal(p.x, 7)
    # ^ transfers without Movable when the value is consumed where it is
    var into_arg = Pinned(8)
    assert_equal(take_pinned(into_arg^), 8)  # into a var argument
    var into_deinit = Pinned(9)
    assert_equal(into_deinit^.unpin(), 9)  # into a deinit method
    var l = Linear(3)
    assert_equal(l^.done(), 3)


# Conformance syntax: a conditional conformance exists only when proven.
@fieldwise_init
struct Box[T: Copyable & Deinitable](Hashable where conforms_to(T, Hashable)):
    var value: Self.T


struct NoHash(Copyable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


def test_conformance_syntax() raises:
    assert_true(conforms_to(Box[Int], Hashable))  # T is Hashable, so Box is
    assert_false(conforms_to(Box[NoHash], Hashable))  # T isn't, so Box isn't
    assert_equal(hash(Box[Int](5)), hash(Box[Int](5)))
    var plain = Box[NoHash](NoHash(1))  # still a Box, just not Hashable
    assert_equal(plain.value.x, 1)


def test_string_not_sized() raises:
    var s = String("café")
    assert_equal(s.byte_length(), 5)
    assert_equal(s.count_codepoints(), 4)
    assert_equal(s.count_graphemes(), 4)


# Diagnostics: messages after else compile as literals.
def chunk[
    w: Int
]() -> Int where w.is_power_of_two() else ("'w' must be a power of two"):
    return w


def test_else_messages() raises:
    assert_equal(chunk[8](), 8)  # a passing condition shows no message


def main() raises:
    test_value_compare_format_floor()
    test_conversions()
    test_sized()
    test_unary_math()
    test_divmod()
    test_comparable_builtins()
    test_hashable()
    test_iterable()
    test_conformance_claims()
    test_movable_opt_outs()
    test_conformance_syntax()
    test_else_messages()
    test_string_not_sized()
