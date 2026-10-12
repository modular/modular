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
# tests.mojo
# Tests for python-to-mojo.mdx code examples and prose claims.
#
# Type-identity claims compare `reflect[...].name()` of two types rather than
# a hard-coded name string, so a change in the name format doesn't fail them.
#
# Page claims, and where each is verified:
#   1. A bracketed literal is an `Array[Int, 3]`, its size stays 3, and its
#      elements can be reassigned.
#      -> test_bracket_literal_is_array
#   2. An explicit `List` or `List[Int]` annotation makes the literal a list.
#      -> test_annotated_literal_is_list
#   3. A `List` has one element type and can grow; a `Dict` has fixed key and
#      value types.
#      -> test_list_append, test_dict_insert
#   4. `Variant` works as the element type of a collection that holds a known
#      set of types.
#      -> test_variant_elements
#   5. List, dictionary, and set comprehensions and `for` and `while` loops
#      are supported.
#      -> test_comprehensions, test_loops
#   6. A `ref` binding shares the value: appending through `b` changes `a`.
#      -> test_ref_shares_value
#   7. `Int` copies implicitly; the copy is independent.
#      -> test_int_copies_implicitly
#   8. A `List` copy made with `.copy()` is independent; ownership can also
#      move with `^`.
#      -> test_list_explicit_copy, test_list_transfer
#   9. A `Variant[String, Int]` variable keeps its type while the held value
#      switches between the allowed types.
#      -> test_variant_switches_held_type
#  10. A variable can be reassigned a value of the same type.
#      -> test_variable_reassignment
#  11. A `mut` argument modifies the caller's value; a `mut self` method
#      modifies the instance.
#      -> test_mut_argument, test_mut_self_method
#  12. A `var` argument takes ownership of the caller's value.
#      -> test_var_argument_takes_ownership
#  13. `Int` is machine width; `Int8`, `Int32`, and `Int64` are sized.
#      -> test_int_is_machine_width, test_sized_integers
#  14. `Int32` and `Float32` are one-element `SIMD` aliases, and `SIMD`
#      holds fixed-size vectors.
#      -> test_scalar_aliases_are_simd, test_simd_vector
#  15. `Int / Int` produces an `Int` (7 / 2 is 3); integer-literal division
#      produces a floating-point result at compile time (7 / 2 is 3.5).
#      -> test_int_division, test_literal_division
#  16. The `Point` initializer uses `out self` to initialize every field;
#      `@fieldwise_init` synthesizes an equivalent initializer.
#      -> test_point_initializer, test_fieldwise_init
#  17. A trait states the interface and the compiler checks conformance.
#      -> test_trait_bound
#  18. A raising function declares `raises`; a non-raising caller handles the
#      error with `try`/`except`.
#      -> test_validate_raises, test_check_handles_error
#  19. A function can declare a typed error.
#      -> test_typed_error
#
# Not tested (the page shows these as compile errors, so a passing build
# can't contain them; each is hand-verified with `mojo build`):
#   - `a.append(4)` on an `Array` (no `append` attribute)
#   - `value = "hello"` after `var value = 1` (can't convert `String` to `Int`)
#   - `value += 1` on a default (immutable) argument
#   - calling `validate()` from a non-raising function without `try`
#   - a bare `var b = a` on a `List` (not `ImplicitlyCopyable`)
#   - "Mojo doesn't have a general `Float` type" (`Float` is an unknown
#     declaration)
#   - adding a field to a struct instance at runtime, and passing a
#     non-`Drawable` type to `sketch()` (no runnable form)
#
# Not tested (no runnable behavior):
#   - the Python blocks
#   - "Memory uses ownership" and "Don't translate Python mechanically"
#     (guidance)
from std.reflection import reflect
from std.sys import bit_width_of, size_of
from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.utils import Variant


def same_type[A: AnyType, B: AnyType]() -> Bool:
    return reflect[A].name() == reflect[B].name()


# --- The default collection isn't List ---


def test_bracket_literal_is_array() raises:
    var a = [1, 2, 3]
    assert_true(same_type[type_of(a), Array[Int, 3]]())
    assert_equal(len(a), 3)
    a[0] = 10
    assert_equal(a[0], 10)
    assert_equal(len(a), 3)


def test_annotated_literal_is_list() raises:
    var a: List = [1, 2, 3]
    var b: List[Int] = [1, 2, 3]
    assert_true(same_type[type_of(a), List[Int]]())
    assert_true(same_type[type_of(b), List[Int]]())
    assert_equal(a, b)


def test_list_append() raises:
    var values: List[String] = ["one", "two", "three"]
    values.append("four")
    assert_equal(len(values), 4)
    assert_equal(values[3], "four")


def test_dict_insert() raises:
    var counts: Dict[String, Int] = {"a": 1, "b": 2}
    counts["c"] = 3
    assert_equal(len(counts), 3)
    assert_equal(counts["c"], 3)


def test_variant_elements() raises:
    comptime Field = Variant[String, Int]
    var row: List[Field] = [Field("Ada"), Field(36)]
    assert_true(row[0].isa[String]())
    assert_true(row[1].isa[Int]())
    assert_equal(row[0][String], "Ada")
    assert_equal(row[1][Int], 36)


def test_comprehensions() raises:
    var squares = [n * n for n in range(5) if n % 2 == 0]
    assert_equal(squares, [0, 4, 16])
    var lengths = {word: word.byte_length() for word in ["red", "green"]}
    assert_equal(lengths["green"], 5)
    var remainders = {n % 3 for n in range(9)}
    assert_equal(len(remainders), 3)


def test_loops() raises:
    var total = 0
    for n in range(1, 5):
        total += n
    assert_equal(total, 10)
    var countdown = 3
    while countdown > 0:
        countdown -= 1
    assert_equal(countdown, 0)


# --- Values aren't Python references ---


def test_ref_shares_value() raises:
    var a: List[Int] = [1, 2, 3]
    ref b = a
    b.append(4)
    assert_equal(a, [1, 2, 3, 4])


def test_int_copies_implicitly() raises:
    var a = 10
    var b = a
    b += 10
    assert_equal(a, 10)
    assert_equal(b, 20)


def test_list_explicit_copy() raises:
    var a: List[Int] = [1, 2, 3]
    var b = a.copy()
    b.append(4)
    assert_equal(a, [1, 2, 3])
    assert_equal(b, [1, 2, 3, 4])


def test_list_transfer() raises:
    var a: List[Int] = [1, 2, 3]
    var b = a^
    assert_equal(b, [1, 2, 3])


# --- Types stay fixed ---


def test_variant_switches_held_type() raises:
    var value: Variant[String, Int] = String("hello")
    assert_true(value.isa[String]())
    value = 10
    assert_true(value.isa[Int]())
    assert_false(value.isa[String]())
    assert_true(same_type[type_of(value), Variant[String, Int]]())


# --- Variables are mutable; function arguments aren't by default ---


def test_variable_reassignment() raises:
    var value = 10
    assert_equal(value, 10)
    value = 20
    assert_equal(value, 20)


def increment(mut value: Int):
    value += 1


def test_mut_argument() raises:
    var value = 20
    increment(value)
    assert_equal(value, 21)


@fieldwise_init
struct Struct:
    var name: String

    def update_name(mut self, new_name: String):
        self.name = new_name


def test_mut_self_method() raises:
    var name_struct = Struct("Mojo")
    assert_equal(name_struct.name, "Mojo")
    name_struct.update_name("Hello")
    assert_equal(name_struct.name, "Hello")


def consume(var values: List[Int]) -> Int:
    return len(values)


def test_var_argument_takes_ownership() raises:
    var values: List[Int] = [1, 2, 3]
    assert_equal(consume(values^), 3)


# --- Numeric representation is explicit ---


def test_int_is_machine_width() raises:
    var count = 0
    var pointer = Pointer(to=count)
    assert_equal(size_of[Int](), size_of[type_of(pointer)]())
    assert_equal(pointer[], 0)


def test_sized_integers() raises:
    assert_equal(bit_width_of[DType.int8](), 8)
    assert_equal(bit_width_of[DType.int32](), 32)
    assert_equal(bit_width_of[DType.int64](), 64)


def test_scalar_aliases_are_simd() raises:
    assert_true(same_type[Int32, SIMD[DType.int32, 1]]())
    assert_true(same_type[Float32, SIMD[DType.float32, 1]]())


def test_simd_vector() raises:
    var v = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    assert_equal(v * 2, SIMD[DType.float32, 4](2.0, 4.0, 6.0, 8.0))


# --- Watch division types ---


def test_int_division() raises:
    var a: Int = 7
    var b: Int = 2
    assert_true(same_type[type_of(a / b), Int]())
    assert_equal(a / b, 3)


def test_literal_division() raises:
    comptime half = 7 / 2
    assert_equal(Float64(half), 3.5)


# --- Structs aren't dynamic Python classes ---


struct Point:
    var x: Int
    var y: Int

    def __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


@fieldwise_init
struct FieldwisePoint:
    var x: Int
    var y: Int


def test_point_initializer() raises:
    var p = Point(3, 4)
    assert_equal(p.x, 3)
    assert_equal(p.y, 4)


def test_fieldwise_init() raises:
    var p = FieldwisePoint(3, 4)
    assert_equal(p.x, 3)
    assert_equal(p.y, 4)


# --- Traits make interfaces explicit ---


trait Drawable:
    def draw(self) -> String:
        ...


@fieldwise_init
struct Circle(Drawable):
    var radius: Int

    def draw(self) -> String:
        return String(t"circle r={self.radius}")


def sketch[T: Drawable](shape: T) -> String:
    return shape.draw()


def test_trait_bound() raises:
    comptime assert conforms_to(Circle, Drawable)
    assert_equal(sketch(Circle(2)), "circle r=2")


# --- Error propagation is explicit ---


def validate(value: Int) raises:
    if value < 0:
        raise Error("value must be nonnegative")


def check(value: Int) -> String:
    try:
        validate(value)
    except e:
        return String(e)
    return "ok"


def test_validate_raises() raises:
    validate(5)
    with assert_raises(contains="value must be nonnegative"):
        validate(-1)


def test_check_handles_error() raises:
    assert_equal(check(5), "ok")
    assert_equal(check(-1), "value must be nonnegative")


@fieldwise_init
struct ParseError(Writable):
    var position: Int


def parse_digit(text: String) raises ParseError -> Int:
    var digit = Int(text[byte=0].as_bytes()[0]) - 48
    if digit < 0 or digit > 9:
        raise ParseError(0)
    return digit


def test_typed_error() raises:
    assert_equal(parse_digit("7"), 7)
    try:
        _ = parse_digit("x")
    except e:
        assert_equal(e.position, 0)
        return
    raise Error("parse_digit should have raised ParseError")


def main() raises:
    test_bracket_literal_is_array()
    test_annotated_literal_is_list()
    test_list_append()
    test_dict_insert()
    test_variant_elements()
    test_comprehensions()
    test_loops()
    test_ref_shares_value()
    test_int_copies_implicitly()
    test_list_explicit_copy()
    test_list_transfer()
    test_variant_switches_held_type()
    test_variable_reassignment()
    test_mut_argument()
    test_mut_self_method()
    test_var_argument_takes_ownership()
    test_int_is_machine_width()
    test_sized_integers()
    test_scalar_aliases_are_simd()
    test_simd_vector()
    test_int_division()
    test_literal_division()
    test_point_initializer()
    test_fieldwise_init()
    test_trait_bound()
    test_validate_raises()
    test_check_handles_error()
    test_typed_error()
