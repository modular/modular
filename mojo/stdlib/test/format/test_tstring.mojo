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

from testing import assert_equal, assert_true, TestSuite


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(t"({self.x}, {self.y})")


def test_basic_tstring():
    assert_equal(String(t"Hello, World!"), "Hello, World!")


def test_single_interpolation():
    var name = "Alice"
    assert_equal(String(t"Hello, {name}!"), "Hello, Alice!")


def test_multiple_interpolations():
    var x = 10
    var y = 20
    assert_equal(String(t"{x} + {y} = {x + y}"), "10 + 20 = 30")


def test_expression_interpolation():
    assert_equal(String(t"Result: {2 * 3 + 1}"), "Result: 7")


def test_empty_tstring():
    var s = t""
    assert_equal(String(s), "")


def test_tstring_only_expression():
    assert_equal(String(t"{42}"), "42")


def test_escaped_braces():
    assert_equal(String(t"Use {{braces}} like this"), "Use {braces} like this")


def test_mixed_escaped_and_interpolation():
    var value = 123
    assert_equal(
        String(t"The value {{value}} = {value}"), "The value {value} = 123"
    )


def test_deeply_nested_escape_braces():
    var x = 42
    assert_equal(String(t"{{{{{x}}}}}"), "{{42}}")


def test_adjacent_interpolations():
    var a = "A"
    var b = "B"
    var c = "C"
    assert_equal(String(t"{a}{b}{c}"), "ABC")


def test_boolean_interpolation():
    assert_equal(
        String(t"True: {True}, False: {False}"), "True: True, False: False"
    )


def test_integer_interpolation():
    var i8 = Int8(127)
    var i16 = Int16(32767)
    var i32 = Int32(2147483647)
    var i64 = Int64(9223372036854775807)
    assert_equal(String(t"Int8: {i8}"), "Int8: 127")
    assert_equal(String(t"Int16: {i16}"), "Int16: 32767")
    assert_equal(String(t"Int32: {i32}"), "Int32: 2147483647")
    assert_equal(String(t"Int64: {i64}"), "Int64: 9223372036854775807")


def test_string_interpolation():
    var msg = "world"
    # TODO(KGEN): Same bug as test_single_interpolation
    assert_equal(String(t"Hello, {msg}"), "Hello, world")


def test_writable_type():
    var p = Point(10, 20)
    assert_equal(String(t"Point: {p}"), "Point: (10, 20)")


def test_nested_expressions():
    var x = 10
    var y = 5
    assert_equal(String(t"Calc: {(x + y) * 2}"), "Calc: 30")


def test_multiple_same_variable():
    var num = 5
    assert_equal(String(t"{num} * {num} = {num * num}"), "5 * 5 = 25")


def test_complex_expression():
    var a = 2
    var b = 3
    var c = 4
    assert_equal(String(t"Result: {a * b * c + b * 2 + a * 5}"), "Result: 40")


def test_tstring_in_variable():
    var x = 100
    var message = t"The value is {x}"
    assert_equal(String(message), "The value is 100")


def test_method_calls():
    var s = String("hello")
    assert_equal(String(t"Uppercase: {s.upper()}"), "Uppercase: HELLO")
    assert_equal(String(t"Length: {s.__len__()}"), "Length: 5")


def test_list_subscripting():
    var numbers = [10, 20, 30, 40, 50]
    assert_equal(String(t"First: {numbers[0]}"), "First: 10")
    assert_equal(String(t"Third: {numbers[2]}"), "Third: 30")
    assert_equal(String(t"Last: {numbers[4]}"), "Last: 50")


def test_attribute_access():
    var p = Point(15, 25)
    assert_equal(String(t"X coordinate: {p.x}"), "X coordinate: 15")
    assert_equal(String(t"Y coordinate: {p.y}"), "Y coordinate: 25")


def test_chained_method_calls():
    var text = String("  hello world  ")
    assert_equal(
        String(t"Stripped and upper: {text.strip().upper()}"),
        "Stripped and upper: HELLO WORLD",
    )


def test_subscript_with_expression():
    var data = [100, 200, 300, 400]
    var index = 2
    assert_equal(
        String(t"Value at index {index}: {data[index]}"),
        "Value at index 2: 300",
    )
    assert_equal(
        String(t"Value at computed index: {data[index + 1]}"),
        "Value at computed index: 400",
    )


def test_method_on_literal():
    assert_equal(
        String(t"Upper case: {String('mojo').upper()}"), "Upper case: MOJO"
    )


def test_complex_nested_expression():
    var values = [5, 10, 15, 20]
    var multiplier = 3
    assert_equal(
        String(t"Computed: {values[1] * multiplier + values[2]}"),
        "Computed: 45",
    )


def test_conditional_expression():
    var x = 10
    var y = 20
    var max_val = x if x > y else y
    assert_equal(String(t"Maximum: {max_val}"), "Maximum: 20")


def test_comparison_in_interpolation():
    var a = 5
    var b = 10
    assert_equal(String(t"{a} < {b}: {a < b}"), "5 < 10: True")
    assert_equal(String(t"{a} > {b}: {a > b}"), "5 > 10: False")


def test_arithmetic_with_subscript():
    var nums = [2, 4, 6, 8]
    assert_equal(
        String(t"Sum of first two: {nums[0] + nums[1]}"), "Sum of first two: 6"
    )
    assert_equal(
        String(t"Product of last two: {nums[2] * nums[3]}"),
        "Product of last two: 48",
    )


def test_string_method_with_args():
    var text = String("hello-world-test")
    assert_equal(
        String(t"Split count: {len(text.split('-'))}"), "Split count: 3"
    )


def test_type_conversion_in_interpolation():
    var num = 42
    var float_num = Float64(num)
    assert_equal(String(t"As float: {float_num}"), "As float: 42.0")


def test_same_quote_nested_string_double():
    assert_equal(String(t"hello {"world"}"), "hello world")


def test_same_quote_nested_string_single():
    assert_equal(String(t'hello {'world'}'), "hello world")


def test_same_quote_multiple_nested():
    assert_equal(String(t"a {"b"} c {"d"}"), "a b c d")


def test_same_quote_triple_quoted():
    assert_equal(String(t"""hello {"world"}"""), "hello world")


def test_mixed_quotes_double_outer():
    assert_equal(String(t"outer {'inner'}"), "outer inner")


def test_mixed_quotes_single_outer():
    assert_equal(String(t'outer {"inner"}'), "outer inner")


def test_nested_string_with_expression():
    var count = 5
    assert_equal(String(t"Found {"item"} {count} times"), "Found item 5 times")


def test_escaped_quote_in_nested_string():
    assert_equal(String(t'test {"say \"hello\""}'), 'test say "hello"')


def test_mutating_tsring_interpolated_value_before_written():
    var x = "C++"
    var s = t"{x} is the best language"
    x = "Mojo"
    assert_equal(String(s), "Mojo is the best language")


def test_materialized_value_in_tstring():
    comptime world = "World"
    assert_equal(String(t"Hello {world}"), "Hello World")


# =============================================================================
# Nested t-string tests (t-strings inside t-string interpolations)
# =============================================================================


def test_nested_tstring_different_quotes():
    var x = 10
    assert_equal(String(t"Outer: {t'Inner: {x}'}"), "Outer: Inner: 10")


def test_nested_tstring_same_quote_double():
    var value = 42
    assert_equal(String(t"Result: {t"{value}"}"), "Result: 42")


def test_nested_tstring_same_quote_single():
    var num = 99
    assert_equal(String(t'Value: {t'{num}'}'), "Value: 99")


def test_nested_tstring_multiple():
    var a = 1
    var b = 2
    assert_equal(
        String(t"First: {t'{a}'}, Second: {t'{b}'}"), "First: 1, Second: 2"
    )


def test_nested_tstring_triple_level():
    var val = 7
    assert_equal(
        String(t"L1: {t'L2: {t"L3: {val}"}'}"),
        "L1: L2: L3: 7",
    )


def test_nested_tstring_triple_level_same_quotes():
    var n = 3
    assert_equal(
        String(t"A {t"B {t"C {n}"}"}"),
        "A B C 3",
    )


def test_nested_tstring_with_expression():
    var x = 5
    assert_equal(String(t"Double: {t'{x * 2}'}"), "Double: 10")


def test_nested_tstring_adjacent():
    var a = 1
    var b = 2
    assert_equal(String(t"{t'{a}'}{t'{b}'}"), "12")


def test_nested_tstring_with_escaped_braces():
    var x = 10
    assert_equal(
        String(t"Outer {{brace}} {t'Inner {x}'}"),
        "Outer {brace} Inner 10",
    )


def test_nested_tstring_both_escaped_braces():
    var y = 20
    assert_equal(
        String(t"Out {{1}} {t'In {{2}} {y}'}"),
        "Out {1} In {2} 20",
    )


def test_nested_tstring_empty_outer():
    var x = 123
    assert_equal(String(t"{t'{x}'}"), "123")


def test_tstring_with_escape_character():
    var x = 123
    assert_equal(String(t"abc\t{x}"), "abc\t123")


def test_tstring_with_newline_escape():
    var val = 42
    assert_equal(String(t"line1\n{val}"), "line1\n42")


def test_tstring_with_multiple_escapes():
    var num = 99
    assert_equal(
        String(t"tab\there\nnewline\r{num}\t"), "tab\there\nnewline\r99\t"
    )


def test_tstring_with_punctuation_at_end():
    var x = 10
    assert_equal(String(t"value: {x}!"), "value: 10!")


def test_tstring_with_backslash_escape():
    var x = 10
    assert_equal(String(t"path\\to\\{x}"), "path\\to\\10")


def test_tstring_concatenation():
    var x = 10
    var y = 20
    # fmt: off
    assert_equal(String(t"{x}" t"{y}"), "1020")
    # fmt: on


def test_tstring_multiline_concatenation():
    var x = 10
    var y = 20
    # fmt: off
    var tstring = (
        t"This is a multiline {x}"
        t" tstring expression that will "
        t"concatenate, {y}!"
    )
    # fmt: on

    assert_equal(
        String(tstring),
        "This is a multiline 10 tstring expression that will concatenate, 20!",
    )


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
