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

from std.format.tstring import _encode_format_string
from std.testing import assert_equal, assert_raises, assert_true, TestSuite


@fieldwise_init
struct Point(Writable):
    var x: Int
    var y: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(t"({self.x}, {self.y})")


def test_basic_tstring() raises:
    assert_equal(String(t"Hello, World!"), "Hello, World!")


def test_single_interpolation() raises:
    var name = "Alice"
    assert_equal(String(t"Hello, {name}!"), "Hello, Alice!")


def test_multiple_interpolations() raises:
    var x = 10
    var y = 20
    assert_equal(String(t"{x} + {y} = {x + y}"), "10 + 20 = 30")


def test_expression_interpolation() raises:
    assert_equal(String(t"Result: {2 * 3 + 1}"), "Result: 7")


def test_empty_tstring() raises:
    var s = t""
    assert_equal(String(s), "")


def test_tstring_only_expression() raises:
    assert_equal(String(t"{42}"), "42")


def test_escaped_braces() raises:
    assert_equal(String(t"Use {{braces}} like this"), "Use {braces} like this")


def test_mixed_escaped_and_interpolation() raises:
    var value = 123
    assert_equal(
        String(t"The value {{value}} = {value}"), "The value {value} = 123"
    )


def test_deeply_nested_escape_braces() raises:
    var x = 42
    assert_equal(String(t"{{{{{x}}}}}"), "{{42}}")


def test_adjacent_interpolations() raises:
    var a = "A"
    var b = "B"
    var c = "C"
    assert_equal(String(t"{a}{b}{c}"), "ABC")


def test_boolean_interpolation() raises:
    assert_equal(
        String(t"True: {True}, False: {False}"), "True: True, False: False"
    )


def test_integer_interpolation() raises:
    var i8 = Int8(127)
    var i16 = Int16(32767)
    var i32 = Int32(2147483647)
    var i64 = Int64(9223372036854775807)
    assert_equal(String(t"Int8: {i8}"), "Int8: 127")
    assert_equal(String(t"Int16: {i16}"), "Int16: 32767")
    assert_equal(String(t"Int32: {i32}"), "Int32: 2147483647")
    assert_equal(String(t"Int64: {i64}"), "Int64: 9223372036854775807")


def test_string_interpolation() raises:
    var msg = "world"
    # TODO(KGEN): Same bug as test_single_interpolation
    assert_equal(String(t"Hello, {msg}"), "Hello, world")


def test_writable_type() raises:
    var p = Point(10, 20)
    assert_equal(String(t"Point: {p}"), "Point: (10, 20)")


def test_nested_expressions() raises:
    var x = 10
    var y = 5
    assert_equal(String(t"Calc: {(x + y) * 2}"), "Calc: 30")


def test_multiple_same_variable() raises:
    var num = 5
    assert_equal(String(t"{num} * {num} = {num * num}"), "5 * 5 = 25")


def test_complex_expression() raises:
    var a = 2
    var b = 3
    var c = 4
    assert_equal(String(t"Result: {a * b * c + b * 2 + a * 5}"), "Result: 40")


def test_tstring_in_variable() raises:
    var x = 100
    var message = t"The value is {x}"
    assert_equal(String(message), "The value is 100")


def test_method_calls() raises:
    var s = String("hello")
    assert_equal(String(t"Uppercase: {s.upper()}"), "Uppercase: HELLO")
    assert_equal(String(t"Length: {s.__len__()}"), "Length: 5")


def test_list_subscripting() raises:
    var numbers = [10, 20, 30, 40, 50]
    assert_equal(String(t"First: {numbers[0]}"), "First: 10")
    assert_equal(String(t"Third: {numbers[2]}"), "Third: 30")
    assert_equal(String(t"Last: {numbers[4]}"), "Last: 50")


def test_attribute_access() raises:
    var p = Point(15, 25)
    assert_equal(String(t"X coordinate: {p.x}"), "X coordinate: 15")
    assert_equal(String(t"Y coordinate: {p.y}"), "Y coordinate: 25")


def test_chained_method_calls() raises:
    var text = String("  hello world  ")
    assert_equal(
        String(t"Stripped and upper: {text.strip().upper()}"),
        "Stripped and upper: HELLO WORLD",
    )


def test_subscript_with_expression() raises:
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


def test_method_on_literal() raises:
    assert_equal(
        String(t"Upper case: {String('mojo').upper()}"), "Upper case: MOJO"
    )


def test_complex_nested_expression() raises:
    var values = [5, 10, 15, 20]
    var multiplier = 3
    assert_equal(
        String(t"Computed: {values[1] * multiplier + values[2]}"),
        "Computed: 45",
    )


def test_conditional_expression() raises:
    var x = 10
    var y = 20
    var max_val = x if x > y else y
    assert_equal(String(t"Maximum: {max_val}"), "Maximum: 20")


def test_comparison_in_interpolation() raises:
    var a = 5
    var b = 10
    assert_equal(String(t"{a} < {b}: {a < b}"), "5 < 10: True")
    assert_equal(String(t"{a} > {b}: {a > b}"), "5 > 10: False")


def test_arithmetic_with_subscript() raises:
    var nums = [2, 4, 6, 8]
    assert_equal(
        String(t"Sum of first two: {nums[0] + nums[1]}"), "Sum of first two: 6"
    )
    assert_equal(
        String(t"Product of last two: {nums[2] * nums[3]}"),
        "Product of last two: 48",
    )


def test_string_method_with_args() raises:
    var text = String("hello-world-test")
    assert_equal(
        String(t"Split count: {len(text.split('-'))}"), "Split count: 3"
    )


def test_type_conversion_in_interpolation() raises:
    var num = 42
    var float_num = Float64(num)
    assert_equal(String(t"As float: {float_num}"), "As float: 42.0")


def test_same_quote_nested_string_double() raises:
    assert_equal(String(t"hello {"world"}"), "hello world")


def test_same_quote_nested_string_single() raises:
    assert_equal(String(t'hello {'world'}'), "hello world")


def test_same_quote_multiple_nested() raises:
    assert_equal(String(t"a {"b"} c {"d"}"), "a b c d")


def test_same_quote_triple_quoted() raises:
    assert_equal(String(t"""hello {"world"}"""), "hello world")


def test_mixed_quotes_double_outer() raises:
    assert_equal(String(t"outer {'inner'}"), "outer inner")


def test_mixed_quotes_single_outer() raises:
    assert_equal(String(t'outer {"inner"}'), "outer inner")


def test_nested_string_with_expression() raises:
    var count = 5
    assert_equal(String(t"Found {"item"} {count} times"), "Found item 5 times")


def test_escaped_quote_in_nested_string() raises:
    assert_equal(String(t'test {"say \"hello\""}'), 'test say "hello"')


def test_mutating_tsring_interpolated_value_before_written() raises:
    var x = "C++"
    var s = t"{x} is the best language"
    x = "Mojo"
    assert_equal(String(s), "Mojo is the best language")


def test_materialized_value_in_tstring() raises:
    comptime world = "World"
    assert_equal(String(t"Hello {world}"), "Hello World")


# =============================================================================
# Nested t-string tests (t-strings inside t-string interpolations)
# =============================================================================


def test_nested_tstring_different_quotes() raises:
    var x = 10
    assert_equal(String(t"Outer: {t'Inner: {x}'}"), "Outer: Inner: 10")


def test_nested_tstring_same_quote_double() raises:
    var value = 42
    assert_equal(String(t"Result: {t"{value}"}"), "Result: 42")


def test_nested_tstring_same_quote_single() raises:
    var num = 99
    assert_equal(String(t'Value: {t'{num}'}'), "Value: 99")


def test_nested_tstring_multiple() raises:
    var a = 1
    var b = 2
    assert_equal(
        String(t"First: {t'{a}'}, Second: {t'{b}'}"), "First: 1, Second: 2"
    )


def test_nested_tstring_triple_level() raises:
    var val = 7
    assert_equal(
        String(t"L1: {t'L2: {t"L3: {val}"}'}"),
        "L1: L2: L3: 7",
    )


def test_nested_tstring_triple_level_same_quotes() raises:
    var n = 3
    assert_equal(
        String(t"A {t"B {t"C {n}"}"}"),
        "A B C 3",
    )


def test_nested_tstring_with_expression() raises:
    var x = 5
    assert_equal(String(t"Double: {t'{x * 2}'}"), "Double: 10")


def test_nested_tstring_adjacent() raises:
    var a = 1
    var b = 2
    assert_equal(String(t"{t'{a}'}{t'{b}'}"), "12")


def test_nested_tstring_with_escaped_braces() raises:
    var x = 10
    assert_equal(
        String(t"Outer {{brace}} {t'Inner {x}'}"),
        "Outer {brace} Inner 10",
    )


def test_nested_tstring_both_escaped_braces() raises:
    var y = 20
    assert_equal(
        String(t"Out {{1}} {t'In {{2}} {y}'}"),
        "Out {1} In {2} 20",
    )


def test_nested_tstring_empty_outer() raises:
    var x = 123
    assert_equal(String(t"{t'{x}'}"), "123")


def test_tstring_with_escape_character() raises:
    var x = 123
    assert_equal(String(t"abc\t{x}"), "abc\t123")


def test_tstring_with_newline_escape() raises:
    var val = 42
    assert_equal(String(t"line1\n{val}"), "line1\n42")


def test_tstring_with_multiple_escapes() raises:
    var num = 99
    assert_equal(
        String(t"tab\there\nnewline\r{num}\t"), "tab\there\nnewline\r99\t"
    )


def test_tstring_with_punctuation_at_end() raises:
    var x = 10
    assert_equal(String(t"value: {x}!"), "value: 10!")


def test_tstring_with_backslash_escape() raises:
    var x = 10
    assert_equal(String(t"path\\to\\{x}"), "path\\to\\10")


def test_tstring_concatenation() raises:
    var x = 10
    var y = 20
    # fmt: off
    assert_equal(String(t"{x}" t"{y}"), "1020")
    # fmt: on


def test_tstring_multiline_concatenation() raises:
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


# =============================================================================
# _encode_format_string tests
# =============================================================================


fn _encode(*strings: String) -> List[Byte]:
    var result = List[Byte]()
    for i in range(len(strings)):
        for byte in strings[i].as_bytes():
            result.append(byte)
        result.append(0)
    return result^


def test_encode_plain_text() raises:
    var got = _encode_format_string("Hello!")
    var want = _encode("Hello!")
    assert_equal(got, want)


def test_encode_empty_string() raises:
    # "" -> "\0"
    assert_equal(_encode_format_string(""), _encode(""))


def test_encode_single_replacement() raises:
    # "Hello, {}!" -> "Hello, \0!\0"
    assert_equal(_encode_format_string("Hello, {}!"), _encode("Hello, ", "!"))


def test_encode_multiple_replacements() raises:
    # "{} + {} = {}" -> "\0 + \0 = \0\0"
    assert_equal(
        _encode_format_string("{} + {} = {}"),
        _encode("", " + ", " = ", ""),
    )


def test_encode_escaped_braces() raises:
    # "{{braces}}" -> "{braces}\0"
    assert_equal(_encode_format_string("{{braces}}"), _encode("{braces}"))


def test_encode_blog_post_example() raises:
    # "Hello, {}, {{I'm}} {}!" -> "Hello, \0, {I'm} \0!\0"
    assert_equal(
        _encode_format_string("Hello, {}, {{I'm}} {}!"),
        _encode("Hello, ", ", {I'm} ", "!"),
    )


def test_encode_only_replacement_field() raises:
    # "{}" -> "\0\0"
    assert_equal(_encode_format_string("{}"), _encode("", ""))


def test_encode_adjacent_replacements() raises:
    # "{}{}{}" -> "\0\0\0\0"
    assert_equal(_encode_format_string("{}{}{}"), _encode("", "", "", ""))


def test_encode_only_escaped_braces() raises:
    # "{{}}" -> "{}\0"
    assert_equal(_encode_format_string("{{}}"), _encode("{}"))


def test_encode_adjacent_escaped_braces() raises:
    # "{{}}{{}}" -> "{}{}\0"
    assert_equal(_encode_format_string("{{}}{{}}"), _encode("{}{}"))


def test_encode_deeply_nested_escaped_braces() raises:
    # "{{{{{}}}}}" -> "{{" + NUL (from {}) + "}}\0"
    assert_equal(_encode_format_string("{{{{{}}}}}"), _encode("{{", "}}"))


def test_encode_escaped_braces_adjacent_to_replacement() raises:
    # "{{}}{}{{}}": escaped pair + replacement + escaped pair -> "{}\0{}\0"
    assert_equal(_encode_format_string("{{}}{}{{}}"), _encode("{}", "{}"))


def test_encode_replacement_at_start() raises:
    # "{}tail" -> "\0tail\0"
    assert_equal(_encode_format_string("{}tail"), _encode("", "tail"))


def test_encode_replacement_at_end() raises:
    # "head{}" -> "head\0\0"
    assert_equal(_encode_format_string("head{}"), _encode("head", ""))


def test_encode_errors() raises:
    for invalid in ["hello }", "}}}", "}"]:
        with assert_raises(contains="single '}'"):
            _ = _encode_format_string(invalid)

    for invalid in ["{0}", "{name}", "{abc", "{!r}", "{:.2f}", "{{{", "{}{"]:
        with assert_raises(
            contains="unclosed/non-empty replacement field in format string"
        ):
            _ = _encode_format_string(invalid)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
