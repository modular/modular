# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.string.string import (
    _calc_initial_buffer_size_int32,
    _calc_initial_buffer_size_int64,
)
from math import isinf, isnan

from memory import memcpy
from python import Python, PythonObject
from testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)


@fieldwise_init
struct AString(Stringable):
    fn __str__(self) -> String:
        return "a string"


def test_stringable():
    assert_equal("hello", String("hello"))
    assert_equal("0", String(0))
    assert_equal("AAA", String(StringSlice("AAA")))
    assert_equal("a string", String(AString()))

    # Should not produce an exclusivity error.
    # Incorrect exclusivity error when printing a StringSlice
    # https://github.com/modular/modular/issues/4790
    print(String("test").removeprefix(""))


def test_constructors():
    # Default construction
    assert_equal(0, len(String()))
    assert_true(not String())
    assert_not_equal("xyz", "abc")

    # Construction from Int
    var s0 = String(0)
    assert_equal("0", String(0))
    assert_equal(1, len(s0))

    var s1 = String(123)
    assert_equal("123", String(123))
    assert_equal(3, len(s1))

    # Construction from StringLiteral
    var s2 = "abc"
    assert_equal("abc", String(s2))
    assert_equal(3, len(s2))

    # Construction with capacity
    var s4 = String(capacity=1)
    assert_true(s4.capacity() <= String.INLINE_CAPACITY)

    # Construction from Codepoint
    var s5 = String(Codepoint(65))
    assert_true(s5.capacity() <= String.INLINE_CAPACITY)
    assert_equal(s5, "A")


def test_copy():
    var s0 = "find"
    var s1 = String(s0)
    s1.unsafe_ptr_mut()[3] = ord("e")
    assert_equal("find", s0)
    assert_equal("fine", s1)


def test_len():
    # String length is in bytes, not codepoints.
    var s0 = "ನಮಸ್ಕಾರ"

    assert_equal(len(s0), 21)
    assert_equal(len(s0.codepoints()), 7)

    # For ASCII string, the byte and codepoint length are the same:
    var s1 = "abc"

    assert_equal(len(s1), 3)
    assert_equal(len(s1.codepoints()), 3)


def test_equality_operators():
    var s0 = "abc"
    var s1 = "def"
    assert_equal(s0, s0)
    assert_not_equal(s0, s1)

    var s2 = "abc"
    assert_equal(s0, s2)
    # Explicitly invoke eq and ne operators
    assert_true(s0 == s2)
    assert_false(s0 != s2)

    # Is case sensitive
    var s3 = "ABC"
    assert_not_equal(s0, s3)

    # Implicit conversion can promote for eq and ne
    assert_equal(s0, "abc")
    assert_not_equal(s0, "notabc")


def test_comparison_operators():
    var abc = "abc"
    var de = "de"
    var ABC = "ABC"
    var ab = "ab"
    var abcd = "abcd"

    # Test less than and greater than
    assert_true(String.__lt__(abc, de))
    assert_false(String.__lt__(de, abc))
    assert_false(String.__lt__(abc, abc))
    assert_true(String.__lt__(ab, abc))
    assert_true(String.__gt__(abc, ab))
    assert_false(String.__gt__(abc, abcd))

    # Test less than or equal to and greater than or equal to
    assert_true(String.__le__(abc, de))
    assert_true(String.__le__(abc, abc))
    assert_false(String.__le__(de, abc))
    assert_true(String.__ge__(abc, abc))
    assert_false(String.__ge__(ab, abc))
    assert_true(String.__ge__(abcd, abc))

    # Test case sensitivity in comparison (assuming ASCII order)
    assert_true(String.__gt__(abc, ABC))
    assert_false(String.__le__(abc, ABC))

    # Testing with implicit conversion
    assert_true(String.__lt__(abc, "defgh"))
    assert_false(String.__gt__(abc, "xyz"))
    assert_true(String.__ge__(abc, "abc"))
    assert_false(String.__le__(abc, "ab"))

    # Test comparisons involving empty strings
    assert_true(String.__lt__("", abc))
    assert_false(String.__lt__(abc, ""))
    assert_true(String.__le__("", ""))
    assert_true(String.__ge__("", ""))


def test_add():
    var s1 = "123"
    var s2 = "abc"
    var s3 = s1 + s2
    assert_equal("123abc", s3)

    var s4 = "x"
    var s5 = s4.join(1, 2, 3)
    assert_equal("1x2x3", s5)

    var s6 = s4.join(s1, s2)
    assert_equal("123xabc", s6)

    var s7 = String()
    assert_equal("abc", s2 + s7)

    assert_equal("abcdef", s2 + "def")
    assert_equal("123abc", "123" + s2)

    var s8 = "abc is "
    var s9 = AString()
    assert_equal("abc is a string", String(s8) + String(s9))


def test_add_string_slice():
    var s1 = "123"
    var s2 = StringSlice("abc")
    var s3 = "abc"
    assert_equal("123abc", s1 + s2)
    assert_equal("123abc", s1 + s3)
    assert_equal("abc123", s2 + s1)
    assert_equal("abc123", s3 + s1)
    s1 += s2
    s1 += s3
    assert_equal("123abcabc", s1)


def test_string_join():
    var sep = ","
    var s0 = "abc"
    var s1 = sep.join(s0, s0, s0, s0)
    assert_equal("abc,abc,abc,abc", s1)

    assert_equal(sep.join(1, 2, 3), "1,2,3")

    assert_equal(sep.join(1, "abc", 3), "1,abc,3")

    var s2 = ",".join(List[UInt8](1, 2, 3))
    assert_equal(s2, "1,2,3")

    var s3 = ",".join(List[UInt8](1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert_equal(s3, "1,2,3,4,5,6,7,8,9")

    var s4 = ",".join(List[UInt8]())
    assert_equal(s4, "")

    var s5 = ",".join(List[UInt8](1))
    assert_equal(s5, "1")

    var s6 = ",".join(List[String]("1", "2", "3"))
    assert_equal(s6, "1,2,3")


def test_ord():
    # Regular ASCII
    assert_equal(ord("A"), 65)
    assert_equal(ord("Z"), 90)
    assert_equal(ord("0"), 48)
    assert_equal(ord("9"), 57)
    assert_equal(ord("a"), 97)
    assert_equal(ord("z"), 122)
    assert_equal(ord("!"), 33)

    # Multi byte character
    assert_equal(ord("α"), 945)
    assert_equal(ord("➿"), 10175)
    assert_equal(ord("🔥"), 128293)

    # Make sure they work in the parameter domain too
    alias single_byte = ord("A")
    assert_equal(single_byte, 65)
    alias single_byte2 = ord("!")
    assert_equal(single_byte2, 33)

    # TODO: change these to parameter domain when it work.
    var multi_byte = ord("α")
    assert_equal(multi_byte, 945)
    var multi_byte2 = ord("➿")
    assert_equal(multi_byte2, 10175)
    var multi_byte3 = ord("🔥")
    assert_equal(multi_byte3, 128293)

    # Test StringSlice overload
    assert_equal(ord("A".as_string_slice()), 65)
    assert_equal(ord("α".as_string_slice()), 945)
    assert_equal(ord("➿".as_string_slice()), 10175)
    assert_equal(ord("🔥".as_string_slice()), 128293)


def test_chr():
    assert_equal("\0", chr(0))
    assert_equal("A", chr(65))
    assert_equal("a", chr(97))
    assert_equal("!", chr(33))
    assert_equal("α", chr(945))
    assert_equal("➿", chr(10175))
    assert_equal("🔥", chr(128293))


def test_string_indexing():
    var str = "Hello Mojo!!"

    assert_equal("H", str[0])
    assert_equal("!", str[-1])
    assert_equal("H", str[-len(str)])
    assert_equal("llo Mojo!!", str[2:])
    assert_equal("lo Mojo!", str[3:-1:1])
    assert_equal("lo Moj", str[3:-3])

    assert_equal("!!ojoM olleH", str[::-1])

    assert_equal("leH", str[2::-1])

    assert_equal("!oo le", str[::-2])

    assert_equal("", str[:-1:-2])
    assert_equal("", str[-50::-1])
    assert_equal("Hello Mojo!!", str[-50::])
    assert_equal("!!ojoM olleH", str[:-50:-1])
    assert_equal("Hello Mojo!!", str[:50:])
    assert_equal("H", str[::50])
    assert_equal("!", str[::-50])
    assert_equal("!", str[50::-50])
    assert_equal("H", str[-50::50])


def test_atol():
    # base 10
    assert_equal(375, atol("375"))
    assert_equal(1, atol("001"))
    assert_equal(5, atol(" 005"))
    assert_equal(13, atol(" 013  "))
    assert_equal(-89, atol("-89"))
    assert_equal(-52, atol(" -52"))
    assert_equal(-69, atol(" -69  "))
    assert_equal(1_100_200, atol(" 1_100_200"))

    # other bases
    assert_equal(10, atol("A", 16))
    assert_equal(15, atol("f ", 16))
    assert_equal(255, atol(" FF", 16))
    assert_equal(255, atol(" 0xff ", 16))
    assert_equal(255, atol(" 0Xff ", 16))
    assert_equal(18, atol("10010", 2))
    assert_equal(18, atol("0b10010", 2))
    assert_equal(18, atol("0B10010", 2))
    assert_equal(10, atol("12", 8))
    assert_equal(10, atol("0o12", 8))
    assert_equal(10, atol("0O12", 8))
    assert_equal(35, atol("Z", 36))
    assert_equal(255, atol("0x_00_ff", 16))
    assert_equal(18, atol("0b0001_0010", 2))
    assert_equal(18, atol("0b_000_1001_0", 2))

    # Negative cases
    with assert_raises(
        contains="String is not convertible to integer with base 10: '9.03'"
    ):
        _ = atol("9.03")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ' 10 1'"
    ):
        _ = atol(" 10 1")

    # start/end with underscore double underscores
    with assert_raises(
        contains="String is not convertible to integer with base 10: '5__5'"
    ):
        _ = atol("5__5")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ' _5'"
    ):
        _ = atol(" _5")

    with assert_raises(
        contains="String is not convertible to integer with base 10: '5_'"
    ):
        _ = atol("5_")

    with assert_raises(
        contains="String is not convertible to integer with base 5: '5'"
    ):
        _ = atol("5", 5)

    with assert_raises(
        contains="String is not convertible to integer with base 10: '0x_ff'"
    ):
        _ = atol("0x_ff")

    with assert_raises(
        contains="String is not convertible to integer with base 3: '_12'"
    ):
        _ = atol("_12", 3)

    with assert_raises(contains="Base must be >= 2 and <= 36, or 0."):
        _ = atol("0", 1)

    with assert_raises(contains="Base must be >= 2 and <= 36, or 0."):
        _ = atol("0", 37)

    with assert_raises(
        contains="String is not convertible to integer with base 16: '_ff'"
    ):
        _ = atol("_ff", base=16)

    with assert_raises(
        contains="String is not convertible to integer with base 2: '  _01'"
    ):
        _ = atol("  _01", base=2)

    with assert_raises(
        contains="String is not convertible to integer with base 10: '0x_ff'"
    ):
        _ = atol("0x_ff")

    with assert_raises(
        contains="String is not convertible to integer with base 10: ''"
    ):
        _ = atol("")

    with assert_raises(
        contains="String expresses an integer too large to store in Int."
    ):
        _ = atol("9223372036854775832")


def test_atol_base_0():
    assert_equal(155, atol(" 155", base=0))
    assert_equal(155_155, atol("155_155 ", base=0))

    assert_equal(0, atol(" 0000", base=0))
    assert_equal(0, atol(" 000_000", base=0))

    assert_equal(3, atol("0b11", base=0))
    assert_equal(3, atol("0B1_1", base=0))

    assert_equal(63, atol("0o77", base=0))
    assert_equal(63, atol(" 0O7_7 ", base=0))

    assert_equal(17, atol("0x11", base=0))
    assert_equal(17, atol("0X1_1", base=0))

    assert_equal(0, atol("0X0", base=0))

    assert_equal(255, atol("0x_00_ff", base=0))

    assert_equal(18, atol("0b_0001_0010", base=0))
    assert_equal(18, atol("0b000_1001_0", base=0))

    assert_equal(10, atol("0o_000_12", base=0))
    assert_equal(10, atol("0o00_12", base=0))

    with assert_raises(
        contains="String is not convertible to integer with base 0: '  0x'"
    ):
        _ = atol("  0x", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '  0b  '"
    ):
        _ = atol("  0b  ", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '00100'"
    ):
        _ = atol("00100", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0r100'"
    ):
        _ = atol("0r100", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0xf__f'"
    ):
        _ = atol("0xf__f", base=0)

    with assert_raises(
        contains="String is not convertible to integer with base 0: '0of_'"
    ):
        _ = atol("0of_", base=0)


def test_atof():
    assert_equal(375.0, atof("375.f"))
    assert_equal(1.0, atof("001."))
    assert_equal(+5.0, atof(" +005."))
    assert_equal(13.0, atof(" 013.f  "))
    assert_equal(-89.0, atof("-89"))
    assert_equal(-0.3, atof(" -0.3"))
    assert_equal(-69e3, atof(" -69E+3  "))
    assert_equal(123.2e1, atof(" 123.2E1  "))
    assert_equal(23e3, atof(" 23E3  "))
    assert_equal(989343e-13, atof(" 989343E-13  "))
    assert_equal(1.123, atof(" 1.123f"))
    assert_equal(0.78, atof(" .78 "))
    assert_equal(121234.0, atof(" 121234.  "))
    assert_equal(985031234.0, atof(" 985031234.F  "))
    assert_equal(FloatLiteral.negative_zero, atof("-0"))
    assert_equal(String(FloatLiteral.nan), String(atof("  nan")))
    assert_equal(FloatLiteral.infinity, atof(" inf "))
    assert_equal(FloatLiteral.negative_infinity, atof("-inf  "))

    # Tests for scientific notation bug fix (using buff[pos] instead of buff[start])
    assert_equal(
        1.23e-2, atof("1.23e-2")
    )  # Previously failed due to wrong buffer indexing
    assert_equal(
        4.56e2, atof("4.56e+2")
    )  # Previously failed due to wrong buffer indexing

    # Tests for case-insensitive NaN and infinity
    assert_true(isnan(atof("NaN")))
    assert_true(isnan(atof("nan")))
    assert_true(isinf(atof("Inf")))
    assert_true(isinf(atof("INFINITY")))
    assert_true(isinf(atof("infinity")))
    assert_true(isinf(atof("-INFINITY")))

    # Tests for leading decimal point (no digits before decimal)
    assert_equal(0.123, atof(".123"))
    assert_equal(-0.123, atof("-.123"))
    assert_equal(0.123, atof("+.123"))

    # Tests for large exponents (overflow handling)
    assert_equal(
        FloatLiteral.infinity, atof("1e309")
    )  # Overflows double precision
    assert_equal(0.0, atof("1e-325"))  # Underflows to zero

    # Negative cases
    with assert_raises(contains="String is not convertible to float: ''"):
        _ = atof("")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 123 asd'. "
            "The last character of '123 asd' should be a "
            "digit or dot to convert it to a float."
        )
    ):
        _ = atof(" 123 asd")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' f.9123 '. "
            "The first character of 'f.9123' should be a "
            "digit or dot to convert it to a float."
        )
    ):
        _ = atof(" f.9123 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 989343E-1A3 '. "
            "Invalid character(s) in the number: '989343E-1A3'"
        )
    ):
        _ = atof(" 989343E-1A3 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 124124124_2134124124 '. "
            "Invalid character(s) in the number: '124124124_2134124124'"
        )
    ):
        _ = atof(" 124124124_2134124124 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' 123.2E '. "
            "The last character of '123.2E' should be a digit "
            "or dot to convert it to a float."
        )
    ):
        _ = atof(" 123.2E ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' --958.23 '. "
            "The first character of '-958.23' should be a digit "
            "or dot to convert it to a float."
        )
    ):
        _ = atof(" --958.23 ")

    with assert_raises(
        contains=(
            "String is not convertible to float: ' ++94. '. "
            "The first character of '+94.' should be "
            "a digit or dot to convert it to a float."
        )
    ):
        _ = atof(" ++94. ")

    with assert_raises(contains="String is not convertible to float"):
        _ = atof(".")  # Just a decimal point with no digits

    with assert_raises(contains="String is not convertible to float"):
        _ = atof("e5")  # Exponent with no mantissa


def test_calc_initial_buffer_size_int32():
    assert_equal(1, _calc_initial_buffer_size_int32(0))
    assert_equal(1, _calc_initial_buffer_size_int32(9))
    assert_equal(2, _calc_initial_buffer_size_int32(10))
    assert_equal(2, _calc_initial_buffer_size_int32(99))
    assert_equal(8, _calc_initial_buffer_size_int32(99999999))
    assert_equal(9, _calc_initial_buffer_size_int32(100000000))
    assert_equal(9, _calc_initial_buffer_size_int32(999999999))
    assert_equal(10, _calc_initial_buffer_size_int32(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int32(4294967295))


def test_calc_initial_buffer_size_int64():
    assert_equal(1, _calc_initial_buffer_size_int64(0))
    assert_equal(1, _calc_initial_buffer_size_int64(9))
    assert_equal(2, _calc_initial_buffer_size_int64(10))
    assert_equal(2, _calc_initial_buffer_size_int64(99))
    assert_equal(9, _calc_initial_buffer_size_int64(999999999))
    assert_equal(10, _calc_initial_buffer_size_int64(1000000000))
    assert_equal(10, _calc_initial_buffer_size_int64(9999999999))
    assert_equal(11, _calc_initial_buffer_size_int64(10000000000))
    assert_equal(20, _calc_initial_buffer_size_int64(18446744073709551615))


def test_contains():
    var str = "Hello world"

    assert_true(str.__contains__(""))
    assert_true(str.__contains__("He"))
    assert_true("lo" in str)
    assert_true(str.__contains__(" "))
    assert_true(str.__contains__("ld"))

    assert_false(str.__contains__("below"))
    assert_true("below" not in str)


def test_find():
    var str = "Hello world"

    assert_equal(0, str.find(""))
    assert_equal(0, str.find("Hello"))
    assert_equal(2, str.find("llo"))
    assert_equal(6, str.find("world"))
    assert_equal(-1, str.find("universe"))

    # Test find() offset is absolute, not relative (issue mojo/#1355)
    var str2 = "...a"
    assert_equal(3, str2.find("a", 0))
    assert_equal(3, str2.find("a", 1))
    assert_equal(3, str2.find("a", 2))
    assert_equal(3, str2.find("a", 3))

    # Test find() support for negative start positions
    assert_equal(4, str.find("o", -10))
    assert_equal(7, str.find("o", -5))

    assert_equal(-1, String("abc").find("abcd"))


def test_replace():
    # Replace empty
    var s1 = "abc"
    assert_equal("xaxbxc", s1.replace("", "x"))
    assert_equal("->a->b->c", s1.replace("", "->"))

    var s2 = "Hello Python"
    assert_equal("Hello Mojo", s2.replace("Python", "Mojo"))
    assert_equal("HELLo Python", s2.replace("Hell", "HELL"))
    assert_equal("Hello Python", s2.replace("HELL", "xxx"))
    assert_equal("HellP oython", s2.replace("o P", "P o"))
    assert_equal("Hello Pything", s2.replace("thon", "thing"))
    assert_equal("He||o Python", s2.replace("ll", "||"))
    assert_equal("He--o Python", s2.replace("l", "-"))
    assert_equal("He-x--x-o Python", s2.replace("l", "-x-"))

    var s3 = "a   complex  test case  with some  spaces"
    assert_equal("a  complex test case with some spaces", s3.replace("  ", " "))


def test_rfind():
    var s1 = "hello world"
    # Basic usage.
    assert_equal(s1.rfind("world"), 6)
    assert_equal(s1.rfind("bye"), -1)

    # Repeated substrings.
    assert_equal(String("ababab").rfind("ab"), 4)

    # Empty string and substring.
    assert_equal(String().rfind("ab"), -1)
    assert_equal(String("foo").rfind(""), 3)

    # Test that rfind(start) returned pos is absolute, not relative to specified
    # start. Also tests positive and negative start offsets.
    assert_equal(s1.rfind("l", 5), 9)
    assert_equal(s1.rfind("l", -5), 9)
    assert_equal(s1.rfind("w", -3), -1)
    assert_equal(s1.rfind("w", -5), 6)

    assert_equal(-1, String("abc").rfind("abcd"))

    # Special characters.
    # TODO(#26444): Support unicode strings.
    # assert_equal(String("こんにちは").rfind("にち"), 2)
    # assert_equal(String("🔥🔥").rfind("🔥"), 1)


def test_split():
    alias S = StaticString
    alias L = List[StaticString]

    # Should add all whitespace-like chars as one
    # test all unicode separators
    # 0 is to build a String with null terminator
    var next_line = List[UInt8](0xC2, 0x85)
    var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
    var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
    # TODO add line and paragraph separator as StringLiteral once unicode
    # escape sequences are accepted
    var univ_sep_var = String(
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        String(bytes=next_line),
        String(bytes=unicode_line_sep),
        String(bytes=unicode_paragraph_sep),
    )
    var s = univ_sep_var + "hello" + univ_sep_var + "world" + univ_sep_var
    assert_equal(StringSlice(s).split(), L("hello", "world"))

    # should split into empty strings between separators
    assert_equal(S("1,,,3").split(","), L("1", "", "", "3"))
    assert_equal(S(",,,").split(","), L("", "", "", ""))
    assert_equal(S(" a b ").split(" "), L("", "a", "b", ""))
    assert_equal(S("abababaaba").split("aba"), L("", "b", "", ""))
    assert_true(len(S("").split()) == 0)
    assert_true(len(S(" ").split()) == 0)
    assert_true(len(S("").split(" ")) == 1)
    assert_true(len(S(",").split(",")) == 2)
    assert_true(len(S(" ").split(" ")) == 2)
    assert_true(len(S("").split("")) == 2)
    assert_true(len(S("  ").split(" ")) == 3)
    assert_true(len(S("   ").split(" ")) == 4)

    # should split into maxsplit + 1 items
    assert_equal(S("1,2,3").split(",", 0), L("1,2,3"))
    assert_equal(S("1,2,3").split(",", 1), L("1", "2,3"))

    # Split in middle
    assert_equal(S("faang").split("n"), L("faa", "g"))

    # No match from the delimiter
    assert_equal(S("hello world").split("x"), L("hello world"))

    # Multiple character delimiter
    assert_equal(S("hello").split("ll"), L("he", "o"))

    res = L("", "bb", "", "", "", "bbb", "")
    assert_equal(S("abbaaaabbba").split("a"), res)
    assert_equal(S("abbaaaabbba").split("a", 8), res)
    s1 = S("abbaaaabbba").split("a", 5)
    assert_equal(s1, L("", "bb", "", "", "", "bbba"))
    assert_equal(S("aaa").split("a", 0), L("aaa"))
    assert_equal(S("a").split("a"), L("", ""))
    assert_equal(S("1,2,3").split("3", 0), L("1,2,3"))
    assert_equal(S("1,2,3").split("3", 1), L("1,2,", ""))
    assert_equal(S("1,2,3,3").split("3", 2), L("1,2,", ",", ""))
    assert_equal(S("1,2,3,3,3").split("3", 2), L("1,2,", ",", ",3"))

    assert_equal(S("Hello 🔥!").split(), L("Hello", "🔥!"))

    s2 = S("Лорем ипсум долор сит амет").split(" ")
    assert_equal(s2, L("Лорем", "ипсум", "долор", "сит", "амет"))
    s3 = S("Лорем ипсум долор сит амет").split("м")
    assert_equal(s3, L("Лоре", " ипсу", " долор сит а", "ет"))

    assert_equal(S("123").split(""), L("", "1", "2", "3", ""))
    assert_equal(S("").join(S("123").split("")), "123")
    assert_equal(S(",1,2,3,").split(","), S("123").split(""))
    assert_equal(S(",").join(S("123").split("")), ",1,2,3,")


def test_splitlines():
    alias L = List[String]
    # Test with no line breaks
    assert_equal(String("hello world").splitlines(), L("hello world"))

    # Test with line breaks
    assert_equal(String("hello\nworld").splitlines(), L("hello", "world"))
    assert_equal(String("hello\rworld").splitlines(), L("hello", "world"))
    assert_equal(String("hello\r\nworld").splitlines(), L("hello", "world"))

    # Test with multiple different line breaks
    s1 = "hello\nworld\r\nmojo\rlanguage\r\n"
    hello_mojo = L("hello", "world", "mojo", "language")
    assert_equal(s1.splitlines(), hello_mojo)
    assert_equal(
        s1.splitlines(keepends=True),
        L("hello\n", "world\r\n", "mojo\r", "language\r\n"),
    )

    # Test with an empty string
    assert_equal(String().splitlines(), L())
    # test \v \f \x1c \x1d
    s2 = "hello\vworld\fmojo\x1clanguage\x1d"
    assert_equal(s2.splitlines(), hello_mojo)
    assert_equal(
        s2.splitlines(keepends=True),
        L("hello\v", "world\f", "mojo\x1c", "language\x1d"),
    )

    # test \x1c \x1d \x1e
    s3 = "hello\x1cworld\x1dmojo\x1elanguage\x1e"
    assert_equal(s3.splitlines(), hello_mojo)
    assert_equal(
        s3.splitlines(keepends=True),
        L("hello\x1c", "world\x1d", "mojo\x1e", "language\x1e"),
    )

    # test \x85 \u2028 \u2029
    var next_line = List[UInt8](0xC2, 0x85)
    var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
    var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)

    for elt in [next_line, unicode_line_sep, unicode_paragraph_sep]:
        u = String(bytes=elt)
        item = String().join("hello", u, "world", u, "mojo", u, "language", u)
        assert_equal(item.splitlines(), hello_mojo)
        assert_equal(
            item.splitlines(keepends=True),
            L("hello" + u, "world" + u, "mojo" + u, "language" + u),
        )


def test_isspace():
    assert_false(String().isspace())

    # test all utf8 and unicode separators
    # 0 is to build a String with null terminator
    var next_line = List[UInt8](0xC2, 0x85)
    var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
    var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
    # TODO add line and paragraph separator as StringLiteral once unicode
    # escape sequences are accepted
    var univ_sep_var = [
        " ",
        "\t",
        "\n",
        "\r",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        String(bytes=next_line),
        String(bytes=unicode_line_sep),
        String(bytes=unicode_paragraph_sep),
    ]

    for i in univ_sep_var:
        assert_true(i.isspace())

    for i in List[String]("not", "space", "", "s", "a", "c"):
        assert_false(i.isspace())

    for i in range(len(univ_sep_var)):
        var sep = String()
        for j in range(len(univ_sep_var)):
            sep += univ_sep_var[i]
            sep += univ_sep_var[j]
        assert_true(sep.isspace())
        _ = sep


def test_ascii_aliases():
    assert_true("a" in String.ASCII_LOWERCASE)
    assert_true("b" in String.ASCII_LOWERCASE)
    assert_true("y" in String.ASCII_LOWERCASE)
    assert_true("z" in String.ASCII_LOWERCASE)

    assert_true("A" in String.ASCII_UPPERCASE)
    assert_true("B" in String.ASCII_UPPERCASE)
    assert_true("Y" in String.ASCII_UPPERCASE)
    assert_true("Z" in String.ASCII_UPPERCASE)

    assert_true("a" in String.ASCII_LETTERS)
    assert_true("b" in String.ASCII_LETTERS)
    assert_true("y" in String.ASCII_LETTERS)
    assert_true("z" in String.ASCII_LETTERS)
    assert_true("A" in String.ASCII_LETTERS)
    assert_true("B" in String.ASCII_LETTERS)
    assert_true("Y" in String.ASCII_LETTERS)
    assert_true("Z" in String.ASCII_LETTERS)

    assert_true("0" in String.DIGITS)
    assert_true("9" in String.DIGITS)

    assert_true("0" in String.HEX_DIGITS)
    assert_true("9" in String.HEX_DIGITS)
    assert_true("A" in String.HEX_DIGITS)
    assert_true("F" in String.HEX_DIGITS)

    assert_true("7" in String.OCT_DIGITS)
    assert_false("8" in String.OCT_DIGITS)

    assert_true("," in String.PUNCTUATION)
    assert_true("." in String.PUNCTUATION)
    assert_true("\\" in String.PUNCTUATION)
    assert_true("@" in String.PUNCTUATION)
    assert_true('"' in String.PUNCTUATION)
    assert_true("'" in String.PUNCTUATION)

    var text = "I love my Mom and Dad so much!!!\n"
    for i in range(len(text)):
        assert_true(text[i] in String.PRINTABLE)


def test_rstrip():
    # with default rstrip chars
    var empty_string = String()
    assert_true(empty_string.rstrip() == "")

    var space_string = " \t\n\r\v\f  "
    assert_true(space_string.rstrip() == "")

    var str0 = "     n "
    assert_true(str0.rstrip() == "     n")

    var str1 = "string"
    assert_true(str1.rstrip() == "string")

    var str2 = "something \t\n\t\v\f"
    assert_true(str2.rstrip() == "something")

    # with custom chars for rstrip
    var str3 = "mississippi"
    assert_true(str3.rstrip("sip") == "m")

    var str4 = "mississippimississippi \n "
    assert_true(str4.rstrip("sip ") == "mississippimississippi \n")
    assert_true(str4.rstrip("sip \n") == "mississippim")


def test_lstrip():
    # with default lstrip chars
    var empty_string = String()
    assert_true(empty_string.lstrip() == "")

    var space_string = " \t\n\r\v\f  "
    assert_true(space_string.lstrip() == "")

    var str0 = "     n "
    assert_true(str0.lstrip() == "n ")

    var str1 = "string"
    assert_true(str1.lstrip() == "string")

    var str2 = " \t\n\t\v\fsomething"
    assert_true(str2.lstrip() == "something")

    # with custom chars for lstrip
    var str3 = "mississippi"
    assert_true(str3.lstrip("mis") == "ppi")

    var str4 = " \n mississippimississippi"
    assert_true(str4.lstrip("mis ") == "\n mississippimississippi")
    assert_true(str4.lstrip("mis \n") == "ppimississippi")


def test_strip():
    # with default strip chars
    var empty_string = String()
    assert_true(empty_string.strip() == "")
    alias comp_empty_string_stripped = String(String().strip())
    assert_true(comp_empty_string_stripped == "")

    var space_string = " \t\n\r\v\f  "
    assert_true(space_string.strip() == "")
    alias comp_space_string_stripped = String(" \t\n\r\v\f  ".strip())
    assert_true(comp_space_string_stripped == "")

    var str0 = "     n "
    assert_true(str0.strip() == "n")
    alias comp_str0_stripped = String("     n ".strip())
    assert_true(comp_str0_stripped == "n")

    var str1 = "string"
    assert_true(str1.strip() == "string")
    alias comp_str1_stripped = String("string".strip())
    assert_true(comp_str1_stripped == "string")

    var str2 = " \t\n\t\v\fsomething \t\n\t\v\f"
    alias comp_str2_stripped = String(" \t\n\t\v\fsomething \t\n\t\v\f".strip())
    assert_true(str2.strip() == "something")
    assert_true(comp_str2_stripped == "something")

    # with custom strip chars
    var str3 = "mississippi"
    assert_true(str3.strip("mips") == "")
    assert_true(str3.strip("mip") == "ssiss")
    alias comp_str3_stripped = String("mississippi".strip("mips"))
    assert_true(comp_str3_stripped == "")

    var str4 = " \n mississippimississippi \n "
    assert_true(str4.strip(" ") == "\n mississippimississippi \n")
    assert_true(str4.strip("\nmip ") == "ssissippimississ")

    alias comp_str4_stripped = String(
        " \n mississippimississippi \n ".strip(" ")
    )
    assert_true(comp_str4_stripped == "\n mississippimississippi \n")


def test_hash():
    fn assert_hash_equals_literal_hash[s: StaticString]() raises:
        assert_equal(hash(s), hash(String(s)))

    assert_hash_equals_literal_hash["a"]()
    assert_hash_equals_literal_hash["b"]()
    assert_hash_equals_literal_hash["c"]()
    assert_hash_equals_literal_hash["d"]()
    assert_hash_equals_literal_hash["this is a longer string"]()
    assert_hash_equals_literal_hash[
        """
Blue: We have to take the amulet to the Banana King.
Charlie: Oh, yes, The Banana King, of course. ABSOLUTELY NOT!
Pink: He, he's counting on us, Charlie! (Pink starts floating) ah...
Blue: If we don't give the amulet to the Banana King, the vortex will open and let out a thousand years of darkness.
Pink: No! Darkness! (Pink is floating in the air)"""
    ]()


def test_startswith():
    var str = "Hello world"

    assert_true(str.startswith("Hello"))
    assert_false(str.startswith("Bye"))

    assert_true(str.startswith("llo", 2))
    assert_true(str.startswith("llo", 2, -1))
    assert_false(str.startswith("llo", 2, 3))


def test_endswith():
    var str = "Hello world"

    assert_true(str.endswith(""))
    assert_true(str.endswith("world"))
    assert_true(str.endswith("ld"))
    assert_false(str.endswith("universe"))

    assert_true(str.endswith("ld", 2))
    assert_true(str.endswith("llo", 2, 5))
    assert_false(str.endswith("llo", 2, 3))


# TODO: Remove the explicit conversion to `String` once #4790 is resolved.
def test_removeprefix():
    assert_equal(String(String("hello world").removeprefix("")), "hello world")
    assert_equal(String(String("hello world").removeprefix("hello")), " world")
    assert_equal(
        String(String("hello world").removeprefix("world")), "hello world"
    )
    assert_equal(String(String("hello world").removeprefix("hello world")), "")
    assert_equal(
        String(String("hello world").removeprefix("llo wor")), "hello world"
    )


# TODO: Remove the explicit conversion to `String` once #4790 is resolved.
def test_removesuffix():
    assert_equal(String(String("hello world").removesuffix("")), "hello world")
    assert_equal(String(String("hello world").removesuffix("world")), "hello ")
    assert_equal(
        String(String("hello world").removesuffix("hello")), "hello world"
    )
    assert_equal(String(String("hello world").removesuffix("hello world")), "")
    assert_equal(
        String(String("hello world").removesuffix("llo wor")), "hello world"
    )


def test_intable():
    assert_equal(Int("123"), 123)
    assert_equal(Int("10", base=8), 8)

    with assert_raises():
        _ = Int("hi")


def test_string_mul():
    assert_equal("*" * 0, "")
    assert_equal("!" * 10, "!!!!!!!!!!")
    assert_equal("ab" * 5, "ababababab")


def test_indexing():
    a = "abc"
    assert_equal(a[False], "a")
    assert_equal(a[Int(1)], "b")
    assert_equal(a[2], "c")


def test_string_codepoints_iter():
    var s = "abc"
    var iter = s.codepoints()
    assert_equal(iter.__next__(), Codepoint.ord("a"))
    assert_equal(iter.__next__(), Codepoint.ord("b"))
    assert_equal(iter.__next__(), Codepoint.ord("c"))
    assert_equal(iter.__has_next__(), False)


def test_string_char_slices_iter():
    var s0 = "abc"
    var s0_iter = s0.codepoint_slices()
    assert_true(s0_iter.__next__() == "a")
    assert_true(s0_iter.__next__() == "b")
    assert_true(s0_iter.__next__() == "c")
    assert_equal(s0_iter.__has_next__(), False)

    var vs = "123"

    # Borrow immutably
    fn conc(vs: String) -> String:
        var c = String()
        for v in vs.codepoint_slices():
            c += v
        return c

    assert_equal(123, atol(conc(vs)))

    concat = String()
    for v in vs.__reversed__():
        concat += v
    assert_equal(321, atol(concat))

    # Borrow immutably
    for v in vs.codepoint_slices():
        concat += v

    assert_equal(321123, atol(concat))

    vs = "mojo🔥"
    var iterator = vs.codepoint_slices()
    assert_equal(5, len(iterator))
    var item = iterator.__next__()
    assert_equal("m", String(item))
    assert_equal(4, len(iterator))
    item = iterator.__next__()
    assert_equal("o", String(item))
    assert_equal(3, len(iterator))
    item = iterator.__next__()
    assert_equal("j", String(item))
    assert_equal(2, len(iterator))
    item = iterator.__next__()
    assert_equal("o", String(item))
    assert_equal(1, len(iterator))
    item = iterator.__next__()
    assert_equal("🔥", String(item))
    assert_equal(0, len(iterator))

    var items = List[String](
        "mojo🔥",
        "السلام عليكم",
        "Dobrý den",
        "Hello",
        "שָׁלוֹם",
        "नमस्ते",
        "こんにちは",
        "안녕하세요",
        "你好",
        "Olá",
        "Здравствуйте",
    )
    var rev = List[String](
        "🔥ojom",
        "مكيلع مالسلا",
        "ned ýrboD",
        "olleH",
        "םֹולָׁש",
        "ेत्समन",
        "はちにんこ",
        "요세하녕안",
        "好你",
        "álO",
        "етйувтсвардЗ",
    )
    var items_amount_characters = [5, 12, 9, 5, 7, 6, 5, 5, 2, 3, 12]
    for item_idx in range(len(items)):
        var item = items[item_idx]
        var ptr = item.unsafe_ptr()
        var amnt_characters = 0
        var byte_idx = 0
        for v in item.codepoint_slices():
            var byte_len = v.byte_length()
            for i in range(byte_len):
                assert_equal(ptr[byte_idx + i], v.unsafe_ptr()[i])
            byte_idx += byte_len
            amnt_characters += 1

        assert_equal(amnt_characters, items_amount_characters[item_idx])
        var concat = String()
        for v in item.__reversed__():
            concat += v
        assert_equal(rev[item_idx], concat)
        item_idx += 1


def test_format_args():
    with assert_raises(contains="Index -1 not in *args"):
        _ = "{-1} {0}".format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = "A {0} B {1}".format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = "A {1} B {0}".format("First")

    with assert_raises(contains="Index 1 not in *args"):
        _ = "A {1} B {0}".format()

    with assert_raises(
        contains="Automatic indexing require more args in *args"
    ):
        _ = "A {} B {}".format("First")

    with assert_raises(
        contains="Cannot both use manual and automatic indexing"
    ):
        _ = "A {} B {1}".format("First", "Second")

    with assert_raises(contains="Index first not in kwargs"):
        _ = "A {first} B {second}".format(1, 2)

    var s = " {} , {} {} !".format("Hello", "Beautiful", "World")
    assert_equal(s, " Hello , Beautiful World !")

    fn curly(c: StaticString) -> String:
        return "there is a single curly " + c + " left unclosed or unescaped"

    with assert_raises(contains=curly("{")):
        _ = "{ {}".format(1)

    with assert_raises(contains=curly("{")):
        _ = "{ {0}".format(1)

    with assert_raises(contains=curly("{")):
        _ = "{}{".format(1)

    with assert_raises(contains=curly("}")):
        _ = "{}}".format(1)

    with assert_raises(contains=curly("{")):
        _ = "{} {".format(1)

    with assert_raises(contains=curly("{")):
        _ = "{".format(1)

    with assert_raises(contains=curly("}")):
        _ = "}".format(1)

    with assert_raises(contains=""):
        _ = "{}".format()

    assert_equal("}}".format(), "}")
    assert_equal("{{".format(), "{")

    assert_equal("{{}}{}{{}}".format("foo"), "{}foo{}")

    assert_equal("{{ {0}".format("foo"), "{ foo")
    assert_equal("{{{0}".format("foo"), "{foo")
    assert_equal("{{0}}".format("foo"), "{0}")
    assert_equal("{{}}".format("foo"), "{}")
    assert_equal("{{0}}".format("foo"), "{0}")
    assert_equal("{{{0}}}".format("foo"), "{foo}")

    var vinput = "{} {}"
    var output = vinput.format("123", 456)
    assert_equal(len(output), 7)

    var vinput2 = "{1}{0}"
    output = vinput2.format("123", 456)
    assert_equal(len(output), 6)
    assert_equal(output, "456123")

    var vinput3 = "123"
    output = vinput3.format()
    assert_equal(len(output), 3)

    var vinput4 = ""
    output = vinput4.format()
    assert_equal(len(output), 0)

    var res = "🔥 Mojo ❤️‍🔥 Mojo 🔥"
    assert_equal("{0} {1} ❤️‍🔥 {1} {0}".format("🔥", "Mojo"), res)

    assert_equal("{0} {1}".format(True, 1.125), "True 1.125")

    assert_equal("{0} {1}".format("{1}", "Mojo"), "{1} Mojo")
    assert_equal("{0} {1} {0} {1}".format("{1}", "Mojo"), "{1} Mojo {1} Mojo")


def test_format_conversion_flags():
    assert_equal("{!r}".format(""), "''")
    var special_str = "a\nb\tc"
    assert_equal(
        "{} {!r}".format(special_str, special_str),
        "a\nb\tc 'a\\nb\\tc'",
    )
    assert_equal(
        "{!s} {!r}".format(special_str, special_str),
        "a\nb\tc 'a\\nb\\tc'",
    )

    var a = "Mojo"
    assert_equal("{} {!r}".format(a, a), "Mojo 'Mojo'")
    assert_equal("{!s} {!r}".format(a, a), "Mojo 'Mojo'")
    assert_equal("{0!s} {0!r}".format(a), "Mojo 'Mojo'")
    assert_equal("{0!s} {0!r}".format(a, "Mojo2"), "Mojo 'Mojo'")

    var b = 21.1
    assert_true(
        "21.1 SIMD[DType.float64, 1](2" in "{} {!r}".format(b, b),
    )
    assert_true(
        "21.1 SIMD[DType.float64, 1](2" in "{!s} {!r}".format(b, b),
    )

    var c = 1e100
    assert_equal(
        "{} {!r}".format(c, c),
        "1e+100 SIMD[DType.float64, 1](1e+100)",
    )
    assert_equal(
        "{!s} {!r}".format(c, c),
        "1e+100 SIMD[DType.float64, 1](1e+100)",
    )

    var d = 42
    assert_equal("{} {!r}".format(d, d), "42 42")
    assert_equal("{!s} {!r}".format(d, d), "42 42")

    assert_true(
        "Mojo SIMD[DType.float64, 1](2" in "{} {!r} {} {!r}".format(a, b, c, d)
    )
    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in "{!s} {!r} {!s} {!r}".format(a, b, c, d)
    )

    var e = True
    assert_equal("{} {!r}".format(e, e), "True True")

    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in "{0} {1!r} {2} {3}".format(a, b, c, d)
    )
    assert_true(
        "Mojo SIMD[DType.float64, 1](2"
        in "{0!s} {1!r} {2} {3!s}".format(a, b, c, d)
    )

    assert_equal(
        "{3} {2} {1} {0}".format(a, d, c, b),
        "21.1 1e+100 42 Mojo",
    )

    assert_true(
        "'Mojo' 42 SIMD[DType.float64, 1](2"
        in "{0!r} {3} {1!r}".format(a, b, c, d)
    )

    assert_true(
        "True 'Mojo' 42 SIMD[DType.float64, 1](2"
        in "{4} {0!r} {3} {1!r}".format(a, b, c, d, True)
    )

    with assert_raises(contains='Conversion flag "x" not recognised.'):
        _ = "{!x}".format(1)

    with assert_raises(contains="Empty conversion flag."):
        _ = "{!}".format(1)

    with assert_raises(contains='Conversion flag "rs" not recognised.'):
        _ = "{!rs}".format(1)

    with assert_raises(contains='Conversion flag "r123" not recognised.'):
        _ = "{!r123}".format(1)

    with assert_raises(contains='Conversion flag "r!" not recognised.'):
        _ = "{!r!}".format(1)

    with assert_raises(contains='Conversion flag "x" not recognised.'):
        _ = "{0!x}".format(1)

    with assert_raises(contains='Conversion flag "r:d" not recognised.'):
        _ = "{!r:d}".format(1)


def test_float_conversion():
    # This is basically just a wrapper around atof which is
    # more throughouly tested above
    assert_equal(String("4.5").__float__(), 4.5)
    assert_equal(Float64("4.5"), 4.5)
    with assert_raises():
        _ = Float64("not a float")


def test_slice_contains():
    assert_true("hello world".as_string_slice().__contains__("world"))
    assert_false("hello world".as_string_slice().__contains__("not-found"))


def test_reserve():
    var s = String()
    assert_equal(s.capacity(), 0)
    s.reserve(1)
    assert_equal(s.capacity(), 8)


def test_uninit_ctor():
    var hello_len = len("hello")
    var s = String(unsafe_uninit_length=hello_len)
    memcpy(s.unsafe_ptr(), StaticString("hello").unsafe_ptr(), hello_len)
    assert_equal(s, "hello")

    # Resize with uninitialized memory.
    var s2 = String()
    s2.resize(unsafe_uninit_length=hello_len)
    memcpy(s2.unsafe_ptr_mut(), StaticString("hello").unsafe_ptr(), hello_len)
    assert_equal(s2, "hello")
    assert_equal(s2._is_inline(), True)

    var s3 = String()
    var long: StaticString = "hellohellohellohellohellohellohellohellohellohel"
    s3.resize(unsafe_uninit_length=len(long))
    memcpy(s3.unsafe_ptr_mut(), long.unsafe_ptr(), len(long))
    assert_equal(s3, long)
    assert_equal(s3._is_inline(), False)


def test_unsafe_cstr():
    var s1: String = "ab"
    var p1 = s1.unsafe_cstr_ptr()
    assert_equal(p1[0], ord("a"))
    assert_equal(p1[1], ord("b"))
    assert_equal(p1[2], 0)

    var s2: String = ""
    var p2 = s2.unsafe_cstr_ptr()
    assert_equal(p2[0], 0)

    var s3 = String()
    var p3 = s3.unsafe_cstr_ptr()
    assert_equal(p3[0], 0)

    # 24 bytes is out of line.
    var s4: String = "abcdefghabcdefghabcdefgh"
    var p4 = s4.unsafe_cstr_ptr()
    assert_equal(p4[0], ord("a"))
    assert_equal(p4[1], ord("b"))
    assert_equal(p4[2], ord("c"))
    assert_equal(p4[23], ord("h"))
    assert_equal(p4[24], 0)


def test_variadic_ctors():
    var s = String("message", 42, 42.2, True, sep=", ")
    assert_equal(s, "message, 42, 42.2, True")

    var s2 = String.write("message", 42, 42.2, True, sep=", ")
    assert_equal(s2, "message, 42, 42.2, True")

    fn forward_variadic_pack[
        *Ts: Writable,
    ](*args: *Ts) -> String:
        return String(args)

    var s3 = forward_variadic_pack(1, ", ", 2.0, ", ", "three")
    assert_equal(s3, "1, 2.0, three")


def test_sso():
    # String literals are initially stored as indirect regardless of length
    var s = "hello"
    assert_equal(s.capacity(), 5)
    assert_equal(len(s), 5)
    assert_equal(s._is_inline(), False)
    assert_equal(s._has_nul_terminator(), True)
    assert_equal(s.unsafe_ptr()[s.byte_length()], 0)

    # Adding a single char should remove the nul terminator and inline it.
    s += "f"
    assert_equal(len(s), 6)
    assert_equal(s.capacity(), String.INLINE_CAPACITY)
    assert_equal(s._is_inline(), True)
    assert_equal(s._has_nul_terminator(), False)
    assert_equal(s, "hellof")

    # Check that unsafe_cstr_ptr adds the nul terminator at the end.
    var ptr = s.unsafe_cstr_ptr()
    assert_equal(s._has_nul_terminator(), True)
    assert_equal(ptr[len(s) - 1], ord("f"))
    assert_equal(ptr[len(s)], 0)

    # Test StringLiterals behave the same when above SSO capacity.
    alias long = "hellohellohellohellohellohellohellohellohellohellohello"
    s = String(long)
    assert_equal(len(s), 55)
    assert_equal(s.capacity(), 55)
    assert_equal(s._is_inline(), False)
    assert_equal(s._has_nul_terminator(), True)
    assert_equal(s.unsafe_ptr()[s.byte_length()], 0)

    # Modifying it should remove the nul terminator.
    s += "f"
    assert_equal(len(s), 56)
    assert_true(s.capacity() >= 56)
    assert_equal(s._is_inline(), False)
    assert_equal(s._has_nul_terminator(), False)
    assert_equal(s, long + "f")

    # Check that unsafe_cstr_ptr adds the nul terminator at the end.
    ptr = s.unsafe_cstr_ptr()
    assert_equal(s._has_nul_terminator(), True)
    assert_equal(ptr[len(s) - 1], ord("f"))
    assert_equal(ptr[len(s)], 0)

    # Empty strings are stored inline.
    s = String()
    assert_equal(s.capacity(), String.INLINE_CAPACITY)
    assert_equal(s._is_inline(), True)
    assert_equal(s._has_nul_terminator(), False)

    s += "f" * String.INLINE_CAPACITY
    assert_equal(len(s), String.INLINE_CAPACITY)
    assert_equal(s.capacity(), String.INLINE_CAPACITY)
    assert_equal(s._is_inline(), True)

    # One more byte.
    s += "f"

    # The capacity should be 2x the previous amount, rounded up to 8.
    alias expected_capacity = (Int(String.INLINE_CAPACITY) * 2 + 7) & ~7
    assert_equal(s.capacity(), expected_capacity)
    assert_equal(s._is_inline(), False)

    # Shrink down to small, but stays out of line to maintain our malloc.
    s.resize(4)
    assert_equal(s.capacity(), expected_capacity)
    assert_equal(s._is_inline(), False)
    assert_equal(s, "ffff")

    # Copying the small out-of-line string should just bump the count.
    var s2 = s.copy()
    assert_equal(s2._is_inline(), False)
    assert_equal(s2, "ffff")

    # Stringizing short things should be inline.
    s = String(42)
    assert_equal(s, "42")
    assert_equal(s._is_inline(), True)


def test_python_object():
    var s = String(PythonObject("hello"))
    assert_equal(s, "hello")

    var p = Python()
    _ = p.eval("class A:\n  def __str__(self): pass")
    var a = p.evaluate("A()")
    with assert_raises(contains="__str__ returned non-string"):
        _ = String(a)


def test_copyinit():
    alias sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    assert_equal(len(sizes), 10)
    var test_current_size = 1

    @parameter
    for sizes_index in range(len(sizes)):
        alias current_size = sizes[sizes_index]
        x = ""
        for i in range(current_size):
            x += String(i)[0]
        y = x
        assert_equal(test_current_size, current_size)
        assert_equal(len(y), current_size)
        # TODO: check pointer equality?
        test_current_size *= 2
    assert_equal(test_current_size, 1024)


def main():
    test_constructors()
    test_copy()
    test_len()
    test_equality_operators()
    test_comparison_operators()
    test_add()
    test_add_string_slice()
    test_stringable()
    test_string_join()
    test_ord()
    test_chr()
    test_string_indexing()
    test_atol()
    test_atol_base_0()
    test_atof()
    test_calc_initial_buffer_size_int32()
    test_calc_initial_buffer_size_int64()
    test_contains()
    test_find()
    test_replace()
    test_rfind()
    test_split()
    test_splitlines()
    test_isspace()
    test_ascii_aliases()
    test_rstrip()
    test_lstrip()
    test_strip()
    test_hash()
    test_startswith()
    test_endswith()
    test_removeprefix()
    test_removesuffix()
    test_intable()
    test_string_mul()
    test_indexing()
    test_string_codepoints_iter()
    test_string_char_slices_iter()
    test_format_args()
    test_format_conversion_flags()
    test_float_conversion()
    test_slice_contains()
    test_uninit_ctor()
    test_unsafe_cstr()
    test_variadic_ctors()
    test_sso()
    test_python_object()
    test_copyinit()
