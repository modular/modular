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
# RUN: %mojo %s

from collections.string.string_slice import get_static_string
from sys.info import alignof, sizeof

from memory import Span, UnsafePointer
from testing import assert_equal, assert_false, assert_raises, assert_true

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #

alias EVERY_CODEPOINT_LENGTH_STR = StringSlice("߷കൈ🔄!")
"""A string that contains at least one of 1-, 2-, 3-, and 4-byte UTF-8
sequences.

Visualized as:

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      ߷കൈ🔄!                    ┃
┣━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━┫
┃   ߷  ┃     ക     ┃     ൈ    ┃       🔄      ┃! ┃
┣━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━┫
┃ 2039  ┃   3349    ┃   3400    ┃    128260     ┃33┃
┣━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━┳━━━╋━━┫
┃223┃183┃224┃180┃149┃224┃181┃136┃240┃159┃148┃132┃33┃
┗━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━┛
  0   1   2   3   4   5   6   7   8   9  10  11  12
```

For further visualization and analysis involving this sequence, see:
<https://connorgray.com/ephemera/project-log#2025-01-13>.
"""

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


fn test_string_slice_layout() raises:
    # Test that the layout of `StringSlice` is the same as `llvm::StringRef`.
    # This is necessary for `StringSlice` to be validly bitcasted to and from
    # `llvm::StringRef`

    # StringSlice should be two words in size.
    assert_equal(sizeof[StringSlice[MutableAnyOrigin]](), 2 * sizeof[Int]())

    var str_slice = StringSlice("")

    var base_ptr = Int(UnsafePointer(to=str_slice))
    var first_word_ptr = Int(UnsafePointer(to=str_slice._slice._data))
    var second_word_ptr = Int(UnsafePointer(to=str_slice._slice._len))

    # 1st field should be at 0-byte offset from base ptr
    assert_equal(first_word_ptr - base_ptr, 0)
    # 2nd field should at 1-word offset from base ptr
    assert_equal(second_word_ptr - base_ptr, sizeof[Int]())


fn test_string_literal_byte_span() raises:
    alias slc = "Hello".as_bytes()

    assert_equal(len(slc), 5)
    assert_equal(slc[0], ord("H"))
    assert_equal(slc[1], ord("e"))
    assert_equal(slc[2], ord("l"))
    assert_equal(slc[3], ord("l"))
    assert_equal(slc[4], ord("o"))


fn test_string_byte_span() raises:
    var string = String("Hello")
    var str_slice = string.as_bytes_mut()

    assert_equal(len(str_slice), 5)
    assert_equal(str_slice[0], ord("H"))
    assert_equal(str_slice[1], ord("e"))
    assert_equal(str_slice[2], ord("l"))
    assert_equal(str_slice[3], ord("l"))
    assert_equal(str_slice[4], ord("o"))

    # ----------------------------------
    # Test subslicing
    # ----------------------------------

    # Slice the whole thing
    var sub1 = str_slice[:5]
    assert_equal(len(sub1), 5)
    assert_equal(sub1[0], ord("H"))
    assert_equal(sub1[1], ord("e"))
    assert_equal(sub1[2], ord("l"))
    assert_equal(sub1[3], ord("l"))
    assert_equal(sub1[4], ord("o"))

    # Slice the end
    var sub2 = str_slice[2:5]
    assert_equal(len(sub2), 3)
    assert_equal(sub2[0], ord("l"))
    assert_equal(sub2[1], ord("l"))
    assert_equal(sub2[2], ord("o"))

    # Slice the first element
    var sub3 = str_slice[0:1]
    assert_equal(len(sub3), 1)
    assert_equal(sub3[0], ord("H"))

    #
    # Test mutation through slice
    #

    sub1[0] = ord("J")
    assert_equal(string, "Jello")

    sub2[2] = ord("y")
    assert_equal(string, "Jelly")

    # ----------------------------------
    # Test empty subslicing
    # ----------------------------------

    var sub4 = str_slice[0:0]
    assert_equal(len(sub4), 0)

    var sub5 = str_slice[2:2]
    assert_equal(len(sub5), 0)

    # Empty slices still have a pointer value
    assert_equal(Int(sub5.unsafe_ptr()) - Int(sub4.unsafe_ptr()), 2)

    # ----------------------------------
    # Test out of range slicing
    # ----------------------------------

    # TODO: Improve error reporting for invalid slice bounds.

    # assert_equal(
    #     # str_slice[3:6]
    #     str_slice._try_slice(slice(3, 6)).unwrap[String](),
    #     String("Slice end is out of bounds"),
    # )

    # assert_equal(
    #     # str_slice[5:6]
    #     str_slice._try_slice(slice(5, 6)).unwrap[String](),
    #     String("Slice start is out of bounds"),
    # )

    # assert_equal(
    #     # str_slice[5:5]
    #     str_slice._try_slice(slice(5, 5)).unwrap[String](),
    #     String("Slice start is out of bounds"),
    # )

    # --------------------------------------------------------
    # Test that malformed partial slicing of codepoints raises
    # --------------------------------------------------------

    # These test what happens if you try to subslice a string in a way that
    # would leave the byte contents of the string containing partial encoded
    # codepoint sequences, invalid UTF-8. Consider a string with the following
    # content, containing both 1-byte and a 4-byte UTF-8 sequence:
    #
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
    # ┃          Hi👋!          ┃ String
    # ┣━━┳━━━┳━━━━━━━━━━━━━━━┳━━┫
    # ┃H ┃ i ┃       👋      ┃! ┃ Codepoint Characters
    # ┣━━╋━━━╋━━━━━━━━━━━━━━━╋━━┫
    # ┃72┃105┃    128075     ┃33┃ Codepoints
    # ┣━━╋━━━╋━━━┳━━━┳━━━┳━━━╋━━┫
    # ┃72┃105┃240┃159┃145┃139┃33┃ Bytes
    # ┗━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━┛
    #  0   1   2   3   4   5   6
    var unicode_str1 = StringSlice("Hi👋!")

    # Test slicing 0:{0–7}
    assert_equal(unicode_str1[0:0], "")
    assert_equal(unicode_str1[0:1], "H")
    assert_equal(unicode_str1[0:2], "Hi")
    assert_equal(unicode_str1[0:6], "Hi👋")
    assert_equal(unicode_str1[0:7], "Hi👋!")

    # -------------------------------------------------------------------
    # Test that slicing through combining codepoint graphemes is allowed
    # -------------------------------------------------------------------

    # The "ö" is a user-perceived character (grapheme) that is composed of two
    # codepoints. This test tests that we can use string slicing to divide that
    # grapheme into constituent codepoints.
    #
    # ┏━━━━━━━━━━━━━━━┓
    # ┃      yö       ┃ String
    # ┣━━━┳━━━┳━━━━━━━┫
    # ┃ y ┃ o ┃   ̈    ┃ Codepoint Characters
    # ┣━━━╋━━━╋━━━━━━━┫
    # ┃121┃111┃  776  ┃ Codepoints
    # ┣━━━╋━━━╋━━━┳━━━┫
    # ┃121┃111┃204┃136┃ Bytes
    # ┗━━━┻━━━┻━━━┻━━━┛
    #   0   1   2   3
    var unicode_str2 = StringSlice("yö")

    assert_equal(unicode_str2[0:1], "y")
    assert_equal(unicode_str2[0:2], "yo")
    assert_equal(unicode_str2[0:4], unicode_str2)
    # NOTE: This renders weirdly, but is a single-codepoint string containing
    #   <https://www.compart.com/en/unicode/U+0308>.
    assert_equal(unicode_str2[2:4], "̈")


fn test_heap_string_from_string_slice() raises:
    alias static_str = "Hello".as_string_slice()

    alias heap_string = String(static_str)

    assert_equal(heap_string, "Hello")


fn test_string_substring() raises:
    var string = String("Hello")
    var str_slice = string.as_string_slice()

    assert_equal(len(str_slice), 5)
    assert_equal(str_slice[0], "H")
    assert_equal(str_slice[1], "e")
    assert_equal(str_slice[2], "l")
    assert_equal(str_slice[3], "l")
    assert_equal(str_slice[4], "o")

    # ----------------------------------
    # Test subslicing
    # ----------------------------------

    # Slice the whole thing
    var sub1 = str_slice[:5]
    assert_equal(len(sub1), 5)
    assert_equal(sub1[0], "H")
    assert_equal(sub1[1], "e")
    assert_equal(sub1[2], "l")
    assert_equal(sub1[3], "l")
    assert_equal(sub1[4], "o")

    # Slice the end
    var sub2 = str_slice[2:5]
    assert_equal(len(sub2), 3)
    assert_equal(sub2[0], "l")
    assert_equal(sub2[1], "l")
    assert_equal(sub2[2], "o")

    # Slice the first element
    var sub3 = str_slice[0:1]
    assert_equal(len(sub3), 1)
    assert_equal(sub3[0], "H")
    assert_equal(sub3[-1], "H")

    # ----------------------------------
    # Test empty subslicing
    # ----------------------------------

    var sub4 = str_slice[0:0]
    assert_equal(len(sub4), 0)

    var sub5 = str_slice[2:2]
    assert_equal(len(sub5), 0)

    # Empty slices still have a pointer value
    assert_equal(Int(sub5.unsafe_ptr()) - Int(sub4.unsafe_ptr()), 2)


fn test_slice_len() raises:
    assert_equal(5, len(StringSlice("12345")))
    assert_equal(4, len(StringSlice("1234")))
    assert_equal(3, len(StringSlice("123")))
    assert_equal(2, len(StringSlice("12")))
    assert_equal(1, len(StringSlice("1")))
    assert_equal(0, len(StringSlice("")))

    # String length is in bytes, not codepoints.
    var s0 = String("ನಮಸ್ಕಾರ")
    assert_equal(len(s0), 21)
    assert_equal(len(s0.codepoints()), 7)

    # For ASCII string, the byte and codepoint length are the same:
    var s1 = String("abc")
    assert_equal(len(s1), 3)
    assert_equal(len(s1.codepoints()), 3)


fn test_slice_char_length() raises:
    var s0 = StringSlice("")
    assert_equal(s0.byte_length(), 0)
    assert_equal(s0.char_length(), 0)

    var s1 = StringSlice("foo")
    assert_equal(s1.byte_length(), 3)
    assert_equal(s1.char_length(), 3)

    # This string contains 1-, 2-, 3-, and 4-byte codepoint sequences.
    var s2 = EVERY_CODEPOINT_LENGTH_STR
    assert_equal(s2.byte_length(), 13)
    assert_equal(s2.char_length(), 5)

    # Just a bit of Zalgo text.
    var s3 = StringSlice("H̵͙̖̼̬̬̲̱͊̇̅͂̍͐͌͘͜͝")
    assert_equal(s3.byte_length(), 37)
    assert_equal(s3.char_length(), 19)

    # Character length is codepoints, not graphemes
    # This is thumbs up + a skin tone modifier codepoint.
    var s4 = StringSlice("👍🏻")
    assert_equal(s4.byte_length(), 8)
    assert_equal(s4.char_length(), 2)
    # TODO: assert_equal(s4.grapheme_count(), 1)


fn test_slice_eq() raises:
    var str1: String = "12345"
    var str2: String = "12345"
    var str3: StaticString = "12345"
    var str4: String = "abc"
    var str5: String = "abcdef"
    var str6: StaticString = "abcdef"

    # eq

    # FIXME: the origin of the StringSlice origin should be the data in the
    # string, not the string itself.
    # assert_true(str1.as_string_slice().__eq__(str1))
    assert_true(str1.as_string_slice().__eq__(str2))
    assert_true(str2.as_string_slice().__eq__(str2.as_string_slice()))
    assert_true(str1.as_string_slice().__eq__(str3))

    # ne

    assert_true(str1.as_string_slice().__ne__(str4))
    assert_true(str1.as_string_slice().__ne__(str5))
    assert_true(str1.as_string_slice().__ne__(str5.as_string_slice()))
    assert_true(str1.as_string_slice().__ne__(str6))


fn test_slice_bool() raises:
    var str1: String = "abc"
    assert_true(str1.as_string_slice().__bool__())
    var str2: String = ""
    assert_true(not str2.as_string_slice().__bool__())


def test_slice_repr():
    # Standard single-byte characters
    assert_equal(StringSlice.__repr__("hello"), "'hello'")
    assert_equal(StringSlice.__repr__(String(0)), "'0'")
    assert_equal(StringSlice.__repr__("A"), "'A'")
    assert_equal(StringSlice.__repr__(" "), "' '")
    assert_equal(StringSlice.__repr__("~"), "'~'")

    # Special single-byte characters
    assert_equal(StringSlice.__repr__("\0"), r"'\x00'")
    assert_equal(StringSlice.__repr__("\x06"), r"'\x06'")
    assert_equal(StringSlice.__repr__("\x09"), r"'\t'")
    assert_equal(StringSlice.__repr__("\n"), r"'\n'")
    assert_equal(StringSlice.__repr__("\x0d"), r"'\r'")
    assert_equal(StringSlice.__repr__("\x0e"), r"'\x0e'")
    assert_equal(StringSlice.__repr__("\x1f"), r"'\x1f'")
    assert_equal(StringSlice.__repr__("'"), '"\'"')
    assert_equal(StringSlice.__repr__("\\"), r"'\\'")
    assert_equal(StringSlice.__repr__("\x7f"), r"'\x7f'")

    # Multi-byte characters
    assert_equal(
        StringSlice.__repr__("Örnsköldsvik"), "'Örnsköldsvik'"
    )  # 2-byte
    assert_equal(StringSlice.__repr__("你好!"), "'你好!'")  # 3-byte
    assert_equal(StringSlice.__repr__("hello 🔥!"), "'hello 🔥!'")  # 4-byte


def test_find():
    haystack = String("abcdefg").as_string_slice()
    haystack_with_special_chars = String("abcdefg@#$").as_string_slice()
    haystack_repeated_chars = String(
        "aaaaaaaaaaaaaaaaaaaaaaaa"
    ).as_string_slice()

    assert_equal(haystack.find(String("a").as_string_slice()), 0)
    assert_equal(haystack.find(String("ab").as_string_slice()), 0)
    assert_equal(haystack.find(String("abc").as_string_slice()), 0)
    assert_equal(haystack.find(String("bcd").as_string_slice()), 1)
    assert_equal(haystack.find(String("de").as_string_slice()), 3)
    assert_equal(haystack.find(String("fg").as_string_slice()), 5)
    assert_equal(haystack.find(String("g").as_string_slice()), 6)
    assert_equal(haystack.find(String("z").as_string_slice()), -1)
    assert_equal(haystack.find(String("zzz").as_string_slice()), -1)

    assert_equal(haystack.find(String("@#$").as_string_slice()), -1)
    assert_equal(
        haystack_with_special_chars.find(String("@#$").as_string_slice()), 7
    )

    assert_equal(
        haystack_repeated_chars.find(String("aaa").as_string_slice()), 0
    )
    assert_equal(
        haystack_repeated_chars.find(String("AAa").as_string_slice()), -1
    )

    assert_equal(
        haystack.find(String("hijklmnopqrstuvwxyz").as_string_slice()), -1
    )

    assert_equal(
        String("").as_string_slice().find(String("abc").as_string_slice()), -1
    )


def test_find_compile_time():
    alias haystack = String("abcdefg").as_string_slice()
    alias haystack_with_special_chars = String("abcdefg@#$").as_string_slice()
    alias haystack_repeated_chars = String(
        "aaaaaaaaaaaaaaaaaaaaaaaa"
    ).as_string_slice()

    alias c1 = haystack.find(String("a").as_string_slice())
    alias c2 = haystack.find(String("ab").as_string_slice())
    alias c3 = haystack.find(String("abc").as_string_slice())
    alias c4 = haystack.find(String("bcd").as_string_slice())
    alias c5 = haystack.find(String("de").as_string_slice())
    alias c6 = haystack.find(String("fg").as_string_slice())
    alias c7 = haystack.find(String("g").as_string_slice())
    alias c8 = haystack.find(String("z").as_string_slice())
    alias c9 = haystack.find(String("zzz").as_string_slice())
    alias c10 = haystack.find(String("@#$").as_string_slice())
    alias c11 = haystack_with_special_chars.find(
        String("@#$").as_string_slice()
    )
    alias c12 = haystack_repeated_chars.find(String("aaa").as_string_slice())
    alias c13 = haystack_repeated_chars.find(String("AAa").as_string_slice())
    alias c14 = haystack.find(String("hijklmnopqrstuvwxyz").as_string_slice())
    alias c15 = String("").as_string_slice().find(
        String("abc").as_string_slice()
    )

    assert_equal(c1, 0)
    assert_equal(c2, 0)
    assert_equal(c3, 0)
    assert_equal(c4, 1)
    assert_equal(c5, 3)
    assert_equal(c6, 5)
    assert_equal(c7, 6)
    assert_equal(c8, -1)
    assert_equal(c9, -1)
    assert_equal(c10, -1)
    assert_equal(c11, 7)
    assert_equal(c12, 0)
    assert_equal(c13, -1)
    assert_equal(c14, -1)
    assert_equal(c15, -1)


def test_is_codepoint_boundary():
    var abc = StringSlice("abc")
    assert_equal(len(abc), 3)
    assert_true(abc.is_codepoint_boundary(0))
    assert_true(abc.is_codepoint_boundary(1))
    assert_true(abc.is_codepoint_boundary(2))
    assert_true(abc.is_codepoint_boundary(3))

    var thumb = StringSlice("👍")
    assert_equal(len(thumb), 4)
    assert_true(thumb.is_codepoint_boundary(0))
    assert_false(thumb.is_codepoint_boundary(1))
    assert_false(thumb.is_codepoint_boundary(2))
    assert_false(thumb.is_codepoint_boundary(3))

    var empty = StringSlice("")
    assert_equal(len(empty), 0)
    assert_true(empty.is_codepoint_boundary(0))
    # Also tests that positions greater then the length don't raise/abort.
    assert_false(empty.is_codepoint_boundary(1))


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
    # escape secuences are accepted
    var univ_sep_var = (
        String(
            " ",
            "\t",
            "\n",
            "\r",
            "\v",
            "\f",
            "\x1c",
            "\x1d",
            "\x1e",
            String(next_line),
            String(unicode_line_sep),
            String(unicode_paragraph_sep),
        )
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
    # assert_true(len(S("").split("")) == 2) # TODO(#3528)
    assert_true(len(S("  ").split(" ")) == 3)
    assert_true(len(S("   ").split(" ")) == 4)

    # should split into maxsplit + 1 items
    assert_equal(S("1,2,3").split(",", 0), L("1,2,3"))
    assert_equal(S("1,2,3").split(",", 1), L("1", "2,3"))

    # TODO(#3528): delete this test
    with assert_raises():
        _ = S("").split("")

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

    # TODO(#3528)
    # assert_equal(S("123").split(""), L("", "1", "2", "3", ""))
    # assert_equal(S("").join(S("123").split("")), "123")
    # assert_equal(S(",1,2,3,").split(","), S("123").split(""))
    # assert_equal(S(",").join(S("123").split("")), ",1,2,3,")


def test_splitlines():
    alias S = StringSlice[StaticConstantOrigin]
    alias L = List[StringSlice[StaticConstantOrigin]]

    # FIXME: remove once StringSlice conforms to TestableCollectionElement
    fn _assert_equal[
        O1: ImmutableOrigin
    ](l1: List[StringSlice[O1]], l2: List[String]) raises:
        assert_equal(len(l1), len(l2))
        for i in range(len(l1)):
            assert_equal(String(l1[i]), l2[i])

    # Test with no line breaks
    assert_equal(S("hello world").splitlines(), L("hello world"))

    # Test with line breaks
    assert_equal(S("hello\nworld").splitlines(), L("hello", "world"))
    assert_equal(S("hello\rworld").splitlines(), L("hello", "world"))
    assert_equal(S("hello\r\nworld").splitlines(), L("hello", "world"))

    # Test with multiple different line breaks
    s1 = S("hello\nworld\r\nmojo\rlanguage\r\n")
    hello_mojo = L("hello", "world", "mojo", "language")
    assert_equal(s1.splitlines(), hello_mojo)
    assert_equal(
        s1.splitlines(keepends=True),
        L("hello\n", "world\r\n", "mojo\r", "language\r\n"),
    )

    # Test with an empty string
    assert_equal(S("").splitlines(), L())
    # test \v \f \x1c \x1d
    s2 = S("hello\vworld\fmojo\x1clanguage\x1d")
    assert_equal(s2.splitlines(), hello_mojo)
    assert_equal(
        s2.splitlines(keepends=True),
        L("hello\v", "world\f", "mojo\x1c", "language\x1d"),
    )

    # test \x1c \x1d \x1e
    s3 = S("hello\x1cworld\x1dmojo\x1elanguage\x1e")
    assert_equal(s3.splitlines(), hello_mojo)
    assert_equal(
        s3.splitlines(keepends=True),
        L("hello\x1c", "world\x1d", "mojo\x1e", "language\x1e"),
    )

    # test \x85 \u2028 \u2029
    var next_line = String(List[UInt8](0xC2, 0x85))
    var unicode_line_sep = String(List[UInt8](0xE2, 0x80, 0xA8))
    var unicode_paragraph_sep = String(List[UInt8](0xE2, 0x80, 0xA9))

    for i in List(next_line, unicode_line_sep, unicode_paragraph_sep):
        u = i[]
        item = String("").join("hello", u, "world", u, "mojo", u, "language", u)
        s = StringSlice(item)
        assert_equal(s.splitlines(), hello_mojo)
        items = List("hello" + u, "world" + u, "mojo" + u, "language" + u)
        _assert_equal(s.splitlines(keepends=True), items)


def test_rstrip():
    # with default rstrip chars
    var empty_string = "".as_string_slice()
    assert_true(empty_string.rstrip() == "")

    var space_string = " \t\n\r\v\f  ".as_string_slice()
    assert_true(space_string.rstrip() == "")

    var str0 = "     n ".as_string_slice()
    assert_true(str0.rstrip() == "     n")

    var str1 = "string".as_string_slice()
    assert_true(str1.rstrip() == "string")

    var str2 = "something \t\n\t\v\f".as_string_slice()
    assert_true(str2.rstrip() == "something")

    # with custom chars for rstrip
    var str3 = "mississippi".as_string_slice()
    assert_true(str3.rstrip("sip") == "m")

    var str4 = "mississippimississippi \n ".as_string_slice()
    assert_true(str4.rstrip("sip ") == "mississippimississippi \n")
    assert_true(str4.rstrip("sip \n") == "mississippim")


def test_lstrip():
    # with default lstrip chars
    var empty_string = "".as_string_slice()
    assert_true(empty_string.lstrip() == "")

    var space_string = " \t\n\r\v\f  ".as_string_slice()
    assert_true(space_string.lstrip() == "")

    var str0 = "     n ".as_string_slice()
    assert_true(str0.lstrip() == "n ")

    var str1 = "string".as_string_slice()
    assert_true(str1.lstrip() == "string")

    var str2 = " \t\n\t\v\fsomething".as_string_slice()
    assert_true(str2.lstrip() == "something")

    # with custom chars for lstrip
    var str3 = "mississippi".as_string_slice()
    assert_true(str3.lstrip("mis") == "ppi")

    var str4 = " \n mississippimississippi".as_string_slice()
    assert_true(str4.lstrip("mis ") == "\n mississippimississippi")
    assert_true(str4.lstrip("mis \n") == "ppimississippi")


def test_strip():
    # with default strip chars
    var empty_string = "".as_string_slice()
    assert_true(empty_string.strip() == "")
    alias comp_empty_string_stripped = "".as_string_slice().strip()
    assert_true(comp_empty_string_stripped == "")

    var space_string = " \t\n\r\v\f  ".as_string_slice()
    assert_true(space_string.strip() == "")
    alias comp_space_string_stripped = " \t\n\r\v\f  ".as_string_slice().strip()
    assert_true(comp_space_string_stripped == "")

    var str0 = "     n ".as_string_slice()
    assert_true(str0.strip() == "n")
    alias comp_str0_stripped = "     n ".as_string_slice().strip()
    assert_true(comp_str0_stripped == "n")

    var str1 = "string".as_string_slice()
    assert_true(str1.strip() == "string")
    alias comp_str1_stripped = ("string").strip()
    assert_true(comp_str1_stripped == "string")

    var str2 = " \t\n\t\v\fsomething \t\n\t\v\f".as_string_slice()
    alias comp_str2_stripped = (" \t\n\t\v\fsomething \t\n\t\v\f").strip()
    assert_true(str2.strip() == "something")
    assert_true(comp_str2_stripped == "something")

    # with custom strip chars
    var str3 = "mississippi".as_string_slice()
    assert_true(str3.strip("mips") == "")
    assert_true(str3.strip("mip") == "ssiss")
    alias comp_str3_stripped = "mississippi".as_string_slice().strip("mips")
    assert_true(comp_str3_stripped == "")

    var str4 = " \n mississippimississippi \n ".as_string_slice()
    assert_true(str4.strip(" ") == "\n mississippimississippi \n")
    assert_true(str4.strip("\nmip ") == "ssissippimississ")

    alias comp_str4_stripped = (
        " \n mississippimississippi \n ".as_string_slice().strip(" ")
    )
    assert_true(comp_str4_stripped == "\n mississippimississippi \n")


def test_startswith():
    var empty = StringSlice("")
    assert_true(empty.startswith(""))
    assert_false(empty.startswith("a"))
    assert_false(empty.startswith("ab"))

    var a = StringSlice("a")
    assert_true(a.startswith(""))
    assert_true(a.startswith("a"))
    assert_false(a.startswith("ab"))

    var ab = StringSlice("ab")
    assert_true(ab.startswith(""))
    assert_true(ab.startswith("a"))
    assert_false(ab.startswith("b"))
    assert_true(ab.startswith("b", start=1))
    assert_true(ab.startswith("a", end=1))
    assert_true(ab.startswith("ab"))


def test_endswith():
    var empty = StringSlice("")
    assert_true(empty.endswith(""))
    assert_false(empty.endswith("a"))
    assert_false(empty.endswith("ab"))

    var a = StringSlice("a")
    assert_true(a.endswith(""))
    assert_true(a.endswith("a"))
    assert_false(a.endswith("ab"))

    var ab = StringSlice("ab")
    assert_true(ab.endswith(""))
    assert_false(ab.endswith("a"))
    assert_true(ab.endswith("b"))
    assert_true(ab.endswith("b", start=1))
    assert_true(ab.endswith("a", end=1))
    assert_true(ab.endswith("ab"))


def test_isupper():
    assert_true(StringSlice("ASDG").isupper())
    assert_false(StringSlice("AsDG").isupper())
    assert_true(StringSlice("ABC123").isupper())
    assert_false(StringSlice("1!").isupper())
    assert_true(StringSlice("É").isupper())
    assert_false(StringSlice("é").isupper())


def test_islower():
    assert_true(StringSlice("asdfg").islower())
    assert_false(StringSlice("asdFDg").islower())
    assert_true(StringSlice("abc123").islower())
    assert_false(StringSlice("1!").islower())
    assert_true(StringSlice("é").islower())
    assert_false(StringSlice("É").islower())


def test_lower():
    assert_equal(StringSlice("HELLO").lower(), "hello")
    assert_equal(StringSlice("hello").lower(), "hello")
    assert_equal(StringSlice("FoOBaR").lower(), "foobar")

    assert_equal(StringSlice("MOJO🔥").lower(), "mojo🔥")

    assert_equal(StringSlice("É").lower(), "é")
    assert_equal(StringSlice("é").lower(), "é")


def test_upper():
    assert_equal(StringSlice("hello").upper(), "HELLO")
    assert_equal(StringSlice("HELLO").upper(), "HELLO")
    assert_equal(StringSlice("FoOBaR").upper(), "FOOBAR")

    assert_equal(StringSlice("mojo🔥").upper(), "MOJO🔥")

    assert_equal(StringSlice("É").upper(), "É")
    assert_equal(StringSlice("é").upper(), "É")


def test_is_ascii_digit():
    assert_false(StringSlice("").is_ascii_digit())
    assert_true(StringSlice("123").is_ascii_digit())
    assert_false(StringSlice("asdg").is_ascii_digit())
    assert_false(StringSlice("123asdg").is_ascii_digit())


def test_is_ascii_printable():
    assert_true(StringSlice("aasdg").is_ascii_printable())
    assert_false(StringSlice("aa\nae").is_ascii_printable())
    assert_false(StringSlice("aa\tae").is_ascii_printable())


def test_rjust():
    assert_equal(StringSlice("hello").rjust(4), "hello")
    assert_equal(StringSlice("hello").rjust(8), "   hello")
    assert_equal(StringSlice("hello").rjust(8, "*"), "***hello")


def test_ljust():
    assert_equal(StringSlice("hello").ljust(4), "hello")
    assert_equal(StringSlice("hello").ljust(8), "hello   ")
    assert_equal(StringSlice("hello").ljust(8, "*"), "hello***")


def test_center():
    assert_equal(StringSlice("hello").center(4), "hello")
    assert_equal(StringSlice("hello").center(8), " hello  ")
    assert_equal(StringSlice("hello").center(8, "*"), "*hello**")


def test_count():
    var str = StringSlice("Hello world")

    assert_equal(12, str.count(""))
    assert_equal(1, str.count("Hell"))
    assert_equal(3, str.count("l"))
    assert_equal(1, str.count("ll"))
    assert_equal(1, str.count("ld"))
    assert_equal(0, str.count("universe"))

    assert_equal(StringSlice("aaaaa").count("a"), 5)
    assert_equal(StringSlice("aaaaaa").count("aa"), 3)


def test_chars_iter():
    # Test `for` loop iteration support
    for char in StringSlice("abc").codepoints():
        assert_true(
            char in (Codepoint.ord("a"), Codepoint.ord("b"), Codepoint.ord("c"))
        )

    # Test empty string chars
    var s0 = StringSlice("")
    var s0_iter = s0.codepoints()

    assert_false(s0_iter.__has_next__())
    assert_true(s0_iter.peek_next() is None)
    assert_true(s0_iter.next() is None)

    # Test simple ASCII string chars
    var s1 = StringSlice("abc")
    var s1_iter = s1.codepoints()

    assert_equal(s1_iter.next().value(), Codepoint.ord("a"))
    assert_equal(s1_iter.next().value(), Codepoint.ord("b"))
    assert_equal(s1_iter.next().value(), Codepoint.ord("c"))
    assert_true(s1_iter.next() is None)

    # Multibyte character decoding: A visual character composed of a combining
    # sequence of 2 codepoints.
    var s2 = StringSlice("á")
    assert_equal(s2.byte_length(), 3)
    assert_equal(s2.char_length(), 2)

    var iter = s2.codepoints()
    assert_equal(iter.__next__(), Codepoint.ord("a"))
    # U+0301 Combining Acute Accent
    assert_equal(iter.__next__().to_u32(), 0x0301)
    assert_equal(iter.__has_next__(), False)

    # A piece of text containing, 1-byte, 2-byte, 3-byte, and 4-byte codepoint
    # sequences.
    var s3 = EVERY_CODEPOINT_LENGTH_STR
    assert_equal(s3.byte_length(), 13)
    assert_equal(s3.char_length(), 5)
    var s3_iter = s3.codepoints()

    # Iterator __len__ returns length in codepoints, not bytes.
    assert_equal(s3_iter.__len__(), 5)
    assert_equal(s3_iter._slice.byte_length(), 13)
    assert_equal(s3_iter.__has_next__(), True)
    assert_equal(s3_iter.__next__(), Codepoint.ord("߷"))

    assert_equal(s3_iter.__len__(), 4)
    assert_equal(s3_iter._slice.byte_length(), 11)
    assert_equal(s3_iter.__next__(), Codepoint.ord("ക"))

    # Combining character, visually comes first, but codepoint-wise comes
    # after the character it combines with.
    assert_equal(s3_iter.__len__(), 3)
    assert_equal(s3_iter._slice.byte_length(), 8)
    assert_equal(s3_iter.__next__(), Codepoint.ord("ൈ"))

    assert_equal(s3_iter.__len__(), 2)
    assert_equal(s3_iter._slice.byte_length(), 5)
    assert_equal(s3_iter.__next__(), Codepoint.ord("🔄"))

    assert_equal(s3_iter.__len__(), 1)
    assert_equal(s3_iter._slice.byte_length(), 1)
    assert_equal(s3_iter.__has_next__(), True)
    assert_equal(s3_iter.__next__(), Codepoint.ord("!"))

    assert_equal(s3_iter.__len__(), 0)
    assert_equal(s3_iter._slice.byte_length(), 0)
    assert_equal(s3_iter.__has_next__(), False)


def test_string_slice_from_pointer():
    var a = StringSlice("AAA")
    var b = StringSlice[StaticConstantOrigin](
        unsafe_from_utf8_ptr=a.unsafe_ptr()
    )
    assert_equal(3, len(a))
    assert_equal(3, len(b))
    var c = String("ABCD")
    var d = StringSlice[__origin_of(c)](
        unsafe_from_utf8_ptr=c.unsafe_cstr_ptr()
    )
    var e = StringSlice[__origin_of(c)](unsafe_from_utf8_ptr=c.unsafe_ptr())
    assert_equal(4, len(c))
    assert_equal(4, len(d))
    assert_equal(4, len(e))
    assert_true("A", d[0])
    assert_true("B", d[1])
    assert_true("C", d[2])
    assert_true("D", d[3])
    assert_true("D", d[-1])


def test_replace():
    assert_equal(StringSlice("").replace("", "hello world"), "")
    assert_equal(
        StringSlice("hello").replace("", "something"),
        "somethinghsomethingesomethinglsomethinglsomethingo",
    )
    assert_equal(StringSlice("hello world").replace("world", ""), "hello ")
    assert_equal(
        StringSlice("hello world").replace("world", "mojo"), "hello mojo"
    )
    assert_equal(
        StringSlice("hello world hello world").replace("world", "mojo"),
        "hello mojo hello mojo",
    )


def test_join():
    assert_equal(StaticString("").join(), "")
    assert_equal(StaticString("").join("a", "b", "c"), "abc")
    assert_equal(StaticString(" ").join("a", "b", "c"), "a b c")
    assert_equal(StaticString(" ").join("a", "b", "c", ""), "a b c ")
    assert_equal(StaticString(" ").join("a", "b", "c", " "), "a b c  ")

    var sep = StaticString(",")
    var s = String("abc")
    assert_equal(sep.join(s, s, s, s), "abc,abc,abc,abc")
    assert_equal(sep.join(1, 2, 3), "1,2,3")
    assert_equal(sep.join(1, "abc", 3), "1,abc,3")

    var s2 = StaticString(",").join(List[UInt8](1, 2, 3))
    assert_equal(s2, "1,2,3")

    var s3 = StaticString(",").join(List[UInt8](1, 2, 3, 4, 5, 6, 7, 8, 9))
    assert_equal(s3, "1,2,3,4,5,6,7,8,9")

    var s4 = StaticString(",").join(List[UInt8]())
    assert_equal(s4, "")

    var s5 = StaticString(",").join(List[UInt8](1))
    assert_equal(s5, "1")


def test_string_slice_intern():
    assert_equal(get_static_string["hello"](), "hello")
    assert_equal(get_static_string[String("hello")](), "hello")
    assert_equal(get_static_string[String(42)](), "42")
    alias simd = SIMD[DType.int64, 4](1, 2, 3, 4)
    assert_equal(get_static_string[String(simd)](), "[1, 2, 3, 4]")
    # Test get_static_string with multiple string arguments.
    assert_equal(get_static_string["a", "b", "c"](), "abc")


def main():
    test_string_slice_layout()
    test_string_literal_byte_span()
    test_string_byte_span()
    test_heap_string_from_string_slice()
    test_slice_len()
    test_slice_char_length()
    test_slice_eq()
    test_slice_bool()
    test_slice_repr()
    test_find()
    test_find_compile_time()
    test_is_codepoint_boundary()
    test_split()
    test_splitlines()
    test_rstrip()
    test_lstrip()
    test_strip()
    test_startswith()
    test_endswith()
    test_isupper()
    test_islower()
    test_lower()
    test_upper()
    test_is_ascii_digit()
    test_is_ascii_printable()
    test_rjust()
    test_ljust()
    test_center()
    test_count()
    test_chars_iter()
    test_string_slice_from_pointer()
    test_replace()
    test_join()
    test_string_slice_intern()
