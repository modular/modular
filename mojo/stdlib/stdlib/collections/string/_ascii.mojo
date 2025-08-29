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

# ===----------------------------------------------------------------------=== #
# isdigit
# ===----------------------------------------------------------------------=== #


@always_inline
fn _isdigit_vec[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `0` = SIMD[DType.uint8, w](Byte(ord("0")))
    alias `9` = SIMD[DType.uint8, w](Byte(ord("9")))
    return v.ge(`0`) & v.le(`9`)


# ===----------------------------------------------------------------------=== #
# isprintable
# ===----------------------------------------------------------------------=== #


@always_inline
fn _is_ascii_printable_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias ` ` = SIMD[DType.uint8, w](Byte(ord(" ")))
    alias `~` = SIMD[DType.uint8, w](Byte(ord("~")))
    return v.ge(` `) & v.le(`~`)


@always_inline
fn _nonprintable_ascii[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (~_is_ascii_printable_vec(v)) & v.lt(0b1000_0000)


@always_inline
fn _is_python_printable_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `\\` = SIMD[DType.uint8, w](Byte(ord(" ")))
    return v.ne(`\\`) & _is_ascii_printable_vec(v)


@always_inline
fn _nonprintable_python[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (~_is_python_printable_vec(v)) & v.lt(0b1000_0000)


# ===----------------------------------------------------------------------=== #
# isupper
# ===----------------------------------------------------------------------=== #


@always_inline
fn _is_ascii_uppercase_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `A` = SIMD[DType.uint8, w](Byte(ord("A")))
    alias `Z` = SIMD[DType.uint8, w](Byte(ord("Z")))
    return v.ge(`A`) & v.le(`Z`)


@always_inline
fn _is_ascii_uppercase(c: Byte) -> Bool:
    return _is_ascii_uppercase_vec(c)


# ===----------------------------------------------------------------------=== #
# islower
# ===----------------------------------------------------------------------=== #


@always_inline
fn _is_ascii_lowercase_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `a` = SIMD[DType.uint8, w](Byte(ord("a")))
    alias `z` = SIMD[DType.uint8, w](Byte(ord("z")))
    return v.ge(`a`) & v.le(`z`)


@always_inline
fn _is_ascii_lowercase(c: Byte) -> Bool:
    return _is_ascii_lowercase_vec(c)


# ===----------------------------------------------------------------------=== #
# toggle_case
# ===----------------------------------------------------------------------=== #


@always_inline
fn _ascii_toggle_case[
    w: Int
](value: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, w]:
    alias `a` = Byte(ord("a"))
    alias `A` = Byte(ord("A"))
    # ASCII only has a 1 upper bit bifference in uppercase and lowercase letters
    return value ^ (`A` ^ `a`)


# ===----------------------------------------------------------------------=== #
# isspace
# ===----------------------------------------------------------------------=== #


fn _is_ascii_space(c: Byte) -> Bool:
    """Determines whether the given character is an ASCII whitespace character:
    `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

    Args:
        c: The character to check.

    Returns:
        True if the character is one of the ASCII whitespace characters.
    """

    # NOTE: a global LUT doesn't work at compile time so we can't use it here.
    alias ` ` = Byte(ord(" "))
    alias `\t` = Byte(ord("\t"))
    alias `\n` = Byte(ord("\n"))
    alias `\r` = Byte(ord("\r"))
    alias `\f` = Byte(ord("\f"))
    alias `\v` = Byte(ord("\v"))
    alias `\x1c` = Byte(ord("\x1c"))
    alias `\x1d` = Byte(ord("\x1d"))
    alias `\x1e` = Byte(ord("\x1e"))

    # This compiles to something very clever that's even faster than a LUT.
    return (
        c == ` `
        or c == `\t`
        or c == `\n`
        or c == `\r`
        or c == `\f`
        or c == `\v`
        or c == `\x1c`
        or c == `\x1d`
        or c == `\x1e`
    )


# ===----------------------------------------------------------------------=== #
# ascii
# ===----------------------------------------------------------------------=== #


fn _chr_ascii(c: UInt8) -> String:
    """Returns a string based on the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.
    """
    var result = String(capacity=1)
    result.append_byte(c)
    return result^


fn _repr_ascii(c: UInt8) -> String:
    """Returns a printable representation of the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a representation of the given code point.
    """
    alias ord_tab = ord("\t")
    alias ord_new_line = ord("\n")
    alias ord_carriage_return = ord("\r")
    alias ord_back_slash = ord("\\")

    if c == ord_back_slash:
        return r"\\"
    elif _is_ascii_printable_vec(c):
        return _chr_ascii(c)
    elif c == ord_tab:
        return r"\t"
    elif c == ord_new_line:
        return r"\n"
    elif c == ord_carriage_return:
        return r"\r"
    else:
        var uc = c.cast[DType.uint8]()
        if uc < 16:
            return hex(uc, prefix=r"\x0")
        else:
            return hex(uc, prefix=r"\x")


@always_inline
fn ascii(value: StringSlice) -> String:
    """Get the ASCII representation of the object.

    Args:
        value: The object to get the ASCII representation of.

    Returns:
        A string containing the ASCII representation of the object.
    """
    alias ord_squote = ord("'")
    var result = String()
    var use_dquote = False

    for idx in range(len(value._slice)):
        var char = value._slice[idx]
        result += _repr_ascii(char)
        use_dquote = use_dquote or (char == ord_squote)

    if use_dquote:
        return '"' + result + '"'
    else:
        return "'" + result + "'"
