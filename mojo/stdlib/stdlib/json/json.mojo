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

"""
A simple, high‐performance JSON parser in pure Mojo.

This module provides functionality for parsing JSON strings into structured
data types and serializing them back to JSON format.
"""

from collections import Dict, InlineArray, List
from collections.optional import _NoneType
from sys import simdwidthof

from memory import Pointer

from utils import Variant, Writer


@fieldwise_init
struct JSONList[mut: Bool, //, origin: Origin[mut]](
    Representable, Copyable, Movable, Writable, Stringable
):
    """
    Represents a JSON array as a list of JSON values.

    Parameters:
        mut: Whether the list is mutable.
        origin: The origin of the list's memory.
    """

    var _storage: List[JSONValue[origin]]

    @no_inline
    fn write_to[W: Writer, //](self, mut writer: W):
        """
        Writes the JSON list to the given writer in JSON array format.

        Parameters:
            W: The type of the writer.

        Args:
            writer: The writer to write to.
        """
        writer.write("[")
        var first = True
        for elem in self._storage:
            if not first:
                writer.write(", ")
            else:
                first = False
            writer.write(elem[])
        writer.write("]")

    fn __str__(self) -> String:
        """
        Returns a string representation of the JSON list.

        Returns:
            A string containing the JSON array representation.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """
        Returns a string representation of the JSON list.

        Returns:
            A string containing the JSON array representation.
        """
        return String(self)


@fieldwise_init
struct JSONDict[mut: Bool, //, origin: Origin[mut]](
    Representable, Copyable, Movable, Writable, Stringable
):
    """
    Represents a JSON object as a dictionary of string keys to JSON values.

    Parameters:
        mut: Whether the dictionary is mutable.
        origin: The origin of the dictionary's memory.
    """

    var _storage: Dict[String, JSONValue[origin]]

    @no_inline
    fn write_to[W: Writer, //](self, mut writer: W):
        """
        Writes the JSON dictionary to the given writer in JSON object format.

        Parameters:
            W: The type of the writer.

        Args:
            writer: The writer to write to.
        """
        writer.write("{")
        try:
            var first = True
            for elem in self._storage:
                if not first:
                    writer.write(", ")
                else:
                    first = False
                writer.write('"', elem[], '": ', self._storage[elem[]])
        except:
            pass
        writer.write("}")

    fn __str__(self) -> String:
        """
        Returns a string representation of the JSON dictionary.

        Returns:
            A string containing the JSON object representation.
        """
        return String.write(self)

    fn __repr__(self) -> String:
        """
        Returns a string representation of the JSON dictionary.

        Returns:
            A string containing the JSON object representation.
        """
        return String(self)


struct JSONValue[mut: Bool, //, origin: Origin[mut]](
    Representable, Copyable, Movable, Writable, Stringable
):
    """
    Represents a JSON value of any type (null, boolean, number, string,
    array, or object).

    Parameters:
        mut: Whether the value is mutable.
        origin: The origin of the value's memory.
    """

    var _kind: UInt8
    var _storage: Variant[
        _NoneType,
        Bool,
        String,
        Float64,
        JSONList[origin],
        JSONDict[origin],
    ]

    alias KIND_NULL: UInt8 = 0
    alias KIND_BOOL: UInt8 = 1
    alias KIND_NUMBER: UInt8 = 2
    alias KIND_STRING: UInt8 = 3
    alias KIND_ARRAY: UInt8 = 4
    alias KIND_OBJECT: UInt8 = 5

    @always_inline
    fn __init__(out self):
        """
        Initializes a new JSON value as null.
        """
        self = Self(_NoneType())

    @always_inline
    @implicit
    fn __init__(out self, v: _NoneType):
        """
        Initializes a new JSON value as null.

        Args:
            v: The null value.
        """
        self._kind = Self.KIND_NULL
        self._storage = v

    @always_inline
    @implicit
    fn __init__(out self, v: Bool):
        """
        Initializes a new JSON value as a boolean.

        Args:
            v: The boolean value.
        """
        self._kind = Self.KIND_BOOL
        self._storage = v

    @always_inline
    @implicit
    fn __init__(out self, s: Float64):
        """
        Initializes a new JSON value as a floating-point number.

        Args:
            s: The floating-point value.
        """
        self._kind = Self.KIND_NUMBER
        self._storage = s

    @always_inline
    @implicit
    fn __init__(out self, s: String):
        """
        Initializes a new JSON value as a string.

        Args:
            s: The string value.
        """
        self._kind = Self.KIND_STRING
        self._storage = s

    @always_inline
    @implicit
    fn __init__(out self, a: List[JSONValue[origin]]):
        """
        Initializes a new JSON value as an array.

        Args:
            a: The list of JSON values.
        """
        self._kind = Self.KIND_ARRAY
        self._storage = JSONList(a)

    @always_inline
    @implicit
    fn __init__(out self, o: Dict[String, JSONValue[origin]]):
        """
        Initializes a new JSON value as an object.

        Args:
            o: The dictionary of string keys to JSON values.
        """
        self._kind = Self.KIND_OBJECT
        self._storage = JSONDict(o)

    @always_inline
    @implicit
    fn __init__(out self, o: JSONDict[origin]):
        """
        Initializes a new JSON value as an object.

        Args:
            o: The JSON dictionary.
        """
        self._kind = Self.KIND_OBJECT
        self._storage = o

    @always_inline
    fn __copyinit__(out self, other: Self):
        """
        Initializes a new JSON value as a copy of another.

        Args:
            other: The JSON value to copy.
        """
        self._kind = other._kind
        self._storage = other._storage

    @always_inline
    fn __moveinit__(out self, owned existing: Self):
        """
        Initializes a new JSON value by moving another.

        Args:
            existing: The JSON value to move.
        """
        self._kind = existing._kind
        self._storage = existing._storage^

    @no_inline
    fn write_to[W: Writer, //](self, mut writer: W):
        """
        Writes a string representation of the JSONValue to the given writer.

        Outputs the JSONValue in its JSON string representation according to the
        JSON standard (RFC 8259).

        Parameters:
            W: The type of the writer, conforming to the `Writer` trait.

        Args:
            writer: The writer instance to output the representation to.
        """
        var kind = self._kind
        if kind == Self.KIND_NULL:
            writer.write("null")
        elif kind == Self.KIND_BOOL:
            if self._storage[Bool]:
                writer.write("true")
            else:
                writer.write("false")
        elif kind == Self.KIND_STRING:
            self._write_string(writer, self._storage[String])
        elif kind == Self.KIND_NUMBER:
            writer.write(self._storage[Float64])
        elif kind == Self.KIND_ARRAY:
            writer.write(self._storage[JSONList[origin]])
        elif kind == Self.KIND_OBJECT:
            writer.write(self._storage[JSONDict[origin]])

    @no_inline
    fn _format_hex(self, value: Int, width: Int = 4) -> String:
        """
        Formats an integer as a hexadecimal string with a specified width.

        This helper function ensures consistent formatting of hex values for
        Unicode escape sequences in JSON strings.

        Args:
            value: The integer value to format as hex
            width: The minimum width of the result (0-padded)

        Returns:
            A string with the hexadecimal representation padded to the specified width
        """
        # Define character code constants for better readability
        alias DIGIT_0 = ord("0")
        alias CHAR_A = ord("a")

        var result = String()
        var remaining = value

        # Collect hex digits in reverse order
        var digits = InlineArray[UInt8, 16](fill=0)
        var count = 0

        while remaining > 0 or count < width:
            var digit = remaining % 16

            if digit < 10:
                digits[count] = UInt8(digit + DIGIT_0)
            else:
                digits[count] = UInt8(digit - 10 + CHAR_A)

            remaining //= 16
            count += 1

        # Reverse the digits to get the correct order
        for i in reversed(range(count)):
            result += chr(Int(digits[i]))

        return result

    @no_inline
    fn _write_string[W: Writer, //](self, mut writer: W, s: String):
        """
        Writes a properly escaped JSON string to the given writer.

        This helper method ensures that control characters, quotes, backslashes,
        and other special characters are properly escaped according to the JSON spec.

        Parameters:
            W: The type of the writer.

        Args:
            writer: The writer instance to output the representation to.
            s: The string to write.
        """
        # Define character code constants
        alias QUOTE = ord('"')
        alias BACKSLASH = ord("\\")
        alias SLASH = ord("/")
        alias BACKSPACE = ord("\b")
        alias FORMFEED = ord("\f")
        alias NEWLINE = ord("\n")
        alias CARRIAGE_RETURN = ord("\r")
        alias TAB = ord("\t")

        writer.write('"')

        # Fast path for empty string
        if len(s) == 0:
            writer.write('"')
            return

        # Process each character in the string
        for i in range(len(s)):
            var c = s[i]
            var code = ord(c)

            # Handle special characters that need escaping
            if code == QUOTE:
                writer.write('\\"')
            elif code == BACKSLASH:
                writer.write("\\\\")
            elif code == BACKSPACE:
                writer.write("\\b")
            elif code == FORMFEED:
                writer.write("\\f")
            elif code == NEWLINE:
                writer.write("\\n")
            elif code == CARRIAGE_RETURN:
                writer.write("\\r")
            elif code == TAB:
                writer.write("\\t")
            elif code < 32:
                # Escape control characters as \uXXXX with 4 hex digits
                writer.write("\\u", self._format_hex(code, 4))
            else:
                # Regular character, write as-is
                writer.write(c)

        writer.write('"')

    fn __str__(self) -> String:
        """
        Returns a string representation of the JSON value.

        Returns:
            A string containing the JSON representation.
        """
        return String.write(self)

    @always_inline
    fn __repr__(self) -> String:
        """
        Returns a string representation of the JSON value.

        Returns:
            A string containing the JSON representation.
        """
        return String(self)

    # Type checking methods - private

    @always_inline
    fn _is_null(self) -> Bool:
        """Check if this value is null."""
        return self._kind == Self.KIND_NULL

    @always_inline
    fn _is_boolean(self) -> Bool:
        """Check if this value is a boolean."""
        return self._kind == Self.KIND_BOOL

    @always_inline
    fn _is_number(self) -> Bool:
        """Check if this value is a number."""
        return self._kind == Self.KIND_NUMBER

    @always_inline
    fn _is_string(self) -> Bool:
        """Check if this value is a string."""
        return self._kind == Self.KIND_STRING

    @always_inline
    fn _is_array(self) -> Bool:
        """Check if this value is an array."""
        return self._kind == Self.KIND_ARRAY

    @always_inline
    fn _is_object(self) -> Bool:
        """Check if this value is an object."""
        return self._kind == Self.KIND_OBJECT

    # Value access methods

    fn _as_boolean(self) raises -> Bool:
        """Get the value as a boolean."""
        if not self._is_boolean():
            raise Error("JSON value is not a boolean")
        return self._storage[Bool]

    fn _as_number(self) raises -> Float64:
        """Get the value as a number."""
        if not self._is_number():
            raise Error("JSON value is not a number")
        return self._storage[Float64]

    fn _as_string(self) raises -> String:
        """Get the value as a string."""
        if not self._is_string():
            raise Error("JSON value is not a string")
        return self._storage[String]

    fn _as_array(self) raises -> JSONList[origin]:
        """Get the value as an array."""
        if not self._is_array():
            raise Error("JSON value is not an array")
        return self._storage[JSONList[origin]]

    fn _as_object(self) raises -> Dict[String, Self]:
        """Get the value as an object."""
        if not self._is_object():
            raise Error("JSON value is not an object")
        return self._storage[JSONDict[origin]]._storage


struct JSONParser[mut: Bool, //, origin: Origin[mut]]:
    """
    A parser for JSON text.

    This struct provides methods to parse JSON text into structured JSONValue
    objects. It works directly on the UTF-8 bytes of the source string.

    Parameters:
        mut: Whether the parser is mutable.
        origin: The origin of the parser's memory.
    """

    # We'll work over the raw UTF-8 bytes of the source.
    var _slice: StringSlice[origin]
    """
    The string slice containing the JSON text to be parsed.

    This field stores the UTF-8 encoded input text that the parser will process
    character by character to construct JSON values.
    """

    var _idx: Int
    """
    The current position in the string slice.

    This index tracks the parser's current position within the _slice field as
    it processes the JSON text. It's incremented as characters are consumed.
    """

    fn __init__(out self, span: StringSlice[origin]):
        """
        Initializes a new JSON parser with the given string slice.

        Args:
            span: The string slice to parse.
        """
        self._slice = span
        self._idx = 0

    @always_inline
    fn _h_as_input(self) -> Bool:
        """
        Checks if there is more input to parse.

        Optimized implementation using cached length for better performance.

        Returns:
            True if there is more input, False otherwise.
        """
        return self._idx < len(self._slice)

    @always_inline
    fn _peek(self) -> Byte:
        """
        Returns the current byte without advancing the parser.

        Returns:
            The current byte, or 0 if at the end of the input.
        """
        if self._idx >= len(self._slice):
            return 0
        return self._slice._slice[self._idx]

    @always_inline
    fn _peek_slice(self) -> StringSlice[origin=origin]:
        """
        Returns a string slice of the current byte without advancing the parser.

        Returns:
            A string slice containing the current byte, or an empty slice if at
            the end of the input.
        """
        if self._idx >= len(self._slice):
            return StringSlice[origin=origin]()
        return StringSlice(
            unsafe_from_utf8=self._slice._slice[self._idx : self._idx + 1]
        )

    @always_inline
    fn _next(mut self) -> Byte:
        """
        Returns the current byte and advances the parser.

        Returns:
            The current byte, or 0 if at the end of the input.
        """
        var c = self._peek()
        self._idx += 1
        return c

    fn _expect(mut self, pat: String) raises:
        """
        Expects the given pattern at the current position and advances past it.

        Args:
            pat: The pattern to expect.

        Raises:
            Error: If the pattern is not found at the current position.
        """
        var p = pat.as_bytes()
        var n = pat.byte_length()

        # Fast path - check if we have enough bytes left
        if self._idx + n > len(self._slice):
            raise Error("Unexpected end of input, expected '", pat, "'")

        for i in range(n):
            if self._next() != p[i]:
                raise Error(
                    "Expected ",
                    pat,
                    " but got ",
                    StringSlice(
                        unsafe_from_utf8=self._slice._slice[self._idx :]
                    ),
                )

    @always_inline
    fn _skip_whitespace(mut self):
        """Advances the parser past any whitespace characters."""
        alias simd_width = simdwidthof[DType.uint8]()

        alias SPACE = ord(" ")
        alias TAB = ord("\t")
        alias LF = ord("\n")
        alias CR = ord("\r")

        if self._idx >= len(self._slice):
            return

        # Quick check for non-whitespace to avoid entering loop (very common case)
        var c = self._slice._slice[self._idx]
        if c > SPACE:  # Most non-whitespace characters are > space (32)
            return

        # Special case: optimize for sequence of spaces (most common whitespace)
        if c == SPACE:  # Space
            # Find run of spaces using vectorized approach
            var end = self._idx + 1

            # Fast bulk processing using SIMD when possible
            if end + simd_width <= len(self._slice):
                while end + simd_width - 1 < len(self._slice):
                    # Create SIMD vector of current chunk
                    var chunk = SIMD[DType.uint8, simd_width]()

                    @parameter
                    for i in range(simd_width):
                        chunk[i] = self._slice._slice[end + i]

                    # Compare entire chunk with space character using SIMD
                    var is_space = chunk == SPACE

                    # Check if all elements are spaces
                    if all(is_space):
                        end += simd_width
                    else:
                        # Find first non-space using SIMD
                        for i in range(simd_width):
                            if not is_space[i]:
                                end += i
                                self._idx = end
                                return
                        break

            # Handle remaining bytes individually
            while end < len(self._slice) and self._slice._slice[end] == SPACE:
                end += 1

            self._idx = end
            return

        # General whitespace processing for mixed whitespace characters
        # Process all standard JSON whitespace: space, tab, CR, LF
        while self._idx + simd_width <= len(self._slice):
            # Load chunk into SIMD vector
            var chunk = SIMD[DType.uint8, simd_width]()

            @parameter
            for i in range(simd_width):
                chunk[i] = self._slice._slice[self._idx + i]

            # Check for whitespace characters using SIMD operations
            var space_mask = chunk == SPACE
            var tab_mask = chunk == TAB
            var lf_mask = chunk == LF
            var cr_mask = chunk == CR

            var ws_mask = space_mask | tab_mask | lf_mask | cr_mask

            # Find first non-whitespace character
            if all(ws_mask):
                # All characters are whitespace
                self._idx += simd_width
            else:
                # Find first non-whitespace
                for i in range(simd_width):
                    if not ws_mask[i]:
                        self._idx += i
                        return

        # Handle remaining bytes individually
        while self._idx < len(self._slice):
            c = self._slice._slice[self._idx]

            # Fast case for most common whitespace (space)
            if c == SPACE:  # Space
                self._idx += 1
                continue

            # Check for other whitespace characters: tab (9), LF (10), CR (13)
            if c <= CR and (c == TAB or c == LF or c == CR):
                self._idx += 1
                continue

            # Not whitespace, exit immediately
            break

    fn parse_value(mut self) raises -> JSONValue[origin]:
        """
        Parses a JSON value at the current position.

        Returns:
            The parsed JSON value.

        Raises:
            Error: If the JSON is invalid.
        """
        # Define character code constants for better readability and performance
        alias QUOTE = ord('"')
        alias OPEN_BRACE = ord("{")
        alias OPEN_BRACKET = ord("[")
        alias CHAR_T = ord("t")
        alias CHAR_F = ord("f")
        alias CHAR_N = ord("n")
        alias DIGIT_0 = ord("0")
        alias DIGIT_9 = ord("9")
        alias MINUS = ord("-")

        self._skip_whitespace()
        var c = self._peek()

        if c == QUOTE:
            return JSONValue[origin](self._parse_string())
        if c == OPEN_BRACE:
            return self._parse_object()
        if c == OPEN_BRACKET:
            return self._parse_array()
        if c == CHAR_T:
            self._expect("true")
            return True
        if c == CHAR_F:
            self._expect("false")
            return False
        if c == CHAR_N:
            self._expect("null")
            return JSONValue[origin]()
        elif (DIGIT_0 <= Int(c) <= DIGIT_9) or c == MINUS:
            # Number value (common in JSON)
            return self._parse_number()
        else:
            # Invalid value type - create an informative error
            var context: String
            if self._idx + 8 < len(self._slice):
                # Show a small preview of the invalid content
                context = (
                    String(
                        bytes=Span[Byte, origin](
                            ptr=self._slice.unsafe_ptr() + self._idx, length=8
                        )
                    )
                    + "..."
                )
            else:
                context = String(
                    bytes=Span[Byte, origin](
                        ptr=self._slice.unsafe_ptr() + self._idx,
                        length=len(self._slice) - self._idx,
                    )
                )

            raise Error("Invalid JSON value starting with: '", context, "'")

    fn _parse_string(mut self) raises -> String:
        """
        Parses a JSON string at the current position, handling Unicode escape sequences.

        Returns:
            The parsed string with all escape sequences properly processed.

        Raises:
            Error: If the string is invalid or contains invalid escape sequences.
        """
        if self._idx >= len(self._slice):
            raise Error("Unexpected end of input while parsing string")

        # Define character code constants for better readability and performance
        alias QUOTE = ord('"')
        alias BACKSLASH = ord("\\")
        alias SLASH = ord("/")
        alias B = ord("b")
        alias F = ord("f")
        alias N = ord("n")
        alias R = ord("r")
        alias T = ord("t")
        alias U = ord("u")  # Unicode escape
        alias DIGIT_0 = ord("0")
        alias DIGIT_9 = ord("9")
        alias CHAR_A_LOWER = ord("a")
        alias CHAR_F_LOWER = ord("f")
        alias CHAR_A_UPPER = ord("A")
        alias CHAR_F_UPPER = ord("F")

        # Skip opening quote
        self._idx += 1

        var start = self._idx
        var end = start
        var h_as_escapes = False

        # Fast scan for end of string or escape sequences
        while end < len(self._slice):
            var b = self._slice._slice[end]

            # Check for end of string (unescaped quote)
            if b == QUOTE:
                break

            # Check for escape sequence
            if b == BACKSLASH:
                h_as_escapes = True
                # Skip the backslash and check the next character
                end += 1
                if end >= len(self._slice):
                    raise Error("Unterminated string, unexpected end of input")

                # Special check for Unicode escape which needs 4 more characters
                var next_char = self._slice._slice[end]
                if next_char == U:
                    end += 5  # Skip 'u' and 4 hex digits
                    if end > len(self._slice):
                        raise Error("Incomplete Unicode escape sequence")
                else:
                    end += 1  # Skip one character for regular escapes
            else:
                # Check for unescaped control characters (must be escaped in JSON)
                if b < 0x20:  # ASCII control characters
                    raise Error("Invalid control character in string")
                # Regular character
                end += 1

        if end >= len(self._slice):
            raise Error("Unterminated string, expected closing '\"'")

        # Simple case - no escape sequences
        if not h_as_escapes:
            var result = String(
                bytes=Span[Byte, origin](
                    ptr=self._slice.unsafe_ptr() + start,
                    length=end - start,
                )
            )
            self._idx = end + 1  # skip the closing quote
            return result

        # For strings with escapes, we need to process them carefully
        var result = String()
        self._idx = start  # Reset position to start processing

        while self._idx < end:
            var c = self._next()

            if c == BACKSLASH:
                # Handle escape sequences
                if self._idx >= len(self._slice):
                    raise Error("Unexpected end of input in escape sequence")

                var escape_char = self._next()

                if escape_char == QUOTE:
                    result += '"'
                elif escape_char == BACKSLASH:
                    result += "\\"
                elif escape_char == SLASH:
                    result += "/"
                elif escape_char == B:
                    result += "\b"
                elif escape_char == F:
                    result += "\f"
                elif escape_char == N:
                    result += "\n"
                elif escape_char == R:
                    result += "\r"
                elif escape_char == T:
                    result += "\t"
                elif escape_char == U:
                    # Handle Unicode escape sequences \uXXXX
                    if self._idx + 4 > len(self._slice):
                        raise Error("Incomplete Unicode escape sequence")

                    # Read exactly 4 hex digits and convert to hexadecimal
                    var code_point = Int(0)

                    for j in range(4):
                        var hex_digit = Int(self._slice._slice[self._idx + j])

                        # Manually convert hex digit to value
                        var digit_value: Int
                        if DIGIT_0 <= hex_digit <= DIGIT_9:
                            digit_value = hex_digit - DIGIT_0
                        elif CHAR_A_LOWER <= hex_digit <= CHAR_F_LOWER:
                            digit_value = hex_digit - CHAR_A_LOWER + 10
                        elif CHAR_A_UPPER <= hex_digit <= CHAR_F_UPPER:
                            digit_value = hex_digit - CHAR_A_UPPER + 10
                        else:
                            raise Error(
                                "Invalid Unicode escape sequence: expected hex"
                                " digit"
                            )

                        # Build the code point value
                        code_point = (code_point << 4) | digit_value

                    self._idx += 4  # Skip the 4 hex digits

                    # Check for surrogate pairs
                    if code_point >= 0xD800 and code_point <= 0xDBFF:
                        # This is a high surrogate, expect a low surrogate next
                        if (
                            self._idx + 2 > end
                            or self._slice._slice[self._idx] != BACKSLASH
                            or self._slice._slice[self._idx + 1] != U
                        ):
                            raise Error(
                                "Invalid surrogate pair: expected \\u after"
                                " high surrogate"
                            )

                        self._idx += 2  # Skip \u

                        if self._idx + 4 > len(self._slice):
                            raise Error(
                                "Incomplete Unicode escape sequence in"
                                " surrogate pair"
                            )

                        # Read the low surrogate using the same method
                        var low_surrogate = 0
                        for j in range(4):
                            var hex_digit = Int(
                                self._slice._slice[self._idx + j]
                            )

                            # Manually convert hex digit to value
                            var digit_value: Int
                            if DIGIT_0 <= hex_digit <= DIGIT_9:
                                digit_value = hex_digit - DIGIT_0
                            elif CHAR_A_LOWER <= hex_digit <= CHAR_F_LOWER:
                                digit_value = hex_digit - CHAR_A_LOWER + 10
                            elif CHAR_A_UPPER <= hex_digit <= CHAR_F_UPPER:
                                digit_value = hex_digit - CHAR_A_UPPER + 10
                            else:
                                raise Error(
                                    "Invalid Unicode escape sequence in low"
                                    " surrogate: expected hex digit"
                                )

                            # Build the low surrogate code point value
                            low_surrogate = (low_surrogate << 4) | digit_value

                        self._idx += 4  # Skip the 4 hex digits

                        if not (0xDC00 <= low_surrogate <= 0xDFFF):
                            raise Error(
                                "Invalid surrogate pair: low surrogate out of"
                                " range"
                            )

                        # Compute the actual code point from surrogate pair
                        code_point = (
                            0x10000
                            + ((code_point - 0xD800) << 10)
                            + (low_surrogate - 0xDC00)
                        )

                    # Convert code point to UTF-8 character
                    if code_point < 0:
                        raise Error("Invalid Unicode code point")

                    # Add the Unicode character to the result string
                    result += chr(code_point)
                else:
                    raise Error(
                        "Invalid escape sequence: '\\", escape_char, "'"
                    )
            else:
                # Regular character - append as-is
                result += String(c)

        self._idx = end + 1  # skip the closing quote
        return result

    fn _parse_number(mut self) raises -> Float64:
        """
        Parses a JSON number at the current position.

        Returns:
            The parsed number as a Float64.

        Raises:
            Error: If the number is invalid.
        """
        var start = self._idx

        # Define character code aliases for better readability
        alias MINUS = ord("-")
        alias DOT = ord(".")
        alias LOWER_E = ord("e")
        alias UPPER_E = ord("E")
        alias PLUS = ord("+")

        # optional leading '-'
        if self._peek() == MINUS:
            self._idx += 1

        # integer part
        while self._peek_slice().is_ascii_digit():
            self._idx += 1

        # optional fraction
        if self._peek() == DOT:
            self._idx += 1
            while self._peek_slice().is_ascii_digit():
                self._idx += 1

        # optional exponent
        if self._peek() == LOWER_E or self._peek() == UPPER_E:
            self._idx += 1
            if self._peek() == PLUS or self._peek() == MINUS:
                self._idx += 1
            while self._peek_slice().is_ascii_digit():
                self._idx += 1

        var sl = StringSlice[__origin_of(self._slice)](
            ptr=self._slice.unsafe_ptr() + start,
            length=self._idx - start,
        )

        return atof(sl)

    fn _parse_array(mut self) raises -> List[JSONValue[origin]]:
        """
        Parses a JSON array at the current position.

        Returns:
            A list of JSON values.

        Raises:
            Error: If the array is invalid.
        """
        # Define character code aliases for better readability
        alias COMMA = ord(",")
        alias CLOSE_BRACKET = ord("]")

        _ = self._next()  # skip '['
        self._skip_whitespace()
        var arr = List[JSONValue[origin]]()

        # Fast path for empty array
        if self._peek() == CLOSE_BRACKET:
            self._idx += 1
            return arr

        while True:
            # Parse array element
            self._skip_whitespace()
            arr.append(self.parse_value())
            self._skip_whitespace()

            # Check for comma or closing bracket
            if self._idx >= len(self._slice):
                raise Error("Unexpected end of input, expected ',' or ']'")

            var c = self._next()
            if c == COMMA:
                continue
            elif c == CLOSE_BRACKET:
                break
            else:
                # Create a string from a single character for the error message
                var char = StringSlice[origin](
                    ptr=self._slice.unsafe_ptr() + (self._idx - 1), length=1
                )
                raise Error("Expected ',' or ']', got '", char, "'")

        return arr

    fn _parse_object(mut self) raises -> JSONDict[origin]:
        """
        Parses a JSON object at the current position.

        Returns:
            A JSON dictionary.

        Raises:
            Error: If the object is invalid.
        """
        # Define character code aliases for better readability
        alias QUOTE = ord('"')
        alias COLON = ord(":")
        alias COMMA = ord(",")
        alias CLOSE_BRACE = ord("}")

        _ = self._next()  # skip '{'
        self._skip_whitespace()
        var obj = Dict[String, JSONValue[origin]]()

        # Fast path for empty object
        if self._peek() == CLOSE_BRACE:
            self._idx += 1
            return JSONDict[origin](obj)

        while True:
            self._skip_whitespace()

            # Parse key (must be a string)
            if self._peek() != QUOTE:
                # Create a string from a single character for the error message
                var char = StringSlice[origin](
                    ptr=self._slice.unsafe_ptr() + self._idx, length=1
                )
                raise Error("Expected string key in object, got '", char, "'")

            var key = self._parse_string()
            self._skip_whitespace()

            # Check for colon separator
            if self._idx >= len(self._slice) or self._next() != COLON:
                raise Error("Expected ':' after object key")

            # Parse value
            self._skip_whitespace()
            var val = self.parse_value()
            obj[key] = val
            self._skip_whitespace()

            # Check for comma or closing brace
            if self._idx >= len(self._slice):
                raise Error("Unexpected end of input, expected ',' or '}'")

            var c = self._next()
            if c == COMMA:
                continue
            elif c == CLOSE_BRACE:
                break
            else:
                # Create a string from a single character for the error message
                var char = StringSlice[origin](
                    ptr=self._slice.unsafe_ptr() + (self._idx - 1), length=1
                )
                raise Error("Expected ',' or '}', got '", char, "'")

        return JSONDict[origin](obj)


# ===-----------------------------------------------------------------------===#
# loads
# ===-----------------------------------------------------------------------===#


fn loads(text: String) raises -> JSONValue[__origin_of(text)]:
    """
    Parses the given JSON text into a JSONValue.

    This function takes a JSON string and returns a structured representation
    as a JSONValue object.

    Args:
        text: The JSON text to parse.

    Returns:
        A JSONValue representing the parsed JSON.

    Raises:
        Error: If the JSON is invalid.
    """
    if len(text) == 0:
        raise Error("Empty JSON input")

    var p = JSONParser(text)
    p._skip_whitespace()
    if not p._h_as_input():
        raise Error("Empty JSON input (only whitespace)")

    # Parse the value with optimized error handling
    try:
        var result = p.parse_value()

        # Strict validation - ensure no trailing content
        p._skip_whitespace()

        if p._h_as_input():
            # Performance optimization: only build error string if needed
            var remaining = len(p._slice) - p._idx
            var show_len = min(remaining, 16)  # Limit to reduce allocation size

            var trailing = String(
                bytes=Span[Byte, __origin_of(text)](
                    ptr=p._slice.unsafe_ptr() + p._idx, length=show_len
                )
            )

            if show_len < remaining:
                trailing += "..."

            raise Error(
                "Unexpected trailing content after JSON value: '", trailing, "'"
            )

        return result
    except e:
        # Re-throw but with better context if needed
        if "at position" in String(e):
            # Error already has position info
            raise e
        else:
            # Add position info to improve debugging
            raise Error(e, " at position ", p._idx)


# ===-----------------------------------------------------------------------===#
# dumps
# ===-----------------------------------------------------------------------===#


fn dumps[
    mut: Bool, //, origin: Origin[mut]
](value: JSONValue[origin]) -> String:
    """
    Serializes a JSONValue to a JSON string.

    This function takes a JSONValue and returns its string representation
    according to the JSON specification (RFC 8259).

    Parameters:
        mut: Whether the value is mutable.
        origin: The origin of the value.

    Args:
        value: The JSONValue to serialize.

    Returns:
        A string containing the JSON representation of the value.
    """
    return String(value)
