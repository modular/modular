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

from utils import Variant
from collections import Dict, List
from utils import Writer
from collections.optional import _NoneType
from memory import Pointer
from sys import simdwidthof


@value
struct JSONList[mut: Bool, //, origin: Origin[mut]](
    Representable, Copyable, Movable
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


@value
struct JSONDict[mut: Bool, //, origin: Origin[mut]](
    Representable, Copyable, Movable
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
    Representable, Copyable, Movable
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
            writer.write('"', self._storage[String], '"')
        elif kind == Self.KIND_NUMBER:
            writer.write(self._storage[Float64])
        elif kind == Self.KIND_ARRAY:
            writer.write(self._storage[JSONList[origin]])
        elif kind == Self.KIND_OBJECT:
            writer.write(self._storage[JSONDict[origin]])

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
    fn _has_input(self) -> Bool:
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
                    "Expected "
                    + pat
                    + " but got "
                    + StringSlice(
                        unsafe_from_utf8=self._slice._slice[self._idx :]
                    )
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
        self._skip_whitespace()
        var c = self._peek()

        if c == ord('"'):
            return JSONValue[origin](self._parse_string())
        if c == ord("{"):
            return self._parse_object()
        if c == ord("["):
            return self._parse_array()
        if c == ord("t"):
            self._expect("true")
            return True
        if c == ord("f"):
            self._expect("false")
            return False
        if c == ord("n"):
            self._expect("null")
            return JSONValue[origin]()
        elif (ord("0") <= Int(c) <= ord("9")) or c == ord("-"):
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

        # Parser should never reach this point due to the else clause above
        # but compiler may expect a return value
        return JSONValue[origin]()

    fn _parse_string(mut self) raises -> String:
        """
        Parses a JSON string at the current position.

        Returns:
            The parsed string.

        Raises:
            Error: If the string is invalid.
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

        # Skip opening quote
        self._idx += 1

        var start = self._idx
        var end = start
        var has_escapes = False

        # Fast scan for end of string or escape sequences
        while end < len(self._slice):
            var b = self._slice._slice[end]

            # Check for end of string (unescaped quote)
            if b == QUOTE:
                break

            # Check for escape sequence
            if b == BACKSLASH:
                has_escapes = True
                # Skip the backslash and the following character
                end += 2
                if end > len(self._slice):
                    raise Error("Unterminated string, unexpected end of input")
            else:
                # Regular character
                end += 1

        if end >= len(self._slice):
            raise Error("Unterminated string, expected closing '\"'")

        # Simple case - no escape sequences
        if not has_escapes:
            var result = String(
                bytes=Span[Byte, origin](
                    ptr=self._slice.unsafe_ptr() + start,
                    length=end - start,
                )
            )
            self._idx = end + 1  # skip the closing quote
            return result

        # For strings with escapes, we need to process them character by character
        var result = String()
        var i = start
        var escaped = False

        while i < end:
            var c = self._slice._slice[i]
            i += 1

            if escaped:
                # Handle escape sequences according to JSON spec
                if c == QUOTE or c == BACKSLASH or c == SLASH:
                    result += String(c)
                elif c == B:
                    result += "\b"
                elif c == F:
                    result += "\f"
                elif c == N:
                    result += "\n"
                elif c == R:
                    result += "\r"
                elif c == T:
                    result += "\t"
                else:
                    # Invalid escape sequence
                    raise Error(
                        "Invalid escape sequence: '\\" + String(c) + "'"
                    )
                escaped = False
            elif c == BACKSLASH:
                escaped = True
            else:
                # Regular character
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

        # optional leading '-'
        if self._peek() == ord("-"):
            self._idx += 1

        # integer part
        while self._peek_slice().is_ascii_digit():
            self._idx += 1

        # optional fraction
        if self._peek() == ord("."):
            self._idx += 1
            while self._peek_slice().is_ascii_digit():
                self._idx += 1

        # optional exponent
        if self._peek() == ord("e") or self._peek() == ord("E"):
            self._idx += 1
            if self._peek() == ord("+") or self._peek() == ord("-"):
                self._idx += 1
            while self._peek_slice().is_ascii_digit():
                self._idx += 1

        var sl = Span[Byte, __origin_of(self._slice)](
            ptr=self._slice.unsafe_ptr() + start,
            length=self._idx - start,
        )

        return atof(String(bytes=sl))

    fn _parse_array(mut self) raises -> List[JSONValue[origin]]:
        """
        Parses a JSON array at the current position.

        Returns:
            A list of JSON values.

        Raises:
            Error: If the array is invalid.
        """
        _ = self._next()  # skip '['
        self._skip_whitespace()
        var arr = List[JSONValue[origin]]()

        # Fast path for empty array
        if self._peek() == ord("]"):
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
            if c == ord(","):
                continue
            elif c == ord("]"):
                break
            else:
                # Create a string from a single character for the error message
                var char_bytes = Span[Byte, origin](
                    ptr=self._slice.unsafe_ptr() + (self._idx - 1), length=1
                )
                raise Error(
                    "Expected ',' or ']', got '", String(bytes=char_bytes), "'"
                )

        return arr

    fn _parse_object(mut self) raises -> JSONDict[origin]:
        """
        Parses a JSON object at the current position.

        Returns:
            A JSON dictionary.

        Raises:
            Error: If the object is invalid.
        """
        _ = self._next()  # skip '{'
        self._skip_whitespace()
        var obj = Dict[String, JSONValue[origin]]()

        # Fast path for empty object
        if self._peek() == ord("}"):
            self._idx += 1
            return JSONDict[origin](obj)

        while True:
            self._skip_whitespace()

            # Parse key (must be a string)
            if self._peek() != ord('"'):
                # Create a string from a single character for the error message
                var char_bytes = Span[Byte, origin](
                    ptr=self._slice.unsafe_ptr() + self._idx, length=1
                )
                raise Error(
                    "Expected string key in object, got '",
                    String(bytes=char_bytes),
                    "'",
                )

            var key = self._parse_string()
            self._skip_whitespace()

            # Check for colon separator
            if self._idx >= len(self._slice) or self._next() != ord(":"):
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
            if c == ord(","):
                continue
            elif c == ord("}"):
                break
            else:
                # Create a string from a single character for the error message
                var char_bytes = Span[Byte, origin](
                    ptr=self._slice.unsafe_ptr() + (self._idx - 1), length=1
                )
                raise Error(
                    "Expected ',' or '}', got '", String(bytes=char_bytes), "'"
                )

        return JSONDict[origin](obj)


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
    if not p._has_input():
        raise Error("Empty JSON input (only whitespace)")

    # Parse the value with optimized error handling
    try:
        var result = p.parse_value()

        # Strict validation - ensure no trailing content
        p._skip_whitespace()

        if p._has_input():
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
