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
"""The `StringSlice` type implementation for efficient string operations.

This module provides the `StringSlice` type, which is a lightweight view into string data
that enables zero-copy string operations. `StringSlice` is designed for high-performance
string manipulation while maintaining memory safety and UTF-8 awareness.

Key Features:
- Zero-copy string views and operations
- UTF-8 aware string slicing and manipulation
- Memory-safe string operations
- Efficient string comparison and search
- Unicode-aware string operations

The `StringSlice` type is particularly useful for:
- High-performance string operations without copying
- Efficient string parsing and tokenization
- Memory-safe string manipulation
- Unicode-aware text processing

Example:
    ```mojo
    from collections.string import StringSlice

    # Create a string slice
    var text = StringSlice("Hello, 世界")

    # Zero-copy slicing
    var hello = text[0:5] # Hello

    # Unicode-aware operations
    var world = text[7:13]  # "世界"

    # String comparison
    if text.startswith("Hello"):
        print("Found greeting")
    ```
"""

from collections import List, Optional
from collections.string.format import _CurlyEntryFormattable, _FormatCurlyEntry
from collections.string._utf8 import (
    _is_valid_utf8,
    _count_utf8_continuation_bytes,
    _utf8_first_byte_sequence_length,
    _utf8_byte_type,
    _is_newline_char_utf8,
)
from collections.string._unicode import (
    is_lowercase,
    is_uppercase,
    to_lowercase,
    to_uppercase,
)
from hashlib._hasher import _HashableWithHasher, _Hasher
from os import PathLike, abort
from sys import bitwidthof, simdwidthof, is_compile_time
from sys.ffi import c_char
from sys.intrinsics import likely, unlikely

from math import align_down
from bit import count_leading_zeros, count_trailing_zeros
from memory import Span, UnsafePointer, memcmp, memcpy, pack_bits
from memory.memory import _memcmp_impl_unconstrained
from utils.write import _WriteBufferStack

alias StaticString = StringSlice[StaticConstantOrigin]
"""An immutable static string slice."""


@value
struct CodepointSliceIter[
    mut: Bool, //,
    origin: Origin[mut],
    forward: Bool = True,
]:
    """Iterator for `StringSlice` over substring slices containing a single
    Unicode codepoint.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.
        forward: The iteration direction. `False` is backwards.

    The `forward` parameter only controls the behavior of the `__next__()`
    method used for normal iteration. Calls to `next()` will always take an
    element from the front of the iterator, and calls to `next_back()` will
    always take an element from the end.
    """

    var _slice: StringSlice[origin]

    # Note:
    #   Marked private since `StringSlice.codepoints()` is the intended public
    #   way to construct this type.
    @doc_private
    fn __init__(out self, str_slice: StringSlice[origin]):
        self._slice = str_slice

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @doc_private
    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> StringSlice[origin]:
        """Get the next codepoint in the underlying string slice.

        This returns the next single-codepoint substring slice encoded in the
        underlying string, and advances the iterator state.

        If `forward` is set to `False`, this will return the next codepoint
        from the end of the string.

        This function will abort if this iterator has been exhausted.

        Returns:
            The next character in the string.
        """

        @parameter
        if forward:
            return self.next().value()
        else:
            return self.next_back().value()

    @always_inline
    fn __has_next__(self) -> Bool:
        """Returns True if there are still elements in this iterator.

        Returns:
            A boolean indicating if there are still elements in this iterator.
        """
        # NOTE:
        #   This intentionally check if the length _in bytes_ is greater
        #   than zero, because checking the codepoint length requires a linear
        #   scan of the string, which is needlessly expensive for this purpose.
        return len(self._slice) > 0

    @always_inline
    fn __len__(self) -> Int:
        """Returns the remaining length of this iterator in `Codepoint`s.

        The value returned from this method indicates the number of subsequent
        calls to `next()` that will return a value.

        Returns:
            Number of codepoints remaining in this iterator.
        """
        return self._slice.char_length()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn peek_next(self) -> Optional[StringSlice[origin]]:
        """Check what the next single-codepoint slice in this iterator is,
        without advancing the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The next codepoint slice in the underlying string, or None if the
            string is empty.

        # Examples

        `peek_next()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from collections.string import StringSlice, Codepoint
        from testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoint_slices()

        assert_equal(iter.peek_next().value(), "1")
        assert_equal(iter.peek_next().value(), "1")
        assert_equal(iter.peek_next().value(), "1")

        # A call to `next()` return the same value as `peek_next()` had,
        # but also advance the iterator.
        assert_equal(iter.next().value(), "1")

        # Later `peek_next()` calls will return the _new_ next character:
        assert_equal(iter.peek_next().value(), "2")
        ```
        .
        """
        if len(self._slice) > 0:
            # SAFETY: Will not read out of bounds because `_slice` is guaranteed
            #   to contain valid UTF-8.
            var curr_ptr = self._slice.unsafe_ptr()
            var byte_len = _utf8_first_byte_sequence_length(curr_ptr[])
            return StringSlice[origin](ptr=curr_ptr, length=byte_len)
        else:
            return None

    fn peek_back(mut self) -> Optional[StringSlice[origin]]:
        """Check what the last single-codepoint slice in this iterator is,
        without advancing the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The last codepoint slice in the underlying string, or None if the
            string is empty.

        # Examples

        `peek_back()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from collections.string import StringSlice, Codepoint
        from testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoint_slices()

        # Repeated calls to `peek_back()` return the same value.
        assert_equal(iter.peek_back().value(), "3")
        assert_equal(iter.peek_back().value(), "3")
        assert_equal(iter.peek_back().value(), "3")

        # A call to `next_back()` return the same value as `peek_back()` had,
        # but also advance the iterator.
        assert_equal(iter.next_back().value(), "3")

        # Later `peek_back()` calls will return the _new_ next character:
        assert_equal(iter.peek_back().value(), "2")
        ```
        .
        """
        if len(self._slice) > 0:
            var byte_len = 1
            var back_ptr = self._slice.unsafe_ptr() + len(self._slice) - 1
            # SAFETY:
            #   Guaranteed not to go out of bounds because UTF-8
            #   guarantees there is always a "start" byte eventually before any
            #   continuation bytes.
            while _utf8_byte_type(back_ptr[]) == 1:
                byte_len += 1
                back_ptr -= 1

            return StringSlice[origin](ptr=back_ptr, length=byte_len)
        else:
            return None

    fn next(mut self) -> Optional[StringSlice[origin]]:
        """Get the next codepoint slice in the underlying string slice, or None
        if the iterator is empty.

        This returns the next single-codepoint substring encoded in the
        underlying string, and advances the iterator state.

        Returns:
            A character if the string is not empty, otherwise None.
        """
        var result: Optional[StringSlice[origin]] = self.peek_next()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var slice_len = len(result.unsafe_value())
            # Advance the pointer in _slice.
            self._slice._slice._data += slice_len
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= slice_len

        return result

    fn next_back(mut self) -> Optional[StringSlice[origin]]:
        """Get the last single-codepoint slice in this iterator is, or None
        if the iterator is empty.

        This returns the last codepoint slice in this iterator, and advances
        the iterator state.

        Returns:
            The last codepoint slice in the underlying string, or None if the
            string is empty.
        """
        var result: Optional[StringSlice[origin]] = self.peek_back()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var slice_len = len(result.unsafe_value())
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= slice_len

        return result


@value
struct CodepointsIter[mut: Bool, //, origin: Origin[mut]]:
    """Iterator over the `Codepoint`s in a string slice, constructed by
    `StringSlice.codepoints()`.

    Parameters:
        mut: Mutability of the underlying string data.
        origin: Origin of the underlying string data.
    """

    var _slice: StringSlice[origin]
    """String slice containing the bytes that have not been read yet.

    When this iterator advances, the pointer in `_slice` is advanced by the
    byte length of each read character, and the slice length is decremented by
    the same amount.
    """

    # Note:
    #   Marked private since `StringSlice.codepoints()` is the intended public
    #   way to construct this type.
    @doc_private
    fn __init__(out self, str_slice: StringSlice[origin]):
        self._slice = str_slice

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @doc_private
    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Codepoint:
        """Get the next codepoint in the underlying string slice.

        This returns the next `Codepoint` encoded in the underlying string, and
        advances the iterator state.

        This function will abort if this iterator has been exhausted.

        Returns:
            The next character in the string.
        """

        return self.next().value()

    @always_inline
    fn __has_next__(self) -> Bool:
        """Returns True if there are still elements in this iterator.

        Returns:
            A boolean indicating if there are still elements in this iterator.
        """
        return Bool(self.peek_next())

    @always_inline
    fn __len__(self) -> Int:
        """Returns the remaining length of this iterator in `Codepoint`s.

        The value returned from this method indicates the number of subsequent
        calls to `next()` that will return a value.

        Returns:
            Number of codepoints remaining in this iterator.
        """
        return self._slice.char_length()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn peek_next(self) -> Optional[Codepoint]:
        """Check what the next codepoint in this iterator is, without advancing
        the iterator state.

        Repeated calls to this method will return the same value.

        Returns:
            The next character in the underlying string, or None if the string
            is empty.

        # Examples

        `peek_next()` does not advance the iterator, so repeated calls will
        return the same value:

        ```mojo
        from collections.string import StringSlice, Codepoint
        from testing import assert_equal

        var input = StringSlice("123")
        var iter = input.codepoints()

        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))
        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))
        assert_equal(iter.peek_next().value(), Codepoint.ord("1"))

        # A call to `next()` return the same value as `peek_next()` had,
        # but also advance the iterator.
        assert_equal(iter.next().value(), Codepoint.ord("1"))

        # Later `peek_next()` calls will return the _new_ next character:
        assert_equal(iter.peek_next().value(), Codepoint.ord("2"))
        ```
        .
        """
        if len(self._slice) > 0:
            # SAFETY: Will not read out of bounds because `_slice` is guaranteed
            #   to contain valid UTF-8.
            codepoint, _ = Codepoint.unsafe_decode_utf8_codepoint(
                self._slice.unsafe_ptr()
            )
            return codepoint
        else:
            return None

    fn next(mut self) -> Optional[Codepoint]:
        """Get the next codepoint in the underlying string slice, or None if
        the iterator is empty.

        This returns the next `Codepoint` encoded in the underlying string, and
        advances the iterator state.

        Returns:
            A character if the string is not empty, otherwise None.
        """
        var result: Optional[Codepoint] = self.peek_next()

        if result:
            # SAFETY: We just checked that `result` holds a value
            var char_len = result.unsafe_value().utf8_byte_length()
            # Advance the pointer in _slice.
            self._slice._slice._data += char_len
            # Decrement the byte-length of _slice.
            self._slice._slice._len -= char_len

        return result


@value
@register_passable("trivial")
struct StringSlice[mut: Bool, //, origin: Origin[mut]](
    Stringable,
    Representable,
    Sized,
    Writable,
    CollectionElement,
    CollectionElementNew,
    EqualityComparable,
    Hashable,
    PathLike,
    FloatableRaising,
    Boolable,
    IntableRaising,
    RepresentableCollectionElement,
    EqualityComparableCollectionElement,
    _CurlyEntryFormattable,
):
    """A non-owning view to encoded string data.

    This type is guaranteed to have the same ABI (size, alignment, and field
    layout) as the `llvm::StringRef` type.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.

    Notes:
        TODO: The underlying string data is guaranteed to be encoded using
        UTF-8.
    """

    var _slice: Span[Byte, origin]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @doc_private
    @always_inline
    @implicit
    fn __init__(out self: StaticString, _kgen: __mlir_type.`!kgen.string`):
        # FIXME(MSTDL-160): !kgen.string's are not guaranteed to be UTF-8
        # encoded, they can be arbitrary binary data.
        var length: Int = __mlir_op.`pop.string.size`(_kgen)
        var ptr = UnsafePointer(__mlir_op.`pop.string.address`(_kgen)).bitcast[
            Byte
        ]()
        self._slice = Span[Byte, StaticConstantOrigin](ptr=ptr, length=length)

    @always_inline
    @implicit
    fn __init__(out self: StaticString, lit: StringLiteral):
        """Construct a new `StringSlice` from a `StringLiteral`.

        Args:
            lit: The literal to construct this `StringSlice` from.
        """
        # Since a StringLiteral has static origin, it will outlive
        # whatever arbitrary `origin` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StringLiteral is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        self = StaticString(lit.value)

    @always_inline("builtin")
    fn __init__(out self, *, unsafe_from_utf8: Span[Byte, origin]):
        """Construct a new `StringSlice` from a sequence of UTF-8 encoded bytes.

        Args:
            unsafe_from_utf8: A `Span[Byte]` encoded in UTF-8.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.
        """
        # FIXME(#3706): can't run at compile time
        # TODO(MOCO-1525):
        #   Support skipping UTF-8 during comptime evaluations, or support
        #   the necessary SIMD intrinsics to allow this to evaluate at compile
        #   time.
        # debug_assert(
        #     _is_valid_utf8(value.as_bytes()), "value is not valid utf8"
        # )
        self._slice = unsafe_from_utf8

    fn __init__(out self, *, unsafe_from_utf8_ptr: UnsafePointer[Byte]):
        """Construct a new StringSlice from a `UnsafePointer[Byte]` pointing to null-terminated UTF-8
        encoded bytes.

        Args:
            unsafe_from_utf8_ptr: An `UnsafePointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to data that is valid for
                `origin`.
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """

        var count = _unsafe_strlen(unsafe_from_utf8_ptr)

        var byte_slice = Span[Byte, origin](
            ptr=unsafe_from_utf8_ptr,
            length=count,
        )

        self = Self(unsafe_from_utf8=byte_slice)

    fn __init__(out self, *, unsafe_from_utf8_cstr_ptr: UnsafePointer[c_char]):
        """Construct a new StringSlice from a `UnsafePointer[c_char]` pointing to null-terminated UTF-8
        encoded bytes.

        Args:
            unsafe_from_utf8_cstr_ptr: An `UnsafePointer[c_char]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to data that is valid for
                `origin`.
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        var ptr = unsafe_from_utf8_cstr_ptr.bitcast[Byte]()
        self = Self(unsafe_from_utf8_ptr=ptr)

    @always_inline("builtin")
    fn __init__(out self, *, ptr: UnsafePointer[Byte], length: UInt):
        """Construct a `StringSlice` from a pointer to a sequence of UTF-8
        encoded bytes and a length.

        Args:
            ptr: A pointer to a sequence of bytes encoded in UTF-8.
            length: The number of bytes of encoded data.

        Safety:
            - `ptr` MUST point to at least `length` bytes of valid UTF-8 encoded
                data.
            - `ptr` must point to data that is live for the duration of
                `origin`.
        """
        self = Self(unsafe_from_utf8=Span[Byte, origin](ptr=ptr, length=length))

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a deep copy of the provided `StringSlice`.

        Returns:
            A copy of the value.
        """
        return Self(unsafe_from_utf8=self._slice)

    @implicit
    fn __init__[
        O: ImmutableOrigin, //
    ](out self: StringSlice[O], ref [O]value: String):
        """Construct an immutable StringSlice.

        Parameters:
            O: The immutable origin.

        Args:
            value: The string value.
        """
        self = StringSlice[O](unsafe_from_utf8=value.as_bytes())

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    # TODO: Change to `__init__(out self, *, from_utf8: Span[..])` once ambiguity
    #   with existing `unsafe_from_utf8` overload is fixed. Would require
    #   signature comparision to take into account required named arguments.
    @staticmethod
    fn from_utf8(from_utf8: Span[Byte, origin]) raises -> StringSlice[origin]:
        """Construct a new `StringSlice` from a buffer containing UTF-8 encoded
        data.

        Args:
            from_utf8: A span of bytes containing UTF-8 encoded data.

        Returns:
            A new validated `StringSlice` pointing to the provided buffer.

        Raises:
            An exception is raised if the provided buffer byte values do not
            form valid UTF-8 encoded codepoints.
        """
        if not _is_valid_utf8(from_utf8):
            raise Error("StringSlice: buffer is not valid UTF-8")

        return StringSlice[origin](unsafe_from_utf8=from_utf8)

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    @no_inline
    fn __str__(self) -> String:
        """Convert this StringSlice to a String.

        Returns:
            A new String.

        Notes:
            This will allocate a new string that copies the string contents from
            the provided string slice.
        """
        var buffer = String._buffer_type(capacity=self.byte_length() + 1)
        buffer.extend(self.as_bytes())
        buffer.append(0)
        return String(buffer=buffer^)

    fn __repr__(self) -> String:
        """Return a Mojo-compatible representation of this string slice.

        Returns:
            Representation of this string slice as a Mojo string literal input
            form syntax.
        """
        var result = String()
        var use_dquote = False
        for s in self.codepoint_slices():
            use_dquote = use_dquote or (s == "'")

            if s == "\\":
                result += r"\\"
            elif s == "\t":
                result += r"\t"
            elif s == "\n":
                result += r"\n"
            elif s == "\r":
                result += r"\r"
            else:
                var codepoint = Codepoint.ord(s)
                if codepoint.is_ascii_printable():
                    result += s
                elif codepoint.to_u32() < 0x10:
                    result += hex(codepoint, prefix=r"\x0")
                elif codepoint.to_u32() < 0x20 or codepoint.to_u32() == 0x7F:
                    result += hex(codepoint, prefix=r"\x")
                else:  # multi-byte character
                    result += s

        if use_dquote:
            return '"' + result + '"'
        else:
            return "'" + result + "'"

    @always_inline
    fn __len__(self) -> Int:
        """Get the string length in bytes.

        This function returns the number of bytes in the underlying UTF-8
        representation of the string.

        To get the number of Unicode codepoints in a string, use
        `len(str.codepoints())`.

        Returns:
            The string length in bytes.

        # Examples

        Query the length of a string, in bytes and Unicode codepoints:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        var s = StringSlice("ನಮಸ್ಕಾರ")

        assert_equal(len(s), 21)
        assert_equal(len(s.codepoints()), 7)
        ```

        Strings containing only ASCII characters have the same byte and
        Unicode codepoint length:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        var s = StringSlice("abc")

        assert_equal(len(s), 3)
        assert_equal(len(s.codepoints()), 3)
        ```
        .
        """
        return self.byte_length()

    fn write_to[W: Writer](self, mut writer: W):
        """Formats this string slice to the provided `Writer`.

        Parameters:
            W: A type conforming to the `Writable` trait.

        Args:
            writer: The object to write to.
        """
        writer.write_bytes(self.as_bytes())

    fn __bool__(self) -> Bool:
        """Check if a string slice is non-empty.

        Returns:
           True if a string slice is non-empty, False otherwise.
        """
        return len(self._slice) > 0

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self._slice._data, self._slice._len)

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_bytes(self.unsafe_ptr(), len(self))

    fn __fspath__(self) -> String:
        """Return the file system path representation of this string.

        Returns:
          The file system path representation as a string.
        """
        return self.__str__()

    @always_inline
    fn __getitem__(self, span: Slice) raises -> Self:
        """Gets the sequence of characters at the specified positions.

        Args:
            span: A slice that specifies positions of the new substring.

        Returns:
            A new StringSlice containing the substring at the specified positions.

        Raises: This function will raise if the specified slice start or end
            position are outside the bounds of the string, or if they do not
            both fall on codepoint boundaries.
        """
        var step: Int
        var start: Int
        var end: Int
        start, end, step = span.indices(len(self))

        if step != 1:
            raise Error("Slice must be within bounds and step must be 1")

        if not self.is_codepoint_boundary(start):
            var msg = String.format(
                "String `Slice` start byte {} must fall on codepoint boundary.",
                start,
            )
            raise Error(msg^)

        if not self.is_codepoint_boundary(end):
            var msg = String.format(
                "String `Slice` end byte {} must fall on codepoint boundary.",
                end,
            )
            raise Error(msg^)

        return Self(unsafe_from_utf8=self._slice[span])

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__(self, rhs_same: Self) -> Bool:
        """Verify if a `StringSlice` is equal to another `StringSlice` with the
        same origin.

        Args:
            rhs_same: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is equal to the input in length and contents.
        """
        return Self.__eq__(self, rhs=rhs_same)

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_disable_nested_origin_exclusivity
    fn __eq__(self, rhs: StringSlice) -> Bool:
        """Verify if a `StringSlice` is equal to another `StringSlice`.

        Args:
            rhs: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is equal to the input in length and contents.
        """

        var s_len = self.byte_length()
        var s_ptr = self.unsafe_ptr()
        var rhs_ptr = rhs.unsafe_ptr()
        if s_len != rhs.byte_length():
            return False
        # same pointer and length, so equal
        elif s_len == 0 or s_ptr == rhs_ptr:
            return True
        return memcmp(s_ptr, rhs_ptr, s_len) == 0

    fn __ne__(self, rhs_same: Self) -> Bool:
        """Verify if a `StringSlice` is not equal to another `StringSlice` with
        the same origin.

        Args:
            rhs_same: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is not equal to the input in length and
            contents.
        """
        return Self.__ne__(self, rhs=rhs_same)

    @__unsafe_disable_nested_origin_exclusivity
    @always_inline
    fn __ne__(self, rhs: StringSlice) -> Bool:
        """Verify if span is not equal to another `StringSlice`.

        Args:
            rhs: The `StringSlice` to compare against.

        Returns:
            If the `StringSlice` is not equal to the input in length and
            contents.
        """
        return not self == rhs

    @always_inline
    fn __lt__(self, rhs: StringSlice) -> Bool:
        """Verify if the `StringSlice` bytes are strictly less than the input in
        overlapping content.

        Args:
            rhs: The other `StringSlice` to compare against.

        Returns:
            If the `StringSlice` bytes are strictly less than the input in
            overlapping content.
        """
        var len1 = len(self)
        var len2 = len(rhs)
        return Int(len1 < len2) > _memcmp_impl_unconstrained(
            self.unsafe_ptr(), rhs.unsafe_ptr(), min(len1, len2)
        )

    @deprecated("Use `str.codepoints()` or `str.codepoint_slices()` instead.")
    fn __iter__(self) -> CodepointSliceIter[origin]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return self.codepoint_slices()

    fn __reversed__(self) -> CodepointSliceIter[origin, False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return CodepointSliceIter[origin, forward=False](self)

    fn __getitem__[I: Indexer](self, idx: I) -> String:
        """Gets the character at the specified position.

        Parameters:
            I: A type that can be used as an index.

        Args:
            idx: The index value.

        Returns:
            A new string containing the character at the specified position.
        """
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time
        var buf = String._buffer_type(capacity=1)
        buf.append(self._slice[idx])
        buf.append(0)
        return String(buf^)

    fn __contains__(self, substr: StringSlice) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return self.find(substr) != -1

    @always_inline
    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return atol(self)

    @always_inline
    fn __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.
        """
        return atof(self)

    fn __add__(self, rhs: StringSlice) -> String:
        """Returns a string with this value prefixed on another string.

        Args:
            rhs: The right side of the result.

        Returns:
            The result string.
        """
        var res = List[Byte](
            capacity=self.byte_length() + rhs.byte_length() + 1
        )
        res.extend(self.as_bytes())
        res.extend(rhs.as_bytes())
        res.append(0)
        return String(buffer=res^)

    fn __radd__(self, lhs: StringSlice) -> String:
        """Returns a string with this value appended to another string.

        Args:
            lhs: The left side of the result.

        Returns:
            The result string.
        """
        var res = List[Byte](
            capacity=self.byte_length() + lhs.byte_length() + 1
        )
        res.extend(lhs.as_bytes())
        res.extend(self.as_bytes())
        res.append(0)
        return String(buffer=res^)

    fn __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n: The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """

        var buffer = List[Byte](capacity=self.byte_length() * n + 1)
        for i in range(n):
            buffer.extend(self.as_bytes())
        buffer.append(0)
        return String(buffer=buffer)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    fn replace(self, old: StringSlice, new: StringSlice) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        if not old:
            return self._interleave(new)

        var occurrences = self.count(old)
        if occurrences == -1:
            return String(self)

        var self_start = self.unsafe_ptr()
        var self_ptr = self.unsafe_ptr()
        var new_ptr = new.unsafe_ptr()

        var self_len = self.byte_length()
        var old_len = old.byte_length()
        var new_len = new.byte_length()

        var res = String._buffer_type()
        res.reserve(self_len + (old_len - new_len) * occurrences + 1)

        for _ in range(occurrences):
            var curr_offset = Int(self_ptr) - Int(self_start)

            var idx = self.find(old, curr_offset)

            debug_assert(idx >= 0, "expected to find occurrence during find")

            # Copy preceding unchanged chars
            for _ in range(curr_offset, idx):
                res.append(self_ptr[])
                self_ptr += 1

            # Insert a copy of the new replacement string
            for i in range(new_len):
                res.append(new_ptr[i])

            self_ptr += old_len

        while True:
            var val = self_ptr[]
            if val == 0:
                break
            res.append(self_ptr[])
            self_ptr += 1

        res.append(0)
        return String(res^)

    fn _interleave(self, val: StringSlice) -> String:
        var res = String._buffer_type()
        var val_ptr = val.unsafe_ptr()
        var self_ptr = self.unsafe_ptr()
        res.reserve(val.byte_length() * self.byte_length() + 1)
        for i in range(self.byte_length()):
            for j in range(val.byte_length()):
                res.append(val_ptr[j])
            res.append(self_ptr[i])
        res.append(0)
        return String(res^)

    fn split[
        sep_mut: Bool,
        sep_origin: Origin[sep_mut], //,
    ](
        self,
        sep: StringSlice[sep_origin],
        maxsplit: Int = -1,
    ) raises -> List[
        String
    ]:
        """Split the string by a separator.

        Parameters:
            sep_mut: Mutability of the `sep` string slice.
            sep_origin: Origin of the `sep` string slice.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.
                Defaults to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Raises:
            If the separator is empty.

        Examples:

        ```mojo
        # Splitting a space
        _ = StringSlice("hello world").split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = StringSlice("hello,,world").split(",") # ["hello", "", "world"]
        # Splitting with maxsplit
        _ = StringSlice("1,2,3").split(",", 1) # ['1', '2,3']
        ```
        .
        """
        var output = List[String]()

        var str_byte_len = self.byte_length() - 1
        var lhs = 0
        var rhs = 0
        var items = 0
        var sep_len = sep.byte_length()
        if sep_len == 0:
            raise Error("Separator cannot be empty.")
        if str_byte_len < 0:
            output.append(String(""))

        while lhs <= str_byte_len:
            rhs = self.find(sep, lhs)
            if rhs == -1:
                output.append(String(self[lhs:]))
                break

            if maxsplit > -1:
                if items == maxsplit:
                    output.append(String(self[lhs:]))
                    break
                items += 1

            output.append(String(self[lhs:rhs]))
            lhs = rhs + sep_len

        if self.endswith(sep) and (len(output) <= maxsplit or maxsplit == -1):
            output.append(String(""))

        return output^

    fn split(
        self, sep: NoneType = None, maxsplit: Int = -1
    ) -> List[StringSlice[origin]]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from String. Defaults
                to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = StringSlice("      ").split() # []
        _ = StringSlice("").split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = StringSlice("      hello    world     ").split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = StringSlice(
            "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029world"
        ).split()  # ["hello", "world"]
        ```
        .
        """

        return self._split_whitespace()

    fn _split_whitespace(self, maxsplit: Int = -1) -> List[StringSlice[origin]]:
        fn num_bytes(b: UInt8) -> Int:
            var flipped = ~b
            return Int(count_leading_zeros(flipped) + (flipped >> 7))

        var output = List[StringSlice[origin]]()
        var str_byte_len = self.byte_length() - 1
        var lhs = 0
        var rhs = 0
        var items = 0
        while lhs <= str_byte_len:
            try:
                # Python adds all "whitespace chars" as one separator
                # if no separator was specified
                for s in self[lhs:].codepoint_slices():
                    if not s.isspace():
                        break
                    lhs += s.byte_length()
                # if it went until the end of the String, then
                # it should be sliced up until the original
                # start of the whitespace which was already appended
                if lhs - 1 == str_byte_len:
                    break
                elif lhs == str_byte_len:
                    # if the last char is not whitespace
                    output.append(self[str_byte_len:])
                    break
                rhs = lhs + num_bytes(self.unsafe_ptr()[lhs])
                for s in self[
                    lhs + num_bytes(self.unsafe_ptr()[lhs]) :
                ].codepoint_slices():
                    if s.isspace():
                        break
                    rhs += s.byte_length()

                if maxsplit > -1:
                    if items == maxsplit:
                        output.append(self[lhs:])
                        break
                    items += 1

                output.append(self[lhs:rhs])
                lhs = rhs
            except e:
                return abort[List[StringSlice[origin]]](
                    "unexpected exception during split()"
                )

        return output

    @always_inline
    fn strip(self, chars: StringSlice) -> Self:
        """Return a copy of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading or trailing characters.

        Examples:

        ```mojo
        print("himojohi".strip("hi")) # "mojo"
        ```
        .
        """

        return self.lstrip(chars).rstrip(chars)

    @always_inline
    fn strip(self) -> Self:
        """Return a copy of the string with leading and trailing whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no leading or trailing whitespaces.

        Examples:

        ```mojo
        print("  mojo  ".strip()) # "mojo"
        ```
        .
        """
        return self.lstrip().rstrip()

    @always_inline
    fn rstrip(self, chars: StringSlice) -> Self:
        """Return a copy of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no trailing characters.

        Examples:

        ```mojo
        print("mojohi".strip("hi")) # "mojo"
        ```
        .
        """

        var r_idx = self.byte_length()
        while r_idx > 0 and self[r_idx - 1] in chars:
            r_idx -= 1

        return Self(unsafe_from_utf8=self.as_bytes()[:r_idx])

    @always_inline
    fn rstrip(self) -> Self:
        """Return a copy of the string with trailing whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no trailing whitespaces.

        Examples:

        ```mojo
        print("mojo  ".strip()) # "mojo"
        ```
        .
        """
        var r_idx = self.byte_length()
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self.__reversed__():
        #     if not s.isspace():
        #         break
        #     r_idx -= 1
        while (
            r_idx > 0 and Codepoint(self.as_bytes()[r_idx - 1]).is_posix_space()
        ):
            r_idx -= 1
        return Self(unsafe_from_utf8=self.as_bytes()[:r_idx])

    @always_inline
    fn lstrip(self, chars: StringSlice) -> Self:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.

        Examples:

        ```mojo
        print("himojo".strip("hi")) # "mojo"
        ```
        .
        """

        var l_idx = 0
        while l_idx < self.byte_length() and self[l_idx] in chars:
            l_idx += 1

        return Self(unsafe_from_utf8=self.as_bytes()[l_idx:])

    @always_inline
    fn lstrip(self) -> Self:
        """Return a copy of the string with leading whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no leading whitespaces.

        Examples:

        ```mojo
        print("  mojo".strip()) # "mojo"
        ```
        .
        """
        var l_idx = 0
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self:
        #     if not s.isspace():
        #         break
        #     l_idx += 1
        while (
            l_idx < self.byte_length()
            and Codepoint(self.as_bytes()[l_idx]).is_posix_space()
        ):
            l_idx += 1
        return Self(unsafe_from_utf8=self.as_bytes()[l_idx:])

    @always_inline
    fn codepoints(self) -> CodepointsIter[origin]:
        """Returns an iterator over the `Codepoint`s encoded in this string slice.

        Returns:
            An iterator type that returns successive `Codepoint` values stored in
            this string slice.

        # Examples

        Print the characters in a string:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        var s = StringSlice("abc")
        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
        assert_equal(iter.__next__(), Codepoint.ord("b"))
        assert_equal(iter.__next__(), Codepoint.ord("c"))
        assert_equal(iter.__has_next__(), False)
        ```

        `codepoints()` iterates over Unicode codepoints, and supports multibyte
        codepoints:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        # A visual character composed of a combining sequence of 2 codepoints.
        var s = StringSlice("á")
        assert_equal(s.byte_length(), 3)

        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
         # U+0301 Combining Acute Accent
        assert_equal(iter.__next__().to_u32(), 0x0301)
        assert_equal(iter.__has_next__(), False)
        ```
        .
        """
        return CodepointsIter(self)

    fn codepoint_slices(self) -> CodepointSliceIter[origin]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return CodepointSliceIter[origin](self)

    @always_inline
    fn as_bytes(self) -> Span[Byte, origin]:
        """Get the sequence of encoded bytes of the underlying string.

        Returns:
            A slice containing the underlying sequence of encoded bytes.
        """
        return self._slice

    @always_inline
    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[Byte, mut=mut, origin=origin]:
        """Gets a pointer to the first element of this string slice.

        Returns:
            A pointer pointing at the first element of this string slice.
        """
        return self._slice.unsafe_ptr()

    @always_inline
    fn byte_length(self) -> Int:
        """Get the length of this string slice in bytes.

        Returns:
            The length of this string slice in bytes.
        """

        return len(self._slice)

    fn char_length(self) -> UInt:
        """Returns the length in Unicode codepoints.

        This returns the number of `Codepoint` codepoint values encoded in the UTF-8
        representation of this string.

        Note: To get the length in bytes, use `StringSlice.byte_length()`.

        Returns:
            The length in Unicode codepoints.

        # Examples

        Query the length of a string, in bytes and Unicode codepoints:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        var s = StringSlice("ನಮಸ್ಕಾರ")

        assert_equal(s.char_length(), 7)
        assert_equal(len(s), 21)
        ```

        Strings containing only ASCII characters have the same byte and
        Unicode codepoint length:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        var s = StringSlice("abc")

        assert_equal(s.char_length(), 3)
        assert_equal(len(s), 3)
        ```

        The character length of a string with visual combining characters is
        the length in Unicode codepoints, not grapheme clusters:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal

        var s = StringSlice("á")
        assert_equal(s.char_length(), 2)
        assert_equal(s.byte_length(), 3)
        ```
        .
        """
        # Every codepoint is encoded as one leading byte + 0 to 3 continuation
        # bytes.
        # The total number of codepoints is equal the number of leading bytes.
        # So we can compute the number of leading bytes (and thereby codepoints)
        # by subtracting the number of continuation bytes length from the
        # overall length in bytes.
        # For a visual explanation of how this UTF-8 codepoint counting works:
        #   https://connorgray.com/ephemera/project-log#2025-01-13
        var continuation_count = _count_utf8_continuation_bytes(self)
        return self.byte_length() - continuation_count

    fn is_codepoint_boundary(self, index: UInt) -> Bool:
        """Returns True if `index` is the position of the first byte in a UTF-8
        codepoint sequence, or is at the end of the string.

        A byte position is considered a codepoint boundary if a valid subslice
        of the string would end (noninclusive) at `index`.

        Positions `0` and `len(self)` are considered to be codepoint boundaries.

        Positions beyond the length of the string slice will return False.

        Args:
            index: An index into the underlying byte representation of the
                string.

        Returns:
            A boolean indicating if `index` gives the position of the first
            byte in a UTF-8 codepoint sequence, or is at the end of the string.

        # Examples

        Check if particular byte positions are codepoint boundaries:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_equal, assert_true
        var abc = StringSlice("abc")
        assert_equal(len(abc), 3)
        assert_true(abc.is_codepoint_boundary(0))
        assert_true(abc.is_codepoint_boundary(1))
        assert_true(abc.is_codepoint_boundary(2))
        assert_true(abc.is_codepoint_boundary(3))
        ```

        Only the index of the first byte in a multi-byte codepoint sequence is
        considered a codepoint boundary:

        ```mojo
        var thumb = StringSlice("👍")
        assert_equal(len(thumb), 4)
        assert_true(thumb.is_codepoint_boundary(0))
        assert_false(thumb.is_codepoint_boundary(1))
        assert_false(thumb.is_codepoint_boundary(2))
        assert_false(thumb.is_codepoint_boundary(3))
        ```

        Visualization showing which bytes are considered codepoint boundaries,
        within a piece of text that includes codepoints whose UTF-8
        representation requires, respectively, 1, 2, 3, and 4-bytes. The
        codepoint boundary byte indices are indicated by a vertical arrow (↑).

        For example, this diagram shows that a slice of bytes formed by the
        half-open range starting at byte 3 and extending up to but not including
        byte 6 (`[3, 6)`) is a valid UTF-8 sequence.

        ```text
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                a©➇𝄞                  ┃ String
        ┣━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┫
        ┃97┃  169  ┃   10119   ┃    119070     ┃ Unicode Codepoints
        ┣━━╋━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━┳━━━┫
        ┃97┃194┃169┃226┃158┃135┃240┃157┃132┃158┃ UTF-8 Bytes
        ┗━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┛
        0  1   2   3   4   5   6   7   8   9  10
        ↑  ↑       ↑           ↑               ↑
        ```

        The following program verifies the above diagram:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_true, assert_false

        var text = StringSlice("a©➇𝄞")
        assert_true(text.is_codepoint_boundary(0))
        assert_true(text.is_codepoint_boundary(1))
        assert_false(text.is_codepoint_boundary(2))
        assert_true(text.is_codepoint_boundary(3))
        assert_false(text.is_codepoint_boundary(4))
        assert_false(text.is_codepoint_boundary(5))
        assert_true(text.is_codepoint_boundary(6))
        assert_false(text.is_codepoint_boundary(7))
        assert_false(text.is_codepoint_boundary(8))
        assert_false(text.is_codepoint_boundary(9))
        assert_true(text.is_codepoint_boundary(10))
        ```
        .
        """
        # TODO: Example: Print the byte indices that are codepoints boundaries:

        if index >= len(self):
            return index == len(self)

        var byte = self.as_bytes()[index]
        # If this is not a continuation byte, then it must be a start byte.
        return _utf8_byte_type(byte) != 1

    fn get_immutable(
        self,
    ) -> StringSlice[ImmutableOrigin.cast_from[origin].result]:
        """
        Return an immutable version of this string slice.

        Returns:
            A string slice covering the same elements, but without mutability.
        """
        return StringSlice[ImmutableOrigin.cast_from[origin].result](
            ptr=self._slice.unsafe_ptr(),
            length=len(self),
        )

    fn startswith(
        self, prefix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Verify if the `StringSlice` starts with the specified prefix between
        start and end positions.

        The `start` and `end` positions must be offsets given in bytes, and
        must be codepoint boundaries.

        Args:
            prefix: The prefix to check.
            start: The start offset in bytes from which to check.
            end: The end offset in bytes from which to check.

        Returns:
            True if the `self[start:end]` is prefixed by the input prefix.
        """
        if end == -1:
            return self.find(prefix, start) == start
        # FIXME: use normalize_index
        return StringSlice[origin](
            ptr=self.unsafe_ptr() + start, length=end - start
        ).startswith(prefix)

    fn endswith(
        self, suffix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Verify if the `StringSlice` end with the specified suffix between
        start and end positions.

        The `start` and `end` positions must be offsets given in bytes, and
        must be codepoint boundaries.

        Args:
            suffix: The suffix to check.
            start: The start offset in bytes from which to check.
            end: The end offset in bytes from which to check.

        Returns:
            True if the `self[start:end]` is suffixed by the input suffix.
        """
        if len(suffix) > len(self):
            return False
        if end == -1:
            return self.rfind(suffix, start) + len(suffix) == len(self)
        # FIXME: use normalize_index
        return StringSlice[origin](
            ptr=self.unsafe_ptr() + start, length=end - start
        ).endswith(suffix)

    fn _from_start(self, start: Int) -> Self:
        """Gets the `StringSlice` pointing to the substring after the specified
        slice start position in bytes. If start is negative, it is interpreted
        as the number of characters from the end of the string to start at.

        Args:
            start: Starting index of the slice in bytes. Must be a codepoint
                boundary.

        Returns:
            A `StringSlice` borrowed from the current string containing the
            characters of the slice starting at start.
        """
        # FIXME: use normalize_index

        var self_len = self.byte_length()

        var abs_start: Int
        if start < 0:
            # Avoid out of bounds earlier than the start
            # len = 5, start = -3,  then abs_start == 2, i.e. a partial string
            # len = 5, start = -10, then abs_start == 0, i.e. the full string
            abs_start = max(self_len + start, 0)
        else:
            # Avoid out of bounds past the end
            # len = 5, start = 2,   then abs_start == 2, i.e. a partial string
            # len = 5, start = 8,   then abs_start == 5, i.e. an empty string
            abs_start = min(start, self_len)

        debug_assert(
            abs_start >= 0, "strref absolute start must be non-negative"
        )
        debug_assert(
            abs_start <= self_len,
            "strref absolute start must be less than source String len",
        )

        # TODO(MSTDL-1161): Assert that `self.is_codepoint_boundary(abs_start)`.

        # TODO: We assumes the StringSlice only has ASCII.
        # When we support utf-8 slicing, we should drop self._slice[abs_start:]
        # and use something smarter.
        return StringSlice(unsafe_from_utf8=self._slice[abs_start:])

    @always_inline
    fn format[*Ts: _CurlyEntryFormattable](self, *args: *Ts) raises -> String:
        """Format a template with `*args`.

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Representable`
                and `Stringable` (to be changed and made more flexible).

        Returns:
            The template with the given values substituted.

        Examples:

        ```mojo
        # Manual indexing:
        print("{0} {1} {0}".format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print("{} {}".format(True, "hello world")) # True hello world
        ```
        .
        """
        return _FormatCurlyEntry.format(self, args)

    fn find(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset in bytes of the first occurrence of `substr`
        starting at `start`. If not found, returns `-1`.

        Args:
            substr: The substring to find.
            start: The offset in bytes from which to find. Must be a codepoint
                boundary.

        Returns:
            The offset in bytes of `substr` relative to the beginning of the
            string.
        """
        if not substr:
            return 0

        if self.byte_length() < substr.byte_length() + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._from_start(start)

        var loc = _memmem(
            haystack_str.unsafe_ptr(),
            haystack_str.byte_length(),
            substr.unsafe_ptr(),
            substr.byte_length(),
        )

        if not loc:
            return -1

        return Int(loc) - Int(self.unsafe_ptr())

    fn rfind(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset in bytes of the last occurrence of `substr` starting at
        `start`. If not found, returns `-1`.

        Args:
            substr: The substring to find.
            start: The offset in bytes from which to find. Must be a valid
                codepoint boundary.

        Returns:
            The offset in bytes of `substr` relative to the beginning of the
            string.
        """
        if not substr:
            return len(self)

        if len(self) < len(substr) + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var haystack_str = self._from_start(start)

        var loc = _memrmem(
            haystack_str.unsafe_ptr(),
            len(haystack_str),
            substr.unsafe_ptr(),
            len(substr),
        )

        if not loc:
            return -1

        return Int(loc) - Int(self.unsafe_ptr())

    fn isspace(self) -> Bool:
        """Determines whether every character in the given StringSlice is a
        python whitespace String. This corresponds to Python's
        [universal separators](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines):
         `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole StringSlice is made up of whitespace characters
            listed above, otherwise False.

        Examples:

        Check if a string contains only whitespace:

        ```mojo
        from collections.string import StringSlice
        from testing import assert_true, assert_false

        # An empty string is not considered to contain only whitespace chars:
        assert_false(StringSlice("").isspace())

        # ASCII space characters
        assert_true(StringSlice(" ").isspace())
        assert_true(StringSlice("\t").isspace())

        # Contains non-space characters
        assert_false(StringSlice(" abc  ").isspace())
        ```
        .
        """

        if self.byte_length() == 0:
            return False

        for s in self.codepoints():
            if not s.is_python_space():
                return False

        return True

    fn isnewline[single_character: Bool = False](self) -> Bool:
        """Determines whether every character in the given StringSlice is a
        python newline character. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Parameters:
            single_character: Whether to evaluate the stringslice as a single
                unicode character (avoids overhead when already iterating).

        Returns:
            True if the whole StringSlice is made up of whitespace characters
                listed above, otherwise False.
        """

        var ptr = self.unsafe_ptr()
        var length = self.byte_length()

        @parameter
        if single_character:
            return length != 0 and _is_newline_char_utf8[include_r_n=True](
                ptr, 0, ptr[0], length
            )
        else:
            var offset = 0
            for s in self.codepoint_slices():
                var b_len = s.byte_length()
                if not _is_newline_char_utf8(ptr, offset, ptr[offset], b_len):
                    return False
                offset += b_len
            return length != 0

    fn splitlines[
        O: ImmutableOrigin, //
    ](self: StringSlice[O], keepends: Bool = False) -> List[StringSlice[O]]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Parameters:
            O: The immutable origin.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """

        # highly performance sensitive code, benchmark before touching
        alias `\r` = UInt8(ord("\r"))
        alias `\n` = UInt8(ord("\n"))

        output = List[StringSlice[O]](capacity=128)  # guessing
        var ptr = self.unsafe_ptr()
        var length = self.byte_length()
        var offset = 0

        while offset < length:
            var eol_start = offset
            var eol_length = 0

            while eol_start < length:
                var b0 = ptr[eol_start]
                var char_len = _utf8_first_byte_sequence_length(b0)
                debug_assert(
                    eol_start + char_len <= length,
                    "corrupted sequence causing unsafe memory access",
                )
                var isnewline = unlikely(
                    _is_newline_char_utf8(ptr, eol_start, b0, char_len)
                )
                var char_end = Int(isnewline) * (eol_start + char_len)
                var next_idx = char_end * Int(char_end < length)
                var is_r_n = b0 == `\r` and next_idx != 0 and ptr[
                    next_idx
                ] == `\n`
                eol_length = Int(isnewline) * char_len + Int(is_r_n)
                if isnewline:
                    break
                eol_start += char_len

            var str_len = eol_start - offset + Int(keepends) * eol_length
            var s = StringSlice[O](ptr=ptr + offset, length=str_len)
            output.append(s)
            offset = eol_start + eol_length

        return output^

    fn count(self, substr: StringSlice) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
            substr: The substring to count.

        Returns:
            The number of occurrences of `substr`.
        """
        if not substr:
            return len(self) + 1

        var res = 0
        var offset = 0

        while True:
            var pos = self.find(substr, offset)
            if pos == -1:
                break
            res += 1

            offset = pos + substr.byte_length()

        return res

    fn is_ascii_digit(self) -> Bool:
        """A string is a digit string if all characters in the string are digits
        and there is at least one character in the string.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits and it's not empty else False.
        """
        if not self:
            return False
        for char in self.codepoints():
            if not char.is_ascii_digit():
                return False
        return True

    fn isupper(self) -> Bool:
        """Returns True if all cased characters in the string are uppercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are uppercase and there
            is at least one cased character, False otherwise.
        """
        return len(self) > 0 and is_uppercase(self)

    fn islower(self) -> Bool:
        """Returns True if all cased characters in the string are lowercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are lowercase and there
            is at least one cased character, False otherwise.
        """
        return len(self) > 0 and is_lowercase(self)

    fn lower(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        # TODO: the _unicode module does not support locale sensitive conversions yet.
        return to_lowercase(self)

    fn upper(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        # TODO: the _unicode module does not support locale sensitive conversions yet.
        return to_uppercase(self)

    fn is_ascii_printable(self) -> Bool:
        """Returns True if all characters in the string are ASCII printable.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are printable else False.
        """
        for char in self.codepoints():
            if not char.is_ascii_printable():
                return False
        return True

    fn rjust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string right justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns right justified string, or self if width is not bigger than self length.
        """
        return self._justify(width - len(self), width, fillchar)

    fn ljust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string left justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns left justified string, or self if width is not bigger than self length.
        """
        return self._justify(0, width, fillchar)

    fn center(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string center justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns center justified string, or self if width is not bigger than self length.
        """
        return self._justify(width - len(self) >> 1, width, fillchar)

    fn _justify(self, start: Int, width: Int, fillchar: StaticString) -> String:
        if len(self) >= width:
            return String(self)
        debug_assert(
            len(fillchar) == 1, "fill char needs to be a one byte literal"
        )
        var fillbyte = fillchar.as_bytes()[0]
        var buffer = List[Byte](capacity=width + 1)
        buffer.resize(width, fillbyte)
        buffer.append(0)
        memcpy(buffer.unsafe_ptr().offset(start), self.unsafe_ptr(), len(self))
        var result = String(buffer=buffer)
        return result^

    fn join[T: WritableCollectionElement](self, elems: List[T, *_]) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            T: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        var string = String()
        var buffer = _WriteBufferStack(string)
        for i in range(len(elems)):
            buffer.write(elems[i])
            if i < len(elems) - 1:
                buffer.write(self)
        buffer.flush()
        return string

    # TODO(MOCO-1791): The corresponding String.__init__ is limited to
    # StaticString. This is because default arguments and param inference aren't
    # powerful enough to declare sep/end as StringSlice.
    fn join[*Ts: Writable](self: StaticString, *elems: *Ts) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            Ts: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        return String(elems, sep=self)


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _get_kgen_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    """Form a `!kgen.string` from compile-time StringSlice values concatenated.

    Parameters:
        string: The first StringSlice value.
        extra: Additional StringSlice values to concatenate.

    Returns:
        The string value as a StringLiteral.
    """
    return _get_kgen_string[string, extra]()


@always_inline("nodebug")
fn _get_kgen_string[
    string: StaticString, extra: VariadicList[StaticString]
]() -> __mlir_type.`!kgen.string`:
    """Form a `!kgen.string` from compile-time StringSlice values concatenated.

    Parameters:
        string: The first string slice to use.
        extra: Additional string slices to concatenate.

    Returns:
        The string value as a StringLiteral.
    """
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.value,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
fn get_static_string[
    string: StaticString, *extra: StaticString
]() -> StaticString:
    """Form a StaticString from compile-time StringSlice values. This
    guarantees that the returned string is compile-time constant in static
    memory.  It also guarantees that there is a 'nul' zero byte at the end,
    which is not included in the returned range.

    Parameters:
        string: The first StringSlice value.
        extra: Additional StringSlice values to concatenate.

    Returns:
        The string value as a StaticString.
    """
    return _get_kgen_string[string, extra]()


fn _to_string_list[
    T: CollectionElement,  # TODO(MOCO-1446): Make `T` parameter inferred
    len_fn: fn (T) -> Int,
    unsafe_ptr_fn: fn (T) -> UnsafePointer[Byte],
](items: List[T]) -> List[String]:
    i_len = len(items)
    i_ptr = items.unsafe_ptr()
    out_ptr = UnsafePointer[String].alloc(i_len)

    for i in range(i_len):
        og_len = len_fn(i_ptr[i])
        f_len = og_len + 1  # null terminator
        p = UnsafePointer[Byte].alloc(f_len)
        og_ptr = unsafe_ptr_fn(i_ptr[i])
        memcpy(p, og_ptr, og_len)
        p[og_len] = 0  # null terminator
        buf = String._buffer_type(ptr=p, length=f_len, capacity=f_len)
        (out_ptr + i).init_pointee_move(String(buf^))
    return List[String](ptr=out_ptr, length=i_len, capacity=i_len)


@always_inline
fn _to_string_list[
    O: ImmutableOrigin, //
](items: List[StringSlice[O]]) -> List[String]:
    """Create a list of Strings **copying** the existing data.

    Parameters:
        O: The origin of the data.

    Args:
        items: The List of string slices.

    Returns:
        The list of created strings.
    """

    fn unsafe_ptr_fn(v: StringSlice[O]) -> UnsafePointer[Byte]:
        return v.unsafe_ptr()

    fn len_fn(v: StringSlice[O]) -> Int:
        return v.byte_length()

    return _to_string_list[items.T, len_fn, unsafe_ptr_fn](items)


@always_inline
fn _to_string_list[
    O: ImmutableOrigin, //
](items: List[Span[Byte, O]]) -> List[String]:
    """Create a list of Strings **copying** the existing data.

    Parameters:
        O: The origin of the data.

    Args:
        items: The List of Bytes.

    Returns:
        The list of created strings.
    """

    fn unsafe_ptr_fn(v: Span[Byte, O]) -> UnsafePointer[Byte]:
        return v.unsafe_ptr()

    fn len_fn(v: Span[Byte, O]) -> Int:
        return len(v)

    return _to_string_list[items.T, len_fn, unsafe_ptr_fn](items)


@always_inline
fn _unsafe_strlen(owned ptr: UnsafePointer[Byte]) -> Int:
    """
    Get the length of a null-terminated string from a pointer.
    Note: the length does NOT include the null terminator.

    Args:
        ptr: The null-terminated pointer to the string.

    Returns:
        The length of the null terminated string without the null terminator.
    """
    var len = 0
    while ptr.load(len):
        len += 1
    return len


@always_inline
fn _memchr[
    dtype: DType, //
](
    source: UnsafePointer[Scalar[dtype]], char: Scalar[dtype], len: Int
) -> UnsafePointer[Scalar[dtype]]:
    if is_compile_time() or len < simdwidthof[dtype]():
        return _memchr_simple(source, char, len)
    else:
        return _memchr_impl(source, char, len)


@always_inline
fn _memchr_simple[
    dtype: DType, //
](
    source: UnsafePointer[Scalar[dtype]], char: Scalar[dtype], len: Int
) -> UnsafePointer[Scalar[dtype]]:
    for i in range(len):
        if source[i] == char:
            return source + i
    return UnsafePointer[Scalar[dtype]]()


@always_inline
fn _memchr_impl[
    dtype: DType, //
](
    source: UnsafePointer[Scalar[dtype]], char: Scalar[dtype], len: Int
) -> UnsafePointer[Scalar[dtype]]:
    if not len:
        return UnsafePointer[Scalar[dtype]]()
    alias bool_mask_width = simdwidthof[DType.bool]()
    var first_needle = SIMD[dtype, bool_mask_width](char)
    var vectorized_end = align_down(len, bool_mask_width)

    for i in range(0, vectorized_end, bool_mask_width):
        var bool_mask = source.load[width=bool_mask_width](i) == first_needle
        var mask = pack_bits(bool_mask)
        if mask:
            return source + Int(i + count_trailing_zeros(mask))

    for i in range(vectorized_end, len):
        if source[i] == char:
            return source + i
    return UnsafePointer[Scalar[dtype]]()


@always_inline
fn _memmem[
    dtype: DType, //
](
    haystack: UnsafePointer[Scalar[dtype]],
    haystack_len: Int,
    needle: UnsafePointer[Scalar[dtype]],
    needle_len: Int,
) -> UnsafePointer[Scalar[dtype]]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return UnsafePointer[Scalar[dtype]]()
    if needle_len == 1:
        return _memchr(haystack, needle[0], haystack_len)

    if is_compile_time() or haystack_len < simdwidthof[dtype]():
        return _memmem_impl_simple(haystack, haystack_len, needle, needle_len)
    else:
        return _memmem_impl(haystack, haystack_len, needle, needle_len)


@always_inline
fn _memmem_impl_simple[
    dtype: DType, //
](
    haystack: UnsafePointer[Scalar[dtype]],
    haystack_len: Int,
    needle: UnsafePointer[Scalar[dtype]],
    needle_len: Int,
) -> UnsafePointer[Scalar[dtype]]:
    for i in range(haystack_len - needle_len + 1):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i

    return UnsafePointer[Scalar[dtype]]()


@always_inline
fn _memmem_impl[
    dtype: DType, //
](
    haystack: UnsafePointer[Scalar[dtype]],
    haystack_len: Int,
    needle: UnsafePointer[Scalar[dtype]],
    needle_len: Int,
) -> UnsafePointer[Scalar[dtype]]:
    alias bool_mask_width = simdwidthof[DType.bool]()
    var vectorized_end = align_down(
        haystack_len - needle_len + 1, bool_mask_width
    )

    var first_needle = SIMD[dtype, bool_mask_width](needle[0])
    var last_needle = SIMD[dtype, bool_mask_width](needle[needle_len - 1])

    for i in range(0, vectorized_end, bool_mask_width):
        var first_block = haystack.load[width=bool_mask_width](i)
        var last_block = haystack.load[width=bool_mask_width](
            i + needle_len - 1
        )

        var eq_first = first_needle == first_block
        var eq_last = last_needle == last_block

        var bool_mask = eq_first & eq_last
        var mask = pack_bits(bool_mask)

        while mask:
            var offset = Int(i + count_trailing_zeros(mask))
            if memcmp(haystack + offset + 1, needle + 1, needle_len - 1) == 0:
                return haystack + offset
            mask = mask & (mask - 1)

    # remaining partial block compare using byte-by-byte
    #
    for i in range(vectorized_end, haystack_len - needle_len + 1):
        if haystack[i] != needle[0]:
            continue

        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i

    return UnsafePointer[Scalar[dtype]]()


@always_inline
fn _memrchr[
    dtype: DType
](
    source: UnsafePointer[Scalar[dtype]], char: Scalar[dtype], len: Int
) -> UnsafePointer[Scalar[dtype]]:
    if not len:
        return UnsafePointer[Scalar[dtype]]()
    for i in reversed(range(len)):
        if source[i] == char:
            return source + i
    return UnsafePointer[Scalar[dtype]]()


@always_inline
fn _memrmem[
    dtype: DType
](
    haystack: UnsafePointer[Scalar[dtype]],
    haystack_len: Int,
    needle: UnsafePointer[Scalar[dtype]],
    needle_len: Int,
) -> UnsafePointer[Scalar[dtype]]:
    if not needle_len:
        return haystack
    if needle_len > haystack_len:
        return UnsafePointer[Scalar[dtype]]()
    if needle_len == 1:
        return _memrchr[dtype](haystack, needle[0], haystack_len)
    for i in reversed(range(haystack_len - needle_len + 1)):
        if haystack[i] != needle[0]:
            continue
        if memcmp(haystack + i + 1, needle + 1, needle_len - 1) == 0:
            return haystack + i
    return UnsafePointer[Scalar[dtype]]()
