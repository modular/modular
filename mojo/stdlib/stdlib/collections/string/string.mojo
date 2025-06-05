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
"""The core `String` type implementation for Mojo.

This module provides the primary `String` type and its fundamental operations.
The `String` type is a mutable string, and is designed to handle UTF-8 encoded
text efficiently while providing a safe and ergonomic interface for string
manipulation.

Related types:

- [`StringSlice`](/mojo/stdlib/collections/string/string_slice/). A non-owning
  view of string data, which can be either mutable or immutable.
- [`StaticString`](/mojo/stdlib/collections/string/string_slice/#aliases). An
  alias for an immutable constant `StringSlice`.
- [`StringLiteral`](/mojo/stdlib/builtin/string_literal/StringLiteral/). A
  string literal. String literals are compile-time values. For use at runtime,
  you usually want wrap a `StringLiteral` in a `String` (for a mutable string)
  or `StaticString` (for an immutable constant string).

Key Features:
- Short string optimization (SSO) and lazy copying of constant string data.
- O(1) copy operation.
- Memory-safe string operations.
- Efficient string concatenation and slicing.
- String-to-number conversions (
  [`atof()`](/mojo/stdlib/collections/string/string/atof),
  [`atol()`](/mojo/stdlib/collections/string/string/atol)).
- Character code conversions (
  [`chr()`](/mojo/stdlib/collections/string/string/chr),
  [`ord()`](/mojo/stdlib/collections/string/string/ord)).
- String formatting with
  [`format()`](/mojo/stdlib/collections/string/string/String/#format).

The `String` type has Unicode support through UTF-8 encoding. A handful of
operations are known to not be Unicode / UTF-8 compliant yet, but will be fixed
as time permits.

This type is in the prelude, so it is automatically imported into every Mojo
program.

Example:

```mojo
# String creation and basic operations
var s1 = String("Hello")
var s2 = String("World")
var combined = s1 + " " + s2  # "Hello World"

# String-to-number conversion
var num = atof("3.14")
var int_val = atol("42")

# Character operations
var char = chr(65)  # "A"
var code = ord("A")  # 65

# String formatting
print(String("Codepoint {} is {}").format(code, char)) # Codepoint 65 is A

# ASCII utilities
var ascii_str = ascii("Hello")  # ASCII-only string
```
"""

from collections import KeyElement, List, Optional
from collections._index_normalization import normalize_index
from collections.string import CodepointsIter
from collections.string._parsing_numbers.parsing_floats import _atof
from collections.string._unicode import (
    is_lowercase,
    is_uppercase,
    to_lowercase,
    to_uppercase,
)
from collections.string.format import _CurlyEntryFormattable, _FormatCurlyEntry
from collections.string.string_slice import (
    CodepointSliceIter,
    _to_string_list,
    _utf8_byte_type,
)
from hashlib._hasher import _HashableWithHasher, _Hasher
from os import PathLike, abort
from os.atomic import Atomic
from sys import bitwidthof, is_compile_time, sizeof
from sys.ffi import c_char
from sys.intrinsics import _type_is_eq

from bit import count_leading_zeros
from memory import Span, UnsafePointer, memcpy, memset
from python import PythonConvertible, PythonObject, ConvertibleFromPython

from utils import IndexList, Variant, Writable, Writer, write_args
from utils.write import write_buffered
from utils._select import _select_register_value as select

# ===----------------------------------------------------------------------=== #
# String Implementation Details
# ===----------------------------------------------------------------------=== #


# This is a private struct used to store the capacity and bitflags for a String.
# It is not exported and should not be used directly.
@register_passable("trivial")
struct _StringCapacityField:
    # When not-inline, this maintains the capacity of the string shifted right
    # by 3 bits, with 3 top bits used for flags. When inline is the length of
    # the string.
    var _storage: UInt

    # This is the number of bytes that can be stored inline in the string value.
    # 'String' is 3 words in size and we use the top byte of the capacity field
    # to store flags.
    alias INLINE_CAPACITY = Int.BITWIDTH // 8 * 3 - 1
    # The start of the length field in the storage: this is the top byte, which
    # gives us 5 bits for the length.
    alias INLINE_LENGTH_START = UInt(Int.BITWIDTH - 8)
    alias INLINE_LENGTH_MASK = UInt(0b1_1111 << Self.INLINE_LENGTH_START)

    # When FLAG_HAS_NUL_TERMINATOR is set, the byte past the end of the string
    # is known to be an accessible 'nul' terminator.
    alias FLAG_HAS_NUL_TERMINATOR = UInt(1) << (UInt.BITWIDTH - 3)
    # UNUSED_FLAG is not used, but might be used in the future.
    alias UNUSED_FLAG = UInt(1) << (UInt.BITWIDTH - 2)
    # When FLAG_IS_INLINE is set, the string data is inline as the first bytes
    # of the string value (the "Small string optimization").
    alias FLAG_IS_INLINE = UInt(1) << (UInt.BITWIDTH - 1)

    # Initialize with a specified capacity.  Note that the provided value may
    # be rounded up, so clients should check the capacity() after construction.
    @always_inline("nodebug")
    fn __init__(out self, capacity: UInt):
        # If the capacity needed fits inline, use the inline form.
        if capacity <= Self.INLINE_CAPACITY and not is_compile_time():
            self._storage = Self.FLAG_IS_INLINE
        else:
            self = Self(out_of_line_capacity=capacity)

    @always_inline("nodebug")
    fn __init__(out self, *, out_of_line_capacity: UInt):
        # memory allocators work on this granularity anyway, so we might as well
        # use it. We store the capacity with the top 3 bits of the storage used
        # for flags.
        self._storage = (out_of_line_capacity + 7) >> 3

    @always_inline("nodebug")
    fn __init__(out self, *, static_const_length: Int):
        # Short constant strings that can fit inline *with a nul terminator*
        # added are stored inline.
        if static_const_length < Self.INLINE_CAPACITY and not is_compile_time():
            self._storage = Self.FLAG_IS_INLINE
        else:
            # We set the capacity to 0 to always force reallocation if we
            # want the capacity to change.
            self._storage = 0

    @always_inline("nodebug")
    fn capacity(self) -> UInt:
        # In the inline form, we hold the data inline.
        if self.is_inline():
            return Self.INLINE_CAPACITY
        return self._storage << 3

    @always_inline("nodebug")
    fn has_nul_terminator(self) -> Bool:
        return self._storage & Self.FLAG_HAS_NUL_TERMINATOR != 0

    @always_inline("nodebug")
    fn set_has_nul_terminator(mut self, value: Bool):
        var unset = self._storage & ~Self.FLAG_HAS_NUL_TERMINATOR
        self._storage = unset | select(value, Self.FLAG_HAS_NUL_TERMINATOR, 0)

    @always_inline("nodebug")
    fn is_inline(self) -> Bool:
        return self._storage & Self.FLAG_IS_INLINE != 0

    @always_inline("nodebug")
    fn get_len(self, len_or_data: Int) -> Int:
        if self.is_inline():
            return (
                self._storage & Self.INLINE_LENGTH_MASK
            ) >> Self.INLINE_LENGTH_START
        else:
            return len_or_data

    @always_inline("nodebug")
    fn set_len(mut self, new_len: Int, mut len_or_data: Int):
        if self.is_inline():
            debug_assert(new_len <= Self.INLINE_CAPACITY)
            self._storage = (self._storage & ~Self.INLINE_LENGTH_MASK) | (
                new_len << Self.INLINE_LENGTH_START
            )
        else:
            len_or_data = new_len


# This is a private struct used to store the reference count of a out-of-line
# mutable string buffer.
struct _StringOutOfLineHeader:
    var refcount: Atomic[DType.index]
    alias _SIZE = sizeof[Self]()

    @always_inline("nodebug")
    fn __init__(out self):
        """Create an initialized instance of this with a refcount of 1."""
        self.refcount = Scalar[DType.index](1)

    @always_inline("nodebug")
    fn add_ref(mut self):
        """Atomically increment the refcount."""
        _ = self.refcount.fetch_add(1)

    @always_inline("nodebug")
    fn drop_ref(mut self):
        """Atomically decrement the refcount and deallocate self if the result
        hits zero."""
        if self.refcount.fetch_sub(1) == 1:
            UnsafePointer(to=self).bitcast[Byte]().free()

    @always_inline("nodebug")
    fn is_unique(mut self) -> Bool:
        """Return true if the refcount is 1."""
        return self.refcount.load() == 1

    @staticmethod
    fn alloc(capacity: Int) -> UnsafePointer[Byte]:
        """Allocate space for a new out-of-line string buffer."""
        var ptr = UnsafePointer[Byte].alloc(capacity + Self._SIZE)

        # Initialize the header.
        __get_address_as_uninit_lvalue(ptr.bitcast[Self]().address) = Self()

        # Return a pointer to right after the header, which is where the string
        # data will be stored.
        return ptr + Self._SIZE

    @always_inline("nodebug")
    @staticmethod
    fn get(ptr: UnsafePointer[Byte, origin=_]) -> ref [ptr.origin] Self:
        # The header is stored before the string data.
        return (ptr - Self._SIZE).bitcast[Self]()[]


# ===----------------------------------------------------------------------=== #
# String
# ===----------------------------------------------------------------------=== #


struct String(
    Sized,
    Defaultable,
    Stringable,
    Representable,
    IntableRaising,
    KeyElement,
    Comparable,
    Boolable,
    Writable,
    Writer,
    ExplicitlyCopyable,
    FloatableRaising,
    _HashableWithHasher,
    PathLike,
    _CurlyEntryFormattable,
    PythonConvertible,
    ConvertibleFromPython,
):
    """Represents a mutable string.

    See the [`string` module](/mojo/stdlib/collections/string/string/) for
    more information and examples.
    """

    # Fields: String has two forms - the declared form here, and the "inline"
    # form when '_capacity_or_data.is_inline()' is true. The inline form
    # clobbers these fields (except the top byte of the capacity field) with
    # the string data.
    var _ptr_or_data: UnsafePointer[UInt8]
    """The underlying storage for the string data."""
    var _len_or_data: Int
    """The number of elements in the string data."""
    var _capacity_or_data: _StringCapacityField
    """The capacity and bit flags for this String."""

    # Useful string aliases.
    alias ASCII_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
    alias ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alias ASCII_LETTERS = Self.ASCII_LOWERCASE + Self.ASCII_UPPERCASE
    alias DIGITS = "0123456789"
    alias HEX_DIGITS = Self.DIGITS + "abcdef" + "ABCDEF"
    alias OCT_DIGITS = "01234567"
    alias PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
    alias PRINTABLE = Self.DIGITS + Self.ASCII_LETTERS + Self.PUNCTUATION + " \t\n\r\v\f"

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __del__(owned self):
        """Destroy the string data."""
        if self._has_mutable_buffer():
            _StringOutOfLineHeader.get(self._ptr_or_data).drop_ref()

    @always_inline("nodebug")
    fn __init__(out self):
        """Construct an empty string."""
        self = Self(capacity=0)

    @always_inline
    fn __init__(out self, *, capacity: Int):
        """Construct an empty string with a given capacity.

        Args:
            capacity: The capacity of the string to allocate.
        """
        var cap_field = _StringCapacityField(capacity)
        if cap_field.is_inline():
            # Tell mojo it is ok for this to be uninitialized.
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(self)
            )
            self._capacity_or_data = cap_field
        else:
            self._len_or_data = 0
            self._ptr_or_data = _StringOutOfLineHeader.alloc(
                cap_field.capacity()
            )
            self._capacity_or_data = cap_field

    @always_inline
    @implicit  # does not allocate.
    fn __init__(out self, data: StaticString):
        """Construct a string from a static constant string without allocating.

        Args:
            data: The static constant string to refer to.
        """
        var length = data.byte_length()
        var cap_field = _StringCapacityField(static_const_length=length)
        # NOTE: we can't set set_has_nul_terminator(True) because there is no
        # guarantee that this wasn't constructed from a
        # Span[Byte, StaticConstantOrigin] without a nul terminator.
        if cap_field.is_inline():
            # Tell mojo it is ok for this to be uninitialized.
            __mlir_op.`lit.ownership.mark_initialized`(
                __get_mvalue_as_litref(self)
            )
            cap_field.set_len(length, self._len_or_data)
            self._capacity_or_data = cap_field
            memcpy(
                UnsafePointer(to=self).bitcast[Byte](),
                data.unsafe_ptr(),
                length,
            )
        else:
            self._ptr_or_data = data.unsafe_ptr()
            self._len_or_data = length
            self._capacity_or_data = cap_field

    @always_inline
    @implicit  # does not allocate.
    fn __init__(out self, data: StringLiteral):
        """Construct a string from a string literal without allocating.

        Args:
            data: The static constant string to refer to.
        """
        self = StaticString(data)
        # All string literals are nul terminated by the compiler but the inline
        # String overwrites it at any given point
        if not self._is_inline():
            self._capacity_or_data.set_has_nul_terminator(True)

    @always_inline
    fn __init__(out self, *, bytes: Span[Byte, *_]):
        """Construct a string by copying the data. This constructor is explicit
        because it can involve memory allocation.

        Args:
            bytes: The bytes to copy.
        """
        var length = len(bytes)
        self = Self(unsafe_uninit_length=length)
        memcpy(self.unsafe_ptr_mut(), bytes.unsafe_ptr(), length)

    @no_inline
    fn __init__[T: Stringable](out self, value: T):
        """Initialize from a type conforming to `Stringable`.

        Parameters:
            T: The type conforming to Stringable.

        Args:
            value: The object to get the string representation of.
        """
        self = value.__str__()

    @no_inline
    fn __init__[T: StringableRaising](out self, value: T) raises:
        """Initialize from a type conforming to `StringableRaising`.

        Parameters:
            T: The type conforming to Stringable.

        Args:
            value: The object to get the string representation of.

        Raises:
            If there is an error when computing the string representation of the type.
        """
        self = value.__str__()

    @no_inline
    fn __init__[
        *Ts: Writable
    ](out self, *args: *Ts, sep: StaticString = "", end: StaticString = ""):
        """
        Construct a string by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Examples:

        Construct a String from several `Writable` arguments:

        ```mojo
        var string = String(1, 2.0, "three", sep=", ")
        print(string) # "1, 2.0, three"
        ```
        .
        """
        self = String()
        write_buffered(self, args, sep=sep, end=end)

    # TODO(MOCO-1791): Default arguments and param inference aren't powerful
    # to declare sep/end as StringSlice.
    @staticmethod
    @no_inline
    fn __init__[
        *Ts: Writable
    ](
        out self,
        args: VariadicPack[_, _, Writable, *Ts],
        sep: StaticString = "",
        end: StaticString = "",
    ):
        """
        Construct a string by passing a variadic pack.

        Args:
            args: A VariadicPack of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Examples:

        ```mojo
        fn variadic_pack_to_string[
            *Ts: Writable,
        ](*args: *Ts) -> String:
            return String(args)

        string = variadic_pack_to_string(1, ", ", 2.0, ", ", "three")
        %# from testing import assert_equal
        %# assert_equal(string, "1, 2.0, three")
        ```
        .
        """
        self = String()
        write_buffered(self, args, sep=sep, end=end)

    fn copy(self) -> Self:
        """Explicitly copy the provided value.

        Returns:
            A copy of the value.
        """
        return self  # Just use the implicit copyinit.

    fn __init__(out self, *, unsafe_uninit_length: UInt):
        """Construct a String with the specified length, with uninitialized
        memory. This is unsafe, as it relies on the caller initializing the
        elements with unsafe operations, not assigning over the uninitialized
        data.

        Args:
            unsafe_uninit_length: The number of bytes to allocate.
        """
        self = Self(capacity=unsafe_uninit_length)
        self._capacity_or_data.set_len(unsafe_uninit_length, self._len_or_data)

    @always_inline
    fn __init__(
        out self,
        *,
        unsafe_from_utf8_ptr: UnsafePointer[c_char, mut=_, origin=_],
    ):
        """Creates a string from a UTF-8 encoded nul-terminated pointer.

        Args:
            unsafe_from_utf8_ptr: An `UnsafePointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        # Copy the data.
        self = String(
            StringSlice[MutableAnyOrigin](
                unsafe_from_utf8_ptr=unsafe_from_utf8_ptr
            )
        )

    @always_inline
    fn __init__(
        out self, *, unsafe_from_utf8_ptr: UnsafePointer[UInt8, mut=_, origin=_]
    ):
        """Creates a string from a UTF-8 encoded nul-terminated pointer.

        Args:
            unsafe_from_utf8_ptr: An `UnsafePointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        # Copy the data.
        self = String(
            StringSlice[MutableAnyOrigin](
                unsafe_from_utf8_ptr=unsafe_from_utf8_ptr
            )
        )

    @always_inline("nodebug")
    fn __moveinit__(out self, owned other: Self):
        """Move initialize the string from another string.

        Args:
            other: The string to move.
        """
        self._ptr_or_data = other._ptr_or_data
        self._len_or_data = other._len_or_data
        self._capacity_or_data = other._capacity_or_data

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy initialize the string from another string.

        Args:
            other: The string to copy.
        """
        # Keep inline strings inline, and static strings static.
        self._ptr_or_data = other._ptr_or_data
        self._len_or_data = other._len_or_data
        self._capacity_or_data = other._capacity_or_data

        # If the other string is out-of-line and not static, increment the
        # refcount of the out-of-line representation.
        if other._has_mutable_buffer():
            _StringOutOfLineHeader.get(self._ptr_or_data).add_ref()

    # ===------------------------------------------------------------------=== #
    # Field getters
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn capacity(self) -> UInt:
        """Get the capacity of the string.

        Returns:
            The capacity of the string.
        """
        return self._capacity_or_data.capacity()

    @always_inline("nodebug")
    fn _is_inline(self) -> Bool:
        return self._capacity_or_data.is_inline()

    @always_inline("nodebug")
    fn _is_indirect_static_constant(self) -> Bool:
        """Checks if the string is a static constant.

        Returns:
            True if the string is a static constant, False otherwise.
        """
        return Bool(self._ptr_or_data) & (self.capacity() == 0)

    @always_inline("nodebug")
    fn _has_mutable_buffer(self) -> Bool:
        return ~(self._is_indirect_static_constant() | self._is_inline())

    # ===------------------------------------------------------------------=== #
    # Factory dunders
    # ===------------------------------------------------------------------=== #

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """Write a byte span to this String.

        Args:
            bytes: The byte span to write to this String. Must NOT be
                null terminated.
        """
        self._iadd(bytes)

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """

        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)

    @staticmethod
    @no_inline
    fn write[
        *Ts: Writable
    ](*args: *Ts, sep: StaticString = "", end: StaticString = "") -> Self:
        """Construct a string by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Returns:
            A string formed by formatting the argument sequence.

        This is used only when reusing the `write_to` method for
        `__str__` in order to avoid an endless loop recalling
        the constructor:

        ```mojo
        fn write_to[W: Writer](self, mut writer: W):
            writer.write_bytes(self.as_bytes())

        fn __str__(self) -> String:
            return String.write(self)
        ```

        Otherwise you can use the `String` constructor directly without calling
        the `String.write` static method:

        ```mojo
        var msg = String("my message", 42, 42.2, True)
        ```
        """
        var string = String()
        write_buffered(string, args, sep=sep, end=end)
        return string^

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

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
        var normalized_idx = normalize_index["String"](idx, len(self))
        var result = String(capacity=1)
        result.append_byte(self.unsafe_ptr()[normalized_idx])
        return result^

    fn __getitem__(self, span: Slice) -> String:
        """Gets the sequence of characters at the specified positions.

        Args:
            span: A slice that specifies positions of the new substring.

        Returns:
            A new string containing the string at the specified positions.
        """
        var start: Int
        var end: Int
        var step: Int
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time

        start, end, step = span.indices(self.byte_length())
        var r = range(start, end, step)
        if step == 1:
            return String(
                StringSlice(ptr=self.unsafe_ptr() + start, length=len(r))
            )

        var result = String(capacity=len(r))
        var ptr = self.unsafe_ptr()
        for i in r:
            result.append_byte(ptr[i])
        return result^

    @always_inline
    fn __eq__(self, other: String) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return self.as_string_slice() == other.as_string_slice()

    @always_inline
    fn __eq__(self, other: StringSlice) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return self.as_string_slice() == other

    @always_inline
    fn __ne__(self, other: String) -> Bool:
        """Compares two Strings if they do not have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are not equal and False otherwise.
        """
        return not (self == other)

    @always_inline
    fn __ne__(self, other: StringSlice) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return self.as_string_slice() != other

    @always_inline
    fn __lt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String is strictly less than the RHS String and False
            otherwise.
        """
        return self.as_string_slice() < rhs.as_string_slice()

    @always_inline
    fn __le__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LE comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is less than or equal to the RHS String.
        """
        return not (rhs < self)

    @always_inline
    fn __gt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using GT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is strictly greater than the RHS String.
        """
        return rhs < self

    @always_inline
    fn __ge__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using GE comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is greater than or equal to the RHS String.
        """
        return not (self < rhs)

    @staticmethod
    fn _add(lhs: Span[Byte], rhs: Span[Byte]) -> String:
        var lhs_len = len(lhs)
        var rhs_len = len(rhs)

        var result = String(unsafe_uninit_length=lhs_len + rhs_len)
        var result_ptr = result.unsafe_ptr_mut()
        memcpy(result_ptr, lhs.unsafe_ptr(), lhs_len)
        memcpy(result_ptr + lhs_len, rhs.unsafe_ptr(), rhs_len)
        return result^

    @always_inline
    fn __add__(self, other: StringSlice) -> String:
        """Creates a string by appending a string slice at the end.

        Args:
            other: The string slice to append.

        Returns:
            The new constructed string.
        """
        return Self._add(self.as_bytes(), other.as_bytes())

    fn append_byte(mut self, byte: Byte):
        """Append a byte to the string.

        Args:
            byte: The byte to append.
        """
        self._capacity_or_data.set_has_nul_terminator(False)
        var len = self.byte_length()
        self.reserve(len + 1)
        (self.unsafe_ptr_mut() + len).init_pointee_move(byte)
        self._capacity_or_data.set_len(len + 1, self._len_or_data)

    @always_inline
    fn __radd__(self, other: StringSlice[mut=False]) -> String:
        """Creates a string by prepending another string slice to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return Self._add(other.as_bytes(), self.as_bytes())

    fn _iadd(mut self, other: Span[mut=False, Byte]):
        var other_len = len(other)
        if other_len == 0:
            return
        # remove the nul terminator if it exists.
        self._capacity_or_data.set_has_nul_terminator(False)
        var old_len = self.byte_length()
        var new_len = old_len + other_len
        self.reserve(new_len)
        memcpy(self.unsafe_ptr_mut() + old_len, other.unsafe_ptr(), other_len)
        self._capacity_or_data.set_len(new_len, self._len_or_data)

    @always_inline
    fn __iadd__(mut self, other: StringSlice[mut=False]):
        """Appends another string slice to this string.

        Args:
            other: The string to append.
        """
        self._iadd(other.as_bytes())

    @deprecated("Use `str.codepoints()` or `str.codepoint_slices()` instead.")
    fn __iter__(self) -> CodepointSliceIter[__origin_of(self)]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return self.codepoint_slices()

    fn __reversed__(self) -> CodepointSliceIter[__origin_of(self), False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return CodepointSliceIter[__origin_of(self), forward=False](self)

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the string is not empty.

        Returns:
            True if the string length is greater than zero, and False otherwise.
        """
        return self.byte_length() > 0

    @always_inline
    fn __len__(self) -> Int:
        """Get the string length of in bytes.

        This function returns the number of bytes in the underlying UTF-8
        representation of the string.

        To get the number of Unicode codepoints in a string, use
        `len(str.codepoints())`.

        Returns:
            The string length in bytes.

        # Examples

        Query the length of a string, in bytes and Unicode codepoints:

        ```mojo
        from testing import assert_equal

        var s = String("ನಮಸ್ಕಾರ")

        assert_equal(len(s), 21)
        assert_equal(len(s.codepoints()), 7)
        ```

        Strings containing only ASCII characters have the same byte and
        Unicode codepoint length:

        ```mojo
        from testing import assert_equal

        var s = String("abc")

        assert_equal(len(s), 3)
        assert_equal(len(s.codepoints()), 3)
        ```
        .
        """
        return self.byte_length()

    @always_inline
    fn __str__(self) -> String:
        """Gets the string itself.

        This method ensures that you can pass a `String` to a method that
        takes a `Stringable` value.

        Returns:
            The string itself.
        """
        return self

    fn __repr__(self) -> String:
        """Return a Mojo-compatible representation of the `String` instance.

        Returns:
            A new representation of the string.
        """
        return StringSlice(self).__repr__()

    fn __fspath__(self) -> String:
        """Return the file system path representation (just the string itself).

        Returns:
          The file system path representation as a string.
        """
        return self

    fn to_python_object(owned self) -> PythonObject:
        """Convert this value to a PythonObject.

        Returns:
            A PythonObject representing the value.
        """
        return PythonObject(self)

    fn __init__(out self, obj: PythonObject) raises:
        """Construct a `String` from a PythonObject.

        Args:
            obj: The PythonObject to convert from.

        Raises:
            An error if the conversion failed.
        """
        var str_obj = obj.__str__()
        self = String(StringSlice(unsafe_borrowed_obj=str_obj))
        # keep python object alive so the copy can occur
        _ = str_obj

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this string to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write_bytes(self.as_bytes())

    fn join[*Ts: Writable](self, *elems: *Ts) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            Ts: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        var sep = StaticString(ptr=self.unsafe_ptr(), length=len(self))
        return String(elems, sep=sep)

    fn join[
        T: Copyable & Movable & Writable, //, buffer_size: Int = 4096
    ](self, elems: List[T, *_]) -> String:
        """Joins string elements using the current string as a delimiter.
        Defaults to writing to the stack if total bytes of `elems` is less than
        `buffer_size`, otherwise will allocate once to the heap and write
        directly into that. The `buffer_size` defaults to 4096 bytes to match
        the default page size on arm64 and x86-64, but you can increase this if
        you're joining a very large `List` of elements to write into the stack
        instead of the heap.

        Parameters:
            T: The type of the elements. Must implement the `Copyable`,
                `Movable` and `Writable` traits.
            buffer_size: The max size of the stack buffer.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        var result = String()
        if not len(elems):
            return result^
        result.write(elems[0])
        for i in range(1, len(elems)):
            result.write(self)
            result.write(elems[i])
        return result^

    @always_inline
    fn codepoints(self) -> CodepointsIter[__origin_of(self)]:
        """Returns an iterator over the `Codepoint`s encoded in this string slice.

        Returns:
            An iterator type that returns successive `Codepoint` values stored in
            this string slice.

        # Examples

        Print the characters in a string:

        ```mojo
        from testing import assert_equal

        var s = String("abc")
        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
        assert_equal(iter.__next__(), Codepoint.ord("b"))
        assert_equal(iter.__next__(), Codepoint.ord("c"))
        assert_equal(iter.__has_next__(), False)
        ```

        `codepoints()` iterates over Unicode codepoints, and supports multibyte
        codepoints:

        ```mojo
        from testing import assert_equal

        # A visual character composed of a combining sequence of 2 codepoints.
        var s = String("á")
        assert_equal(s.byte_length(), 3)

        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
         # U+0301 Combining Acute Accent
        assert_equal(iter.__next__().to_u32(), 0x0301)
        assert_equal(iter.__has_next__(), False)
        ```
        .
        """
        return self.as_string_slice().codepoints()

    fn codepoint_slices(self) -> CodepointSliceIter[__origin_of(self)]:
        """Returns an iterator over single-character slices of this string.

        Each returned slice points to a single Unicode codepoint encoded in the
        underlying UTF-8 representation of this string.

        Returns:
            An iterator of references to the string elements.

        # Examples

        Iterate over the character slices in a string:

        ```mojo
        from testing import assert_equal, assert_true

        var s = String("abc")
        var iter = s.codepoint_slices()
        assert_true(iter.__next__() == "a")
        assert_true(iter.__next__() == "b")
        assert_true(iter.__next__() == "c")
        assert_equal(iter.__has_next__(), False)
        ```
        .
        """
        return self.as_string_slice().codepoint_slices()

    @always_inline("nodebug")
    fn unsafe_ptr(
        self,
    ) -> UnsafePointer[Byte, mut=False, origin = __origin_of(self)]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        if self._capacity_or_data.is_inline():
            # The string itself holds the data.
            return UnsafePointer(to=self).bitcast[Byte]()
        else:
            return self._ptr_or_data

    fn unsafe_ptr_mut(
        mut self,
    ) -> UnsafePointer[Byte, mut=True, origin = __origin_of(self)]:
        """Retrieves a mutable pointer to the underlying memory, copying to a
        new buffer if this was previously pointing to a static constant.

        Returns:
            The pointer to the underlying memory.
        """
        # If out of line, make sure it is uniquely owned and mutable.
        if not self._is_inline():
            self._make_unique_mutable()
        return self.unsafe_ptr().origin_cast[True, __origin_of(self)]()

    fn unsafe_cstr_ptr(
        mut self,
    ) -> UnsafePointer[c_char, mut=True, origin = __origin_of(self)]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be null, or NUL terminated.

        Returns:
            The pointer to the underlying memory.
        """
        # Add a nul terminator.
        # Reallocate the out-of-line static strings to ensure mutability.
        if not self._capacity_or_data.has_nul_terminator() or (
            self._is_indirect_static_constant()
        ):
            var len = self.byte_length()
            self.reserve(len + 1)  # This will reallocate if constant.
            self.unsafe_ptr_mut()[len] = 0
            self._capacity_or_data.set_has_nul_terminator(True)

        return self.unsafe_ptr_mut().bitcast[c_char]()

    @always_inline
    fn as_bytes(self) -> Span[Byte, __origin_of(self)]:
        """Returns a contiguous slice of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        return Span[Byte, __origin_of(self)](
            ptr=self.unsafe_ptr(), length=self.byte_length()
        )

    @always_inline
    fn as_bytes_mut(mut self) -> Span[Byte, __origin_of(self)]:
        """Returns a mutable contiguous slice of the bytes owned by this string.
        This name has a _mut suffix so the as_bytes() method doesn't have to
        guarantee mutability.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """
        return Span[Byte, __origin_of(self)](
            ptr=self.unsafe_ptr_mut(), length=self.byte_length()
        )

    @always_inline
    fn as_string_slice(self) -> StringSlice[__origin_of(self)]:
        """Returns a string slice of the data owned by this string.

        Returns:
            A string slice pointing to the data owned by this string.
        """
        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in String so this is actually
        #   guaranteed to be valid.
        return StringSlice(unsafe_from_utf8=self.as_bytes())

    @always_inline
    fn as_string_slice_mut(mut self) -> StringSlice[__origin_of(self)]:
        """Returns a mutable string slice of the data owned by this string.

        Returns:
            A string slice pointing to the data owned by this string.
        """
        return StringSlice(unsafe_from_utf8=self.as_bytes_mut())

    @always_inline
    fn byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes.
        """
        return self._capacity_or_data.get_len(self._len_or_data)

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
        return self.as_string_slice().count(substr)

    fn __contains__(self, substr: StringSlice) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return substr in self.as_string_slice()

    fn find(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        return self.as_string_slice().find(substr, start)

    fn rfind(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        return self.as_string_slice().rfind(substr, start=start)

    fn isspace(self) -> Bool:
        """Determines whether every character in the given String is a
        python whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole String is made up of whitespace characters
                listed above, otherwise False.
        """
        return self.as_string_slice().isspace()

    # TODO(MSTDL-590): String.split() should return `StringSlice`s.
    fn split(self, sep: StringSlice, maxsplit: Int = -1) raises -> List[String]:
        """Split the string by a separator.

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
        _ = String("hello world").split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = String("hello,,world").split(",") # ["hello", "", "world"]
        # Splitting with maxsplit
        _ = String("1,2,3").split(",", 1) # ['1', '2,3']
        ```
        .
        """
        return _to_string_list(
            self.as_string_slice().split(sep, maxsplit=maxsplit)
        )

    fn split(self, sep: NoneType = None, maxsplit: Int = -1) -> List[String]:
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
        _ = String("      ").split() # []
        _ = String("").split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = String("      hello    world     ").split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = String(
            "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029world"
        ).split()  # ["hello", "world"]
        ```
        .
        """
        return _to_string_list(
            self.as_string_slice().split(sep, maxsplit=maxsplit)
        )

    fn splitlines(self, keepends: Bool = False) -> List[String]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        return _to_string_list(self.as_string_slice().splitlines(keepends))

    fn replace(self, old: StringSlice, new: StringSlice) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        return StringSlice(self).replace(old, new)

    fn strip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading or trailing characters.
        """

        return self.lstrip(chars).rstrip(chars)

    fn strip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading and trailing whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no leading or trailing whitespaces.
        """
        return self.lstrip().rstrip()

    fn rstrip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no trailing characters.
        """

        return self.as_string_slice().rstrip(chars)

    fn rstrip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with trailing whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no trailing whitespaces.
        """
        return self.as_string_slice().rstrip()

    fn lstrip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.
        """

        return self.as_string_slice().lstrip(chars)

    fn lstrip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A copy of the string with no leading whitespaces.
        """
        return self.as_string_slice().lstrip()

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.as_string_slice())

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_bytes(self.unsafe_ptr(), self.byte_length())

    fn lower(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        return self.as_string_slice().lower()

    fn upper(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        return self.as_string_slice().upper()

    fn startswith(
        self, prefix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        Args:
            prefix: The prefix to check.
            start: The start offset from which to check.
            end: The end offset from which to check.

        Returns:
            True if the `self[start:end]` is prefixed by the input prefix.
        """
        return self.as_string_slice().startswith(prefix, start, end)

    fn endswith(
        self, suffix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        Args:
            suffix: The suffix to check.
            start: The start offset from which to check.
            end: The end offset from which to check.

        Returns:
            True if the `self[start:end]` is suffixed by the input suffix.
        """
        return self.as_string_slice().endswith(suffix, start, end)

    @always_inline
    fn removeprefix(
        self, prefix: StringSlice, /
    ) -> StringSlice[__origin_of(self)]:
        """Returns a new string with the prefix removed if it was present.

        Args:
            prefix: The prefix to remove from the string.

        Returns:
            `string[len(prefix):]` if the string starts with the prefix string,
            or a copy of the original string otherwise.

        Examples:

        ```mojo
        print(String('TestHook').removeprefix('Test')) # 'Hook'
        print(String('BaseTestCase').removeprefix('Test')) # 'BaseTestCase'
        ```
        """
        return self.as_string_slice().removeprefix(prefix)

    @always_inline
    fn removesuffix(
        self, suffix: StringSlice, /
    ) -> StringSlice[__origin_of(self)]:
        """Returns a new string with the suffix removed if it was present.

        Args:
            suffix: The suffix to remove from the string.

        Returns:
            `string[:-len(suffix)]` if the string ends with the suffix string,
            or a copy of the original string otherwise.

        Examples:

        ```mojo
        print(String('TestHook').removesuffix('Hook')) # 'Test'
        print(String('BaseTestCase').removesuffix('Test')) # 'BaseTestCase'
        ```
        """
        return self.as_string_slice().removesuffix(suffix)

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

    fn __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n : The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """
        return self.as_string_slice() * n

    @always_inline
    fn format[*Ts: _CurlyEntryFormattable](self, *args: *Ts) raises -> String:
        """Produce a formatted string using the current string as a template.

        The template, or "format string" can contain literal text and/or
        replacement fields delimited with curly braces (`{}`). Returns a copy of
        the format string with the replacement fields replaced with string
        representations of the `args` arguments.

        For more information, see the discussion in the
        [`format` module](/mojo/stdlib/collections/string/format/).

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Representable`
                and `Stringable` (to be changed and made more flexible).

        Returns:
            The template with the given values substituted.

        Example:

        ```mojo
        # Manual indexing:
        print(String("{0} {1} {0}").format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print(String("{} {}").format(True, "hello world")) # True hello world
        ```
        """
        return _FormatCurlyEntry.format(self, args)

    fn isdigit(self) -> Bool:
        """A string is a digit string if all characters in the string are digits
        and there is at least one character in the string.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits and it's not empty else False.
        """
        return self.as_string_slice().is_ascii_digit()

    fn isupper(self) -> Bool:
        """Returns True if all cased characters in the string are uppercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are uppercase and there
            is at least one cased character, False otherwise.
        """
        return self.as_string_slice().isupper()

    fn islower(self) -> Bool:
        """Returns True if all cased characters in the string are lowercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are lowercase and there
            is at least one cased character, False otherwise.
        """
        return self.as_string_slice().islower()

    fn isprintable(self) -> Bool:
        """Returns True if all characters in the string are ASCII printable.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are printable else False.
        """
        return self.as_string_slice().is_ascii_printable()

    fn rjust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string right justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns right justified string, or self if width is not bigger than self length.
        """
        return self.as_string_slice().rjust(width, fillchar)

    fn ljust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string left justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns left justified string, or self if width is not bigger than self length.
        """
        return self.as_string_slice().ljust(width, fillchar)

    fn center(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string center justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns center justified string, or self if width is not bigger than self length.
        """
        return self.as_string_slice().center(width, fillchar)

    fn resize(mut self, length: Int, fill_byte: UInt8 = 0):
        """Resize the string to a new length.

        Args:
            length: The new length of the string.
            fill_byte: The byte to fill any new space with.

        Notes:
            If the new length is greater than the current length, the string is
            extended by the difference, and the new bytes are initialized to
            `fill_byte`.
        """
        self._capacity_or_data.set_has_nul_terminator(False)
        var old_len = self.byte_length()
        if length > old_len:
            self.reserve(length)
            memset(self.unsafe_ptr_mut() + old_len, fill_byte, length - old_len)
        self._capacity_or_data.set_len(length, self._len_or_data)

    @always_inline
    fn resize(mut self, *, unsafe_uninit_length: Int):
        """Resizes the string to the given new size leaving any new data
        uninitialized.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        string is extended and the new data is left uninitialized.

        Args:
            unsafe_uninit_length: The new size.
        """
        self._capacity_or_data.set_has_nul_terminator(False)
        if unsafe_uninit_length > self.capacity():
            self.reserve(unsafe_uninit_length)
        self._capacity_or_data.set_len(unsafe_uninit_length, self._len_or_data)

    @always_inline
    fn reserve(mut self, new_capacity: UInt):
        """Reserves the requested capacity.

        Args:
            new_capacity: The new capacity in stored bytes.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the data is moved.
        """
        if new_capacity <= self.capacity():
            return
        self._realloc_mutable(new_capacity)

    # This is called when the string is known to be indirect. This checks to
    # make sure the indirect representation is uniquely owned and mutable,
    # copying if necessary.
    fn _make_unique_mutable(mut self):
        debug_assert(not self._is_inline())

        # If already mutable and uniquely owned, we're done.
        if (
            not self._is_indirect_static_constant()
            and _StringOutOfLineHeader.get(self._ptr_or_data).is_unique()
        ):
            return

        # Otherwise, copy to a new buffer to ensure mutability.
        self._realloc_mutable(self.byte_length())

    # This is the out-of-line implementation of reserve called when we need
    # to grow the capacity of the string. Make sure our capacity at least
    # doubles to avoid O(n^2) behavior, and make use of extra space if it exists.
    @no_inline
    fn _realloc_mutable(mut self, capacity: UInt):
        # Get these fields before we change _capacity_or_data
        var len = self.byte_length()
        var ptr = self.unsafe_ptr()
        var should_drop_ref = self._has_mutable_buffer()

        # We always use the inline representation for short strings (even when
        # they are constant) so any need to grow will use an indirect represent.
        var new_capacity = _StringCapacityField(
            out_of_line_capacity=max(capacity, self.capacity() * 2)
        )
        var new_ptr = _StringOutOfLineHeader.alloc(new_capacity.capacity())
        memcpy(new_ptr, ptr, len)
        if should_drop_ref:
            _StringOutOfLineHeader.get(ptr.origin_cast[mut=True]()).drop_ref()

        self._len_or_data = len
        self._ptr_or_data = new_ptr
        self._capacity_or_data = new_capacity


# ===----------------------------------------------------------------------=== #
# ord
# ===----------------------------------------------------------------------=== #


fn ord(s: StringSlice) -> Int:
    """Returns an integer that represents the codepoint of a single-character
    string.

    Given a string containing a single character `Codepoint`, return an integer
    representing the codepoint of that character. For example, `ord("a")`
    returns the integer `97`. This is the inverse of the `chr()` function.

    This function is in the prelude, so you don't need to import it.

    Args:
        s: The input string, which must contain only a single- character.

    Returns:
        An integer representing the code point of the given character.
    """
    return Int(Codepoint.ord(s))


# ===----------------------------------------------------------------------=== #
# chr
# ===----------------------------------------------------------------------=== #


fn chr(c: Int) -> String:
    """Returns a String based on the given Unicode code point. This is the
    inverse of the `ord()` function.

    This function is in the prelude, so you don't need to import it.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.

    Example:
    ```mojo
    print(chr(97), chr(8364)) # "a €"
    ```
    """

    if c < 0b1000_0000:  # 1 byte ASCII char
        var str = String(capacity=1)
        str.append_byte(c)
        return str^

    var char_opt = Codepoint.from_u32(c)
    if not char_opt:
        # TODO: Raise ValueError instead.
        return abort[String](
            String("chr(", c, ") is not a valid Unicode codepoint")
        )

    # SAFETY: We just checked that `char` is present.
    return String(char_opt.unsafe_value())


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
    return result


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
    elif Codepoint(c).is_ascii_printable():
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


# ===----------------------------------------------------------------------=== #
# atol
# ===----------------------------------------------------------------------=== #


fn atol(str_slice: StringSlice, base: Int = 10) raises -> Int:
    """Parses and returns the given string as an integer in the given base.

    If base is set to 0, the string is parsed as an integer literal, with the
    following considerations:
    - '0b' or '0B' prefix indicates binary (base 2)
    - '0o' or '0O' prefix indicates octal (base 8)
    - '0x' or '0X' prefix indicates hexadecimal (base 16)
    - Without a prefix, it's treated as decimal (base 10)

    This follows [Python's integer literals format](
    https://docs.python.org/3/reference/lexical_analysis.html#integers).

    This function is in the prelude, so you don't need to import it.

    Args:
        str_slice: A string to be parsed as an integer in the given base.
        base: Base used for conversion, value must be between 2 and 36, or 0.

    Returns:
        An integer value that represents the string.

    Raises:
        If the given string cannot be parsed as an integer value or if an
        incorrect base is provided.

    Examples:

    ```text
    >>> atol("32")
    32
    >>> atol("FF", 16)
    255
    >>> atol("0xFF", 0)
    255
    >>> atol("0b1010", 0)
    10
    ```
    """

    if (base != 0) and (base < 2 or base > 36):
        raise Error("Base must be >= 2 and <= 36, or 0.")
    if not str_slice:
        raise Error(_str_to_base_error(base, str_slice))

    var real_base: Int
    var ord_num_max: Int

    var ord_letter_max = (-1, -1)
    var result = 0
    var is_negative: Bool
    var has_prefix: Bool
    var start: Int
    var str_len = str_slice.byte_length()

    start, is_negative = _trim_and_handle_sign(str_slice, str_len)

    alias ord_0 = ord("0")
    alias ord_letter_min = (ord("a"), ord("A"))
    alias ord_underscore = ord("_")

    if base == 0:
        var real_base_new_start = _identify_base(str_slice, start)
        real_base = real_base_new_start[0]
        start = real_base_new_start[1]
        has_prefix = real_base != 10
        if real_base == -1:
            raise Error(_str_to_base_error(base, str_slice))
    else:
        start, has_prefix = _handle_base_prefix(start, str_slice, str_len, base)
        real_base = base

    if real_base <= 10:
        ord_num_max = ord(String(real_base - 1))
    else:
        ord_num_max = ord("9")
        ord_letter_max = (
            ord("a") + (real_base - 11),
            ord("A") + (real_base - 11),
        )

    var buff = str_slice.unsafe_ptr()
    var found_valid_chars_after_start = False
    var has_space_after_number = False

    # Prefixed integer literals with real_base 2, 8, 16 may begin with leading
    # underscores under the conditions they have a prefix
    var was_last_digit_underscore = not (real_base in (2, 8, 16) and has_prefix)
    for pos in range(start, str_len):
        var ord_current = Int(buff[pos])
        if ord_current == ord_underscore:
            if was_last_digit_underscore:
                raise Error(_str_to_base_error(base, str_slice))
            else:
                was_last_digit_underscore = True
                continue
        else:
            was_last_digit_underscore = False
        if ord_0 <= ord_current <= ord_num_max:
            result += ord_current - ord_0
            found_valid_chars_after_start = True
        elif ord_letter_min[0] <= ord_current <= ord_letter_max[0]:
            result += ord_current - ord_letter_min[0] + 10
            found_valid_chars_after_start = True
        elif ord_letter_min[1] <= ord_current <= ord_letter_max[1]:
            result += ord_current - ord_letter_min[1] + 10
            found_valid_chars_after_start = True
        elif Codepoint(UInt8(ord_current)).is_posix_space():
            has_space_after_number = True
            start = pos + 1
            break
        else:
            raise Error(_str_to_base_error(base, str_slice))
        if pos + 1 < str_len and not Codepoint(buff[pos + 1]).is_posix_space():
            var nextresult = result * real_base
            if nextresult < result:
                raise Error(
                    _str_to_base_error(base, str_slice)
                    + " String expresses an integer too large to store in Int."
                )
            result = nextresult

    if was_last_digit_underscore or (not found_valid_chars_after_start):
        raise Error(_str_to_base_error(base, str_slice))

    if has_space_after_number:
        for pos in range(start, str_len):
            if not Codepoint(buff[pos]).is_posix_space():
                raise Error(_str_to_base_error(base, str_slice))
    if is_negative:
        result = -result
    return result


@always_inline
fn _trim_and_handle_sign(str_slice: StringSlice, str_len: Int) -> (Int, Bool):
    """Trims leading whitespace, handles the sign of the number in the string.

    Args:
        str_slice: A StringSlice containing the number to parse.
        str_len: The length of the string.

    Returns:
        A tuple containing:
        - The starting index of the number after whitespace and sign.
        - A boolean indicating whether the number is negative.
    """
    var buff = str_slice.unsafe_ptr()
    var start: Int = 0
    while start < str_len and Codepoint(buff[start]).is_posix_space():
        start += 1
    var p: Bool = buff[start] == ord("+")
    var n: Bool = buff[start] == ord("-")
    return start + (Int(p) or Int(n)), n


@always_inline
fn _handle_base_prefix(
    pos: Int, str_slice: StringSlice, str_len: Int, base: Int
) -> (Int, Bool):
    """Adjusts the starting position if a valid base prefix is present.

    Handles "0b"/"0B" for base 2, "0o"/"0O" for base 8, and "0x"/"0X" for base
    16. Only adjusts if the base matches the prefix.

    Args:
        pos: Current position in the string.
        str_slice: The input StringSlice.
        str_len: Length of the input string.
        base: The specified base.

    Returns:
        A tuple containing:
            - Updated position after the prefix, if applicable.
            - A boolean indicating if the prefix was valid for the given base.
    """
    var start = pos
    var buff = str_slice.unsafe_ptr()
    if start + 1 < str_len:
        var prefix_char = chr(Int(buff[start + 1]))
        if buff[start] == ord("0") and (
            (base == 2 and (prefix_char == "b" or prefix_char == "B"))
            or (base == 8 and (prefix_char == "o" or prefix_char == "O"))
            or (base == 16 and (prefix_char == "x" or prefix_char == "X"))
        ):
            start += 2
    return start, start != pos


fn _str_to_base_error(base: Int, str_slice: StringSlice) -> String:
    return String(
        "String is not convertible to integer with base ",
        base,
        ": '",
        str_slice,
        "'",
    )


fn _identify_base(str_slice: StringSlice, start: Int) -> Tuple[Int, Int]:
    var length = str_slice.byte_length()
    # just 1 digit, assume base 10
    if start == (length - 1):
        return 10, start
    if str_slice[start] == "0":
        var second_digit = str_slice[start + 1]
        if second_digit == "b" or second_digit == "B":
            return 2, start + 2
        if second_digit == "o" or second_digit == "O":
            return 8, start + 2
        if second_digit == "x" or second_digit == "X":
            return 16, start + 2
        # checking for special case of all "0", "_" are also allowed
        var was_last_character_underscore = False
        for i in range(start + 1, length):
            if str_slice[i] == "_":
                if was_last_character_underscore:
                    return -1, -1
                else:
                    was_last_character_underscore = True
                    continue
            else:
                was_last_character_underscore = False
            if str_slice[i] != "0":
                return -1, -1
    elif ord("1") <= ord(str_slice[start]) <= ord("9"):
        return 10, start
    else:
        return -1, -1

    return 10, start


fn _atof_error[reason: StaticString = ""](str_ref: StringSlice) -> Error:
    @parameter
    if reason:
        return Error(
            "String is not convertible to float: '",
            str_ref,
            "' because ",
            reason,
        )
    return Error("String is not convertible to float: '", str_ref, "'")


fn atof(str_slice: StringSlice) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    This function is in the prelude, so you don't need to import it.

    Raises:
        If the given string cannot be parsed as an floating point value, for
        example in `atof("hi")`.

    Args:
        str_slice: A string to be parsed as a floating point.

    Returns:
        An floating point value that represents the string, or otherwise raises.
    """
    return _atof(str_slice)


# ===----------------------------------------------------------------------=== #
# Other utilities
# ===----------------------------------------------------------------------=== #


fn _toggle_ascii_case(char: UInt8) -> UInt8:
    """Assuming char is a cased ASCII character, this function will return the
    opposite-cased letter.
    """

    # ASCII defines A-Z and a-z as differing only in their 6th bit,
    # so converting is as easy as a bit flip.
    return char ^ (1 << 5)


fn _calc_initial_buffer_size_int32(n0: Int) -> Int:
    # See https://commaok.xyz/post/lookup_tables/ and
    # https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
    # for a description.
    alias lookup_table = VariadicList[Int](
        4294967296,
        8589934582,
        8589934582,
        8589934582,
        12884901788,
        12884901788,
        12884901788,
        17179868184,
        17179868184,
        17179868184,
        21474826480,
        21474826480,
        21474826480,
        21474826480,
        25769703776,
        25769703776,
        25769703776,
        30063771072,
        30063771072,
        30063771072,
        34349738368,
        34349738368,
        34349738368,
        34349738368,
        38554705664,
        38554705664,
        38554705664,
        41949672960,
        41949672960,
        41949672960,
        42949672960,
        42949672960,
    )
    var n = UInt32(n0)
    var log2 = Int(
        (bitwidthof[DType.uint32]() - 1) ^ count_leading_zeros(n | 1)
    )
    return (n0 + lookup_table[Int(log2)]) >> 32


fn _calc_initial_buffer_size_int64(n0: UInt64) -> Int:
    var result: Int = 1
    var n = n0
    while True:
        if n < 10:
            return result
        if n < 100:
            return result + 1
        if n < 1_000:
            return result + 2
        if n < 10_000:
            return result + 3
        n //= 10_000
        result += 4


fn _calc_initial_buffer_size(n0: Int) -> Int:
    var sign = 0 if n0 > 0 else 1

    # Add 1 for the terminator
    return sign + n0._decimal_digit_count() + 1


fn _calc_initial_buffer_size(n: Float64) -> Int:
    return 128 + 1  # Add 1 for the terminator


fn _calc_initial_buffer_size[dtype: DType](n0: Scalar[dtype]) -> Int:
    @parameter
    if dtype.is_integral():
        var n = abs(n0)
        var sign = 0 if n0 > 0 else 1
        alias is_32bit_system = Int.BITWIDTH == 32

        @parameter
        if is_32bit_system or bitwidthof[dtype]() <= 32:
            return sign + _calc_initial_buffer_size_int32(Int(n)) + 1
        else:
            return (
                sign
                + _calc_initial_buffer_size_int64(n.cast[DType.uint64]())
                + 1
            )

    return 128 + 1  # Add 1 for the terminator


fn _calc_format_buffer_size[dtype: DType]() -> Int:
    """Returns a buffer size in bytes that is large enough to store a formatted
    number of the specified dtype.
    """

    # TODO:
    #   Use a smaller size based on the `dtype`, e.g. we don't need as much
    #   space to store a formatted int8 as a float64.
    @parameter
    if dtype.is_integral():
        return 64 + 1
    else:
        return 128 + 1  # Add 1 for the terminator
