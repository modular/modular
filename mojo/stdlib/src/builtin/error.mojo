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
"""Implements the Error class.

These are Mojo built-ins, so you don't need to import them.
"""

from collections.string import StringSlice, StaticString
from sys import alignof, sizeof
from sys.ffi import c_char

from memory import UnsafePointer, memcpy
from memory.memory import _free

from collections.string.format import _CurlyEntryFormattable

from utils.write import write_buffered

# ===-----------------------------------------------------------------------===#
# Error
# ===-----------------------------------------------------------------------===#


@register_passable
struct Error(
    Stringable,
    Boolable,
    Representable,
    Writable,
    CollectionElement,
    CollectionElementNew,
    _CurlyEntryFormattable,
):
    """This type represents an Error."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var data: UnsafePointer[UInt8]
    """A pointer to the beginning of the string data being referenced."""

    var loaded_length: Int
    """The length of the string being referenced.
    Error instances conditionally own their error message. To reduce
    the size of the error instance we use the sign bit of the length field
    to store the ownership value. When loaded_length is negative it indicates
    ownership and a free is executed in the destructor.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Default constructor."""
        self.data = UnsafePointer[UInt8]()
        self.loaded_length = 0

    @always_inline
    @implicit
    fn __init__(out self, value: StringLiteral):
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.
        """
        self.data = value.unsafe_ptr()
        self.loaded_length = len(value)

    @implicit
    fn __init__(out self, src: String):
        """Construct an Error object with a given string.

        Args:
            src: The error message.
        """
        var length = src.byte_length()
        var dest = UnsafePointer[UInt8].alloc(length + 1)
        memcpy(
            dest=dest,
            src=src.unsafe_ptr(),
            count=length,
        )
        dest[length] = 0
        self.data = dest
        self.loaded_length = -length

    @implicit
    fn __init__(out self, src: StringSlice):
        """Construct an Error object with a given string ref.

        Args:
            src: The error message.
        """
        var length = len(src)
        var dest = UnsafePointer[UInt8].alloc(length + 1)
        memcpy(
            dest=dest,
            src=src.unsafe_ptr(),
            count=length,
        )
        dest[length] = 0
        self.data = dest
        self.loaded_length = -length

    @no_inline
    fn __init__[
        *Ts: Writable
    ](out self, *args: *Ts, sep: StaticString = "", end: StaticString = ""):
        """
        Construct an Error by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.
        """
        var output = String()
        write_buffered(output, args, sep=sep, end=end)
        self = Error(output)

    fn copy(self) -> Self:
        """Copy the object.

        Returns:
            A copy of the value.
        """
        return self

    fn __del__(owned self):
        """Releases memory if allocated."""
        if self.loaded_length < 0:
            self.data.free()

    fn __copyinit__(out self, existing: Self):
        """Creates a deep copy of an existing error.

        Args:
            existing: The error to copy from.
        """
        if existing.loaded_length < 0:
            var length = -existing.loaded_length
            var dest = UnsafePointer[UInt8].alloc(length + 1)
            memcpy(dest, existing.data, length)
            dest[length] = 0
            self.data = dest
        else:
            self.data = existing.data
        self.loaded_length = existing.loaded_length

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __bool__(self) -> Bool:
        """Returns True if the error is set and false otherwise.

        Returns:
          True if the error object contains a value and False otherwise.
        """
        return self.data.__bool__()

    @no_inline
    fn __str__(self) -> String:
        """Converts the Error to string representation.

        Returns:
            A String of the error message.
        """
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this error to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        if not self:
            return
        writer.write(
            StringSlice[__origin_of(self)](
                unsafe_from_utf8_cstr_ptr=self.unsafe_cstr_ptr()
            )
        )

    @no_inline
    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        return String(
            "Error(",
            repr(
                String(
                    StringSlice[__origin_of(self)](
                        unsafe_from_utf8_cstr_ptr=self.unsafe_cstr_ptr()
                    )
                )
            ),
            ")",
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn unsafe_cstr_ptr(self) -> UnsafePointer[c_char]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be NUL terminated, and not null.

        Returns:
            The pointer to the underlying memory.
        """
        return self.data.bitcast[c_char]()


@doc_private
fn __mojo_debugger_raise_hook():
    """This function is used internally by the Mojo Debugger."""
    pass
