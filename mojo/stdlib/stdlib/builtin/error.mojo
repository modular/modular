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


from collections.string.format import _CurlyEntryFormattable
from io.write import _WriteBufferStack
from sys import _libc, external_call, is_gpu
from sys.ffi import c_char, CStringSlice

from memory import (
    ArcPointer,
    OwnedPointer,
    memcpy,
)
from io.write import _WriteBufferStack, _TotalWritableBytes


# ===-----------------------------------------------------------------------===#
# StackTrace
# ===-----------------------------------------------------------------------===#
@register_passable
struct StackTrace(ImplicitlyCopyable, Stringable):
    """Holds a stack trace of a location when StackTrace is constructed."""

    var value: ArcPointer[OwnedPointer[UInt8]]
    """A reference counting pointer to a char array containing the stack trace.

        Note: This owned pointer _can be null_. We'd use Optional[OwnedPointer] but
        we don't have good niche optimization and Optional[T] requires T: Copyable
    """

    @no_inline
    fn __init__(out self):
        """Construct an empty stack trace."""
        self.value = ArcPointer(OwnedPointer[UInt8](unsafe_from_raw_pointer={}))

    @no_inline
    fn __init__(out self, *, depth: Int):
        """Construct a new stack trace.

        Args:
            depth: The depth of the stack trace.
                   When `depth` is zero, entire stack trace is collected.
                   When `depth` is negative, no stack trace is collected.
        """

        @parameter
        if is_gpu():
            self = StackTrace()
            return

        if depth < 0:
            self = StackTrace()
            return

        var buffer = UnsafePointer[UInt8, MutOrigin.external]()
        var num_bytes = external_call["KGEN_CompilerRT_GetStackTrace", Int](
            UnsafePointer(to=buffer), depth
        )
        # When num_bytes is zero, the stack trace was not collected.
        if num_bytes == 0:
            self.value = ArcPointer(
                OwnedPointer[UInt8](unsafe_from_raw_pointer={})
            )
            return

        self.value = ArcPointer[OwnedPointer[UInt8]](
            OwnedPointer(unsafe_from_raw_pointer=buffer)
        )

    fn __str__(self) -> String:
        """Converts the StackTrace to string representation.

        Returns:
            A String of the stack trace.
        """
        if not self.value[].unsafe_ptr():
            return (
                "stack trace was not collected. Enable stack trace collection"
                " with environment variable `MOJO_ENABLE_STACK_TRACE_ON_ERROR`"
            )
        return String(unsafe_from_utf8_ptr=self.value[].unsafe_ptr())


# ===-----------------------------------------------------------------------===#
# Error
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ErrorWriter(Writer):
    var data: List[Byte]

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        self.data.extend(bytes)

    fn write[*Ts: Writable](mut self, *args: *Ts):
        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)


struct Error(
    Boolable,
    Defaultable,
    ImplicitlyCopyable,
    Representable,
    Stringable,
    Writable,
    _CurlyEntryFormattable,
):
    """This type represents an Error."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var data: String
    """The message of the error."""

    var stack_trace: StackTrace
    """The stack trace of the error.
    By default the stack trace is not collected for the Error, unless user
    sets the stack_trace_depth parameter to value >= 0.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    @implicit
    fn __init__(out self, var value: String, *, depth: Int = -1):
        """Construct an Error object with a given String.

        Args:
            value: The error message.
            depth: The depth of the stack trace to collect.
        """
        self.data = value^
        self.stack_trace = StackTrace(depth=depth)

    @always_inline
    fn __init__(out self):
        """Default constructor."""
        self = Error(String())

    @always_inline
    @implicit
    fn __init__(out self, value: StringLiteral):
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.
        """
        self = Error(String(value), depth=0)

    @no_inline
    @implicit
    fn __init__(out self, arg: Some[Writable]):
        """Construct an Error from a Writable argument.

        Args:
            arg: A Writable argument.
        """
        self = Error(String(arg), depth=0)

    @no_inline
    fn __init__[*Ts: Writable](out self, *args: *Ts):
        """Construct an Error by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.
        """
        self = Error(String(args), depth=0)

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
        return self.data

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        if not self:
            return
        writer.write(self.data)

    @no_inline
    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        return String("Error('", self.data, "')")

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn get_stack_trace(self) -> StackTrace:
        """Returns the stack trace of the error.

        Returns:
            The stringable stack trace of the error.
        """
        return self.stack_trace


@doc_private
fn __mojo_debugger_raise_hook():
    """This function is used internally by the Mojo Debugger."""
    pass
