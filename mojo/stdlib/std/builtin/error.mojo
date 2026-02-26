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
"""Implements the Error class.

These are Mojo built-ins, so you don't need to import them.
"""

from collections.string.string_slice import _unsafe_strlen
from format._utils import FormatStruct
from memory import (
    ArcPointer,
    OwnedPointer,
    alloc,
    memcpy,
)
from ffi import external_call
import format._utils as fmt
from sys import is_gpu
from sys.info import size_of, align_of


# ===-----------------------------------------------------------------------===#
# StackTrace
# ===-----------------------------------------------------------------------===#


struct StackTrace(Copyable, Movable, Writable):
    """Holds a stack trace captured at a specific location.

    A `StackTrace` instance always contains a valid stack trace. Use the
    `collect_if_enabled()` static method to conditionally capture a stack
    trace, which returns `None` if stack trace collection is disabled or
    unavailable.
    """

    var _data: OwnedPointer[UInt8]
    """An owned pointer to a null-terminated C string containing the stack trace."""

    fn __init__(
        out self,
        *,
        unsafe_from_raw_pointer: UnsafePointer[UInt8, MutExternalOrigin],
    ):
        """Construct a StackTrace from a raw pointer to a C string.

        Args:
            unsafe_from_raw_pointer: A pointer to a null-terminated C string
                containing the stack trace. The StackTrace takes ownership.

        Safety:
            The pointer must be valid and point to a null-terminated string.
            The caller transfers ownership to this StackTrace.
        """
        self._data = OwnedPointer(
            unsafe_from_raw_pointer=unsafe_from_raw_pointer
        )

    fn __init__(out self, *, copy: Self):
        """Copy constructor - copies the stack trace string.

        Args:
            copy: The existing StackTrace to copy from.
        """
        # Copy the null-terminated string
        var src_ptr = copy._data.unsafe_ptr()
        var str_len = Int(_unsafe_strlen(src_ptr))
        var new_ptr = alloc[UInt8](str_len + 1)
        memcpy(dest=new_ptr, src=src_ptr, count=str_len + 1)
        self._data = OwnedPointer(unsafe_from_raw_pointer=new_ptr)

    fn __init__(out self, *, deinit take: Self):
        """Move constructor.

        Args:
            take: The existing StackTrace to move from.
        """
        self._data = take._data^

    @staticmethod
    @no_inline
    fn collect_if_enabled(depth: Int = 0) -> Optional[StackTrace]:
        """Collect a stack trace if enabled by environment variable.

        This method checks the `MOJO_ENABLE_STACK_TRACE_ON_ERROR` environment
        variable and collects a stack trace only if it is enabled. Returns
        `None` if stack traces are disabled, on GPU, or if collection fails.

        Args:
            depth: The maximum depth of the stack trace to collect.
                   When `depth` is zero, the entire stack trace is collected.
                   When `depth` is negative, no stack trace is collected.

        Returns:
            An `Optional[StackTrace]` containing the stack trace if collection
            succeeded, or `None` if disabled or unavailable.
        """

        comptime if is_gpu():
            return None

        if depth < 0:
            return None

        var buffer = UnsafePointer[UInt8, MutExternalOrigin]()
        var num_bytes = external_call["KGEN_CompilerRT_GetStackTrace", Int](
            UnsafePointer(to=buffer), depth
        )
        # When num_bytes is zero, the stack trace was not collected.
        if num_bytes == 0:
            return None

        return StackTrace(unsafe_from_raw_pointer=buffer)

    @deprecated("Stringable is deprecated. Use Writable instead.")
    fn __str__(self) -> String:
        """Converts the StackTrace to string representation.

        Returns:
            A String of the stack trace.
        """
        return String(unsafe_from_utf8_ptr=self._data.unsafe_ptr())

    fn write_to(self, mut writer: Some[Writer]):
        """Writes the StackTrace to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write_string(
            StringSlice(unsafe_from_utf8_ptr=self._data.unsafe_ptr())
        )

    fn write_repr_to(self, mut writer: Some[Writer]):
        """Writes the StackTrace to the provided Writer in repr format.

        Args:
            writer: The object to write to.
        """
        FormatStruct(writer, "StackTrace").fields(self)


# ===-----------------------------------------------------------------------===#
# Error
# ===-----------------------------------------------------------------------===#


struct Error(
    Copyable,
    Writable,
):
    """This type represents an Error."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _error: String
    """The backing error message."""

    var _stack_trace: Optional[StackTrace]
    """The stack trace of the error, if collected.

    By default, stack trace is collected for errors created from string
    literals. Stack trace collection can be controlled via the
    `MOJO_ENABLE_STACK_TRACE_ON_ERROR` environment variable.
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
            depth: The depth of the stack trace to collect. When negative,
                no stack trace is collected.
        """
        self._error = value^
        self._stack_trace = StackTrace.collect_if_enabled(depth)

    @always_inline
    @implicit
    fn __init__(out self, value: StringLiteral):
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.
        """
        self._error = String(value)
        self._stack_trace = StackTrace.collect_if_enabled(0)

    @no_inline
    @implicit
    fn __init__(out self, value: Some[Writable]):
        """Construct an Error object from a Writable argument.

        Args:
            value: The Writable argument to store in the error message.
        """
        self._error = String(value)
        self._stack_trace = StackTrace.collect_if_enabled(0)

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

    @deprecated("Stringable is deprecated. Use Writable instead.")
    @no_inline
    fn __str__(self) -> String:
        """Converts the Error to string representation.

        Returns:
            A String of the error message.
        """
        return self._error

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self._error.write_to(writer)

    @no_inline
    fn write_repr_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "Error").fields(fmt.Repr(self._error))

    @deprecated("Representable is deprecated. Use Writable instead.")
    @no_inline
    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        var output = String()
        self.write_repr_to(output)
        return output^

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn get_stack_trace(self) -> Optional[String]:
        """Returns the stack trace of the error, if available.

        Returns:
            An `Optional[String]` containing the stack trace if one was
            collected, or `None` if stack trace collection was disabled
            or unavailable.
        """
        if self._stack_trace:
            return String(self._stack_trace.value())
        return None


@doc_private
fn __mojo_debugger_raise_hook():
    """This function is used internally by the Mojo Debugger."""
    pass
