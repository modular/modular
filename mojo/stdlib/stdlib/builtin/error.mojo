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
from sys.ffi import c_char

from memory import ArcPointer, memcpy, OwnedPointer
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

    @always_inline("nodebug")
    fn __init__(out self):
        """Construct an empty stack trace."""
        self.value = ArcPointer(
            OwnedPointer[UInt8](unsafe_from_raw_pointer=UnsafePointer[UInt8]())
        )

    @always_inline("nodebug")
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

        var buffer = UnsafePointer[UInt8]()
        var num_bytes = external_call["KGEN_CompilerRT_GetStackTrace", Int](
            UnsafePointer(to=buffer), depth
        )
        # When num_bytes is zero, the stack trace was not collected.
        if num_bytes == 0:
            self.value = ArcPointer(
                OwnedPointer(unsafe_from_raw_pointer=UnsafePointer[UInt8]())
            )
            return

        var ptr = UnsafePointer[UInt8]().alloc(num_bytes + 1)
        ptr.store(num_bytes, 0)
        self.value = ArcPointer[OwnedPointer[UInt8]](
            OwnedPointer(unsafe_from_raw_pointer=ptr)
        )
        memcpy(dest=self.value[].unsafe_ptr(), src=buffer, count=num_bytes)
        # Explicitly free the buffer using free() instead of the Mojo allocator.
        _libc.free(buffer.bitcast[NoneType]())

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


@register_passable
struct Error(
    Boolable,
    Defaultable,
    ImplicitlyCopyable,
    Movable,
    Representable,
    Stringable,
    Writable,
    _CurlyEntryFormattable,
):
    """This type represents an Error."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var data: UnsafePointer[UInt8, mut=False]
    """A pointer to the beginning of the string data being referenced."""

    var loaded_length: Int
    """The length of the string being referenced.
    Error instances conditionally own their error message. To reduce
    the size of the error instance we use the sign bit of the length field
    to store the ownership value. When loaded_length is negative it indicates
    ownership and a free is executed in the destructor.
    """
    var stack_trace: StackTrace
    """The stack trace of the error.
    By default the stack trace is not collected for the Error, unless user
    sets the stack_trace_depth parameter to value >= 0.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Default constructor."""
        self.data = UnsafePointer[UInt8]()
        self.loaded_length = 0
        self.stack_trace = StackTrace(depth=-1)

    @always_inline
    @implicit
    fn __init__(out self, value: StringLiteral):
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.

        """
        self.data = value.unsafe_ptr()
        self.loaded_length = value.byte_length()
        self.stack_trace = StackTrace(depth=0)

    @no_inline
    fn __init__[*Ts: Writable](out self, *args: *Ts):
        """Construct an Error by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.
        """

        @parameter
        fn _write(mut writer: Some[Writer]):
            @parameter
            for i in range(args.__len__()):
                args[i].write_to(writer)

        # Count the total length of bytes to allocate only once
        var arg_bytes = _TotalWritableBytes()
        _write(arg_bytes)
        arg_bytes.size += 1  # nul terminator
        var writer = _ErrorWriter(List[Byte](capacity=arg_bytes.size))
        var buffer = _WriteBufferStack(writer)
        _write(buffer)
        buffer.flush()
        writer.data.append(0)  # nul terminator
        self.loaded_length = -(len(writer.data) - 1)
        self.data = writer.data.steal_data()
        self.stack_trace = StackTrace(depth=0)

    fn __del__(deinit self):
        """Releases memory if allocated."""
        if self.loaded_length < 0:
            # Safety: if loaded_length < 0, we own the data allowing us to
            # safely free (and mutate) it.
            self.data.unsafe_mut_cast[True]().free()

    fn __copyinit__(out self, existing: Self):
        """Creates a deep copy of an existing error.

        Args:
            existing: The error to copy from.
        """
        if existing.loaded_length < 0:
            var length = -existing.loaded_length
            var dest = UnsafePointer[UInt8].alloc(length + 1)
            memcpy(dest=dest, src=existing.data, count=length)
            dest[length] = 0
            self.data = dest
        else:
            self.data = existing.data
        self.loaded_length = existing.loaded_length
        self.stack_trace = existing.stack_trace

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
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        if not self:
            return
        writer.write(self.as_string_slice())

    @no_inline
    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        return String("Error(", repr(self.as_string_slice()), ")")

    fn byte_length(self) -> Int:
        """Get the length of the Error string in bytes.

        Returns:
            The length of the Error string in bytes.

        Notes:
            This does not include the trailing null terminator in the count.
        """
        return abs(self.loaded_length)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn unsafe_cstr_ptr(self) -> UnsafePointer[c_char, mut=False]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be NUL terminated, and not null.

        Returns:
            The pointer to the underlying memory.
        """
        return self.data.bitcast[c_char]()

    fn as_string_slice(self) -> StringSlice[ImmutableAnyOrigin]:
        """Returns a string slice of the data maybe owned by the Error.

        Returns:
            A string slice pointing to the data maybe owned by the Error.

        Notes:
            Since the data is not guaranteed to be owned by the Error, the
            resulting StringSlice is given an ImmutableAnyOrigin.
        """
        return StringSlice[ImmutableAnyOrigin](
            ptr=self.data, length=UInt(self.byte_length())
        )

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
