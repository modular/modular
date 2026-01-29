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

from collections.string.string_slice import _unsafe_strlen
from memory import (
    ArcPointer,
    OwnedPointer,
    alloc,
    memcpy,
)
from sys import external_call, is_gpu
from sys.info import size_of, align_of


# ===-----------------------------------------------------------------------===#
# StackTrace
# ===-----------------------------------------------------------------------===#


struct StackTrace(Copyable, Movable, Stringable):
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

    fn __copyinit__(out self, existing: Self):
        """Copy constructor - copies the stack trace string.

        Args:
            existing: The existing StackTrace to copy from.
        """
        # Copy the null-terminated string
        var src_ptr = existing._data.unsafe_ptr()
        var str_len = Int(_unsafe_strlen(src_ptr))
        var new_ptr = alloc[UInt8](str_len + 1)
        memcpy(dest=new_ptr, src=src_ptr, count=str_len + 1)
        self._data = OwnedPointer(unsafe_from_raw_pointer=new_ptr)

    fn __moveinit__(out self, deinit existing: Self):
        """Move constructor.

        Args:
            existing: The existing StackTrace to move from.
        """
        self._data = existing._data^

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

        @parameter
        if is_gpu():
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

    fn __str__(self) -> String:
        """Converts the StackTrace to string representation.

        Returns:
            A String of the stack trace.
        """
        return String(unsafe_from_utf8_ptr=self._data.unsafe_ptr())


# ===-----------------------------------------------------------------------===#
# Error
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _TypeErasedWriter(Writer):
    """A type-erased `Writer`.

    Stores an opaque pointer to any `Writer` instance along with a function pointer
    specialized for that concrete type's `write_string` method.
    """

    var _writer: OpaquePointer[MutAnyOrigin]
    """Opaque pointer to the concrete writer instance."""

    var _write_fn: fn(OpaquePointer[MutAnyOrigin], StringSlice[ImmutAnyOrigin])
    """Function pointer specialized for the concrete writer type that calls the
    writer's `write_string` method."""

    fn __init__[
        W: Writer, //, origin: MutOrigin
    ](out self, ref [origin]writer: W):
        """Construct an erased writer, capturing the concrete type `W`."""
        self._writer = _make_opaque(writer)
        self._write_fn = Self._write_to_impl[W]

    @always_inline
    fn write_string(mut self, string: StringSlice):
        """Dispatch to the concrete writer via the stored function pointer."""
        var bytes = string.as_bytes()
        self._write_fn(
            self._writer,
            StringSlice[ImmutAnyOrigin](
                unsafe_from_utf8={ptr = bytes.unsafe_ptr(), length = len(bytes)}
            ),
        )

    @staticmethod
    fn _write_to_impl[
        W: Writer
    ](
        writer: OpaquePointer[MutAnyOrigin],
        string: StringSlice[ImmutAnyOrigin],
    ):
        """Implementation that casts back to `W` and calls write_string.

        This function is stored in `_write_fn` and captures the concrete type `W`,
        allowing type-safe dispatch despite the erased function signature.

        Parameters:
            W: The concrete writer type.

        Args:
            writer: Opaque pointer to the writer.
            string: StringSlice to write.
        """
        writer.bitcast[W]()[].write_string(string)


@fieldwise_init
struct _VTableErrorOp(Equatable, TrivialRegisterType):
    """Operation codes for vtable dispatch.

    These discriminator values tell the vtable which operation to perform on
    the type-erased error data.
    """

    var _value: Int8

    comptime DEL: Self = Self(0)
    comptime COPY: Self = Self(1)
    comptime WRITE_TO: Self = Self(2)


fn _make_opaque[T: AnyType, //](ref t: T) -> OpaquePointer[MutAnyOrigin]:
    """Convert a typed reference to an opaque pointer."""
    return (
        UnsafePointer(to=t)
        .bitcast[NoneType]()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


# TODO: Add inlined error support if sizeof(ErrorType) < sizeof(Pointer) and
# it is trivially movable.
struct _TypeErasedError(Copyable, Writable):
    """A type-erased error using manual vtable dispatch and `ArcPointer` for storage.

    ## Key Design Elements

    1. **ArcPointer Storage**: The concrete error type `T` is wrapped in `ArcPointer[T]`
       then type-erased to `OpaquePointer`. This makes `_TypeErasedError` copyable
       even when `T` is not `Copyable`.

    2. **Vtable Dispatch**: A function pointer captures the concrete type `T` at
       construction time. Operations (DEL, COPY, WRITE_TO) dispatch through this vtable,
       which casts back to `ArcPointer[T]` and performs the type-specific work.

    3. **Memory Layout**: Just two fields - an opaque pointer (disguised `ArcPointer[T]`)
       and a function pointer.
    """

    comptime _ErrorArcPointer = OpaquePointer[MutExternalOrigin]
    """Erased `ArcPointer[T]` storage."""

    comptime _VTableInput = UnsafePointer[Self._ErrorArcPointer, MutAnyOrigin]
    comptime _VTableOutput = OpaquePointer[MutAnyOrigin]

    # TODO: Allow inlining the error type directly (avoid allocation) if the
    # error is small enough and is trivially movable.
    # Will need to update copy logic to account for the different storage.
    var _error: Self._ErrorArcPointer
    """Type-erased `ArcPointer[T]` holding the actual error data."""

    var _vtable: fn(
        _VTableErrorOp,
        Self._VTableInput,
        Self._VTableOutput,
    )
    """Function pointer specialized for the concrete error type, dispatches operations."""

    @always_inline
    fn __init__[
        ErrorType: Movable & ImplicitlyDestructible & Writable
    ](out self, var error: ErrorType):
        """Construct from a concrete error type.

        Wraps `error` in `ArcPointer[ErrorType]`, stores it type-erased in `_error`,
        and captures a vtable specialized for `ErrorType`. Compile-time assertions
        verify that `ArcPointer[ErrorType]` fits in the opaque storage.

        Parameters:
            ErrorType: The concrete error type being stored.

        Args:
            error: The concrete error value to store.
        """
        self._error = {}
        self._vtable = Self._vtable_impl[ErrorType]

        __comptime_assert (
            size_of[ArcPointer[ErrorType]]() <= size_of[Self._ErrorArcPointer]()
        )
        __comptime_assert (
            align_of[ArcPointer[ErrorType]]()
            <= align_of[Self._ErrorArcPointer]()
        )

        var arc = ArcPointer[ErrorType](error^)
        UnsafePointer(to=self._error).bitcast[
            ArcPointer[ErrorType]
        ]().init_pointee_move(arc^)

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy the erased error via vtable."""
        self._error = {}
        self._vtable = other._vtable
        self._vtable(
            _VTableErrorOp.COPY,
            other.error_ptr(),
            _make_opaque(self._error),
        )

    @always_inline
    fn __del__(deinit self):
        """Destroy via vtable."""
        self._vtable(
            _VTableErrorOp.DEL,
            self.error_ptr(),
            {},  # DEL does not need an output pointer
        )

    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        """Write the error via vtable dispatch."""
        var erased = _TypeErasedWriter(writer)
        self._vtable(
            _VTableErrorOp.WRITE_TO,
            self.error_ptr(),
            _make_opaque(erased),
        )

    @always_inline
    fn error_ptr(self) -> Self._VTableInput:
        """Get a pointer to `_error` for passing to vtable functions."""
        return (
            UnsafePointer(to=self._error)
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )

    @staticmethod
    fn _vtable_impl[
        ErrorType: Movable & ImplicitlyDestructible & Writable
    ](op: _VTableErrorOp, input: Self._VTableInput, output: Self._VTableOutput):
        """Vtable dispatcher that captures `ErrorType` and routes operations.

        Parameters:
            ErrorType: The concrete error type this vtable handles.

        Args:
            op: The operation to perform (DEL, COPY, or WRITE_TO).
            input: Pointer to the erased `ArcPointer[ErrorType]`.
            output: Output pointer (destination for COPY, writer for WRITE_TO).
        """
        if op == _VTableErrorOp.DEL:
            Self._del_impl[ErrorType](input)
        elif op == _VTableErrorOp.COPY:
            Self._copy_impl[ErrorType](input, output)
        elif op == _VTableErrorOp.WRITE_TO:
            Self._write_to_impl[ErrorType](input, output)

    @staticmethod
    @always_inline
    fn _del_impl[
        ErrorType: Movable & ImplicitlyDestructible & Writable
    ](input: Self._VTableInput):
        """Destroy the `ArcPointer[ErrorType]`."""
        input.bitcast[ArcPointer[ErrorType]]().destroy_pointee()

    @staticmethod
    @always_inline
    fn _copy_impl[
        ErrorType: Movable & ImplicitlyDestructible & Writable
    ](input: Self._VTableInput, output: Self._VTableOutput):
        """Copy-initialize destination `ArcPointer` from source.

        Parameters:
            ErrorType: The concrete error type.

        Args:
            input: Pointer to the source erased `ArcPointer[ErrorType]`.
            output: Pointer to the destination erased `ArcPointer[ErrorType]`.
        """
        output.bitcast[ArcPointer[ErrorType]]().init_pointee_copy(
            input.as_immutable().bitcast[ArcPointer[ErrorType]]()[]
        )

    @staticmethod
    @always_inline
    fn _write_to_impl[
        ErrorType: Movable & ImplicitlyDestructible & Writable
    ](input: Self._VTableInput, output: Self._VTableOutput):
        """Write the error if `ErrorType` is `Writable`, otherwise write placeholder.

        Parameters:
            ErrorType: The concrete error type.

        Args:
            input: Pointer to the erased `ArcPointer[ErrorType]`.
            output: Pointer to the erased `_TypeErasedWriter`.
        """
        var writer = output.bitcast[_TypeErasedWriter]()
        ref error = input.as_immutable().bitcast[ArcPointer[ErrorType]]()[][]
        error.write_to(writer[])


struct Error(
    Copyable,
    Representable,
    Stringable,
    Writable,
):
    """This type represents an Error."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _error: _TypeErasedError
    """The type erased error."""

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
        self._error = _TypeErasedError(value^)
        self._stack_trace = StackTrace.collect_if_enabled(depth)

    @always_inline
    @implicit
    fn __init__(out self, value: StringLiteral):
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.
        """
        self._error = _TypeErasedError(value)
        self._stack_trace = StackTrace.collect_if_enabled(0)

    @no_inline
    @implicit
    fn __init__(
        out self, var error: Some[Movable & ImplicitlyDestructible & Writable]
    ):
        """Construct an `Error` from a `Writable` argument.

        Args:
            error: The error to store.
        """
        self._error = _TypeErasedError(error^)
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

    @no_inline
    fn __str__(self) -> String:
        """Converts the Error to string representation.

        Returns:
            A String of the error message.
        """
        return String(self._error)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        """
        Formats this error to the provided Writer.

        Args:
            writer: The object to write to.
        """
        self._error.write_to(writer)

    @no_inline
    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        return String("Error('", self._error, "')")

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
