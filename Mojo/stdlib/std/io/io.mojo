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
"""Provides utilities for working with input/output.

These are Mojo built-ins, so you don't need to import them.
"""

from std._plugin import CurrentPlugin
from std.collections.string.string_span import get_static_string
from std.format._utils import _FixedWriteBuffer, _FlushingWriteBuffer
from std.sys import _libc as libc
from std.ffi import (
    c_size_t,
    c_ssize_t,
    external_call,
    CStringSpan,
    OptionalPointer,
)
from std.memory.unsafe_pointer import unsafe_cast
from std.sys import (
    is_gpu,
    stdin,
    stdout,
)
from std.sys._libc import dup, fclose, fdopen, fflush, FILE_ptr
from std.sys.info import CompilationTarget

from .file_descriptor import RawFileDescriptor


# FIXME(MOCO-3871): Alias is to workaround function type comparison bug.
comptime _PrintEmitPluginHookFnType = def[O: Origin](
    cstr: CStringSpan[O],
    file_value: RawFileDescriptor,
) thin -> None
"""Plugin-hook signature for `PluginHooks.print_emit_fn`; keep in sync with the `print` emit path."""


# TODO(OSSMDLR-2752): Drop `fmt: off` once mblack stops splitting
# `*args: *types` into `* args: * types`.
# fmt: off
comptime _PrintfEmitPluginHookFnType[*types: AnyType] = def(
    *args: *types, file: RawFileDescriptor
) thin -> None
"""Plugin-hook signature for `PluginHooks.printf_emit_fn`; keep in sync with `_printf`."""
# fmt: on


# ===----------------------------------------------------------------------=== #
#  _file_handle
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct _fdopen[mode: StaticString = "a"](ImplicitlyCopyable, RegisterPassable):
    var handle: FILE_ptr

    def __init__(out self, stream_id: RawFileDescriptor):
        """Creates a file handle to the stdout/stderr stream.

        Args:
            stream_id: The stream id
        """

        self.handle = fdopen(
            dup(stream_id._value),
            Self.mode.as_c_string_span(),
        )

    def __enter__(self) -> Self:
        """Open the file handle for use within a context manager"""
        return self

    def __exit__(self):
        """Closes the file handle."""
        _ = fclose(self.handle)

    def readline(self) raises -> String:
        """Reads an entire line from stdin or until EOF. Lines are delimited by a newline character.

        Returns:
            The line read from the stdin.

        Examples:

        ```mojo
        from std.io.io import _fdopen
        from std.sys import stdin

        var line = _fdopen["r"](stdin).readline()
        print(line)
        ```

        Assuming the above program is named `my_program.mojo`, feeding it `Hello, World` via stdin would output:

        ```bash
        echo "Hello, World" | mojo run my_program.mojo

        # Output from print:
        Hello, World
        ```
        """
        return self.read_until_delimiter("\n")

    def read_until_delimiter(self, delimiter: StringSlice) raises -> String:
        """Reads an entire line from a stream, up to the `delimiter`.
        Does not include the delimiter in the result.

        Args:
            delimiter: The delimiter to read until.

        Returns:
            The text read from the stdin.

        Examples:

        ```mojo
        from std.io.io import _fdopen
        from std.sys import stdin

        var line = _fdopen["r"](stdin).read_until_delimiter(",")
        print(line)
        ```

        Assuming the above program is named `my_program.mojo`, feeding it `Hello, World` via stdin would output:

        ```bash
        echo "Hello, World" | mojo run my_program.mojo

        # Output from print:
        Hello
        ```
        """
        # getdelim will allocate the buffer using malloc().
        var buffer = OptionalPointer[UInt8, MutUntrackedOrigin]()
        var n = c_size_t(0)
        # ssize_t getdelim(char **restrict lineptr, size_t *restrict n,
        #                  int delimiter, FILE *restrict stream);
        var bytes_read = external_call["getdelim", c_ssize_t](
            Pointer(to=buffer),
            Pointer(to=n),
            ord(delimiter),
            self.handle,
        )
        # Per man getdelim(3), getdelim will return -1 if an error occurs
        # (or the user sends EOF without providing any input). We must
        # raise an error in this case because otherwise, String() will crash mojo
        # if the user sends EOF with no input.
        if bytes_read == -1:
            libc.free(unsafe_cast[Type=NoneType, origin=MutAnyOrigin](buffer))
            # TODO: check errno to ensure we haven't encountered EINVAL or ENOMEM instead
            raise Error("EOF")
        # Copy the buffer (excluding the delimiter itself) into a Mojo String.
        var s = String(
            StringSlice(
                unsafe_from_utf8=Span(
                    unsafe_ptr=buffer.unsafe_value(), length=bytes_read - 1
                )
            )
        )
        # Explicitly free the buffer using free() instead of the Mojo allocator.
        libc.free(unsafe_cast[Type=NoneType, origin=MutAnyOrigin](buffer))
        return s^


# ===----------------------------------------------------------------------=== #
#  _flush
# ===----------------------------------------------------------------------=== #


@inline(.never)
def _flush(file: RawFileDescriptor = stdout):
    with _fdopen(file) as fd:
        _ = fflush(fd.handle)


# ===----------------------------------------------------------------------=== #
#  _printf
# ===----------------------------------------------------------------------=== #


def _printf_cpu[
    fmt: StaticString, *types: AnyType
](*args: *types, file: RawFileDescriptor = stdout):
    with _fdopen(file) as fd:
        # int fprintf(FILE *restrict stream, const char *restrict fmt, ...);
        # The pack is loaded so the variadic arguments are the values
        # themselves rather than references to them.
        _ = external_call["KGEN_CompilerRT_fprintf", Int32, num_fixed_args=2](
            fd,
            get_static_string[fmt]().as_c_string_span(),
            args.get_loaded_kgen_pack(),
        )


@inline(.never)
def _printf[
    fmt: StaticString, *types: AnyType
](*args: *types, file: RawFileDescriptor = stdout):
    if __is_run_in_comptime_interpreter:
        _printf_cpu[fmt](*args, file=file)
        return

    comptime if case .Some(_printf_emit) = CurrentPlugin.printf_emit_fn[
        fmt, *types
    ]:
        _printf_emit(*args, file=file)
    elif not is_gpu():
        _printf_cpu[fmt](*args, file=file)
    elif case .Some(_emit) = CurrentPlugin.print_emit_fn:
        # The hook takes a formatted byte buffer and there is no printf
        # formatter on device, so the arguments are dropped and only `fmt` is
        # written.
        var buffer = _FixedWriteBuffer()
        buffer.write_string(fmt)
        var cstr = buffer.nul_terminate()

        # FIXME: The origin param of `_emit` should be inferred from `cstr`.
        _emit[origin_of(buffer).unsafe_mut_cast[False]()](cstr, file)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


# ===----------------------------------------------------------------------=== #
#  _snprintf
# ===----------------------------------------------------------------------=== #


@inline(.never)
def _snprintf[
    fmt: StaticString, *types: AnyType
](str: MutPointer[UInt8, _], size: Int, *args: *types) -> Int:
    """Writes a format string into an output pointer.

    Parameters:
        fmt: A format string.
        types: The types of arguments interpolated into the format string.

    Args:
        str: A pointer into which the format string is written.
        size: At most, `size - 1` bytes are written into the output string.
        args: Arguments interpolated into the format string.

    Returns:
        The number of bytes written into the output string.
    """

    # int snprintf(char *restrict s, size_t n, const char *restrict fmt, ...);
    # The pack is loaded so the variadic arguments are the values themselves
    # rather than references to them.
    return Int(
        external_call["snprintf", Int32, num_fixed_args=3](
            str,
            size,
            get_static_string[fmt]().as_c_string_span(),
            args.get_loaded_kgen_pack(),
        )
    )


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@inline(.never)
def print[
    *Ts: Writable
](
    *values: *Ts,
    sep: StringSlice = " ",
    end: StringSlice = "\n",
    flush: Bool = False,
    var file: RawFileDescriptor = stdout,
):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

    This function accepts any number of values, but their types must implement
    the [`Writable`](/docs/std/format/Writable/) trait. Most built-in types
    (like `Int`, `Float64`, `Bool`, `String`) implement the
    [`Writable`](/docs/std/format/Writable/) trait.

    For string formatting, you can use the
    [`format()`](/docs/std/collections/string/string/String/#format) method
    or, preferably, a template string
    ([`TString`](/docs/std/format/tstring/TString/), written `t"..."`)
    which interpolates expressions directly without allocating an
    intermediate `String`.

    Examples:

    ```mojo
    print("Hello, World!")                   # Hello, World!

    print("The answer is", 42)               # The answer is 42

    print("{} is {}".format("Mojo", "🔥"))   # Mojo is 🔥

    var name = "Mojo"
    print(t"{name} is 🔥")                   # Mojo is 🔥
    ```

    Parameters:
        Ts: The elements types.

    Args:
        values: The elements to print.
        sep: The separator used between elements.
        end: The String to write after printing the elements.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """

    comptime assert Ts.all_conforms_to[
        Writable
    ]()  # satisfy _write_to where clause.

    if __is_run_in_comptime_interpreter:
        var buffer = _FlushingWriteBuffer(file)
        values._write_to(buffer, sep=sep, end=end)

        buffer.flush()
        if flush:
            _flush(file=file)

        return

    comptime if case .Some(_emit) = CurrentPlugin.print_emit_fn:
        var buffer = _FixedWriteBuffer()
        values._write_to(buffer, sep=sep, end=end)

        var cstr = buffer.nul_terminate()

        # FIXME: The origin param of `_emit` should be inferred from `cstr`.
        _emit[origin_of(buffer).unsafe_mut_cast[False]()](cstr, file)
    elif is_gpu():
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()
    else:
        var buffer = _FlushingWriteBuffer(file)
        values._write_to(buffer, sep=sep, end=end)

        buffer.flush()

        if flush:
            _flush(file=file)


# ===----------------------------------------------------------------------=== #
#  input
# ===----------------------------------------------------------------------=== #


def input(prompt: String = "") raises -> String:
    """Reads a line of input from the user.

    Reads a line from standard input, converts it to a string, and returns that string.
    If the prompt argument is present, it is written to standard output without a trailing newline.

    Args:
        prompt: An optional string to be printed before reading input.

    Returns:
        A string containing the line read from the user input.

    Examples:
    ```mojo
    name = input("Enter your name: ")
    print("Hello", name)
    ```

    If the user enters "Mojo" it prints "Hello Mojo".

    Raises:
        If the operation fails.
    """
    if prompt != "":
        print(prompt, end="")
    return _fdopen["r"](stdin).readline()
