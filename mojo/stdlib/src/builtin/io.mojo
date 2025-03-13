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
"""Provides utilities for working with input/output.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import InlineArray
from collections.string import StringSlice
from sys import _libc as libc
from sys import (
    bitwidthof,
    external_call,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    is_compile_time,
    stdout,
)
from sys._amdgpu import printf_append_args, printf_append_string_n, printf_begin
from sys._libc import dup, fclose, fdopen, fflush
from sys.ffi import OpaquePointer, c_char
from sys.intrinsics import _type_is_eq

from builtin.dtype import _get_dtype_printf_format
from builtin.file_descriptor import FileDescriptor
from memory import UnsafePointer, bitcast, memcpy

from utils import StaticString, write_args, write_buffered

# ===----------------------------------------------------------------------=== #
#  _file_handle
# ===----------------------------------------------------------------------=== #


@value
@register_passable("trivial")
struct _fdopen[mode: StringLiteral = "a"]:
    var handle: OpaquePointer

    @implicit
    fn __init__(out self, stream_id: FileDescriptor):
        """Creates a file handle to the stdout/stderr stream.

        Args:
            stream_id: The stream id
        """

        self.handle = fdopen(dup(stream_id.value), mode.unsafe_cstr_ptr())

    fn __enter__(self) -> Self:
        """Open the file handle for use within a context manager"""
        return self

    fn __exit__(self):
        """Closes the file handle."""
        _ = fclose(self.handle)

    fn readline(self) raises -> String:
        """Reads an entire line from stdin or until EOF. Lines are delimited by a newline character.

        Returns:
            The line read from the stdin.

        Examples:

        ```mojo
        from builtin.io import _fdopen

        var line = _fdopen["r"](0).readline()
        print(line)
        ```

        Assuming the above program is named `my_program.mojo`, feeding it `Hello, World` via stdin would output:

        ```bash
        echo "Hello, World" | mojo run my_program.mojo

        # Output from print:
        Hello, World
        ```
        .
        """
        return self.read_until_delimiter("\n")

    fn read_until_delimiter(self, delimiter: StringSlice) raises -> String:
        """Reads an entire line from a stream, up to the `delimiter`.
        Does not include the delimiter in the result.

        Args:
            delimiter: The delimiter to read until.

        Returns:
            The text read from the stdin.

        Examples:

        ```mojo
        from builtin.io import _fdopen

        var line = _fdopen["r"](0).read_until_delimiter(",")
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
        var buffer = UnsafePointer[UInt8]()
        # ssize_t getdelim(char **restrict lineptr, size_t *restrict n,
        #                  int delimiter, FILE *restrict stream);
        var bytes_read = external_call[
            "getdelim",
            Int,
            UnsafePointer[UnsafePointer[UInt8]],
            UnsafePointer[UInt64],
            Int,
            OpaquePointer,
        ](
            UnsafePointer.address_of(buffer),
            UnsafePointer.address_of(UInt64(0)),
            ord(delimiter),
            self.handle,
        )
        # Per man getdelim(3), getdelim will return -1 if an error occurs
        # (or the user sends EOF without providing any input). We must
        # raise an error in this case because otherwise, String() will crash mojo
        # if the user sends EOF with no input.
        if bytes_read == -1:
            if buffer:
                libc.free(buffer.bitcast[NoneType]())
            # TODO: check errno to ensure we haven't encountered EINVAL or ENOMEM instead
            raise Error("EOF")
        # Copy the buffer (excluding the delimiter itself) into a Mojo String.
        var s = String(
            StringSlice[buffer.origin](ptr=buffer, length=bytes_read - 1)
        )
        # Explicitly free the buffer using free() instead of the Mojo allocator.
        libc.free(buffer.bitcast[NoneType]())
        return s


# ===----------------------------------------------------------------------=== #
#  _flush
# ===----------------------------------------------------------------------=== #


@no_inline
fn _flush(file: FileDescriptor = stdout):
    with _fdopen(file) as fd:
        _ = fflush(fd.handle)


# ===----------------------------------------------------------------------=== #
#  _printf
# ===----------------------------------------------------------------------=== #


fn _printf_cpu[
    fmt: StringLiteral, *types: AnyType
](args: VariadicPack[_, AnyType, *types], file: FileDescriptor = stdout):
    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.

    with _fdopen(file) as fd:
        # FIXME: external_call should handle this
        _ = __mlir_op.`pop.external_call`[
            func = "KGEN_CompilerRT_fprintf".value,
            variadicType = __mlir_attr[
                `(`,
                `!kgen.pointer<none>,`,
                `!kgen.pointer<scalar<si8>>`,
                `) -> !pop.scalar<si32>`,
            ],
            _type=Int32,
        ](fd, fmt.unsafe_cstr_ptr(), args.get_loaded_kgen_pack())


@no_inline
fn _printf[
    fmt: StringLiteral, *types: AnyType
](*args: *types, file: FileDescriptor = stdout):
    if is_compile_time():
        _printf_cpu[fmt](args, file)
    else:

        @parameter
        if is_nvidia_gpu():
            # The argument pack will contain references for each value in the pack,
            # but we want to pass their values directly into the C printf call. Load
            # all the members of the pack.
            var loaded_pack = args.get_loaded_kgen_pack()

            _ = external_call["vprintf", Int32](
                fmt.unsafe_cstr_ptr(), Pointer.address_of(loaded_pack)
            )
        elif is_amd_gpu():
            # This is adapted from Triton's third party method for lowering
            # AMD printf calls:
            # https://github.com/triton-lang/triton/blob/1c28e08971a0d70c4331432994338ee05d31e633/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp#L321
            fn _to_uint64[T: AnyType, //](value: T) -> UInt64:
                @parameter
                if _type_is_eq[T, UInt64]():
                    return rebind[UInt64](value)
                elif _type_is_eq[T, UInt32]():
                    return UInt64(rebind[UInt32](value))
                elif _type_is_eq[T, UInt16]():
                    return UInt64(rebind[UInt16](value))
                elif _type_is_eq[T, UInt8]():
                    return UInt64(rebind[UInt8](value))
                elif _type_is_eq[T, Int64]():
                    return UInt64(rebind[Int64](value))
                elif _type_is_eq[T, Int32]():
                    return UInt64(rebind[Int32](value))
                elif _type_is_eq[T, Int16]():
                    return UInt64(rebind[Int16](value))
                elif _type_is_eq[T, Int8]():
                    return UInt64(rebind[Int8](value))
                elif _type_is_eq[T, Float16]():
                    return bitcast[DType.uint64](
                        Float64(rebind[Float16](value))
                    )
                elif _type_is_eq[T, Float32]():
                    return bitcast[DType.uint64](
                        Float64(rebind[Float32](value))
                    )
                elif _type_is_eq[T, Float64]():
                    return bitcast[DType.uint64](rebind[Float64](value))
                elif _type_is_eq[T, Int]():
                    return UInt64(rebind[Int](value))
                elif _type_is_eq[T, UInt]():
                    return UInt64(rebind[UInt](value))
                elif _type_is_eq[UnsafePointer[UInt8], UInt]():
                    return UInt64(Int(rebind[UnsafePointer[UInt8]](value)))
                elif _type_is_eq[UnsafePointer[Int8], UInt]():
                    return UInt64(Int(rebind[UnsafePointer[Int8]](value)))
                elif _type_is_eq[OpaquePointer, UInt]():
                    return UInt64(Int(rebind[OpaquePointer](value)))
                return 0

            alias args_len = len(VariadicList(types))

            var message = printf_begin()
            message = printf_append_string_n(
                message, fmt.as_bytes(), args_len == 0
            )
            alias k_args_per_group = 7

            @parameter
            for group in range(0, args_len, k_args_per_group):
                alias bound = min(group + k_args_per_group, args_len)
                alias num_args = bound - group

                var arguments = InlineArray[UInt64, k_args_per_group](fill=0)

                @parameter
                for i in range(num_args):
                    arguments[i] = _to_uint64(args[group + i])
                message = printf_append_args(
                    message,
                    num_args,
                    arguments[0],
                    arguments[1],
                    arguments[2],
                    arguments[3],
                    arguments[4],
                    arguments[5],
                    arguments[6],
                    Int32(Int(bound == args_len)),
                )

        else:
            _printf_cpu[fmt](args, file)


# ===----------------------------------------------------------------------=== #
#  _snprintf
# ===----------------------------------------------------------------------=== #


@no_inline
fn _snprintf[
    fmt: StringLiteral, *types: AnyType
](str: UnsafePointer[UInt8], size: Int, *args: *types) -> Int:
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

    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C snprintf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()

    # FIXME: external_call should handle this
    return Int(
        __mlir_op.`pop.external_call`[
            func = "snprintf".value,
            variadicType = __mlir_attr[
                `(`,
                `!kgen.pointer<scalar<si8>>,`,
                `!pop.scalar<index>, `,
                `!kgen.pointer<scalar<si8>>`,
                `) -> !pop.scalar<si32>`,
            ],
            _type=Int32,
        ](str, size, fmt.unsafe_cstr_ptr(), loaded_pack)
    )


# ===----------------------------------------------------------------------=== #
#  print
# ===----------------------------------------------------------------------=== #


@no_inline
fn print[
    *Ts: Writable
](
    *values: *Ts,
    sep: StaticString = " ",
    end: StaticString = "\n",
    flush: Bool = False,
    owned file: FileDescriptor = stdout,
):
    """Prints elements to the text stream. Each element is separated by `sep`
    and followed by `end`.

    Parameters:
        Ts: The elements types.

    Args:
        values: The elements to print.
        sep: The separator used between elements.
        end: The String to write after printing the elements.
        flush: If set to true, then the stream is forcibly flushed.
        file: The output stream.
    """

    if is_compile_time():
        write_buffered(file, values, sep=sep, end=end)
    else:

        @parameter
        if is_amd_gpu():
            write_buffered[buffer_size=512](file, values, sep=sep, end=end)
        elif is_nvidia_gpu():
            write_buffered[use_heap=True](file, values, sep=sep, end=end)
        else:
            write_buffered(file, values, sep=sep, end=end)

    if is_compile_time():
        if flush:
            _flush(file=file)
    else:
        # Isn't this weird
        @parameter
        if not is_gpu():
            if flush:
                _flush(file=file)


# ===----------------------------------------------------------------------=== #
#  input
# ===----------------------------------------------------------------------=== #


fn input(prompt: String = "") raises -> String:
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
    """
    if prompt != "":
        print(prompt, end="")
    return _fdopen["r"](0).readline()


fn _get_stdout_stream() -> OpaquePointer:
    return external_call[
        "KGEN_CompilerRT_IO_get_stdout_stream", OpaquePointer
    ]()
