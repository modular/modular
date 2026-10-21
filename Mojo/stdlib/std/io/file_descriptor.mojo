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
"""Higher level abstraction for file stream.

These are Mojo built-ins, so you don't need to import them.

For example, here's how to print to a file

```mojo
var f = open("my_file.txt", "w")
var fd = RawFileDescriptor(f)
print("hello", file=fd)
f.close()
```

"""
from std.sys import CompilationTarget, is_gpu
from std.ffi import (
    c_ssize_t,
    c_int,
    external_call,
    get_errno,
)

from std.collections import Span
from std.memory import MaybeUninit
from std.os import abort
from std.utils._nicheable import UnsafeSingleNicheable


struct RawFileDescriptor(
    TrivialRegisterPassable, UnsafeSingleNicheable, Writer
):
    """File descriptor of a file.

    A file descriptor is always a non-negative integer, so `-1` is never a
    valid descriptor. `RawFileDescriptor` exposes `-1` as its niche, which
    makes `Optional[RawFileDescriptor]` the same size as
    `RawFileDescriptor`; use `Optional[RawFileDescriptor]` (not a `-1`
    value) to represent an absent descriptor.
    """

    var _value: c_int
    """The underlying value of the file descriptor."""

    def __init__(out self, value: c_int = 1):
        """Constructs the file descriptor from an integer.

        Args:
            value: The file identifier (Default 1 = stdout).

        Constraints:
            `value` must be non-negative. File descriptors are never
            negative, and `-1` is the type's niche value, so a negative
            `value` aborts at runtime. Use `Optional[RawFileDescriptor]` to
            represent an absent descriptor.
        """
        if value < 0:
            abort(
                "invalid file descriptor: file descriptors are non-negative;"
                " use Optional[RawFileDescriptor] to represent an absent"
                " descriptor"
            )
        self._value = value

    @doc_hidden
    def __init__(out self, *, unsafe_unchecked_value: c_int):
        """Constructs the file descriptor without validating the value.

        The standard streams are built with this at compile time. The checked
        constructor reports through `print`, which reaches `stdout`, so
        building `stdout` through the checked constructor would recurse.

        Args:
            unsafe_unchecked_value: The file identifier, assumed non-negative.
        """
        self._value = unsafe_unchecked_value

    def __init__(out self, f: FileHandle):
        """Constructs the file descriptor from a file handle.

        Args:
            f: The file handle.
        """
        self._value = c_int(f._get_raw_fd())

    # ===------------------------------------------------------------------===#
    # UnsafeSingleNicheable
    # ===------------------------------------------------------------------===#

    comptime _InvalidFd: c_int = -1

    @staticmethod
    @inline(.always)
    @doc_hidden
    def write_niche(memory: MutPointer[MaybeUninit[Self], _]):
        memory.unsafe_bitcast[c_int]().write(Self._InvalidFd)

    @staticmethod
    @inline(.always)
    @doc_hidden
    def isa_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> Bool:
        return memory.unsafe_bitcast[c_int]()[] == Self._InvalidFd

    @inline(.always)
    def write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """
        var written = external_call["write", c_ssize_t](
            self._value,
            bytes.unsafe_ptr(),
            len(bytes).__mlir_index__(),
        )
        assert written == len(bytes), "expected amount of bytes not written"

    def write_string(mut self, string: StringSlice):
        """
        Write a `StringSlice` to this `RawFileDescriptor`.

        This method is required by the `Writer` trait.

        Args:
            string: The `StringSlice` to write to this `RawFileDescriptor`.
        """
        self.write_bytes(string.as_bytes())

    @inline(.always)
    def read_bytes(mut self, buffer: MutSpan[Byte, _]) raises -> Int:
        """Read a number of bytes from the file into a buffer.

        Args:
            buffer: A `Span[Byte]` to read bytes into. Read up to `len(buffer)` number of bytes.

        Returns:
            Actual number of bytes read.

        Notes:
            [Reference](https://pubs.opengroup.org/onlinepubs/9799919799/functions/read.html).

        Raises:
            If the operation fails.
        """

        comptime assert (
            not is_gpu()
        ), "`read_bytes()` is not yet implemented for GPUs."

        comptime assert (
            CompilationTarget.is_macos() or CompilationTarget.is_linux()
        ), "`read_bytes()` is not yet implemented for unknown platform."
        var read = external_call["read", c_ssize_t](
            self._value, buffer.unsafe_ptr(), len(buffer)
        )
        if read < 0:
            raise Error("Failed to read bytes.")
        return read

    def isatty(self) -> Bool:
        """Checks whether a file descriptor refers to a terminal.

        Returns `True` if the file descriptor is open and connected to a
        tty(-like) device, otherwise `False`. On GPUs, the function always
        returns `False`.

        Returns:
            `True` if the file descriptor is connected to a terminal, `False` otherwise.

        Examples:
            ```mojo
            from std.sys import stdout

            # Check if stdout is a terminal
            if stdout.isatty():
                print("Running in a terminal")
            else:
                print("Output is redirected")
            ```
        """

        comptime if is_gpu():
            return False
        return external_call["isatty", c_int](self._value) != 0

    def fchdir(self) raises:
        """Changes the current working directory to the one
           represented by this fd.

        Raises:
            If the operations fails. In particular if the fd is
            not a directory.
        """
        var result = external_call["fchdir", c_int](self._value)
        if result < 0:
            var err = get_errno()
            raise Error("fchdir failed err: ", String(err))
