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


struct RawFileDescriptor(TrivialRegisterPassable, Writer):
    """File descriptor of a file."""

    var value: c_int
    """The underlying value of the file descriptor."""

    def __init__(out self, value: c_int = 1):
        """Constructs the file descriptor from an integer.

        Args:
            value: The file identifier (Default 1 = stdout).
        """
        self.value = value

    def __init__(out self, f: FileHandle):
        """Constructs the file descriptor from a file handle.

        Args:
            f: The file handle.
        """
        self.value = c_int(f._get_raw_fd())

    @inline(.always)
    def write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """
        var written: c_ssize_t
        if __is_run_in_comptime_interpreter:
            written = external_call["write", c_ssize_t](
                self.value.__mlir_index__(),
                bytes.unsafe_ptr(),
                len(bytes).__mlir_index__(),
            )
        else:
            written = external_call["write", c_ssize_t](
                self.value,
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
            self.value, buffer.unsafe_ptr(), len(buffer)
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
        return external_call["isatty", c_int](self.value) != 0

    def fchdir(self) raises:
        """Changes the current working directory to the one
           represented by this fd.

        Raises:
            If the operations fails. In particular if the fd is
            not a directory.
        """
        var result = external_call["fchdir", c_int](self.value)
        if result < 0:
            var err = get_errno()
            raise Error("fchdir failed err: ", String(err))
