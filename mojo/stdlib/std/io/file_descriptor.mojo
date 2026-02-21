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
var f = open("my_file.txt", "r")
print("hello", file=f^)
f.close()
```

"""
from os import abort
from sys import (
    CompilationTarget,
    is_amd_gpu,
    is_compile_time,
    is_gpu,
    is_nvidia_gpu,
)
from ffi import (
    c_ssize_t,
    c_int,
    external_call,
    _external_call_const,
    get_errno,
)

from memory import Span


struct FileDescriptor(TrivialRegisterPassable, Writer):
    """File descriptor of a file."""

    var value: Int
    """The underlying value of the file descriptor."""

    fn __init__(out self, value: Int = 1):
        """Constructs the file descriptor from an integer.

        Args:
            value: The file identifier (Default 1 = stdout).
        """
        self.value = value

    fn __init__(out self, f: FileHandle):
        """Constructs the file descriptor from a file handle.

        Args:
            f: The file handle.
        """
        self.value = f._get_raw_fd()

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """
        written = external_call["write", c_ssize_t](
            self.value, bytes.unsafe_ptr(), len(bytes)
        )
        debug_assert(
            written == len(bytes),
            "expected amount of bytes not written",
        )

    fn write_string(mut self, string: StringSlice):
        """
        Write a `StringSlice` to this `FileDescriptor`.

        This method is required by the `Writer` trait.

        Args:
            string: The `StringSlice` to write to this `FileDescriptor`.
        """
        self.write_bytes(string.as_bytes())

    @always_inline
    fn read_bytes(mut self, buffer: Span[mut=True, Byte]) raises -> UInt:
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

        comptime if CompilationTarget.is_macos() or CompilationTarget.is_linux():
            var read = external_call["read", c_ssize_t](
                self.value, buffer.unsafe_ptr(), len(buffer)
            )
            if read < 0:
                raise Error("Failed to read bytes.")
            return UInt(read)
        else:
            constrained[
                False,
                "`read_bytes()` is not yet implemented for unknown platform.",
            ]()
            abort()

    fn isatty(self) -> Bool:
        """Checks whether a file descriptor refers to a terminal.

        Returns `True` if the file descriptor is open and connected to a
        tty(-like) device, otherwise `False`. On GPUs, the function always
        returns `False`.

        Returns:
            `True` if the file descriptor is connected to a terminal, `False` otherwise.

        Examples:
            ```mojo
            # Check if stdout is a terminal
            if stdout.isatty():
                print("Running in a terminal")
            else:
                print("Output is redirected")
            ```
        """

        comptime if is_gpu():
            return False
        return _external_call_const["isatty", c_int](c_int(self.value)) != 0
