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
    is_amd_gpu,
    is_compile_time,
    is_gpu,
    is_nvidia_gpu,
    os_is_linux,
    os_is_macos,
)
from sys._amdgpu import printf_append_string_n, printf_begin
from sys.ffi import c_ssize_t, external_call

from builtin.io import _printf
from memory import Span, UnsafePointer


@register_passable("trivial")
struct FileDescriptor(Writer):
    """File descriptor of a file."""

    var value: Int
    """The underlying value of the file descriptor."""

    fn __init__(out self, value: Int = 1):
        """Constructs the file descriptor from an integer.

        Args:
            value: The file identifier (Default 1 = stdout).
        """
        self.value = value

    @implicit
    fn __init__(out self, f: FileHandle):
        """Constructs the file descriptor from a file handle.

        Args:
            f: The file handle.
        """
        self.value = f._get_raw_fd()

    @always_inline
    fn __write_bytes_cpu(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """

        written = external_call["write", Int32](
            self.value, bytes.unsafe_ptr(), len(bytes)
        )
        debug_assert(
            written == len(bytes),
            "expected amount of bytes not written. expected: ",
            len(bytes),
            "but got: ",
            written,
        )

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """

        if is_compile_time():
            self.__write_bytes_cpu(bytes)
        else:

            @parameter
            if is_nvidia_gpu():
                _printf["%*s"](len(bytes), bytes.unsafe_ptr())
            elif is_amd_gpu():
                var msg = printf_begin()
                _ = printf_append_string_n(msg, bytes, is_last=True)
            else:
                self.__write_bytes_cpu(bytes)

    @always_inline
    fn read_bytes(mut self, buffer: Span[mut=True, Byte]) raises -> UInt:
        """Read a number of bytes from the file into a buffer.

        Args:
            buffer: A `Span[Byte]` to read bytes into. Read up to `len(buffer)` number of bytes.

        Returns:
            Actual number of bytes read.

        Notes:
            [Reference](https://pubs.opengroup.org/onlinepubs/9799919799/functions/read.html).
        """

        constrained[
            not is_gpu(), "`read_bytes()` is not yet implemented for GPUs."
        ]()

        @parameter
        if os_is_macos() or os_is_linux():
            var read = external_call["read", c_ssize_t](
                self.value, buffer.unsafe_ptr(), len(buffer)
            )
            if read < 0:
                raise Error("Failed to read bytes.")
            return read
        else:
            constrained[
                False,
                "`read_bytes()` is not yet implemented for unknown platform.",
            ]()
            return abort[UInt]()

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """

        @parameter
        for i in range(args.__len__()):
            args[i].write_to(self)
