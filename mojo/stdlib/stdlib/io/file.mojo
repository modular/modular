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
"""Provides APIs to read and write files.

These are Mojo built-ins, so you don't need to import them.

For example, here's how to read a file:

```mojo
var  f = open("my_file.txt", "r")
print(f.read())
f.close()
```

Or use a `with` statement to close the file automatically:

```mojo
with open("my_file.txt", "r") as f:
  print(f.read())
```

"""

from io.write import _WriteBufferStack
from os import PathLike, abort
from sys import external_call, size_of
from sys._libc_errno import ErrNo, get_errno
from sys.ffi import c_ssize_t

from memory import Span


# This type is used to pass into CompilerRT functions.  It is an owning
# pointer+length that is tightly coupled to the llvm::StringRef memory layout.
@register_passable
struct _OwnedStringRef(Boolable, Defaultable):
    var data: UnsafePointer[UInt8, MutOrigin.external]
    var length: Int

    fn __init__(out self):
        self.data = {}
        self.length = 0

    fn __del__(deinit self):
        if self.data:
            self.data.free()

    fn consume_as_error(var self) -> Error:
        result = Error()
        result.data = self.data
        result.loaded_length = -self.length

        # Don't free self.data in our dtor.
        self.data = {}
        return result

    fn __bool__(self) -> Bool:
        return self.length != 0


struct FileHandle(Defaultable, Movable, Writer):
    """File handle to an opened file."""

    var handle: OpaquePointer[MutOrigin.external]
    """The underlying pointer to the file handle."""

    fn __init__(out self):
        """Default constructor."""
        self.handle = {}

    fn __init__(out self, path: StringSlice, mode: StringSlice) raises:
        """Construct the FileHandle using the file path and mode.

        Args:
            path: The file path.
            mode: The mode to open the file in: {"r", "w", "rw", "a"}.

        Raises:
            If file open mode is not one of the supported modes.
            If there is an error when opening the file.
        """
        if not (mode == "r" or mode == "w" or mode == "rw" or mode == "a"):
            raise Error(
                'ValueError: invalid mode: "',
                mode,
                '". Can only be one of: {"r", "w", "rw", "a"}',
            )
        var err_msg = _OwnedStringRef()
        var handle = external_call[
            "KGEN_CompilerRT_IO_FileOpen", type_of(self.handle)
        ](path, mode, Pointer(to=err_msg))

        if err_msg:
            self.handle = {}
            raise err_msg^.consume_as_error()

        self.handle = handle

    fn __del__(deinit self):
        """Closes the file handle."""
        try:
            self.close()
        except:
            pass

    fn close(mut self) raises:
        """Closes the file handle.

        Raises:
            If the operation fails.
        """
        if not self.handle:
            return

        var err_msg = _OwnedStringRef()
        external_call["KGEN_CompilerRT_IO_FileClose", NoneType](
            self.handle, Pointer(to=err_msg)
        )

        if err_msg:
            raise err_msg^.consume_as_error()

        self.handle = {}

    fn read(self, size: Int = -1) raises -> String:
        """Reads data from a file and sets the file handle seek position. If
        size is left as the default of -1, it will read to the end of the file.
        Setting size to a number larger than what's in the file will set
        the String length to the total number of bytes, and read all the data.

        Args:
            size: Requested number of bytes to read (Default: -1 = EOF).

        Returns:
          The contents of the file.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        Read the entire file into a String:

        ```mojo
        var file = open("/tmp/example.txt", "r")
        var string = file.read()
        print(string)
        ```

        Read the first 8 bytes, skip 2 bytes, and then read the next 8 bytes:

        ```mojo
        import os
        var file = open("/tmp/example.txt", "r")
        var word1 = file.read(8)
        print(word1)
        _ = file.seek(2, os.SEEK_CUR)
        var word2 = file.read(8)
        print(word2)
        ```

        Read the last 8 bytes in the file, then the first 8 bytes
        ```mojo
        _ = file.seek(-8, os.SEEK_END)
        var last_word = file.read(8)
        print(last_word)
        _ = file.seek(8, os.SEEK_SET) # os.SEEK_SET is the default start of file
        var first_word = file.read(8)
        print(first_word)
        ```
        """

        var list = self.read_bytes(size)
        return String(bytes=list)

    fn read[
        dtype: DType, origin: Origin[True]
    ](self, buffer: Span[Scalar[dtype], origin]) raises -> Int:
        """Read data from the file into the Span.

        This will read n bytes from the file into the input Span where
        `0 <= n <= len(buffer)`.

        0 is returned when the file is at EOF, or a 0-sized buffer is
        passed in.

        Parameters:
            dtype: The type that the data will be represented as.
            origin: The origin of the passed in Span.

        Args:
            buffer: The mutable Span to read data into.

        Returns:
            The total amount of data that was read in bytes.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        ```mojo
        import os
        from sys.info import size_of

        comptime file_name = "/tmp/example.txt"
        var file = open(file_name, "r")

        # Allocate and load 8 elements
        var buffer = InlineArray[Float32, size=8](fill=0)
        var bytes = file.read(buffer)
        print("bytes read", bytes)

        var first_element = buffer[0]
        print(first_element)

        # Skip 2 elements
        _ = file.seek(2 * size_of[DType.float32](), os.SEEK_CUR)

        # Allocate and load 8 more elements from file handle seek position
        var buffer2 = InlineArray[Float32, size=8](fill=0)
        var bytes2 = file.read(buffer2)

        var eleventh_element = buffer2[0]
        var twelvth_element = buffer2[1]
        print(eleventh_element, twelvth_element)
        ```
        """

        if not self.handle:
            raise Error("invalid file handle")

        var fd = self._get_raw_fd()
        var bytes_read = external_call["read", c_ssize_t](
            fd,
            buffer.unsafe_ptr(),
            len(buffer) * size_of[dtype](),
        )

        if bytes_read < 0:
            var err = get_errno()
            raise Error("Failed to read from file: " + String(err))

        return Int(bytes_read)

    fn read_bytes(self, size: Int = -1) raises -> List[UInt8]:
        """Reads data from a file and sets the file handle seek position. If
        size is left as default of -1, it will read to the end of the file.
        Setting size to a number larger than what's in the file will be handled
        and set the List length to the total number of bytes in the file.

        Args:
            size: Requested number of bytes to read (Default: -1 = EOF).

        Returns:
            The contents of the file.

        Raises:
            An error if this file handle is invalid, or if the file read
            returned a failure.

        Examples:

        Reading the entire file into a List[Int8]:

        ```mojo
        var file = open("/tmp/example.txt", "r")
        var string = file.read_bytes()
        ```

        Reading the first 8 bytes, skipping 2 bytes, and then reading the next
        8 bytes:

        ```mojo
        import os
        var file = open("/tmp/example.txt", "r")
        var list1 = file.read(8)
        _ = file.seek(2, os.SEEK_CUR)
        var list2 = file.read(8)
        ```

        Reading the last 8 bytes in the file, then the first 8 bytes:

        ```mojo
        import os
        var file = open("/tmp/example.txt", "r")
        _ = file.seek(-8, os.SEEK_END)
        var last_data = file.read(8)
        _ = file.seek(8, os.SEEK_SET) # os.SEEK_SET is the default start of file
        var first_data = file.read(8)
        ```
        """
        if not self.handle:
            raise Error("invalid file handle")

        # Start out with the correct size if we know it, otherwise use 256.
        var result = List[UInt8](
            unsafe_uninit_length=size if size >= 0 else 256
        )

        var fd = self._get_raw_fd()
        var num_read = 0
        while True:
            # Read bytes into the list buffer and get the number of bytes
            # successfully read. This may return with a partial read, and
            # signifies EOF with a result of zero bytes.
            var chunk_bytes_to_read = len(result) - num_read
            var chunk_bytes_read = external_call["read", c_ssize_t](
                fd,
                result.unsafe_ptr() + num_read,
                chunk_bytes_to_read,
            )

            if chunk_bytes_read < 0:
                var err = get_errno()
                raise Error("Failed to read from file: " + String(err))

            num_read += Int(chunk_bytes_read)

            # If we read all of the 'size' bytes then we're done.
            if num_read == size or chunk_bytes_read == 0:
                result.shrink(num_read)  # Trim off any tail.
                break

            # If we are reading to EOF, keep reading the next chunk, taking
            # bigger bites each time.
            if size < 0:
                result.resize(unsafe_uninit_length=num_read * 2)

        return result^

    fn seek(self, offset: UInt64, whence: UInt8 = os.SEEK_SET) raises -> UInt64:
        """Seeks to the given offset in the file.

        Args:
            offset: The byte offset to seek to.
            whence: The reference point for the offset:
                os.SEEK_SET = 0: start of file (Default).
                os.SEEK_CUR = 1: current position.
                os.SEEK_END = 2: end of file.

        Raises:
            An error if this file handle is invalid, or if file seek returned a
            failure.

        Returns:
            The resulting byte offset from the start of the file.

        Examples:

        Skip 32 bytes from the current read position:

        ```mojo
        import os
        var f = open("/tmp/example.txt", "r")
        _ = f.seek(32, os.SEEK_CUR)
        ```

        Start from 32 bytes from the end of the file:

        ```mojo
        import os
        var f = open("/tmp/example.txt", "r")
        _ = f.seek(-32, os.SEEK_END)
        ```
        """
        if not self.handle:
            raise "invalid file handle"

        debug_assert(
            whence >= 0 and whence < 3,
            "Second argument to `seek` must be between 0 and 2.",
        )

        var fd = self._get_raw_fd()
        # lseek returns off_t which is typically Int64 on Unix systems
        var pos = external_call["lseek", Int64](fd, Int64(offset), Int(whence))

        if pos < 0:
            var err = get_errno()
            raise Error("Failed to seek in file: " + String(err))

        return UInt64(pos)

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.

        Notes:
            Passing an invalid file handle (e.g., after calling `close()`) is
            undefined behavior. In debug builds, this will trigger an assertion.
        """
        debug_assert(self.handle, "invalid file handle in write_bytes()")

        var fd = self._get_raw_fd()
        var bytes_written = external_call["write", c_ssize_t](
            fd, bytes.unsafe_ptr(), len(bytes)
        )

        debug_assert(bytes_written >= 0, "write() syscall failed")
        debug_assert(bytes_written == len(bytes), "incomplete write to file")

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.

        Notes:
            Passing an invalid file handle (e.g., after calling `close()`) is
            undefined behavior. In debug builds, this will trigger an assertion.
        """
        debug_assert(self.handle, "invalid file handle in write()")

        var file = FileDescriptor(self._get_raw_fd())
        var buffer = _WriteBufferStack(file)

        @parameter
        for i in range(args.__len__()):
            args[i].write_to(buffer)

        buffer.flush()

    fn _write(
        self,
        ptr: UnsafePointer[mut=False, UInt8, address_space=_],
        len: Int,
    ) raises:
        """Write the data to the file.

        Args:
          ptr: The pointer to the data to write.
          len: The length of the data buffer (in bytes).
        Raises:
            If the file handle is invalid, the write fails, or the write is incomplete.
        """
        if not self.handle:
            raise Error("invalid file handle")

        var fd = self._get_raw_fd()
        var bytes_written = external_call["write", c_ssize_t](
            fd, ptr.address, len
        )

        if bytes_written < 0:
            var err = get_errno()
            raise Error("Failed to write to file: " + String(err))

        if bytes_written != len:
            raise Error("Incomplete write to file")

    fn __enter__(var self) -> Self:
        """The function to call when entering the context.

        Returns:
            The file handle.
        """
        return self^

    fn _get_raw_fd(self) -> Int:
        return Int(
            external_call[
                "KGEN_CompilerRT_IO_GetFD",
                Int64,
            ](self.handle)
        )


fn open[
    PathLike: os.PathLike
](path: PathLike, mode: StringSlice) raises -> FileHandle:
    """Opens the file specified by path using the mode provided, returning a
    FileHandle.

    Parameters:
        PathLike: The a type conforming to the os.PathLike trait.

    Args:
        path: The path to the file to open.
        mode: The mode to open the file in: {"r", "w", "rw", "a"}.

    Returns:
        A file handle.

    Raises:
        If file open mode is not one of the supported modes.
        If there is an error when opening the file.
    """
    return FileHandle(path.__fspath__(), mode)
