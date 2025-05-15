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

from os import PathLike, abort
from sys import external_call, sizeof
from sys.ffi import OpaquePointer

from memory import AddressSpace, Span, UnsafePointer

from utils import write_buffered


# This type is used to pass into CompilerRT functions.  It is an owning
# pointer+length that is tightly coupled to the llvm::StringRef memory layout.
@register_passable
struct _OwnedStringRef(Boolable):
    var data: UnsafePointer[UInt8]
    var length: Int

    fn __init__(out self):
        self.data = UnsafePointer[UInt8]()
        self.length = 0

    fn __del__(owned self):
        if self.data:
            self.data.free()

    fn consume_as_error(owned self) -> Error:
        result = Error()
        result.data = self.data
        result.loaded_length = -self.length

        # Don't free self.data in our dtor.
        self.data = UnsafePointer[UInt8]()
        return result

    fn __bool__(self) -> Bool:
        return self.length != 0


struct FileHandle(Writer):
    """File handle to an opened file."""

    var handle: OpaquePointer
    """The underlying pointer to the file handle."""

    fn __init__(out self):
        """Default constructor."""
        self.handle = OpaquePointer()

    fn __init__(out self, path: StringSlice, mode: StringSlice) raises:
        """Construct the FileHandle using the file path and mode.

        Args:
          path: The file path.
          mode: The mode to open the file in (the mode can be "r" or "w" or "rw").
        """
        var err_msg = _OwnedStringRef()
        var handle = external_call[
            "KGEN_CompilerRT_IO_FileOpen", OpaquePointer
        ](path, mode, Pointer(to=err_msg))

        if err_msg:
            self.handle = OpaquePointer()
            raise err_msg^.consume_as_error()

        self.handle = handle

    fn __del__(owned self):
        """Closes the file handle."""
        try:
            self.close()
        except:
            pass

    fn close(mut self) raises:
        """Closes the file handle."""
        if not self.handle:
            return

        var err_msg = _OwnedStringRef()
        external_call["KGEN_CompilerRT_IO_FileClose", NoneType](
            self.handle, Pointer(to=err_msg)
        )

        if err_msg:
            raise err_msg^.consume_as_error()

        self.handle = OpaquePointer()

    fn __moveinit__(out self, owned existing: Self):
        """Moves constructor for the file handle.

        Args:
          existing: The existing file handle.
        """
        self.handle = existing.handle
        existing.handle = OpaquePointer()

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
        .
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
        from sys.info import sizeof

        alias file_name = "/tmp/example.txt"
        var file = open(file_name, "r")

        # Allocate and load 8 elements
        var ptr = UnsafePointer[Float32].alloc(8)
        var buffer = InlineArray[Float32, size=8](fill=0)
        var bytes = file.read(buffer)
        print("bytes read", bytes)

        var first_element = ptr[0]
        print(first_element)

        # Skip 2 elements
        _ = file.seek(2 * sizeof[DType.float32](), os.SEEK_CUR)

        # Allocate and load 8 more elements from file handle seek position
        var ptr2 = UnsafePointer[Float32].alloc(8)
        var buffer2 = InlineArray[Float32, size=8](fill=0)
        var bytes2 = file.read(buffer2)

        var eleventh_element = ptr2[0]
        var twelvth_element = ptr2[1]
        print(eleventh_element, twelvth_element)
        ```
        """

        if not self.handle:
            raise Error("invalid file handle")

        var err_msg = _OwnedStringRef()

        var bytes_read = external_call["KGEN_CompilerRT_IO_FileReadBytes", Int](
            self.handle,
            buffer.unsafe_ptr(),
            len(buffer) * sizeof[dtype](),
            Pointer(to=err_msg),
        )

        if err_msg:
            raise (err_msg^).consume_as_error()

        return bytes_read

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
        .
        """
        if not self.handle:
            raise Error("invalid file handle")

        # Start out with the correct size if we know it, otherwise use 256.
        var result = List[UInt8](
            unsafe_uninit_length=size if size >= 0 else 256
        )

        var err_msg = _OwnedStringRef()
        var num_read = 0
        while True:
            # Read bytes into the list buffer and get the number of bytes
            # successfully read. This may return with a partial read, and
            # signifies EOF with a result of zero bytes.
            var chunk_bytes_to_read = len(result) - num_read
            var chunk_bytes_read = external_call[
                "KGEN_CompilerRT_IO_FileReadBytes", Int
            ](
                self.handle,
                result.unsafe_ptr() + num_read,
                chunk_bytes_to_read,
                Pointer(to=err_msg),
            )
            if err_msg:
                raise err_msg^.consume_as_error()

            num_read += chunk_bytes_read

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
        .
        """
        if not self.handle:
            raise "invalid file handle"

        debug_assert(
            whence >= 0 and whence < 3,
            "Second argument to `seek` must be between 0 and 2.",
        )
        var err_msg = _OwnedStringRef()
        var pos = external_call["KGEN_CompilerRT_IO_FileSeek", UInt64](
            self.handle, offset, whence, Pointer(to=err_msg)
        )

        if err_msg:
            raise err_msg^.consume_as_error()

        return pos

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """
        Write a span of bytes to the file.

        Args:
            bytes: The byte span to write to this file.
        """
        var err_msg = _OwnedStringRef()
        external_call["KGEN_CompilerRT_IO_FileWrite", NoneType](
            self.handle,
            bytes.unsafe_ptr(),
            len(bytes),
            Pointer(to=err_msg),
        )

        if err_msg:
            abort(err_msg^.consume_as_error())

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """
        var file = FileDescriptor(self._get_raw_fd())
        write_buffered(file, args)

    fn _write[
        address_space: AddressSpace
    ](
        self, ptr: UnsafePointer[UInt8, address_space=address_space], len: Int
    ) raises:
        """Write the data to the file.

        Params:
          address_space: The address space of the pointer.

        Args:
          ptr: The pointer to the data to write.
          len: The length of the pointer (in bytes).
        """
        if not self.handle:
            raise Error("invalid file handle")

        var err_msg = _OwnedStringRef()
        external_call["KGEN_CompilerRT_IO_FileWrite", NoneType](
            self.handle,
            ptr.address,
            len,
            Pointer(to=err_msg),
        )

        if err_msg:
            raise err_msg^.consume_as_error()

    fn __enter__(owned self) -> Self:
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
      mode: The mode to open the file in (the mode can be "r" or "w").

    Returns:
      A file handle.
    """
    return FileHandle(path.__fspath__(), mode)
