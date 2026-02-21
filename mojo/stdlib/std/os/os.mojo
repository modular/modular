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
"""Provides functions to access operating-system dependent functionality, including
file system operations.

You can import a method from the `os` package. For example:

```mojo
from os import listdir
```
"""

from collections import InlineArray, List
from collections.string.string_slice import _unsafe_strlen
from io import FileDescriptor
from ffi import c_char, c_int, external_call, get_errno
from sys import CompilationTarget, is_gpu

from .path import isdir, split
from .pathlike import PathLike

# TODO move this to a more accurate location once nt/posix like modules are in stdlib
comptime sep = "/"
"""The path separator for the current platform."""


# ===----------------------------------------------------------------------=== #
# SEEK Constants
# ===----------------------------------------------------------------------=== #


comptime SEEK_SET: UInt8 = 0
"""Seek from the beginning of the file."""
comptime SEEK_CUR: UInt8 = 1
"""Seek from the current position."""
comptime SEEK_END: UInt8 = 2
"""Seek from the end of the file."""


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


struct _dirent_linux(Copyable):
    comptime MAX_NAME_SIZE = 256
    var d_ino: Int64
    """File serial number."""
    var d_off: Int64
    """Seek offset value."""
    var d_reclen: Int16
    """Length of the record."""
    var d_type: Int8
    """Type of file."""
    var name: InlineArray[c_char, Self.MAX_NAME_SIZE]
    """Name of entry."""


struct _dirent_macos(Copyable):
    comptime MAX_NAME_SIZE = 1024
    var d_ino: Int64
    """File serial number."""
    var d_off: Int64
    """Seek offset value."""
    var d_reclen: Int16
    """Length of the record."""
    var d_namlen: Int16
    """Length of the name."""
    var d_type: Int8
    """Type of file."""
    var name: InlineArray[c_char, Self.MAX_NAME_SIZE]
    """Name of entry."""


struct _DirHandle:
    """Handle to an open directory descriptor opened via opendir."""

    var _handle: OpaquePointer[MutExternalOrigin]

    fn __init__(out self, var path: String) raises:
        """Construct the _DirHandle using the path provided.

        Args:
          path: The path to open.
        """
        if not isdir(path):
            raise Error("the directory '", path, "' does not exist")

        self._handle = external_call["opendir", type_of(self._handle)](
            path.as_c_string_slice().unsafe_ptr()
        )

        if not self._handle:
            var err = get_errno()
            raise Error(
                "unable to open the directory '",
                path,
                "'",
                " Err: ",
                String(err),
            )

    fn __del__(deinit self):
        """Closes the handle opened via popen."""
        _ = external_call["closedir", Int32](self._handle)

    fn list(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
          A string containing the output of running the command.
        """

        comptime if CompilationTarget.is_linux():
            return self._list_linux()
        else:
            return self._list_macos()

    fn _list_linux(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
            A string containing the output of running the command.
        """
        var res = List[String]()

        while True:
            var ep = external_call[
                "readdir", UnsafePointer[_dirent_linux, MutExternalOrigin]
            ](self._handle)
            if not ep:
                break
            ref name = ep.take_pointee().name
            var name_ptr = name.unsafe_ptr().bitcast[Byte]()
            var name_str = StringSlice[origin_of(name)](
                ptr=name_ptr,
                length=Int(
                    _unsafe_strlen(name_ptr, _dirent_linux.MAX_NAME_SIZE)
                ),
            )
            if name_str == "." or name_str == "..":
                continue
            res.append(String(name_str))

        return res^

    fn _list_macos(self) -> List[String]:
        """Reads all the data from the handle.

        Returns:
            A string containing the output of running the command.
        """
        var res = List[String]()

        while True:
            var ep = external_call[
                "readdir", UnsafePointer[_dirent_macos, MutExternalOrigin]
            ](self._handle)
            if not ep:
                break
            ref name = ep.take_pointee().name
            var name_ptr = name.unsafe_ptr().bitcast[Byte]()
            var name_str = StringSlice[origin_of(name)](
                ptr=name_ptr,
                length=Int(
                    _unsafe_strlen(name_ptr, _dirent_macos.MAX_NAME_SIZE)
                ),
            )
            if name_str == "." or name_str == "..":
                continue
            res.append(String(name_str))

        return res^


# ===----------------------------------------------------------------------=== #
# getuid
# ===----------------------------------------------------------------------=== #
fn getuid() -> Int:
    """Retrieve the user ID of the calling process.

    Returns:
        The user ID of the calling process.

    Constraints:
        This function is constrained to run on Linux or macOS operating systems only.
    """
    return Int(external_call["getuid", UInt32]())


# ===----------------------------------------------------------------------=== #
# listdir
# ===----------------------------------------------------------------------=== #


fn listdir[PathLike: os.PathLike](path: PathLike) raises -> List[String]:
    """Gets the list of entries contained in the path provided.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Returns:
      Returns the list of entries in the path provided.

    Raises:
        If the operation fails.
    """
    var dir = _DirHandle(path.__fspath__())
    return dir.list()


# ===----------------------------------------------------------------------=== #
# abort
# ===----------------------------------------------------------------------=== #


@always_inline
fn abort() -> Never:
    """Terminates execution, using a target dependent trap instruction if
    available.
    """

    __mlir_op.`llvm.intr.trap`()

    # We need to satisfy the noreturn checker.
    while True:
        pass


@always_inline
fn abort[*, prefix: StaticString = "ABORT:"](message: String) -> Never:
    """Calls a target dependent trap instruction if available.

    Parameters:
        prefix: A static string prefix to include before the message.

    Args:
        message: The message to include when aborting.
    """

    comptime if not is_gpu():
        print(prefix, message, flush=True)

    abort()


# ===----------------------------------------------------------------------=== #
# remove/unlink
# ===----------------------------------------------------------------------=== #
fn remove[PathLike: os.PathLike](path: PathLike) raises:
    """Removes the specified file.

    If the path is a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file.


    Raises:
        If the operation fails.
    """
    var fspath = path.__fspath__()
    var error = external_call["unlink", Int32](
        fspath.as_c_string_slice().unsafe_ptr()
    )

    if error != 0:
        var err = get_errno()
        raise Error("Can not remove file: ", fspath, " Err: ", String(err))


fn unlink[PathLike: os.PathLike](path: PathLike) raises:
    """Removes the specified file.

    If the path is a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the file.


    Raises:
        If the operation fails.
    """
    remove(path.__fspath__())


# ===----------------------------------------------------------------------=== #
# symlink
# ===----------------------------------------------------------------------=== #


fn symlink[
    TargetType: os.PathLike, LinkType: os.PathLike
](target: TargetType, linkpath: LinkType) raises:
    """Creates a symlink.

    If linkpath already exists it will not be overwritten.
    See `symlink(2)`

    Parameters:
        TargetType: The path type of the link target.
        LinkType: The path type of the link.

    Args:
        target: The target of the symbolic link.
        linkpath: The path of the symbolic link to create.

    Raises:
        If the operation fails.
    """
    var target_fspath = target.__fspath__()
    var linkpath_fspath = linkpath.__fspath__()

    var error = external_call["symlink", c_int](
        target_fspath.as_c_string_slice().unsafe_ptr(),
        linkpath_fspath.as_c_string_slice().unsafe_ptr(),
    )

    if error != 0:
        var err = get_errno()
        raise Error(
            "Can not create symlink from ",
            linkpath_fspath,
            " to ",
            target_fspath,
            " Err: ",
            String(err),
        )


# ===----------------------------------------------------------------------=== #
# link
# ===----------------------------------------------------------------------=== #


fn link[
    OldType: os.PathLike, NewType: os.PathLike
](oldpath: OldType, newpath: NewType) raises:
    """Creates a new hard-link to an existing file.

    Parameters:
        OldType: The path type of the existing file.
        NewType: The path type of the file to create.

    Args:
        oldpath: The exsting file.
        newpath: The new file.

    Raises:
        If the operation fails.
    """
    var oldpath_fspath = oldpath.__fspath__()
    var newpath_fspath = newpath.__fspath__()

    var error = external_call["link", Int32](
        oldpath_fspath.as_c_string_slice().unsafe_ptr(),
        newpath_fspath.as_c_string_slice().unsafe_ptr(),
    )

    if error != 0:
        var err = get_errno()
        raise Error(
            "Can not create link from ",
            newpath_fspath,
            " to ",
            oldpath_fspath,
            " Err: ",
            String(err),
        )


# ===----------------------------------------------------------------------=== #
# mkdir/rmdir
# ===----------------------------------------------------------------------=== #


fn mkdir[PathLike: os.PathLike](path: PathLike, mode: Int = 0o777) raises:
    """Creates a directory at the specified path.

    If the directory can not be created an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
      mode: The mode to create the directory with.

    Raises:
        If the operation fails.
    """

    var fspath = path.__fspath__()
    var error = external_call["mkdir", Int32](
        fspath.as_c_string_slice().unsafe_ptr(), mode
    )
    if error != 0:
        var err = get_errno()
        raise Error("Can not create directory: ", fspath, " Err: ", String(err))


fn makedirs[
    PathLike: os.PathLike
](path: PathLike, mode: Int = 0o777, exist_ok: Bool = False) raises -> None:
    """Creates a specified leaf directory along with any necessary intermediate
    directories that don't already exist.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.
      mode: The mode to create the directory with.
      exist_ok: Ignore error if `True` and path exists (default `False`).

    Raises:
        If the operation fails.
    """
    var head, tail = split(path)
    if not tail:
        head, tail = split(head)
    if head and tail and not os.path.exists(head):
        try:
            makedirs(head, exist_ok=exist_ok)
        except:
            # Defeats race condition when another thread created the path
            pass
        # xxx/newdir/. exists if xxx/newdir exists
        if tail == ".":
            return None
    try:
        mkdir(path, mode)
    except e:
        if not exist_ok:
            raise Error(
                e,
                "\nset `makedirs(path, exist_ok=True)` to allow existing dirs",
            )
        if not os.path.isdir(path):
            raise Error("path not created: ", path.__fspath__(), "\n", e)


fn rmdir[PathLike: os.PathLike](path: PathLike) raises:
    """Removes the specified directory.

    If the path is not a directory or it can not be deleted, an error is raised.
    Absolute and relative paths are allowed, relative paths are resolved from cwd.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Raises:
        If the operation fails.
    """
    var fspath = path.__fspath__()
    var error = external_call["rmdir", Int32](
        fspath.as_c_string_slice().unsafe_ptr()
    )
    if error != 0:
        var err = get_errno()
        raise Error("Can not remove directory: ", fspath, " Err: ", String(err))


fn removedirs[PathLike: os.PathLike](path: PathLike) raises -> None:
    """Removes a leaf directory and all empty intermediate ones.

    Directories corresponding to rightmost path segments will be pruned away
    until either the whole path is consumed or an error occurs. Errors during
    this latter phase are ignored, which occur when a directory was not empty.

    Parameters:
      PathLike: The a type conforming to the os.PathLike trait.

    Args:
      path: The path to the directory.

    Raises:
        If the operation fails.
    """
    rmdir(path)
    var head, tail = os.path.split(path)
    if not tail:
        head, tail = os.path.split(head)
    while head and tail:
        try:
            rmdir(head)
        except:
            break
        head, tail = os.path.split(head)


# ===----------------------------------------------------------------------=== #
# isatty
# ===----------------------------------------------------------------------=== #


fn isatty(fd: Int) -> Bool:
    """Checks whether a file descriptor refers to a terminal.

    Returns `True` if the file descriptor `fd` is open and connected to a
    tty(-like) device, otherwise `False`. On GPUs, the function always returns
    `False`.

    Args:
        fd: A file descriptor.

    Returns:
        `True` if `fd` is connected to a terminal, `False` otherwise.

    Examples:
        ```mojo
        from os import isatty

        # Check if stdout (fd=1) is a terminal
        if isatty(1):
            print("Running in a terminal")
        else:
            print("Output is redirected")
        ```
    """

    return FileDescriptor(fd).isatty()
