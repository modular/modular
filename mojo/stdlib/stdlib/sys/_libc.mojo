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
"""Implements low-level bindings to functions from the C standard library.

The functions in this module are intended to be thin wrappers around their
C standard library counterparts. These are used to implement higher level
functionality in the rest of the Mojo standard library.
"""

from sys import CompilationTarget
from sys.ffi import c_char, c_int, c_size_t, get_errno

# ===-----------------------------------------------------------------------===#
# stdlib.h — core C standard library operations
# ===-----------------------------------------------------------------------===#


@always_inline
fn free(ptr: UnsafePointer[mut=True, NoneType, **_]):
    # manually construct the call to free and attach the
    # correct attributes
    __mlir_op.`pop.external_call`[
        func = __mlir_attr[`"free" : !kgen.string`],
        _type=None,
        argAttrs = __mlir_attr.`[{llvm.allocptr}]`,
        funcAttrs = __mlir_attr.`[["allockind", "4"], ["alloc-family", "malloc"]]`,
    ](ptr)


@always_inline
fn exit(status: c_int):
    external_call["exit", NoneType](status)


# ===-----------------------------------------------------------------------===#
# stdio.h — input/output operations
# ===-----------------------------------------------------------------------===#

comptime FILE_ptr = OpaquePointer[MutOrigin.external]


@always_inline
fn fdopen(fd: c_int, mode: UnsafePointer[mut=False, c_char]) -> FILE_ptr:
    return external_call["fdopen", FILE_ptr](fd, mode)


@always_inline
fn fclose(stream: FILE_ptr) -> c_int:
    return external_call["fclose", c_int](stream)


@always_inline
fn fflush(stream: FILE_ptr) -> c_int:
    return external_call["fflush", c_int](stream)


@always_inline
fn popen(
    command: UnsafePointer[mut=False, c_char],
    type: UnsafePointer[mut=False, c_char],
) -> FILE_ptr:
    return external_call["popen", FILE_ptr](command, type)


@always_inline
fn pclose(stream: FILE_ptr) -> c_int:
    return external_call["pclose", c_int](stream)


@always_inline
fn setvbuf(
    stream: FILE_ptr,
    buffer: UnsafePointer[mut=True, c_char],
    mode: c_int,
    size: c_size_t,
) -> c_int:
    return external_call["setvbuf", c_int](stream, buffer)


struct BufferMode:
    """
    Modes for use in `setvbuf` to control buffer output.
    """

    alias buffered = 0
    """Equivalent to `_IOFBF`."""
    alias line_buffered = 1
    """Equivalent to `_IOLBF`."""
    alias unbuffered = 2
    """Equivalent to `_IONBF`."""


# ===-----------------------------------------------------------------------===#
# unistd.h
# ===-----------------------------------------------------------------------===#


@always_inline
fn dup(oldfd: c_int) -> c_int:
    return external_call["dup", c_int](oldfd)


@always_inline
fn execvp[
    origin: ImmutOrigin, //,
](
    file: UnsafePointer[mut=False, c_char],
    argv: UnsafePointer[mut=False, UnsafePointer[c_char, origin]],
) -> c_int:
    """[`execvp`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/exec.html)
    — execute a file.

    Args:
        file: NULL terminated UnsafePointer[c_char] (C string), containing path to executable.
        argv: The UnsafePointer[c_char] array must be terminated with a NULL pointer.
    """
    return external_call["execvp", c_int](file, argv)


@always_inline
fn vfork() -> c_int:
    """[`vfork()`](https://pubs.opengroup.org/onlinepubs/009696799/functions/vfork.html).
    """
    return external_call["vfork", c_int]()


struct SignalCodes:
    alias HUP = 1  # (hang up)
    alias INT = 2  # (interrupt)
    alias QUIT = 3  # (quit)
    alias ABRT = 6  # (abort)
    alias KILL = 9  # (non-catchable, non-ignorable kill)
    alias ALRM = 14  # (alarm clock)
    alias TERM = 15  # (software termination signal)


@always_inline
fn kill(pid: c_int, sig: c_int) -> c_int:
    """[`kill()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/kill.html)
    — send a signal to a process or group of processes."""
    return external_call["kill", c_int](pid, sig)


@always_inline
fn pipe(fildes: UnsafePointer[mut=True, c_int]) -> c_int:
    """[`pipe()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pipe.html) — create an interprocess channel.
    """
    return external_call["pipe", c_int](fildes)


@always_inline
fn close(fd: c_int) -> c_int:
    """[`close()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/close.html)
    — close a file descriptor.
    """
    return external_call["close", c_int](fd)


@always_inline
fn write(fd: c_int, buf: OpaquePointer[mut=False], nbyte: c_size_t) -> c_int:
    """[`write()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/write.html)
    — write to a file descriptor.
    """
    return external_call["write", c_int](fd, buf, nbyte)


# ===-----------------------------------------------------------------------===#
# fcntl.h - Control over file descriptors
# ===-----------------------------------------------------------------------===#


struct FcntlCommands:
    alias F_GETFD: c_int = 1
    alias F_SETFD: c_int = 2


struct FcntlFDFlags:
    alias FD_CLOEXEC: c_int = 1


@always_inline
fn fcntl[*types: Intable](fd: c_int, cmd: c_int, *args: *types) -> c_int:
    """[`fcntl()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/fcntl.html)
    — file control.
    """
    return external_call["fcntl", c_int](fd, cmd, args)


# ===-----------------------------------------------------------------------===#
# dlfcn.h — dynamic library operations
# ===-----------------------------------------------------------------------===#


@always_inline
fn dlerror(out result: UnsafePointer[c_char, MutOrigin.external]):
    result = external_call["dlerror", type_of(result)]()


@always_inline
fn dlopen(
    filename: UnsafePointer[mut=False, c_char], flags: c_int
) -> OpaquePointer[MutOrigin.external]:
    return external_call["dlopen", OpaquePointer[MutOrigin.external]](
        filename, flags
    )


@always_inline
fn dlclose(handle: OpaquePointer[mut=True]) -> c_int:
    return external_call["dlclose", c_int](handle)


@always_inline
fn dlsym[
    # Default `dlsym` result is an OpaquePointer.
    result_type: AnyType = NoneType
](
    handle: OpaquePointer,
    name: UnsafePointer[mut=False, c_char],
    out result: UnsafePointer[result_type, MutOrigin.external],
):
    result = external_call["dlsym", type_of(result)](handle, name)


fn realpath(
    path: UnsafePointer[mut=False, c_char],
    resolved_path: UnsafePointer[mut=True, c_char] = {},
    out result: UnsafePointer[c_char, MutOrigin.external],
):
    """Expands all symbolic links and resolves references to /./, /../ and extra
    '/' characters in the null-terminated string named by path to produce a
    canonicalized absolute pathname.  The resulting pathname is stored as a
    null-terminated string, up to a maximum of PATH_MAX bytes, in the buffer
    pointed to by resolved_path.  The resulting path will have no symbolic link,
    /./ or /../ components.

    If resolved_path is a NULL pointer, then realpath() uses malloc(3) to
    allocate a buffer of up to PATH_MAX bytes to hold the resolved pathname, and
    returns a pointer to this buffer. The caller is responsible for deallocating
    the buffer in this scenario.

    Args:
        path: The path to resolve.
        resolved_path: The buffer to store the resolved path. If this is a NULL
            pointer then libc will allocate a buffer of up to PATH_MAX bytes to
            hold the resolved pathname. The caller is responsible for
            deallocating the buffer in this scenario.

    Returns:
        A pointer to the resolved path.
    """
    return external_call["realpath", type_of(result)](path, resolved_path)
