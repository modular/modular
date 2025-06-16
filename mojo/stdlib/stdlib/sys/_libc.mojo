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

from sys import os_is_windows
from sys.ffi import OpaquePointer, c_char, c_int, c_size_t


# ===-----------------------------------------------------------------------===#
# stdlib.h — core C standard library operations
# ===-----------------------------------------------------------------------===#


@always_inline
fn free(ptr: UnsafePointer[NoneType, **_]):
    external_call["free", NoneType](ptr)


@always_inline
fn exit(status: c_int):
    external_call["exit", NoneType](status)


# ===-----------------------------------------------------------------------===#
# stdio.h — input/output operations
# ===-----------------------------------------------------------------------===#

alias FILE_ptr = OpaquePointer


@always_inline
fn fdopen(fd: c_int, mode: UnsafePointer[c_char, **_]) -> FILE_ptr:
    alias name = "_fdopen" if os_is_windows() else "fdopen"

    return external_call[name, FILE_ptr](fd, mode)


@always_inline
fn fclose(stream: FILE_ptr) -> c_int:
    return external_call["fclose", c_int](stream)


@always_inline
fn fflush(stream: FILE_ptr) -> c_int:
    return external_call["fflush", c_int](stream)


@always_inline
fn popen(
    command: UnsafePointer[c_char],
    type: UnsafePointer[c_char],
) -> FILE_ptr:
    return external_call["popen", FILE_ptr](command, type)


@always_inline
fn pclose(stream: FILE_ptr) -> c_int:
    return external_call["pclose", c_int](stream)


@always_inline
fn setvbuf(
    stream: FILE_ptr, buffer: UnsafePointer[c_char], mode: c_int, size: c_size_t
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
    alias name = "_dup" if os_is_windows() else "dup"

    return external_call[name, c_int](oldfd)


@always_inline
fn execvp(
    file: UnsafePointer[c_char], argv: UnsafePointer[UnsafePointer[c_char]]
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
fn pipe(fildes: UnsafePointer[c_int]) -> c_int:
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
fn write(fd: c_int, buf: OpaquePointer, nbyte: c_size_t) -> c_int:
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
fn dlerror() -> UnsafePointer[c_char]:
    return external_call["dlerror", UnsafePointer[c_char]]()


@always_inline
fn dlopen(filename: UnsafePointer[c_char, **_], flags: c_int) -> OpaquePointer:
    return external_call["dlopen", OpaquePointer](filename, flags)


@always_inline
fn dlclose(handle: OpaquePointer) -> c_int:
    return external_call["dlclose", c_int](handle)


@always_inline
fn dlsym[
    # Default `dlsym` result is an OpaquePointer.
    result_type: AnyType = NoneType
](
    handle: OpaquePointer,
    name: UnsafePointer[c_char, **_],
) -> UnsafePointer[
    result_type
]:
    return external_call["dlsym", UnsafePointer[result_type]](handle, name)
