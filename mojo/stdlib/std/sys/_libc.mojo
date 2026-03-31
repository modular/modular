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
"""Implements low-level bindings to functions from the C standard library.

The functions in this module are intended to be thin wrappers around their
C standard library counterparts. These are used to implement higher level
functionality in the rest of the Mojo standard library.
"""

from std.ffi import (
    c_char,
    c_int,
    c_size_t,
    c_pid_t,
    external_call,
    get_errno,
    CStringSlice,
    _CPointer,
)
from std.sys import CompilationTarget
from std.memory._nonnull import NonNullUnsafePointer

# ===-----------------------------------------------------------------------===#
# stdlib.h — core C standard library operations
# ===-----------------------------------------------------------------------===#


@always_inline
def free(ptr: UnsafePointer[mut=True, NoneType, ...]):
    # manually construct the call to free and attach the
    # correct attributes
    __mlir_op.`pop.external_call`[
        func=__mlir_attr[`"free" : !kgen.string`],
        _type=None,
        argAttrs=__mlir_attr.`[{llvm.allocptr}]`,
        funcAttrs=__mlir_attr.`[["allockind", "4"], ["alloc-family", "malloc"]]`,
    ](ptr)


@always_inline
def free(ptr: _CPointer[NoneType, ExternalOrigin[mut=True]]):
    """Frees memory previously allocated by `malloc`, `calloc`, or `realloc`.

    This overload accepts an `_CPointer` because it is valid in C to call
    `free` on a null pointer (it is a no-op).

    Args:
        ptr: A c-pointer to the memory to free.
    """
    free(
        UnsafePointer(to=ptr).bitcast[
            UnsafePointer[NoneType, ExternalOrigin[mut=True]]
        ]()[]
    )


@always_inline
def exit(status: c_int):
    external_call["exit", NoneType](status)


# ===-----------------------------------------------------------------------===#
# stdio.h — input/output operations
# ===-----------------------------------------------------------------------===#

comptime FILE_ptr = _CPointer[NoneType, ExternalOrigin[mut=True]]


@always_inline
def fdopen(fd: c_int, mode: UnsafePointer[mut=False, c_char, _]) -> FILE_ptr:
    return external_call["fdopen", FILE_ptr](fd, mode)


@always_inline
def fclose(stream: FILE_ptr) -> c_int:
    return external_call["fclose", c_int](stream)


@always_inline
def fflush(stream: FILE_ptr) -> c_int:
    return external_call["fflush", c_int](stream)


@always_inline
def popen(
    command: UnsafePointer[mut=False, c_char, _],
    type: UnsafePointer[mut=False, c_char, _],
) -> FILE_ptr:
    return external_call["popen", FILE_ptr](command, type)


@always_inline
def pclose(stream: FILE_ptr) -> c_int:
    return external_call["pclose", c_int](stream)


@always_inline
def setvbuf(
    stream: FILE_ptr,
    buffer: UnsafePointer[mut=True, c_char, _],
    mode: c_int,
    size: c_size_t,
) -> c_int:
    return external_call["setvbuf", c_int](stream, buffer)


struct BufferMode:
    """
    Modes for use in `setvbuf` to control buffer output.
    """

    comptime buffered = 0
    """Equivalent to `_IOFBF`."""
    comptime line_buffered = 1
    """Equivalent to `_IOLBF`."""
    comptime unbuffered = 2
    """Equivalent to `_IONBF`."""


# ===-----------------------------------------------------------------------===#
# spawn.h - Spawn process
# ===-----------------------------------------------------------------------===#


@always_inline
def posix_spawnp[
    argv_origin: ImmutOrigin,
    //,
](
    pid: NonNullUnsafePointer[mut=True, c_pid_t, _],
    file: CStringSlice[_],
    argv: NonNullUnsafePointer[Optional[CStringSlice[argv_origin]], _],
    envp: _CPointer[Optional[CStringSlice[ImmutAnyOrigin]], ImmutAnyOrigin],
) -> c_int:
    """[`posix_spawn`](https://pubs.opengroup.org/onlinepubs/007904975/functions/posix_spawn.html)
    — function creates a new process (child process) from the specified process image.

    Args:
        pid: UnsafePointer[c_pid_t], dest. for process id if spawned successfully.
        file: NULL terminated UnsafePointer[c_char] (C string), containing path to executable.
        argv: The UnsafePointer[c_char] array must be terminated with a NULL pointer.
        envp: The UnsafePointer[c_char] array must be terminated with a NULL pointer.
    """
    # TODO: Implement `const posix_spawn_file_actions_t`, `*file_actions, const posix_spawnattr_t *restrict attrp,`
    # to allow full control of how process is spawned
    return external_call["posix_spawnp", c_int](
        pid,
        file,
        _CPointer[NoneType, ExternalOrigin[mut=False]](),
        _CPointer[NoneType, ExternalOrigin[mut=False]](),
        argv,
        envp,
    )


# ===-----------------------------------------------------------------------===#
# unistd.h
# ===-----------------------------------------------------------------------===#


@always_inline
def dup(oldfd: c_int) -> c_int:
    return external_call["dup", c_int](oldfd)


@always_inline
def execvp[
    origin: ImmutOrigin,
    //,
](
    file: UnsafePointer[mut=False, c_char, _],
    argv: UnsafePointer[mut=False, _CPointer[mut=False, c_char, origin], _],
) -> c_int:
    """[`execvp`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/exec.html)
    — execute a file.

    Args:
        file: NULL terminated UnsafePointer[c_char] (C string), containing path to executable.
        argv: The UnsafePointer[c_char] array must be terminated with a NULL pointer.
    """
    return external_call["execvp", c_int](file, argv)


@always_inline
def vfork() -> c_int:
    """[`vfork()`](https://pubs.opengroup.org/onlinepubs/009696799/functions/vfork.html).
    """
    return external_call["vfork", c_int]()


struct SignalCodes:
    comptime HUP = 1  # (hang up)
    comptime INT = 2  # (interrupt)
    comptime QUIT = 3  # (quit)
    comptime ABRT = 6  # (abort)
    comptime KILL = 9  # (non-catchable, non-ignorable kill)
    comptime ALRM = 14  # (alarm clock)
    comptime TERM = 15  # (software termination signal)


@always_inline
def kill(pid: c_int, sig: c_int) -> c_int:
    """[`kill()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/kill.html)
    — send a signal to a process or group of processes."""
    return external_call["kill", c_int](pid, sig)


@always_inline
def pipe(fildes: UnsafePointer[mut=True, c_int, _]) -> c_int:
    """[`pipe()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/pipe.html) — create an interprocess channel.
    """
    return external_call["pipe", c_int](fildes)


@always_inline
def close(fd: c_int) -> c_int:
    """[`close()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/close.html)
    — close a file descriptor.
    """
    return external_call["close", c_int](fd)


@always_inline
def write(
    fd: c_int, buf: OpaquePointer[mut=False, _], nbyte: c_size_t
) -> c_int:
    """[`write()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/write.html)
    — write to a file descriptor.
    """
    return external_call["write", c_int](fd, buf, nbyte)


# ===-----------------------------------------------------------------------===#
# sys/wait.h - Control over file descriptors
# ===-----------------------------------------------------------------------===#


struct WaitFlags:
    """Flags for `waitpid`."""

    comptime WNOHANG: c_int = 1


# pid_t waitpid(pid_t pid, int *wstatus, int options);
@always_inline
def waitpid(
    pid: c_pid_t,
    status: UnsafePointer[mut=True, c_int, _],
    options: c_int,
) -> c_pid_t:
    """[`waitpid()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/waitpid.html)
    — Wait on child process to finish executing.
    """
    return external_call["waitpid", c_pid_t](pid, status, options)


# ===-----------------------------------------------------------------------===#
# fcntl.h - Control over file descriptors
# ===-----------------------------------------------------------------------===#


struct FcntlCommands:
    comptime F_GETFD: c_int = 1
    comptime F_SETFD: c_int = 2


struct FcntlFDFlags:
    comptime FD_CLOEXEC: c_int = 1


@always_inline
def fcntl[*types: Intable](fd: c_int, cmd: c_int, *args: *types) -> c_int:
    """[`fcntl()`](https://pubs.opengroup.org/onlinepubs/9799919799/functions/fcntl.html)
    — file control.
    """
    return external_call["fcntl", c_int](fd, cmd, args)


# ===-----------------------------------------------------------------------===#
# dlfcn.h — dynamic library operations
# ===-----------------------------------------------------------------------===#


@always_inline
def dlerror(out result: _CPointer[c_char, MutExternalOrigin]):
    result = external_call["dlerror", type_of(result)]()


@always_inline
def dlopen(
    filename: UnsafePointer[mut=False, c_char, _], flags: c_int
) -> _CPointer[NoneType, MutExternalOrigin]:
    return external_call["dlopen", _CPointer[NoneType, MutExternalOrigin]](
        filename, flags
    )


@always_inline
def dlclose(handle: OpaquePointer[mut=True, _]) -> c_int:
    return external_call["dlclose", c_int](handle)


@always_inline
def dlsym[
    # Default `dlsym` result is an OpaquePointer.
    result_type: AnyType = NoneType
](
    handle: OpaquePointer,
    name: UnsafePointer[mut=False, c_char, _],
    out result: _CPointer[result_type, MutExternalOrigin],
):
    result = external_call["dlsym", type_of(result)](handle, name)


def realpath(
    path: UnsafePointer[mut=False, c_char, _],
    resolved_path: _CPointer[mut=True, c_char, _] = _CPointer[
        c_char, MutExternalOrigin
    ](),
    out result: _CPointer[c_char, MutExternalOrigin],
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
