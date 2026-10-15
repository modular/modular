#!/usr/bin/env python3
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

"""Run a wasm64 module under Wasmtime with real WASI I/O.

Wasmtime is the engine here: it compiles and runs the module, and memory64
is enabled. But Wasmtime's built-in WASI preview1 implementation defines its
imports with wasm32 signatures (i32 pointers), so a wasm64 module that
imports `wasi_snapshot_preview1` with i64 pointers cannot use it. This
runner therefore provides its own WASI preview1 host with **i64 pointers**,
backed by the real operating system (stdio, files, clocks, random).

Usage:
  wasmtime-runner.py MODULE [ARG...] [--dir PATH] [--env K=V]

  MODULE   a .wasm file exporting `_start`, run as a WASI command
           (argv = [MODULE, ARG...]).
"""

from __future__ import annotations

import argparse
import errno
import os
import select
import stat
import sys
import time
from collections.abc import Callable, Sequence
from typing import BinaryIO

import wasmtime

# WASI preview1 errno values.
ESUCCESS = 0
EACCES = 2
EBADF = 8
EEXIST = 20
EINVAL = 28
EIO = 29
EISDIR = 31
ENAMETOOLONG = 37
ENOENT = 44
ENOSPC = 51
ENOTDIR = 54
ENOTEMPTY = 55
ENOTTY = 59
EOVERFLOW = 61
EPERM = 63
EPIPE = 64
ERANGE = 68

FILETYPE_UNKNOWN = 0
FILETYPE_CHARACTER_DEVICE = 2
FILETYPE_DIRECTORY = 3
FILETYPE_REGULAR_FILE = 4
FILETYPE_SYMBOLIC_LINK = 7

FDFLAG_APPEND = 1
LOOKUP_SYMLINK_FOLLOW = 1

# wasm64 iovec layout: pointer u64 @0, length u64 @8, stride 16.
IOV_SIZE = 16

# WASI path_open oflags and fdflags.
OFLAGS_CREAT = 1 << 0
OFLAGS_DIRECTORY = 1 << 1
OFLAGS_EXCL = 1 << 2
OFLAGS_TRUNC = 1 << 3

RIGHTS_FD_WRITE = 1 << 6
RIGHTS_ALL = (1 << 64) - 1


# Host errno to WASI preview1 errno.
_LINUX_TO_WASI = {
    errno.EPERM: EPERM,
    errno.ENOENT: ENOENT,
    errno.EIO: EIO,
    errno.EBADF: EBADF,
    errno.EACCES: EACCES,
    errno.EEXIST: EEXIST,
    errno.ENOTDIR: ENOTDIR,
    errno.EISDIR: EISDIR,
    errno.EINVAL: EINVAL,
    errno.ENOTTY: ENOTTY,
    errno.ENOSPC: ENOSPC,
    errno.EPIPE: EPIPE,
    errno.ERANGE: ERANGE,
    errno.ENAMETOOLONG: ENAMETOOLONG,
    errno.ENOTEMPTY: ENOTEMPTY,
    errno.EOVERFLOW: EOVERFLOW,
}


def wasi_errno_of(exc: OSError) -> int:
    return _LINUX_TO_WASI.get(exc.errno or 0, EIO)


class Fd:
    """One entry in the WASI fd table."""

    def __init__(
        self,
        kind: str,
        fileobj: BinaryIO | None = None,
        path: str | None = None,
        preopen: str | None = None,
    ):
        self.kind = kind  # 'stdio' | 'file' | 'dir'
        self.fileobj = fileobj
        self.path = path  # absolute path for dirs/preopens
        self.preopen = preopen  # the path this fd was preopened as


class WasiHost:
    def __init__(
        self,
        preopens: Sequence[str],
        argv: Sequence[str | bytes] = (),
    ):
        self.argv = [
            a.encode("utf-8") if isinstance(a, str) else a for a in argv
        ]
        # Pre-encoded like argv: sizes_get must count UTF-8 bytes, not chars,
        # and Linux env can carry non-UTF-8 bytes (surrogateescape round-trip).
        self.environ = [
            f"{k}={v}".encode("utf-8", "surrogateescape")
            for k, v in os.environ.items()
        ]
        self.fds = {
            0: Fd("stdio", fileobj=sys.stdin.buffer),
            1: Fd("stdio", fileobj=sys.stdout.buffer),
            2: Fd("stdio", fileobj=sys.stderr.buffer),
        }
        self.next_fd = 3
        for path in preopens:
            self.fds[self.next_fd] = Fd(
                "dir", path=os.path.abspath(path), preopen=path
            )
            self.next_fd += 1

    def memory(self, caller: wasmtime.Caller) -> wasmtime.Memory:
        return caller.get("memory")

    def read_bytes(
        self, caller: wasmtime.Caller, ptr: int, length: int
    ) -> bytes:
        mem = self.memory(caller)
        return bytes(mem.read(caller, ptr, ptr + length))

    def write_bytes(
        self, caller: wasmtime.Caller, ptr: int, data: bytes
    ) -> None:
        mem = self.memory(caller)
        mem.write(caller, data, ptr)

    def read_cstr(self, caller: wasmtime.Caller, ptr: int) -> str:
        mem = self.memory(caller)
        data = mem.read(caller, ptr, mem.data_len(caller))
        end = data.find(b"\x00")
        return bytes(data[:end] if end >= 0 else data).decode(
            "utf-8", "replace"
        )

    def write_cstr(self, caller: wasmtime.Caller, ptr: int, text: str) -> None:
        self.write_bytes(caller, ptr, text.encode("utf-8") + b"\x00")

    def read_iovs(
        self, caller: wasmtime.Caller, iovs: int, iovs_len: int
    ) -> bytes:
        """Read an iovec array into one bytes object."""
        parts = []
        for i in range(iovs_len):
            base = iovs + i * IOV_SIZE
            buf = int.from_bytes(self.read_bytes(caller, base, 8), "little")
            length = int.from_bytes(
                self.read_bytes(caller, base + 8, 8), "little"
            )
            parts.append(self.read_bytes(caller, buf, length))
        return b"".join(parts)

    def iovs_total(
        self, caller: wasmtime.Caller, iovs: int, iovs_len: int
    ) -> int:
        """Total byte length of an iovec array."""
        total = 0
        for i in range(iovs_len):
            base = iovs + i * IOV_SIZE
            total += int.from_bytes(
                self.read_bytes(caller, base + 8, 8), "little"
            )
        return total

    def write_iovs(
        self,
        caller: wasmtime.Caller,
        iovs: int,
        iovs_len: int,
        data: bytes,
    ) -> int:
        """Write data into an iovec array; returns bytes written."""
        written = 0
        for i in range(iovs_len):
            base = iovs + i * IOV_SIZE
            buf = int.from_bytes(self.read_bytes(caller, base, 8), "little")
            length = int.from_bytes(
                self.read_bytes(caller, base + 8, 8), "little"
            )
            chunk = data[written : written + length]
            self.write_bytes(caller, buf, chunk)
            written += len(chunk)
            if written >= len(data):
                break
        return written

    def resolve(self, fd: int, path: str) -> str | None:
        """Resolve a guest path against a directory fd, rejecting paths that
        escape it lexically. Symlinks inside the preopen can still resolve
        outside it (as with wasmtime's own preview1 host)."""
        entry = self.fds.get(fd)
        if entry is None or entry.kind != "dir" or not entry.path:
            return None
        root = os.path.abspath(entry.path)
        rel = path.removeprefix("/")
        joined = os.path.normpath(os.path.join(root, rel))
        if os.path.commonpath([root, joined]) != root:
            return None
        return joined


def make_host_functions(
    linker: wasmtime.Linker, store: wasmtime.Store, host: WasiHost
) -> None:
    i32 = wasmtime.ValType.i32()
    i64 = wasmtime.ValType.i64()

    def define(
        name: str,
        params: list[wasmtime.ValType],
        results: list[wasmtime.ValType],
        func: Callable[..., int | None],
    ) -> None:
        # wasi-libc wasm64 imports preview1 names with a `_wasm64` suffix.
        linker.define_func(
            "wasi_snapshot_preview1",
            name + "_wasm64",
            wasmtime.FuncType(params, results),
            func,
            access_caller=True,
        )

    def fd_write(
        caller: wasmtime.Caller,
        fd: int,
        iovs: int,
        iovs_len: int,
        nwritten: int,
    ) -> int:
        entry = host.fds.get(fd)
        if entry is None or entry.fileobj is None:
            return EBADF
        data = host.read_iovs(caller, iovs, iovs_len)
        written = 0
        try:
            view = memoryview(data)
            while view:
                n = entry.fileobj.write(view)
                if not n:
                    break
                view = view[n:]
                written += n
            entry.fileobj.flush()
        except OSError as exc:
            return wasi_errno_of(exc)
        if nwritten != 0:
            host.write_bytes(caller, nwritten, written.to_bytes(8, "little"))
        return ESUCCESS

    define("fd_write", [i32, i64, i64, i64], [i32], fd_write)

    def fd_read(
        caller: wasmtime.Caller,
        fd: int,
        iovs: int,
        iovs_len: int,
        nread: int,
    ) -> int:
        entry = host.fds.get(fd)
        if entry is None or entry.fileobj is None:
            return EBADF
        total = host.iovs_total(caller, iovs, iovs_len)
        try:
            data = entry.fileobj.read(total)
        except OSError as exc:
            return wasi_errno_of(exc)
        written = host.write_iovs(caller, iovs, iovs_len, data)
        if nread != 0:
            host.write_bytes(caller, nread, written.to_bytes(8, "little"))
        return ESUCCESS

    define("fd_read", [i32, i64, i64, i64], [i32], fd_read)

    def fd_seek(
        caller: wasmtime.Caller,
        fd: int,
        offset: int,
        whence: int,
        newoffset: int,
    ) -> int:
        entry = host.fds.get(fd)
        if entry is None or entry.fileobj is None:
            return EBADF
        try:
            pos = entry.fileobj.seek(offset, whence)
        except OSError as exc:
            return wasi_errno_of(exc)
        if newoffset != 0:
            host.write_bytes(caller, newoffset, pos.to_bytes(8, "little"))
        return ESUCCESS

    define("fd_seek", [i32, i64, i32, i64], [i32], fd_seek)

    def fd_close(caller: wasmtime.Caller, fd: int) -> int:
        entry = host.fds.pop(fd, None)
        if entry is None:
            return EBADF
        if entry.kind == "file" and entry.fileobj is not None:
            try:
                entry.fileobj.close()
            except OSError:
                pass
        return ESUCCESS

    define("fd_close", [i32], [i32], fd_close)

    def fd_fdstat_get(caller: wasmtime.Caller, fd: int, buf: int) -> int:
        entry = host.fds.get(fd)
        if entry is None:
            return EBADF
        if entry.kind == "dir":
            ftype = FILETYPE_DIRECTORY
        elif entry.kind == "file":
            ftype = FILETYPE_REGULAR_FILE
        else:
            ftype = FILETYPE_CHARACTER_DEVICE
        # filetype u8 @0, fdflags u16 @2, rights u64 @8 and @16.
        stat = bytearray(24)
        stat[0] = ftype
        stat[8:16] = RIGHTS_ALL.to_bytes(8, "little")
        stat[16:24] = RIGHTS_ALL.to_bytes(8, "little")
        host.write_bytes(caller, buf, bytes(stat))
        return ESUCCESS

    define("fd_fdstat_get", [i32, i64], [i32], fd_fdstat_get)

    def fd_fdstat_set_flags(
        caller: wasmtime.Caller, fd: int, fdflags: int
    ) -> int:
        # wasi-libc imports this for every module; the runner applies append
        # at open time, so accept and ignore other flag changes.
        entry = host.fds.get(fd)
        if entry is None:
            return EBADF
        return ESUCCESS

    define("fd_fdstat_set_flags", [i32, i32], [i32], fd_fdstat_set_flags)

    def fd_pread(
        caller: wasmtime.Caller,
        fd: int,
        iovs: int,
        iovs_len: int,
        offset: int,
        nread: int,
    ) -> int:
        entry = host.fds.get(fd)
        if entry is None or entry.fileobj is None:
            return EBADF
        total = host.iovs_total(caller, iovs, iovs_len)
        try:
            pos = entry.fileobj.tell()
            entry.fileobj.seek(offset)
            data = entry.fileobj.read(total)
            entry.fileobj.seek(pos)
        except OSError as exc:
            return wasi_errno_of(exc)
        written = host.write_iovs(caller, iovs, iovs_len, data)
        if nread != 0:
            host.write_bytes(caller, nread, written.to_bytes(8, "little"))
        return ESUCCESS

    define("fd_pread", [i32, i64, i64, i64, i64], [i32], fd_pread)

    def args_sizes_get(
        caller: wasmtime.Caller, argc_ptr: int, argv_buf_size_ptr: int
    ) -> int:
        buf_size = sum(len(a) + 1 for a in host.argv)
        host.write_bytes(caller, argc_ptr, len(host.argv).to_bytes(8, "little"))
        host.write_bytes(
            caller, argv_buf_size_ptr, buf_size.to_bytes(8, "little")
        )
        return ESUCCESS

    define("args_sizes_get", [i64, i64], [i32], args_sizes_get)

    def args_get(caller: wasmtime.Caller, argv_ptrs: int, argv_buf: int) -> int:
        offset = 0
        for i, arg in enumerate(host.argv):
            host.write_bytes(
                caller,
                argv_ptrs + i * 8,
                (argv_buf + offset).to_bytes(8, "little"),
            )
            host.write_bytes(caller, argv_buf + offset, arg + b"\x00")
            offset += len(arg) + 1
        return ESUCCESS

    define("args_get", [i64, i64], [i32], args_get)

    def environ_sizes_get(
        caller: wasmtime.Caller, count: int, buf_size: int
    ) -> int:
        total = sum(len(e) + 1 for e in host.environ)
        if count != 0:
            host.write_bytes(
                caller, count, len(host.environ).to_bytes(8, "little")
            )
        if buf_size != 0:
            host.write_bytes(caller, buf_size, total.to_bytes(8, "little"))
        return ESUCCESS

    define("environ_sizes_get", [i64, i64], [i32], environ_sizes_get)

    def environ_get(
        caller: wasmtime.Caller, env_ptrs: int, env_buf: int
    ) -> int:
        ptr = env_buf
        for i, entry in enumerate(host.environ):
            host.write_bytes(
                caller, env_ptrs + i * 8, ptr.to_bytes(8, "little")
            )
            host.write_bytes(caller, ptr, entry + b"\x00")
            ptr += len(entry) + 1
        return ESUCCESS

    define("environ_get", [i64, i64], [i32], environ_get)

    def clock_time_get(
        caller: wasmtime.Caller, clock_id: int, precision: int, time_ptr: int
    ) -> int:
        if clock_id == 0:
            ns = time.time_ns()
        elif clock_id == 1:
            ns = time.monotonic_ns()
        else:
            return EINVAL
        if time_ptr != 0:
            host.write_bytes(caller, time_ptr, ns.to_bytes(8, "little"))
        return ESUCCESS

    define("clock_time_get", [i32, i64, i64], [i32], clock_time_get)

    def random_get(caller: wasmtime.Caller, buf: int, buf_len: int) -> int:
        host.write_bytes(caller, buf, os.urandom(buf_len))
        return ESUCCESS

    define("random_get", [i64, i64], [i32], random_get)

    def proc_exit(caller: wasmtime.Caller, code: int) -> None:
        raise SystemExit(code)

    define("proc_exit", [i32], [], proc_exit)

    def path_open(
        caller: wasmtime.Caller,
        dirfd: int,
        dirflags: int,
        path_ptr: int,
        path_len: int,
        oflags: int,
        rights_base: int,
        rights_inheriting: int,
        fdflags: int,
        opened_fd: int,
    ) -> int:
        path = host.read_bytes(caller, path_ptr, path_len).decode(
            "utf-8", "replace"
        )
        resolved = host.resolve(dirfd, path)
        if resolved is None:
            return EBADF
        if os.path.isdir(resolved):
            if not (oflags & OFLAGS_DIRECTORY):
                return EISDIR
            fd = host.next_fd
            host.next_fd += 1
            host.fds[fd] = Fd("dir", path=resolved)
            if opened_fd != 0:
                host.write_bytes(caller, opened_fd, fd.to_bytes(4, "little"))
            return ESUCCESS
        if oflags & OFLAGS_DIRECTORY:
            return ENOENT if not os.path.exists(resolved) else ENOTDIR
        if oflags & OFLAGS_EXCL and not oflags & OFLAGS_CREAT:
            return EINVAL
        writable = bool(rights_base & RIGHTS_FD_WRITE)
        flags = os.O_RDWR if writable else os.O_RDONLY
        if oflags & OFLAGS_CREAT:
            flags |= os.O_CREAT
        if oflags & OFLAGS_TRUNC:
            flags |= os.O_TRUNC
        if oflags & OFLAGS_EXCL:
            flags |= os.O_EXCL
        if fdflags & FDFLAG_APPEND:
            flags |= os.O_APPEND
        try:
            fobj = os.fdopen(
                os.open(resolved, flags),
                "r+b" if writable else "rb",
                buffering=0,
            )
        except PermissionError:
            # EACCES and EPERM both surface here; WASI callers expect not-perm.
            return EPERM
        except OSError as exc:
            # FileNotFoundError/FileExistsError/IsADirectoryError/NotADirectoryError
            # carry unique errnos that wasi_errno_of maps identically.
            return wasi_errno_of(exc)
        fd = host.next_fd
        host.next_fd += 1
        host.fds[fd] = Fd("file", fileobj=fobj, path=resolved)
        if opened_fd != 0:
            host.write_bytes(caller, opened_fd, fd.to_bytes(4, "little"))
        return ESUCCESS

    define(
        "path_open",
        [i32, i32, i64, i64, i32, i64, i64, i32, i64],
        [i32],
        path_open,
    )

    def fd_prestat_get(caller: wasmtime.Caller, fd: int, buf: int) -> int:
        entry = host.fds.get(fd)
        if entry is None or entry.kind != "dir" or not entry.preopen:
            return EBADF
        # tag u8 @0, pr_name_len u64 @8.
        prestat = bytearray(16)
        prestat[0] = 0
        prestat[8:16] = len(entry.preopen.encode("utf-8")).to_bytes(8, "little")
        host.write_bytes(caller, buf, bytes(prestat))
        return ESUCCESS

    define("fd_prestat_get", [i32, i64], [i32], fd_prestat_get)

    def fd_prestat_dir_name(
        caller: wasmtime.Caller, fd: int, path_ptr: int, path_len: int
    ) -> int:
        entry = host.fds.get(fd)
        if entry is None or entry.kind != "dir" or entry.preopen is None:
            return EBADF
        name = entry.preopen.encode("utf-8")
        if len(name) > path_len:
            return ENAMETOOLONG
        host.write_bytes(caller, path_ptr, name)
        return ESUCCESS

    define("fd_prestat_dir_name", [i32, i64, i64], [i32], fd_prestat_dir_name)

    def wasi_filetype(st: os.StatResult) -> int:
        mode = st.st_mode
        if stat.S_ISDIR(mode):
            return FILETYPE_DIRECTORY
        if stat.S_ISCHR(mode):
            return FILETYPE_CHARACTER_DEVICE
        if stat.S_ISLNK(mode):
            return FILETYPE_SYMBOLIC_LINK
        if stat.S_ISREG(mode):
            return FILETYPE_REGULAR_FILE
        return FILETYPE_UNKNOWN

    def u64(n: int) -> bytes:
        return (int(n) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")

    def write_filestat(
        caller: wasmtime.Caller, buf: int, st: os.StatResult
    ) -> None:
        out = bytearray(64)
        out[0:8] = u64(st.st_dev)
        out[8:16] = u64(st.st_ino)
        out[16] = wasi_filetype(st)
        out[24:32] = u64(st.st_nlink)
        out[32:40] = u64(st.st_size)
        out[40:48] = u64(st.st_atime_ns)
        out[48:56] = u64(st.st_mtime_ns)
        out[56:64] = u64(st.st_ctime_ns)
        host.write_bytes(caller, buf, bytes(out))

    def fd_filestat_get(caller: wasmtime.Caller, fd: int, buf: int) -> int:
        entry = host.fds.get(fd)
        if entry is None:
            return EBADF
        try:
            if entry.fileobj is not None:
                st = os.fstat(entry.fileobj.fileno())
            elif entry.path:
                st = os.stat(entry.path)
            else:
                return EBADF
        except OSError as exc:
            return wasi_errno_of(exc)
        write_filestat(caller, buf, st)
        return ESUCCESS

    define("fd_filestat_get", [i32, i64], [i32], fd_filestat_get)

    def path_filestat_get(
        caller: wasmtime.Caller,
        dirfd: int,
        flags: int,
        path_ptr: int,
        path_len: int,
        buf: int,
    ) -> int:
        path = host.read_bytes(caller, path_ptr, path_len).decode(
            "utf-8", "replace"
        )
        resolved = host.resolve(dirfd, path)
        if resolved is None:
            return EBADF
        try:
            if flags & LOOKUP_SYMLINK_FOLLOW:
                st = os.stat(resolved)
            else:
                st = os.lstat(resolved)
        except FileNotFoundError:
            return ENOENT
        except OSError as exc:
            return wasi_errno_of(exc)
        write_filestat(caller, buf, st)
        return ESUCCESS

    define(
        "path_filestat_get", [i32, i32, i64, i64, i64], [i32], path_filestat_get
    )

    def path_create_directory(
        caller: wasmtime.Caller, dirfd: int, path_ptr: int, path_len: int
    ) -> int:
        path = host.read_bytes(caller, path_ptr, path_len).decode(
            "utf-8", "replace"
        )
        resolved = host.resolve(dirfd, path)
        if resolved is None:
            return EBADF
        try:
            os.mkdir(resolved)
        except PermissionError:
            # EACCES and EPERM both surface here; WASI callers expect not-perm.
            return EPERM
        except OSError as exc:
            # FileExistsError/FileNotFoundError/NotADirectoryError carry
            # unique errnos that wasi_errno_of maps identically.
            return wasi_errno_of(exc)
        return ESUCCESS

    define(
        "path_create_directory", [i32, i64, i64], [i32], path_create_directory
    )

    def path_remove_directory(
        caller: wasmtime.Caller, dirfd: int, path_ptr: int, path_len: int
    ) -> int:
        path = host.read_bytes(caller, path_ptr, path_len).decode(
            "utf-8", "replace"
        )
        resolved = host.resolve(dirfd, path)
        if resolved is None:
            return EBADF
        try:
            os.rmdir(resolved)
        except IsADirectoryError:
            # Not in the errno table's shape: rmdir's EISDIR maps to WASI
            # ENOTDIR, not EISDIR.
            return ENOTDIR
        except PermissionError:
            # EACCES and EPERM both surface here; WASI callers expect not-perm.
            return EPERM
        except OSError as exc:
            # FileNotFoundError/NotADirectoryError carry unique errnos that
            # wasi_errno_of maps identically.
            return wasi_errno_of(exc)
        return ESUCCESS

    define(
        "path_remove_directory", [i32, i64, i64], [i32], path_remove_directory
    )

    def path_unlink_file(
        caller: wasmtime.Caller, dirfd: int, path_ptr: int, path_len: int
    ) -> int:
        path = host.read_bytes(caller, path_ptr, path_len).decode(
            "utf-8", "replace"
        )
        resolved = host.resolve(dirfd, path)
        if resolved is None:
            return EBADF
        try:
            os.unlink(resolved)
        except PermissionError:
            # EACCES and EPERM both surface here; WASI callers expect not-perm.
            return EPERM
        except OSError as exc:
            # FileNotFoundError/IsADirectoryError/NotADirectoryError carry
            # unique errnos that wasi_errno_of maps identically.
            return wasi_errno_of(exc)
        return ESUCCESS

    define("path_unlink_file", [i32, i64, i64], [i32], path_unlink_file)

    def fd_readdir(
        caller: wasmtime.Caller,
        fd: int,
        buf: int,
        buf_len: int,
        cookie: int,
        bufused: int,
    ) -> int:
        # d_next u64, d_ino u64, d_namlen u32, d_type u8, pad to 24, then the name.
        entry = host.fds.get(fd)
        if entry is None:
            return EBADF
        if entry.kind != "dir":
            return ENOTDIR
        try:
            # Cookies index a fresh scan; entries are stable only while the
            # directory is not mutated across the read sequence.
            names = [".", ".."] + sorted(os.listdir(entry.path))
        except OSError as exc:
            return wasi_errno_of(exc)
        offset = 0
        index = cookie
        while index < len(names):
            name = names[index].encode("utf-8")
            full = os.path.join(entry.path, names[index])
            try:
                st = os.lstat(full)
                ino = st.st_ino
                ftype = wasi_filetype(st)
            except OSError:
                ino = 0
                ftype = FILETYPE_UNKNOWN
            # Write an entry only if it fits whole; a partial dirent must be
            # left for the caller to retry at the same cookie.
            if offset + 24 + len(name) > buf_len:
                break
            dirent = bytearray(24)
            dirent[0:8] = (index + 1).to_bytes(8, "little")
            dirent[8:16] = (ino & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
            dirent[16:20] = len(name).to_bytes(4, "little")
            dirent[20] = ftype
            host.write_bytes(caller, buf + offset, bytes(dirent))
            offset += 24
            host.write_bytes(caller, buf + offset, name)
            offset += len(name)
            index += 1
        host.write_bytes(caller, bufused, offset.to_bytes(8, "little"))
        return ESUCCESS

    define("fd_readdir", [i32, i64, i64, i64, i64], [i32], fd_readdir)

    # __wasi_subscription_t is 48 bytes, __wasi_event_t is 32, on wasm64.
    SUB_SIZE = 48
    EVENT_SIZE = 32
    EVENTTYPE_CLOCK = 0
    EVENTTYPE_FD_READ = 1
    EVENTTYPE_FD_WRITE = 2
    CLOCK_ABSTIME = 1

    def poll_oneoff(
        caller: wasmtime.Caller,
        in_ptr: int,
        out_ptr: int,
        nsubs: int,
        nevents_ptr: int,
    ) -> int:
        now_rt = time.time_ns()
        now_mono = time.monotonic_ns()
        reported = []  # (userdata, error, tag) — completed subscriptions only.
        pending_clocks = []  # (userdata, delay_ns)
        pending_fds = []  # (userdata, tag, fileobj)

        for i in range(nsubs):
            raw = host.read_bytes(caller, in_ptr + i * SUB_SIZE, SUB_SIZE)
            userdata = int.from_bytes(raw[0:8], "little")
            tag = raw[8]
            if tag == EVENTTYPE_CLOCK:
                clock_id = int.from_bytes(raw[16:20], "little")
                timeout = int.from_bytes(raw[24:32], "little")
                flags = int.from_bytes(raw[40:42], "little")
                if clock_id not in (0, 1):
                    reported.append((userdata, EINVAL, tag))
                    continue
                now = now_rt if clock_id == 0 else now_mono
                delay = (
                    max(0, timeout - now) if flags & CLOCK_ABSTIME else timeout
                )
                pending_clocks.append((userdata, delay))
            elif tag in (EVENTTYPE_FD_READ, EVENTTYPE_FD_WRITE):
                fd = int.from_bytes(raw[16:20], "little")
                entry = host.fds.get(fd)
                if entry is None or entry.fileobj is None:
                    reported.append((userdata, EBADF, tag))
                else:
                    pending_fds.append((userdata, tag, entry.fileobj))
            else:
                reported.append((userdata, EINVAL, tag))

        def ready_fds(timeout: float | None) -> list[tuple[int, int, int]]:
            """Report pending fd subscriptions ready after `timeout` seconds."""
            if not pending_fds:
                return []
            reads = [f for _, t, f in pending_fds if t == EVENTTYPE_FD_READ]
            writes = [f for _, t, f in pending_fds if t == EVENTTYPE_FD_WRITE]
            try:
                r, w, _ = select.select(reads, writes, [], timeout)
            except (OSError, ValueError):
                return []
            return [
                (userdata, ESUCCESS, tag)
                for userdata, tag, f in pending_fds
                if (tag == EVENTTYPE_FD_READ and f in r)
                or (tag == EVENTTYPE_FD_WRITE and f in w)
            ]

        reported += ready_fds(0)
        if not reported:
            if pending_clocks:
                # Wait for the soonest clock, then report due clocks and any
                # fds that became ready meanwhile; later clocks stay pending.
                soonest = min(delay for _, delay in pending_clocks)
                time.sleep(soonest / 1e9)
                reported += ready_fds(0)
                reported += [
                    (userdata, ESUCCESS, EVENTTYPE_CLOCK)
                    for userdata, delay in pending_clocks
                    if delay <= soonest
                ]
            else:
                # Only fd subscriptions and none ready: block like poll(2).
                reported += ready_fds(None)

        written = 0
        for userdata, err, tag in reported:
            ev = bytearray(EVENT_SIZE)
            ev[0:8] = userdata.to_bytes(8, "little")
            ev[8:10] = err.to_bytes(2, "little")
            ev[10] = tag
            host.write_bytes(caller, out_ptr + written * EVENT_SIZE, bytes(ev))
            written += 1
        if nevents_ptr != 0:
            host.write_bytes(caller, nevents_ptr, written.to_bytes(8, "little"))
        return ESUCCESS

    define("poll_oneoff", [i64, i64, i64, i64], [i32], poll_oneoff)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="path to the .wasm module")
    parser.add_argument("args", nargs="*", help="argv passed to the module")
    parser.add_argument(
        "--dir",
        action="append",
        default=[],
        help="preopen a directory (repeatable)",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="set an environment variable K=V (repeatable)",
    )
    opts = parser.parse_args()

    config = wasmtime.Config()
    config.wasm_memory64 = True
    engine = wasmtime.Engine(config)
    store = wasmtime.Store(engine)

    for kv in opts.env:
        key, _, value = kv.partition("=")
        os.environ[key] = value

    preopens = ["."] + opts.dir
    host = WasiHost(preopens, argv=[os.path.basename(opts.module)] + opts.args)

    module = wasmtime.Module(engine, open(opts.module, "rb").read())
    linker = wasmtime.Linker(engine)
    make_host_functions(linker, store, host)
    linker.define_unknown_imports_as_traps(module)

    instance = linker.instantiate(store, module)

    exports = instance.exports(store)
    if "_start" not in exports:
        print("module has no _start export", file=sys.stderr)
        return 1
    try:
        exports["_start"](store)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
