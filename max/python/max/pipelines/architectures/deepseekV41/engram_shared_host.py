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
"""One host copy of the Engram tables, shared by every process on the host.

The Engram tables are large and read from pinned host memory. Rather than
pin a private copy per process, the first loader fills a sealed memfd and
every other process on the host that loads the same tables maps and pins that
memfd. Unlike a ``/dev/shm`` file, a memfd is freed once no process maps it,
so a crash leaks no memory. Whenever sharing is not possible, the loader can
fall back to a private copy.
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import hashlib
import json
import logging
import math
import os
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("max.pipelines")

DISABLE_ENV = "MAX_ENGRAM_SHARED_HOST"
"""Set to ``0`` to always pin a private copy."""

DIR_ENV = "MAX_ENGRAM_SHARED_HOST_DIR"
"""Overrides the registry directory (every sharing process must agree)."""

_REGISTRY_NAME = "max-engram-shared"
_RECORD_MAX_BYTES = 4096

_MAGIC = b"MAXENGR1"
_HEADER = struct.Struct("<8sQQ")  # magic, state, metadata length
_STATE_FILLING = 0
_STATE_COMPLETE = 1
# The header gets its own 2 MiB block and every table starts on a 2 MiB
# boundary, so the copy can be backed by transparent huge pages.
_ALIGN = 2 << 20
_LOCK_TIMEOUT_S = 1800.0
_LOCK_POLL_S = 0.2

_PROT_NONE = 0x0
_PROT_READ = 0x1
_PROT_WRITE = 0x2
_MAP_SHARED = 0x01
_MAP_PRIVATE = 0x02
_MAP_FIXED = 0x10
_MAP_ANONYMOUS = 0x20
_MADV_HUGEPAGE = 14
_MAP_FAILED = ctypes.c_void_p(-1).value
# Linux's values; the hermetic Python's ``os``/``fcntl`` lack them.
_MFD_CLOEXEC = 0x1
_MFD_ALLOW_SEALING = 0x2
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SHRINK = 0x2
_F_SEAL_GROW = 0x4
_REQUIRED_SEALS = _F_SEAL_SHRINK | _F_SEAL_GROW


class SharedTablesUnavailable(RuntimeError):
    """Raised when the shared copy cannot be used."""


def enabled() -> bool:
    """Whether sharing is possible here and not turned off."""
    return (
        _LIBC is not None
        and hasattr(_LIBC, "memfd_create")
        and os.environ.get(DISABLE_ENV, "1") != "0"
    )


def registry_dir() -> Path:
    """Returns this user's directory of per-key registry directories.

    Each ``<key>/provider-*.json`` record in it names a live process's memfd
    (pid, fd, device and inode), which other processes open through
    ``/proc/<pid>/fd/<fd>``. The copy so outlives its creator while any
    process maps it.

    The directory is private to the effective user, so another local user
    can neither publish records into it nor pre-create it: it is under
    ``$XDG_RUNTIME_DIR`` when that is a private directory of this user's,
    else ``/dev/shm/max-engram-shared-<euid>``.
    """
    configured = os.environ.get(DIR_ENV)
    if configured:
        return Path(configured)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and os.path.isabs(runtime):
        try:
            if _is_private_dir(os.lstat(runtime)):
                return Path(runtime) / _REGISTRY_NAME
        except OSError:
            pass
    return Path(f"/dev/shm/{_REGISTRY_NAME}-{os.geteuid()}")


def _is_private_dir(st: os.stat_result) -> bool:
    """Whether ``st`` (from ``lstat``) is a real directory only we can use."""
    return (
        stat.S_ISDIR(st.st_mode)
        and st.st_uid == os.geteuid()
        and st.st_mode & 0o077 == 0
    )


def _private_dir(path: Path, *, create: bool) -> bool:
    """Checks that ``path`` is a directory only this user can access.

    Never follows a symlink and never changes an existing directory's owner
    or mode: a directory that fails the check is refused, not repaired.

    Args:
        path: The directory to check.
        create: Creates ``path`` with mode ``0o700`` if it is missing.

    Returns:
        Whether ``path`` exists (always ``True`` when ``create`` is set).

    Raises:
        SharedTablesUnavailable: If ``path`` is a symlink or not a directory,
            belongs to another user, grants group or other access, or
            cannot be created.
    """
    if create:
        try:
            os.mkdir(path, 0o700)
        except FileExistsError:
            pass
        except OSError as e:
            raise SharedTablesUnavailable(f"registry {path}: {e}") from e
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        if create:
            raise SharedTablesUnavailable(
                f"registry {path} vanished after creation"
            ) from None
        return False
    except OSError as e:
        raise SharedTablesUnavailable(f"registry {path}: {e}") from e
    if not _is_private_dir(st):
        raise SharedTablesUnavailable(
            f"registry {path} is not a directory private to uid "
            f"{os.geteuid()} (owner {st.st_uid}, "
            f"mode {stat.filemode(st.st_mode)})"
        )
    return True


def _check_key(key: str) -> None:
    if not key or key in (".", "..") or "/" in key or "\0" in key:
        raise SharedTablesUnavailable(f"invalid registry key {key!r}")


def _owned_by_us(st: os.stat_result) -> bool:
    """Whether ``st`` is a regular file of ours that only we can write."""
    return (
        stat.S_ISREG(st.st_mode)
        and st.st_uid == os.geteuid()
        and st.st_mode & 0o022 == 0
    )


# ----------------------------------------------------------------- libc


def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    if hasattr(libc, "memfd_create"):
        libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    return libc


_LIBC = _libc() if sys.platform == "linux" else None


def _checked_mmap(
    address: int | None, size: int, prot: int, flags: int, fd: int
) -> int:
    assert _LIBC is not None
    mapped = _LIBC.mmap(address, size, prot, flags, fd, 0)
    if mapped is None or mapped == _MAP_FAILED:
        err = ctypes.get_errno()
        raise SharedTablesUnavailable(f"mmap failed: {os.strerror(err)}")
    return mapped


def _mmap(fd: int, size: int, prot: int) -> int:
    """Maps ``fd`` shared at a ``_ALIGN``-aligned address.

    The kernel aligns shared-memory mappings to huge pages only when shmem
    huge pages are enabled, so this reserves ``size + _ALIGN`` bytes, maps
    the file over the aligned start, and releases the unused ends.
    """
    assert _LIBC is not None
    reserved = _checked_mmap(
        None, size + _ALIGN, _PROT_NONE, _MAP_PRIVATE | _MAP_ANONYMOUS, -1
    )
    aligned = -(-reserved // _ALIGN) * _ALIGN
    try:
        _checked_mmap(aligned, size, prot, _MAP_SHARED | _MAP_FIXED, fd)
    except SharedTablesUnavailable:
        _LIBC.munmap(reserved, size + _ALIGN)
        raise
    if aligned > reserved:
        _LIBC.munmap(reserved, aligned - reserved)
    tail = reserved + size + _ALIGN - (aligned + size)
    if tail:
        _LIBC.munmap(aligned + size, tail)
    return aligned


def _memfd_create(name: str) -> int:
    assert _LIBC is not None
    fd = _LIBC.memfd_create(name.encode(), _MFD_CLOEXEC | _MFD_ALLOW_SEALING)
    if fd < 0:
        err = ctypes.get_errno()
        raise SharedTablesUnavailable(
            f"memfd_create failed: {os.strerror(err)}"
        )
    return fd


def _munmap(address: int, size: int) -> None:
    assert _LIBC is not None
    _LIBC.munmap(address, size)


def _byte_view(address: int, nbytes: int) -> np.ndarray:
    """A writable uint8 numpy view of ``nbytes`` at ``address``."""
    return np.ctypeslib.as_array(
        (ctypes.c_uint8 * nbytes).from_address(address)
    )


# --------------------------------------------------------------- layout


@dataclass(frozen=True)
class TableLayout:
    """One table's place in the shared copy."""

    name: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def layout(
    shapes: Mapping[str, Sequence[int]],
) -> tuple[list[TableLayout], int]:
    """Each byte table's offset (in ``shapes`` order) and the total size."""
    tables: list[TableLayout] = []
    offset = _ALIGN
    for name, shape in shapes.items():
        nbytes = math.prod(shape)
        tables.append(
            TableLayout(name, tuple(int(d) for d in shape), offset, nbytes)
        )
        offset += -(-max(nbytes, 1) // _ALIGN) * _ALIGN
    return tables, offset


def _metadata(key: str, tables: Sequence[TableLayout]) -> bytes:
    return json.dumps(
        {
            "key": key,
            "tables": [
                [t.name, list(t.shape), t.offset, t.nbytes] for t in tables
            ],
        },
        sort_keys=True,
    ).encode()


def _write_header(address: int, state: int, metadata: bytes) -> None:
    if _HEADER.size + len(metadata) > _ALIGN:
        raise SharedTablesUnavailable("table metadata does not fit the header")
    header = _byte_view(address, _ALIGN)
    header[_HEADER.size : _HEADER.size + len(metadata)] = np.frombuffer(
        metadata, np.uint8
    )
    header[: _HEADER.size] = np.frombuffer(
        _HEADER.pack(_MAGIC, state, len(metadata)), np.uint8
    )


def _header_matches(address: int, metadata: bytes) -> bool:
    header = _byte_view(address, _ALIGN)
    magic, state, length = _HEADER.unpack(bytes(header[: _HEADER.size]))
    return (
        magic == _MAGIC
        and state == _STATE_COMPLETE
        and length == len(metadata)
        and bytes(header[_HEADER.size : _HEADER.size + length]) == metadata
    )


# ------------------------------------------------------------------ key


@dataclass(frozen=True)
class _Mapping:
    low: int
    high: int
    offset: int
    path: str


def _file_mappings() -> list[_Mapping]:
    out: list[_Mapping] = []
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                low, _, high = fields[0].partition("-")
                out.append(
                    _Mapping(
                        int(low, 16),
                        int(high, 16),
                        int(fields[2], 16),
                        fields[5].rstrip("\n"),
                    )
                )
    return out


def _file_location(
    mappings: Sequence[_Mapping], array: np.ndarray
) -> tuple[str, int] | None:
    """``(path, file offset)`` of a contiguous array mapped from a file."""
    start = array.ctypes.data
    last = start + max(array.nbytes, 1) - 1

    def find(address: int) -> _Mapping | None:
        return next((m for m in mappings if m.low <= address < m.high), None)

    first, final = find(start), find(last)
    if first is None or final is None or first.path != final.path:
        return None
    file_offset = first.offset + (start - first.low)
    if final.offset + (last - final.low) != file_offset + (last - start):
        return None
    return first.path, file_offset


def source_key(sources: Mapping[str, np.ndarray]) -> str | None:
    """Returns the tables' content identity, or ``None`` if not file-backed.

    The key hashes each table's source file identity, file offset, shape and
    dtype, so processes loading the same checkpoint agree on it.

    Args:
        sources: Each table's name mapped to its bytes as read from the
            checkpoint (a view into the memory-mapped weights file).
    """
    if sys.platform != "linux":
        return None
    mappings = _file_mappings()
    parts = []
    for name, array in sources.items():
        if not array.flags.c_contiguous:
            return None
        location = _file_location(mappings, array)
        if location is None or location[0].endswith(" (deleted)"):
            return None
        path, file_offset = location
        st = os.stat(path)
        parts.append(
            [
                name,
                list(array.shape),
                str(array.dtype),
                os.path.realpath(path),
                st.st_dev,
                st.st_ino,
                st.st_size,
                st.st_mtime_ns,
                file_offset,
            ]
        )
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:32]


# ------------------------------------------------------------- registry


def _lock(path: Path, timeout_s: float) -> int:
    fd = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600
    )
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
        os.close(fd)
        raise SharedTablesUnavailable(f"{path} is not a file of ours")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise SharedTablesUnavailable(
                    f"timed out after {timeout_s:.0f}s waiting for {path}"
                ) from None
            time.sleep(_LOCK_POLL_S)


def _unlock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _read_record(path: Path) -> dict[str, int] | None:
    """The record at ``path``, or ``None`` if unreadable or not trusted.

    Only a regular file (not a symlink) that this user owns and that no
    one else can write is trusted.
    """
    try:
        fd = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
        )
        with os.fdopen(fd, "rb") as f:
            if not _owned_by_us(os.fstat(f.fileno())):
                return None
            record = json.loads(f.read(_RECORD_MAX_BYTES))
        return {k: int(record[k]) for k in ("pid", "fd", "dev", "ino", "size")}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _open_provider(
    record: Mapping[str, int], size: int, flags: int
) -> int | None:
    """The memfd ``record`` names, if it is still that memfd; else ``None``.

    Only a memfd that this user created, held by a process this user owns,
    is opened.
    """
    if record["size"] != size:
        return None
    try:
        if os.stat(f"/proc/{record['pid']}").st_uid != os.geteuid():
            return None
    except OSError:
        return None
    try:
        fd = os.open(
            f"/proc/{record['pid']}/fd/{record['fd']}", flags | os.O_CLOEXEC
        )
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        seals = fcntl.fcntl(fd, _F_GET_SEALS)
    except OSError:
        os.close(fd)
        return None
    # A memfd's mode is always 0777, so only its owner is checked; it is
    # reachable only through its holder's fd table, which the pid check
    # restricts to this user's processes.
    if (
        (st.st_dev, st.st_ino) != (record["dev"], record["ino"])
        or st.st_uid != os.geteuid()
        or st.st_size != size
        or seals & _REQUIRED_SEALS != _REQUIRED_SEALS
    ):
        os.close(fd)
        return None
    return fd


def complete_copy_exists(key: str, shapes: Mapping[str, Sequence[int]]) -> bool:
    """Whether a live process already publishes a complete copy of ``key``.

    Attaching to such a copy adds no host memory, which lets a host memory
    estimate skip the tables. Takes no lock and prunes nothing.
    """
    if not enabled():
        return False
    tables, size = layout(shapes)
    metadata = _metadata(key, tables)
    root = registry_dir()
    directory = root / key
    try:
        _check_key(key)
        if not (
            _private_dir(root, create=False)
            and _private_dir(directory, create=False)
        ):
            return False
        records = sorted(directory.glob("provider-*.json"))
    except (OSError, SharedTablesUnavailable):
        return False
    for path in records:
        record = _read_record(path)
        if record is None:
            continue
        fd = _open_provider(record, size, os.O_RDONLY)
        if fd is None:
            continue
        try:
            address = _mmap(fd, _ALIGN, _PROT_READ)
        except SharedTablesUnavailable:
            os.close(fd)
            continue
        try:
            if _header_matches(address, metadata):
                return True
        finally:
            _munmap(address, _ALIGN)
            os.close(fd)
    return False


# --------------------------------------------------------------- tables

Register = Callable[[int, int], Callable[[], None]]
"""Pins ``(address, size)``; returns the matching unpin."""

_OPEN: list[SharedHostTables] = []
_OPEN_LOCK = threading.Lock()


class SharedHostTables:
    """This process's mapping of the host's one shared copy."""

    def __init__(
        self,
        *,
        key: str,
        fd: int,
        address: int,
        size: int,
        tables: Sequence[TableLayout],
        created: bool,
        directory: Path,
    ) -> None:
        self.key = key
        self.created = created
        """Whether this process filled the copy (else it attached)."""
        self.address = address
        self.size = size
        self._fd = fd
        self._directory = directory
        self._record: Path | None = None
        self._unregister: Callable[[], None] | None = None
        self.arrays: dict[str, np.ndarray] = {
            t.name: _byte_view(address + t.offset, t.nbytes).reshape(t.shape)
            for t in tables
        }
        """Each table's bytes, by name, in the shared mapping."""

    def _publish(self) -> None:
        st = os.fstat(self._fd)
        record = {
            "pid": os.getpid(),
            "fd": self._fd,
            "dev": st.st_dev,
            "ino": st.st_ino,
            "size": self.size,
        }
        path = self._directory / f"provider-{os.getpid()}-{self._fd}.json"
        tmp = path.with_suffix(".tmp")
        try:
            fd = os.open(
                tmp,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_TRUNC
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
            )
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(record))
            os.replace(tmp, path)
        except OSError as e:
            raise SharedTablesUnavailable(f"publish {path}: {e}") from e
        self._record = path

    def _unpublish(self) -> None:
        if self._record is not None:
            try:
                self._record.unlink()
            except OSError:
                pass
            self._record = None

    def close(self) -> None:
        """Unpins and unmaps this process's view of the copy.

        The copy lives on while any other process maps it. Arrays from
        :attr:`arrays` must not be used afterwards.
        """
        with _OPEN_LOCK:
            if self in _OPEN:
                _OPEN.remove(self)
        self._unpublish()
        if self._unregister is not None:
            self._unregister()
            self._unregister = None
        if self._fd >= 0:
            _munmap(self.address, self.size)
            os.close(self._fd)
            self._fd = -1
        self.arrays = {}


def _close_all_records() -> None:
    with _OPEN_LOCK:
        open_tables = list(_OPEN)
    for tables in open_tables:
        tables._unpublish()


atexit.register(_close_all_records)


def open_shared_tables(
    key: str,
    shapes: Mapping[str, Sequence[int]],
    fill: Callable[[Mapping[str, np.ndarray]], None],
    *,
    register: Register | None = None,
    read_only_attach: bool = False,
    directory: Path | None = None,
    lock_timeout_s: float = _LOCK_TIMEOUT_S,
) -> SharedHostTables:
    """Attaches to the host's complete copy of ``key``, or creates it.

    An exclusive lock per key serializes creating against attaching, so
    concurrent loaders attach rather than fill a second copy. Stale records
    (dead pid, reused fd) are pruned, and a copy that is incomplete,
    mismatched or not sealed against resizing is never attached. Attachers
    never write the copy.

    Args:
        key: The tables' content identity (:func:`source_key`).
        shapes: Each byte table's shape, by name, in a fixed order.
        fill: Writes every table's bytes into the given arrays; only called
            when this process creates the copy.
        register: Pins the mapping for the GPU (:class:`CudaHostRegistrar`);
            ``None`` leaves it unpinned.
        read_only_attach: Maps an attached copy read-only (only when
            ``register`` accepts read-only memory).
        directory: Registry root; defaults to :func:`registry_dir`. It and
            its per-key directory must be real directories that only this
            user can access; they are created with mode ``0o700``.
        lock_timeout_s: How long to wait for a concurrent creator.

    Raises:
        SharedTablesUnavailable: On any failure, including a registry
            directory that another user owns or can access; nothing is left
            behind, and the tables need a private copy instead.
    """
    if not enabled():
        raise SharedTablesUnavailable(
            f"disabled ({DISABLE_ENV}=0 or not Linux)"
        )
    _check_key(key)
    tables, size = layout(shapes)
    metadata = _metadata(key, tables)
    root = directory or registry_dir()
    try:
        root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as e:
        raise SharedTablesUnavailable(f"registry {root}: {e}") from e
    _private_dir(root, create=True)
    directory = root / key
    _private_dir(directory, create=True)
    try:
        lock_fd = _lock(directory / "lock", lock_timeout_s)
    except OSError as e:
        raise SharedTablesUnavailable(f"registry {directory}: {e}") from e
    try:
        shared = _attach(
            key, directory, tables, size, metadata, read_only_attach
        ) or _create(key, directory, tables, size, metadata, fill)
        try:
            if register is not None:
                shared._unregister = register(shared.address, shared.size)
            shared._publish()
        except BaseException:
            shared.close()
            raise
    finally:
        _unlock(lock_fd)
    with _OPEN_LOCK:
        _OPEN.append(shared)
    return shared


def _attach(
    key: str,
    directory: Path,
    tables: Sequence[TableLayout],
    size: int,
    metadata: bytes,
    read_only: bool,
) -> SharedHostTables | None:
    for path in sorted(directory.glob("provider-*.json")):
        record = _read_record(path)
        fd = (
            None
            if record is None
            else _open_provider(
                record, size, os.O_RDONLY if read_only else os.O_RDWR
            )
        )
        address = None
        if fd is not None:
            try:
                address = _mmap(
                    fd,
                    size,
                    _PROT_READ if read_only else _PROT_READ | _PROT_WRITE,
                )
            except SharedTablesUnavailable:
                address = None
            if address is not None and not _header_matches(address, metadata):
                _munmap(address, size)
                address = None
        if fd is None or address is None:
            logger.info("Engram shared host tables: dropping stale %s", path)
            if fd is not None:
                os.close(fd)
            try:
                path.unlink()
            except OSError:
                pass
            continue
        return SharedHostTables(
            key=key,
            fd=fd,
            address=address,
            size=size,
            tables=tables,
            created=False,
            directory=directory,
        )
    return None


def _create(
    key: str,
    directory: Path,
    tables: Sequence[TableLayout],
    size: int,
    metadata: bytes,
    fill: Callable[[Mapping[str, np.ndarray]], None],
) -> SharedHostTables:
    fd = _memfd_create(f"max-engram-{key[:16]}")
    try:
        os.ftruncate(fd, size)
    except OSError as e:
        os.close(fd)
        raise SharedTablesUnavailable(f"memfd: {e}") from e
    try:
        address = _mmap(fd, size, _PROT_READ | _PROT_WRITE)
    except BaseException:
        os.close(fd)
        raise
    assert _LIBC is not None
    _LIBC.madvise(address, size, _MADV_HUGEPAGE)  # best effort
    shared = SharedHostTables(
        key=key,
        fd=fd,
        address=address,
        size=size,
        tables=tables,
        created=True,
        directory=directory,
    )
    try:
        _write_header(address, _STATE_FILLING, metadata)
        fill(shared.arrays)
        fcntl.fcntl(fd, _F_ADD_SEALS, _REQUIRED_SEALS)
        _write_header(address, _STATE_COMPLETE, metadata)
    except BaseException:
        shared.close()
        raise
    return shared


# ----------------------------------------------------------------- CUDA

_CU_MEMHOSTREGISTER_PORTABLE = 0x01
_CU_MEMHOSTREGISTER_DEVICEMAP = 0x02
_CU_MEMHOSTREGISTER_READ_ONLY = 0x08
_CU_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91
_CU_ATTR_READ_ONLY_HOST_REGISTER_SUPPORTED = 113


class CudaHostRegistrar:
    """Pins host memory with ``cuMemHostRegister`` for every device.

    Registers ``PORTABLE`` (visible to every CUDA context in the process).
    Device code reads the registered memory through its host address, so the
    device must support ``CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM``.

    Raises:
        SharedTablesUnavailable: If CUDA is unavailable or the device cannot
            use registered host pointers.
    """

    def __init__(self, device_ordinal: int) -> None:
        try:
            self._cu = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise SharedTablesUnavailable(f"libcuda: {e}") from e
        self._cu.cuMemHostRegister_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._cu.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self._check(self._cu.cuInit(0), "cuInit")
        self._device = ctypes.c_int()
        self._check(
            self._cu.cuDeviceGet(ctypes.byref(self._device), device_ordinal),
            "cuDeviceGet",
        )
        if not self._attribute(
            _CU_ATTR_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM
        ):
            raise SharedTablesUnavailable(
                "device cannot use registered host pointers directly"
            )
        self.read_only = bool(
            self._attribute(_CU_ATTR_READ_ONLY_HOST_REGISTER_SUPPORTED)
        )
        """Whether read-only (``PROT_READ``) memory can be registered."""

    def _check(self, result: int, what: str) -> None:
        if result != 0:
            name = ctypes.c_char_p()
            self._cu.cuGetErrorName(result, ctypes.byref(name))
            raise SharedTablesUnavailable(
                f"{what} failed: {result} ({(name.value or b'?').decode()})"
            )

    def _attribute(self, attribute: int) -> int:
        value = ctypes.c_int()
        self._check(
            self._cu.cuDeviceGetAttribute(
                ctypes.byref(value), attribute, self._device
            ),
            "cuDeviceGetAttribute",
        )
        return value.value

    def _in_context(self, call: Callable[[], int], what: str) -> None:
        context = ctypes.c_void_p()
        self._check(
            self._cu.cuDevicePrimaryCtxRetain(
                ctypes.byref(context), self._device
            ),
            "cuDevicePrimaryCtxRetain",
        )
        try:
            self._check(
                self._cu.cuCtxPushCurrent_v2(context), "cuCtxPushCurrent"
            )
            try:
                self._check(call(), what)
            finally:
                self._cu.cuCtxPopCurrent_v2(ctypes.byref(ctypes.c_void_p()))
        finally:
            self._cu.cuDevicePrimaryCtxRelease_v2(self._device)

    def __call__(self, address: int, size: int) -> Callable[[], None]:
        flags = _CU_MEMHOSTREGISTER_PORTABLE | _CU_MEMHOSTREGISTER_DEVICEMAP
        if self.read_only:
            flags |= _CU_MEMHOSTREGISTER_READ_ONLY
        start = time.perf_counter()
        self._in_context(
            lambda: self._cu.cuMemHostRegister_v2(address, size, flags),
            "cuMemHostRegister",
        )
        logger.info(
            "Registered %.1f GiB of shared Engram tables in %.1fs.",
            size / (1 << 30),
            time.perf_counter() - start,
        )

        def unregister() -> None:
            self._in_context(
                lambda: self._cu.cuMemHostUnregister(address),
                "cuMemHostUnregister",
            )

        return unregister
