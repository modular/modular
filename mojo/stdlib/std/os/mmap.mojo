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
"""Memory-mapped files and anonymous regions (POSIX `mmap`).

Maps a file or fresh anonymous memory into the process address space, so its
bytes can be read and written directly — no explicit `read`/`write` calls.

You can import these APIs from the `os` package. For example:

```mojo
from std.os import MemoryMap, PROT_READ

# Map a file read-only and access its bytes with no explicit I/O.
with open("data.bin", "r") as f:
    var m = MemoryMap.map(f, prot=PROT_READ)
    var first_byte = m.bytes()[0]
```

Supported on macOS and Linux, which share the libc `mmap` ABI; only a few flag
*values* differ per OS and are selected with `platform_map`.
"""

from std.ffi import c_int, c_size_t, c_ssize_t, external_call, get_errno
from std.io import FileHandle
from std.sys.info import platform_map

# `off_t` is 64-bit on every target Mojo supports (macOS and 64-bit Linux).
comptime _c_off_t = Int64
comptime _Bytes = Pointer[UInt8, MutUntrackedOrigin]

# `sysconf` selector for the page size (macOS 29, Linux 30).
comptime _SC_PAGESIZE = platform_map[
    T=Int, "_SC_PAGESIZE", linux=30, macos=29
]()


# ===-----------------------------------------------------------------------===#
# Typed flags
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Prot(TrivialRegisterPassable, Writable):
    """Memory-protection flags for a mapping (the `PROT_*` values).

    Combine flags with `|`, e.g. `PROT_READ | PROT_WRITE`.
    """

    var value: Int32
    """The raw protection bitmask."""

    def __or__(self, other: Self) -> Self:
        """Combines two protection flag sets.

        Args:
            other: The flags to union with `self`.

        Returns:
            The combined flags.
        """
        return Self(self.value | other.value)

    def write_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of the flags.

        Args:
            writer: The writer to write to.
        """
        writer.write("Prot(", self.value, ")")


comptime PROT_NONE = Prot(0x0)
"""Pages may not be accessed."""
comptime PROT_READ = Prot(0x1)
"""Pages may be read."""
comptime PROT_WRITE = Prot(0x2)
"""Pages may be written."""
comptime PROT_EXEC = Prot(0x4)
"""Pages may be executed."""


@fieldwise_init
struct MapFlags(TrivialRegisterPassable, Writable):
    """Flags controlling the kind and visibility of a mapping (`MAP_*`).

    Combine flags with `|`, e.g. `MAP_PRIVATE | MAP_ANONYMOUS`.
    """

    var value: Int32
    """The raw mapping-flags bitmask."""

    def __or__(self, other: Self) -> Self:
        """Combines two mapping flag sets.

        Args:
            other: The flags to union with `self`.

        Returns:
            The combined flags.
        """
        return Self(self.value | other.value)

    def write_to(self, mut writer: Some[Writer]):
        """Writes a debug representation of the flags.

        Args:
            writer: The writer to write to.
        """
        writer.write("MapFlags(", self.value, ")")


comptime MAP_SHARED = MapFlags(0x1)
"""Updates are visible to other mappings of the same region and, for a
file-backed mapping, are written back to the file."""
comptime MAP_PRIVATE = MapFlags(0x2)
"""Updates are private (copy-on-write) and not written back to the file."""
comptime MAP_FIXED = MapFlags(0x10)
"""Place the mapping at exactly the requested address (advanced use)."""
comptime MAP_ANONYMOUS = MapFlags(
    Int32(platform_map[T=Int, "MAP_ANONYMOUS", linux=0x20, macos=0x1000]())
)
"""The mapping is not backed by a file; its contents are zero-initialized."""


# ===-----------------------------------------------------------------------===#
# Thin libc bindings (private)
# ===-----------------------------------------------------------------------===#


@always_inline
def _mmap(
    addr: Int,
    length: c_size_t,
    prot: c_int,
    flags: c_int,
    fd: c_int,
    offset: _c_off_t,
) -> _Bytes:
    return external_call["mmap", _Bytes](addr, length, prot, flags, fd, offset)


@always_inline
def _munmap(addr: _Bytes, length: c_size_t) -> c_int:
    return external_call["munmap", c_int](addr, length)


@always_inline
def _msync(addr: _Bytes, length: c_size_t, flags: c_int) -> c_int:
    return external_call["msync", c_int](addr, length, flags)


def page_size() -> Int:
    """Returns the system memory page size, in bytes.

    This is the granularity at which mappings are aligned (for example 4096 on
    x86-64 Linux, 16384 on Apple Silicon).

    Returns:
        The page size in bytes.
    """
    return Int(external_call["sysconf", c_ssize_t](c_int(_SC_PAGESIZE)))


def _file_size(fd: c_int) raises -> Int:
    # lseek(fd, 0, SEEK_END), then restore the offset. Avoids depending on the
    # per-OS `struct stat` layout for the common "map the whole file" case.
    var end = external_call["lseek", _c_off_t](fd, _c_off_t(0), c_int(2))
    if end < 0:
        raise Error("lseek failed: ", String(get_errno()))
    _ = external_call["lseek", _c_off_t](fd, _c_off_t(0), c_int(0))
    return Int(end)


# ===-----------------------------------------------------------------------===#
# MemoryMap
# ===-----------------------------------------------------------------------===#


struct MemoryMap(Movable, Sized):
    """An owned memory-mapped region.

    The region is unmapped automatically when the `MemoryMap` is destroyed
    (RAII, like `FileHandle`). Access the bytes through `bytes()` (a `Span`) or
    `unsafe_ptr()`; both borrow from `self`, so the mapping stays alive while a
    borrowed view is in use.

    A non-page-aligned file `offset` is supported: the region is mapped from the
    page boundary at or below `offset`, and `bytes()`/`unsafe_ptr()` point at the
    exact requested offset.
    """

    var _base: _Bytes
    """Page-aligned base address returned by `mmap` (what `munmap` frees)."""
    var _offset: Int
    """Bytes from `_base` to the caller's requested offset."""
    var _mapped_len: Int
    """Bytes actually mapped (`len + _offset`)."""
    var _len: Int
    """Bytes the caller requested (what `len()`/`bytes()` expose)."""

    def __init__(
        out self, base: _Bytes, offset: Int, mapped_len: Int, length: Int
    ):
        """Constructs a `MemoryMap` from an existing mapping. Internal; use the
        `map()` and `anonymous()` factory methods instead.

        Args:
            base: The page-aligned base address returned by `mmap`.
            offset: The byte delta from `base` to the requested offset.
            mapped_len: The number of bytes actually mapped.
            length: The number of bytes the caller requested.
        """
        self._base = base
        self._offset = offset
        self._mapped_len = mapped_len
        self._len = length

    def __deinit__(deinit self):
        """Unmaps the region. Any error is ignored, as in `FileHandle`."""
        _ = _munmap(self._base, c_size_t(self._mapped_len))

    @staticmethod
    def map(
        file: FileHandle,
        length: Int = -1,
        *,
        offset: Int = 0,
        prot: Prot = PROT_READ | PROT_WRITE,
        flags: MapFlags = MAP_SHARED,
    ) raises -> Self:
        """Maps a region of an open file into memory.

        The mapping is independent of `file`: per POSIX it remains valid after
        the file is closed.

        Args:
            file: An open file to map. Only its descriptor is used, during this
                call; the `MemoryMap` does not retain `file`.
            length: The number of bytes to map. Defaults to the remainder of the
                file from `offset`.
            offset: The byte offset into the file at which to start the mapping.
                Need not be page-aligned.
            prot: The memory-protection flags for the mapping.
            flags: The mapping flags. `MAP_ANONYMOUS` is not valid here; use
                `MemoryMap.anonymous()` instead.

        Returns:
            A `MemoryMap` owning the mapped region.

        Raises:
            If determining the default length fails, or if `mmap` fails.
        """
        var fd = c_int(file._get_raw_fd())
        var len = length
        if len < 0:
            len = _file_size(fd) - offset
        var delta = offset % page_size()
        var base = _mmap(
            0,
            c_size_t(len + delta),
            prot.value,
            flags.value,
            fd,
            _c_off_t(offset - delta),
        )
        _check(base)
        return Self(base, delta, len + delta, len)

    @staticmethod
    def anonymous(
        length: Int, *, prot: Prot = PROT_READ | PROT_WRITE
    ) raises -> Self:
        """Maps `length` bytes of fresh, zero-initialized anonymous memory.

        Args:
            length: The number of bytes to map.
            prot: The memory-protection flags for the mapping.

        Returns:
            A `MemoryMap` owning the mapped region.

        Raises:
            If `mmap` fails.
        """
        var base = _mmap(
            0,
            c_size_t(length),
            prot.value,
            (MAP_PRIVATE | MAP_ANONYMOUS).value,
            -1,
            0,
        )
        _check(base)
        return Self(base, 0, length, length)

    @always_inline
    def __len__(self) -> Int:
        """Returns the number of mapped bytes.

        Returns:
            The mapped length in bytes.
        """
        return self._len

    @always_inline
    def unsafe_ptr(mut self) -> Pointer[UInt8, origin_of(self)]:
        """Returns a pointer to the mapped bytes, borrowing from `self`.

        Returns:
            A pointer to the first mapped byte.
        """
        return self._base.unsafe_offset(self._offset).unsafe_origin_cast[
            origin_of(self)
        ]()

    @always_inline
    def bytes(mut self) -> Span[UInt8, origin_of(self)]:
        """Returns a `Span` over the mapped bytes, borrowing from `self`.

        Returns:
            A span covering the mapped region.
        """
        return {unsafe_ptr = self.unsafe_ptr(), length = self._len}

    def flush(self, *, blocking: Bool = True) raises:
        """Flushes changes in the mapping to its backing store (`msync`).

        Args:
            blocking: If True (the default), block until the write completes
                (`MS_SYNC`); otherwise schedule it and return (`MS_ASYNC`).

        Raises:
            If `msync` fails.
        """
        # MS_ASYNC is 0x1 on both OSes; MS_SYNC differs.
        var ms_sync = platform_map[T=Int, "MS_SYNC", linux=0x4, macos=0x10]()
        var flags = ms_sync if blocking else 0x1
        if _msync(self._base, c_size_t(self._mapped_len), c_int(flags)) != 0:
            raise Error("msync failed: ", String(get_errno()))


def _check(p: _Bytes) raises:
    # mmap reports failure with MAP_FAILED == (void*)-1, NOT null.
    if Int(p) == -1:
        raise Error("mmap failed: ", String(get_errno()))
