# Memory mapping in the standard library (`os.MemoryMap`)

## Summary

Add POSIX memory mapping to the standard library: an owning `MemoryMap` type
(in `std.os`) that maps a file or fresh anonymous memory into the process
address space, plus the underlying `mmap`/`munmap`/`msync` bindings and typed
`Prot`/`MapFlags` flag sets. This lets Mojo programs read and write file bytes
directly (no explicit `read`/`write`), build zero-copy IPC/shared-memory
buffers, and back custom allocators — capabilities that today require
hand-rolled `external_call`s in every project.

## Motivation

Memory mapping is a fundamental OS primitive that the stdlib currently lacks
(there is no `mmap`, `MAP_*`, or `PROT_*` anywhere in `std/`). Real uses that
are blocked or forced into per-project FFI today:

- Reading large files without copying them into a heap buffer.
- Shared-memory IPC (multiple processes mapping one file-backed region).
- Zero-copy interop with other languages/runtimes over a fixed byte layout.
- Custom allocators and arenas backed by anonymous mappings.

Every one of these currently reimplements the same `external_call["mmap", ...]`
plumbing, including the per-OS constant differences that are easy to get wrong
(e.g. `MAP_ANONYMOUS` is `0x20` on Linux but `0x1000` on macOS, `O_CREAT` and
`MS_SYNC` differ, etc.). A single well-tested stdlib type removes that footgun.

## Proposed API

```mojo
from std.os import MemoryMap, PROT_READ, PROT_WRITE, MAP_SHARED

# File-backed: map a file and access its bytes directly.
with open("data.bin", "r") as f:
    var m = MemoryMap.map(f, prot=PROT_READ)
    var first = m.bytes()[0]

# Anonymous: zero-initialized scratch memory.
var scratch = MemoryMap.anonymous(64 * 1024)
scratch.bytes()[0] = 1
```

```mojo
struct MemoryMap(Movable, Sized):
    @staticmethod
    def map(file: FileHandle, length: Int = -1, *, offset: Int = 0,
            prot: Prot = PROT_READ | PROT_WRITE,
            flags: MapFlags = MAP_SHARED) raises -> Self
    @staticmethod
    def anonymous(length: Int, *, prot: Prot = PROT_READ | PROT_WRITE) raises -> Self

    def bytes(mut self) -> Span[UInt8, origin_of(self)]        # borrowed view
    def unsafe_ptr(mut self) -> Pointer[UInt8, origin_of(self)]
    def flush(self, *, blocking: Bool = True) raises           # msync
    def __len__(self) -> Int
    def __deinit__(deinit self)                                # munmap (RAII)
```

Flag sets and free functions:

```mojo
struct Prot(TrivialRegisterPassable, Writable): var value: Int32 ...   # | composable
struct MapFlags(TrivialRegisterPassable, Writable): var value: Int32 ...

comptime PROT_NONE, PROT_READ, PROT_WRITE, PROT_EXEC: Prot
comptime MAP_SHARED, MAP_PRIVATE, MAP_FIXED, MAP_ANONYMOUS: MapFlags

def page_size() -> Int
```

## Design decisions

### 1. Typed flag sets (`Prot`, `MapFlags`) — recommended

`Prot` and `MapFlags` are thin newtypes over `Int32` that compose with `|` and
only combine with their own kind. The compiler then rejects passing a `Prot`
where a `MapFlags` is expected, and each per-OS value is defined in exactly one
place.

This is a *new* convention for the stdlib, which today expresses flags either as
loose module-level integers (`O_RDONLY` in `io/file.mojo`) or as integer-valued
struct namespaces (`SignalCodes`, `FcntlCommands` in `sys/_libc.mojo`) — neither
of which is type-safe. We recommend typed flags here because `mmap`'s `prot` and
`flags` arguments are distinct bitmask domains that are genuinely easy to swap
by accident, and because this is fresh surface with no back-compat constraint.

**Alternative (not chosen):** plain `comptime PROT_READ = 0x1` integers,
matching `io/file.mojo`'s `O_*`. Simpler and consistent with existing code, but
loses the type safety and lets `MemoryMap.map(f, prot=MAP_SHARED)` compile. If
the leads prefer stdlib-wide consistency over the added safety, this is the
fallback; the `MemoryMap` surface is otherwise unchanged.

Note we deliberately do **not** add a typed `OFlags`: `MemoryMap` does not open
files (it takes an already-open `FileHandle`), and open-flags already exist as
`O_*` in `io/file.mojo`. Introducing typed open-flags would duplicate and
conflict with that and belongs in a separate change if desired.

### 2. Standalone `MemoryMap`, not a method on `FileHandle`

The owning type is standalone rather than, say, `FileHandle.map()`, for three
reasons:

1. Anonymous mappings have no file, so `MemoryMap.anonymous()` must exist
   independently regardless.
2. A POSIX mapping outlives the file descriptor: after `mmap`, the fd may be
   closed and the mapping stays valid. Coupling the map's identity to a
   `FileHandle` would imply the wrong lifetime relationship. `MemoryMap.map`
   therefore only reads the descriptor during the call and does **not** retain
   the `FileHandle`.
3. Ecosystem precedent is uniformly standalone: Rust `Mmap::map(&file)`, Zig
   `std.posix.mmap`, Python `mmap.mmap(fileno, ...)`.

A convenience `FileHandle.map(...) -> MemoryMap` sugar method could be added
later; it is intentionally out of scope for the initial API.

### 3. RAII ownership with borrowed, origin-tied views

`MemoryMap` unmaps in `__deinit__`, mirroring `FileHandle` (which closes its fd
in `__deinit__`). Byte access is through `bytes()`/`unsafe_ptr()`, which return
`Span`/`Pointer` parameterized by `origin_of(self)` — the same borrowing pattern
`ManagedAllocation.unsafe_span`/`unsafe_ptr` use (`memory/alloc.mojo`). The
origin ties the borrowed view's lifetime to the `MemoryMap`, so the mapping
cannot be unmapped while a view is live.

This relies on origin-based lifetime extension keeping the owner alive across
the borrowed view's later uses. That is the exact guarantee `ManagedAllocation`
already depends on, so it should hold; it must be confirmed on the target
toolchain as part of landing this. If a robust owning+RAII cannot be guaranteed,
the conservative fallback is a linear, explicit `unmap(deinit self)` (no
`__deinit__`), which is unconditionally safe but drops the auto-cleanup
convenience.

### 4. Arbitrary offsets (page-alignment shim)

`mmap` requires a page-aligned offset, but callers want to map from any byte
offset. `MemoryMap.map` rounds the offset down to a page boundary, grows the
mapped length by the delta, and returns views that point at the exact requested
offset. This matches Rust's `memmap2` and is transparent to callers.

### 5. Per-OS values via `platform_map`

Only a few constants differ between the supported OSes. They are selected with
`std.sys.info.platform_map`, e.g.:

```mojo
comptime MAP_ANONYMOUS = MapFlags(Int32(
    platform_map[T=Int, "MAP_ANONYMOUS", linux=0x20, macos=0x1000]()))
```

`PROT_*`, `MAP_SHARED/PRIVATE/FIXED` are identical on both OSes and are plain
literals. The `mmap` ABI itself is identical (64-bit `off_t`, byte offset, plain
`mmap` symbol), so a single `external_call` shape works on both targets.

## Implementation

A reference implementation accompanies this proposal:

- `std/os/mmap.mojo` — `Prot`/`MapFlags`, the `mmap`/`munmap`/`msync` bindings,
  `page_size()`, and `MemoryMap`.
- `std/os/__init__.mojo` — re-exports `MemoryMap`, the flag types, and
  constants.
- `test/os/test_mmap.mojo` — anonymous read/write, file-backed read-only,
  non-page-aligned offset, and shared write + `flush` round-trip.

The low-level `mmap`/`munmap`/`msync` bindings could alternatively live in
`sys/_libc.mojo` alongside `close`/`write`; they are kept in `os/mmap.mojo` here
to keep the change self-contained for review.

Errors use `get_errno()` so failures surface real messages
(`"mmap failed: Invalid argument"`).

## Scope / future work

- Windows (`CreateFileMappingW`/`MapViewOfFile`, 64 KB allocation-granularity
  alignment) is a separate backend, not attempted here — Mojo has no Windows
  target today and, like Zig's std, it should not be unified under one
  signature.
- `madvise`/`mprotect`, `MAP_FIXED`-based remapping, huge pages, and a
  `FileHandle.map()` convenience are natural follow-ups.
- A `length=-1` "whole file" default uses `lseek`; it could move to an
  `fstat`-by-fd helper once one exists in `os.fstat`.

## Prior art

- Rust: `memmap2` (`Mmap`/`MmapMut`).
- Zig: `std.posix.mmap` + per-OS `PROT`/`MAP` tables.
- Python: the `mmap` module.
