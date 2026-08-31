# spike/stack_switch/leaf_externs.mojo — M1.4 (#128), memory half.
#
# LEAF MODULE, same discipline as spike/abi/externs_leaf.mojo and every
# externs.mojo under mojito_sys/: ONLY @extern declarations, the comptime
# pointer aliases they need, and tiny non-raising probe_* call shims. No
# imports, no Movable structs, no raise sites. This keeps the raw libc
# bindings out of the same frame as NativeStack's raising, Movable-struct
# machinery (the #49/#29/#30 misbind/crash shape); spike/abi's own
# ordinary_frame_test.mojo already measured that combination to be SAFE for
# this exact mmap/munmap/mprotect call shape on this toolchain (FINDINGS.md,
# M1.2), so this split is precautionary consistency with repo convention
# rather than a proven-necessary workaround for this leg specifically.
#
# Bindings mirror spike/abi/externs_leaf.mojo's already-measured-correct
# signatures verbatim (same symbols, same argument widths) rather than
# re-deriving them, since #124 already proved these exact shapes byte-exact
# against a C oracle on macOS arm64.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a).

@extern("mmap")
def mjo_mmap(
    addr: UInt64, length: UInt64, prot: Int32, flags: Int32, fd: Int32, offset: Int64
) abi("C") -> UInt64: ...

@extern("munmap")
def mjo_munmap(addr: UInt64, length: UInt64) abi("C") -> Int32: ...

@extern("mprotect")
def mjo_mprotect(addr: UInt64, length: UInt64, prot: Int32) abi("C") -> Int32: ...

@extern("sysconf")
def mjo_sysconf(name: Int32) abi("C") -> Int64: ...

# errno accessor: Darwin's libc exposes __error() -> int*; glibc exposes
# __errno_location() -> int*. Only the platform-matching one is ever called
# (comptime-if at the call site, mirroring spike/abi/libc_calls_test.mojo).
@extern("__error")
def mjo_error_ptr_macos() abi("C") -> UInt64: ...
@extern("__errno_location")
def mjo_errno_location_linux() abi("C") -> UInt64: ...


def probe_mmap(addr: UInt64, length: UInt64, prot: Int32, flags: Int32, fd: Int32, offset: Int64) -> UInt64:
    return mjo_mmap(addr, length, prot, flags, fd, offset)

def probe_munmap(addr: UInt64, length: UInt64) -> Int32:
    return mjo_munmap(addr, length)

def probe_mprotect(addr: UInt64, length: UInt64, prot: Int32) -> Int32:
    return mjo_mprotect(addr, length, prot)

def probe_sysconf(name: Int32) -> Int64:
    return mjo_sysconf(name)

def probe_error_ptr_macos() -> UInt64:
    return mjo_error_ptr_macos()

def probe_errno_location_linux() -> UInt64:
    return mjo_errno_location_linux()
