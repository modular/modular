# spike/stack_switch/native_stack.mojo — M1.4 (#128), memory half.
#
# Mojo-owned, non-moving, guarded native stack built directly over
# mmap/mprotect/munmap (spike/stack_switch/leaf_externs.mojo) — no C
# substrate anywhere in the loop. This is the Mojo-first counterpart to
# native/posix/mjs_stack.c: same layout, same acceptance bar (spec §10,
# SYS-6), but the syscalls are made directly from Mojo rather than through
# a C function this migration is trying to retire (spec §0).
#
# Layout of one reservation (mirrors native/posix/mjs_stack.c exactly, so
# the two are differentially comparable — see tests/spike/ns6_c_oracle_diff.mojo):
#
#   base                     base+guard              base+guard+usable = top
#   | guard pages            | uncommitted span  | committed top span  |
#   | PROT_NONE              | PROT_NONE          | PROT_READ|PROT_WRITE|
#   +-----------------------+-------------------+----------------------+
#
#   - ONE fixed mmap for the whole reservation; the virtual address never
#     changes for the life of the stack (SYS-6). Moving the Mojo value that
#     owns it (a plain struct move) does not touch the mapping at all.
#   - guard_bytes is rounded UP to whole pages (minimum one page) and
#     painted PROT_NONE, so underflow off the bottom of the usable range
#     faults instead of corrupting adjacent memory.
#   - initial_commit_bytes (rounded up to pages, clamped to the usable
#     span) is committed PROT_READ|PROT_WRITE at the TOP of the usable
#     range; the rest of the usable span stays PROT_NONE (reserved only).
#   - top = highest usable address. mmap returns a page-aligned base and
#     every size here is a page multiple, so top is always 16-byte
#     aligned (AAPCS64 SP-at-entry requirement).
#   - release happens exactly once, in __del__, on every path (including
#     an error raised after a partially-constructed stack — see
#     tests/spike/ns5_dtor_exactly_once_on_raise.mojo).
#
# DESTRUCTOR SPELLING (load-bearing, filed as mojito-sys#200): on the
# pinned Mojo 1.0.0b2 (2cf4d08a) toolchain, `__deinit__(deinit self)` --
# the spelling this migration's own planning material names as current,
# and what a dozen existing mojito_sys/* wrapper files already use -- is
# measured to be SILENTLY NEVER INVOKED (compiles cleanly, never runs; see
# docs/defects/m1-4-deinit-silently-inert.mojo). `__del__(deinit self)` --
# the spelling spike/context_switch/SPIKE_REPORT.md item 1 recorded the S0
# spike settling on -- IS invoked reliably on this toolchain, under both
# `mojo run` and `mojo build`. This type therefore uses `__del__`,
# deliberately, not `__deinit__`.
#
# Move-only: NativeStack conforms to Movable and NOT Copyable. No
# `__init__(out self, *, copy: Self)` exists, so `var b = a` (without `^`)
# is a compile error ("value of type 'NativeStack' cannot be implicitly
# copied, it does not conform to 'ImplicitlyCopyable'") -- double release
# by accidental copy is impossible by construction, not by convention (see
# tests/spike/ns4_copy_is_compile_error.mojo, a compile-fail test).
#
# Compiler workarounds carried over from mojito_sys/memory/stack.mojo /
# virtual_memory.mojo (b2, issue #29/#30 fold):
#   - every raising member here has EXACTLY ONE raise site, funneled
#     through `_raise_stack_error(rc)` at the tail -- no String literal
#     reaches a `raise` through a control-flow merge in a body that also
#     lowers an @extern call;
#   - page rounding uses bit masks, never `/` or `%` (host page sizes are
#     powers of two) -- integer division/remainder in the same raising,
#     extern-lowering body has been observed to SIGSEGV this compiler.

from std.memory import stack_allocation

from leaf_externs import (
    probe_mmap,
    probe_munmap,
    probe_mprotect,
    probe_sysconf,
    probe_error_ptr_macos,
    probe_errno_location_linux,
)

comptime _PROT_NONE = Int32(0x00)
comptime _PROT_READ = Int32(0x01)
comptime _PROT_WRITE = Int32(0x02)
comptime _PROT_RW = Int32(0x03)
comptime _MAP_PRIVATE_ANON = Int32(0x1002)  # MAP_PRIVATE(0x02) | MAP_ANON(0x1000), macOS
comptime _SC_PAGESIZE = Int32(29)  # macOS; see module note for Linux (30)

comptime _MAP_FAILED = UInt64(0xFFFFFFFFFFFFFFFF)


def _read_errno() -> Int32:
    comptime IS_MACOS = True  # this leg targets macOS arm64 (see README)
    var addr: UInt64
    comptime if IS_MACOS:
        addr = probe_error_ptr_macos()
    else:
        addr = probe_errno_location_linux()
    var p = UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=Int(addr))
    return p[]


def _raise_stack_error(rc: Int32) raises:
    raise Error("NativeStack: syscall failed, errno=" + String(rc))


def page_size() -> Int:
    return Int(probe_sysconf(_SC_PAGESIZE))


def _round_up(n: Int, page: Int) -> Int:
    var mask = page - 1
    return (n + mask) & ~mask


struct NativeStack(Movable):
    """A Mojo-owned, non-moving, guarded native stack reservation, built
    directly over mmap/mprotect/munmap. See module header for the full
    contract; mirrors native/posix/mjs_stack.c's layout exactly.
    """

    var base: Int
    var guard_bytes: Int
    var reserved_bytes: Int
    var committed_bytes: Int
    var top: Int

    def __init__(out self):
        self.base = 0
        self.guard_bytes = 0
        self.reserved_bytes = 0
        self.committed_bytes = 0
        self.top = 0

    @staticmethod
    def create(
        reserve_bytes: Int,
        initial_commit_bytes: Int,
        guard_bytes: Int,
    ) raises -> Self:
        """Reserve a guarded stack. Raises on any syscall failure or an
        invalid guard_bytes (must be a positive page multiple, mirroring
        mjs_stack_alloc's frozen -EINVAL contract)."""
        var ps = page_size()
        var rc = Int32(0)

        if guard_bytes <= 0 or (guard_bytes & (ps - 1)) != 0:
            rc = Int32(-22)  # EINVAL

        var guard = 0
        var total = 0
        var commit = 0
        var base_addr = UInt64(0)

        if rc == 0:
            guard = guard_bytes
            var usable = _round_up(reserve_bytes if reserve_bytes > 0 else ps, ps)
            total = guard + usable
            commit = _round_up(initial_commit_bytes, ps)
            if commit > usable:
                commit = usable

            base_addr = probe_mmap(0, UInt64(total), _PROT_NONE, _MAP_PRIVATE_ANON, -1, 0)
            if base_addr == _MAP_FAILED:
                rc = _read_errno()

        if rc == 0 and commit > 0:
            var commit_off = total - commit
            var mprc = probe_mprotect(base_addr + UInt64(commit_off), UInt64(commit), _PROT_RW)
            if mprc != 0:
                rc = _read_errno()
                _ = probe_munmap(base_addr, UInt64(total))
                base_addr = 0

        if rc != 0:
            _raise_stack_error(rc)

        var ns = NativeStack()
        ns.base = Int(base_addr)
        ns.guard_bytes = guard
        ns.reserved_bytes = total
        ns.committed_bytes = commit
        ns.top = ns.base + total
        return ns^

    def __moveinit__(mut self, mut existing: Self):
        self.base = existing.base
        self.guard_bytes = existing.guard_bytes
        self.reserved_bytes = existing.reserved_bytes
        self.committed_bytes = existing.committed_bytes
        self.top = existing.top
        existing.base = 0
        existing.guard_bytes = 0
        existing.reserved_bytes = 0
        existing.committed_bytes = 0
        existing.top = 0

    def __del__(deinit self):
        if self.base != 0:
            _ = probe_munmap(UInt64(self.base), UInt64(self.reserved_bytes))

    def is_live(self) -> Bool:
        return self.base != 0

    def base_address(self) -> Int:
        return self.base

    def guard_low_address(self) -> Int:
        return self.base + self.guard_bytes

    def top_address(self) -> Int:
        return self.top

    def total_reserved(self) -> Int:
        return self.reserved_bytes

    def committed(self) -> Int:
        return self.committed_bytes

    def check_geometry(self) -> Bool:
        """base < guard_low <= top and top is 16-byte aligned."""
        var gl = self.guard_low_address()
        if self.base < gl and gl <= self.top:
            return self.top % 16 == 0
        return False
