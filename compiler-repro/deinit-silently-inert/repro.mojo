# M1.4 (#128) compiler-defect reproducer: `__deinit__(deinit self)` — the
# spelling documented as current for Mojo 1.0.0b2 in this migration's own
# planning material, and already used throughout `mojito_sys/memory/stack.mojo`,
# `mojito_sys/ctx/context.mojo` and roughly a dozen other production wrapper
# files in this repo — is never invoked on the pinned toolchain. The struct
# parses and compiles cleanly (no warning, no error); the method is simply
# dead code. `__del__(deinit self)` — the spelling the S0 spike settled on and
# recorded in `spike/context_switch/SPIKE_REPORT.md` item 1 — IS invoked
# reliably, on the identical struct shape, under BOTH `mojo run` (JIT) and
# `mojo build` (AOT).
#
# This matters beyond naming: every production file above relies solely on
# `__deinit__` for release (mmap/munmap, heap frees) with no fallback
# `.close()`, so on this toolchain those types release nothing on ordinary
# scope exit today. `mojito_sys/ctx/context.mojo`'s own header comment
# ("b2 1.0.0b2 does not run `__deinit__` on locals (probe-verified)") shows
# this was independently suspected there and is confirmed precisely here.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64 (also reproduces under
# `mojo build` + run the compiled binary, so it is not JIT-specific).
#
# Run: mojo run m1-4-deinit-silently-inert.mojo
# Expected (bug): prints
#   old-style __del__:    counter = 1  (dtor DID run)
#   new-style __deinit__: counter = 0  (dtor did NOT run — silently skipped)
# A fixed toolchain would print counter = 1 for both.

struct OldStyle:
    var p: UnsafePointer[Int, MutAnyOrigin]

    def __init__(out self, p: UnsafePointer[Int, MutAnyOrigin]):
        self.p = p

    def __del__(deinit self):
        self.p[] += 1


struct NewStyle:
    var p: UnsafePointer[Int, MutAnyOrigin]

    def __init__(out self, p: UnsafePointer[Int, MutAnyOrigin]):
        self.p = p

    # Full "new-style" lifecycle spelling (matches mojito_sys/memory/stack.mojo
    # and mojito_sys/ctx/context.mojo verbatim): the move-ctor uses the
    # `deinit move: Self` form and the destructor is `__deinit__`. Neither
    # form change matters below since this struct is never moved; only the
    # destructor spelling is under test.
    def __init__(out self, *, deinit move: Self):
        self.p = move.p

    def __deinit__(deinit self):
        self.p[] += 1


def scope_exit_old(p: UnsafePointer[Int, MutAnyOrigin]):
    var v = OldStyle(p)
    _ = v.p  # keep `v` live until scope end


def scope_exit_new(p: UnsafePointer[Int, MutAnyOrigin]):
    var v = NewStyle(p)
    _ = v.p


def main():
    var n_old = 0
    scope_exit_old(UnsafePointer[Int, MutAnyOrigin](to=n_old))
    print("old-style __del__:    counter =", n_old, " (expect 1)")

    var n_new = 0
    scope_exit_new(UnsafePointer[Int, MutAnyOrigin](to=n_new))
    print("new-style __deinit__: counter =", n_new, " (expect 1, BUG: reads 0)")
