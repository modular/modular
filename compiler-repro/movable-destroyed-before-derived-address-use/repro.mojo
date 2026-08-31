# M1.4 (#128) compiler-defect reproducer: a Movable resource (NativeStack)
# is destroyed -- __del__ runs, munmap'ing its backing memory -- BEFORE a
# subsequent call finishes using addresses derived from it, even though
# that call is the very next statement and textually depends on the
# memory still being mapped. Deterministic (5/5 in my testing), not the
# separate flaky #202 compiler-crash pattern -- this is a runtime SIGSEGV
# on unmapped memory, with the compiler producing a normal binary that
# then faults.
#
# Mechanism: `stack`'s LAST Mojo-visible use is the pair of accessor calls
# below (`stack.guard_low_address()` / `stack.top_address()`), which each
# return a plain `Int` -- the compiler has no way to know that Int is a
# derived address into `stack`'s own mapping, so once those two reads
# complete it treats `stack` as dead and destroys it right there. The very
# next statement, `write_it(...)`, then dereferences that now-freed address
# and SIGSEGVs. `write_it` itself doesn't need to be `@extern`, doesn't
# need to `raise`, and doesn't need to live in a different module -- a
# plain same-file Mojo function reproduces it. Adding ANY extra reference
# to `stack` textually AFTER the call that consumes the derived addresses
# (see `probe6`-style fix noted at the bottom) reliably avoids it, which is
# the workaround used throughout tests/spike/t8_*.mojo through t13_*.mojo.
#
# Bisection notes:
#   - reproduces with a plain (non-raising, non-extern) local function --
#     ruling out "extern FFI boundary" and "raises" as necessary
#     ingredients (both were tried in isolation and still crashed).
#   - reproduces whether `stack` is a fresh `var stack = NativeStack.create(...)`
#     or a pre-declared-then-reassigned-via-try/except `stack` (mirroring
#     the T1-T7 driver shape) -- ruling out "reassignment vs fresh
#     declaration" as the deciding factor.
#   - reproduces with or without unrelated trailing statements after the
#     crashing call (a trailing `while` loop referencing unrelated locals
#     does not save it) -- ruling out "more code after" as sufficient
#     protection on its own.
#   - does NOT reproduce once an explicit extra reference to `stack` is
#     added textually after the crashing call (`_ = stack.base_address()`);
#     this is the fix applied throughout the M1.4 switch-half tests.
#   - tests/spike/t1_address_stability.mojo through t7 use this exact
#     "derive two Ints, pass them to a later call, never touch `stack`
#     again" shape too, and do NOT crash -- those drivers are considerably
#     larger (stack_allocation'd context buffers, structs, multi-statement
#     switch loops) and something about that shape avoids triggering the
#     early free here, but I could not isolate a single factor that
#     explains the difference within the scope of this leg. Treat T1-T7's
#     safety as UNCONFIRMED-BY-PRINCIPLE rather than proven-safe-by-design
#     until someone bisects that gap specifically.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64.
#
# $ mojo build -I spike/stack_switch docs/defects/m1-4-native-stack-use-after-free-across-call.mojo -o /tmp/repro
# $ /tmp/repro
# Stack dump without symbol names ...
# ... EXC_BAD_ACCESS, deterministic, 5/5 in my testing

from native_stack import NativeStack, page_size


def write_it(stack_low: Int, stack_top: Int) -> Int:
    """Writes one byte 16 below the top of [stack_low, stack_top) and
    reads it back. No @extern, no raises -- a plain Mojo function is
    enough to reproduce this."""
    var p = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=stack_top - 16)
    p[] = 0x5A
    return Int(p[])


def main() raises:
    var ps = page_size()
    var stack = NativeStack.create(256 * 1024, 256 * 1024, ps)

    # `stack`'s last Mojo-visible use is these two accessor reads. On this
    # toolchain that is enough for `stack` to be destroyed (munmap) before
    # `write_it` below runs, even though `write_it` is the very next
    # statement and dereferences memory `stack` still owns at this point
    # in program order.
    var r = write_it(stack.guard_low_address(), stack.top_address())

    # A trailing loop over unrelated locals does not save it -- included
    # here to rule that out explicitly (see bisection notes above).
    var extra = 0
    var i = 0
    while i < 3:
        extra += i
        i += 1

    print("readback=", r, "extra=", extra)
