# M1.2 (#124) compiler-defect / structural-constraint reproducer: two
# related facts about Mojo 1.0.0b2's `@extern` mechanism, both load-
# bearing for how any future platform-divergent OS/libc binding must be
# structured in this migration.
#
# FACT 1 — no module-level conditional compilation. A bare `comptime if`
# cannot wrap a top-level declaration ("must be contained in a
# function"), and `@extern(...)` refuses a comptime-computed symbol-name
# string ("requires a string literal argument") — so there is NO way to
# make ONE @extern declaration resolve to a DIFFERENT literal C symbol
# per compilation target. See the commented-out block below for FACT 1a
# (uncomment it in isolation to see its own error) and the live
# `ERRNO_SYM` declaration further down for FACT 1b (live because a
# comptime-computed string is accepted right up until `@extern` itself
# rejects it, so it does not need commenting out to demonstrate).
#
# FACT 2 — the C SYMBOL is `@extern`'s uniqueness key, not the local Mojo
# function name. Declaring `@extern("fcntl")` twice in one module under
# two different local names AND two different arities (a 2-arg and a
# 3-arg form, mirroring how a real variadic call site might want to call
# it either way) is a compile-time "duplicate functions named 'fcntl'"
# error even though the Mojo-side names differ. See the two
# `mjo_fcntl2`/`mjo_fcntl3` declarations at the bottom of this file.
#
# The WORKING pattern for both (used throughout spike/abi/):
#   - platform-exclusive raw symbols: declare BOTH platforms' externs
#     unconditionally, then select the CALL (not the declaration) with a
#     `comptime if` — the untaken branch's call site, and with it the
#     need to resolve that symbol, is pruned entirely at compile time
#     (spike/abi/libc_calls_test.mojo's read_errno() does exactly this
#     for __error/__errno_location);
#   - multiple arities of one symbol: declare only the WIDEST fixed
#     arity once, and pass a dummy/unused value for call sites that need
#     fewer arguments (spike/abi/externs_leaf.mojo's fcntl/open probes).
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64.
# Run: mojo run m1-2-extern-declaration-constraints.mojo
# (this file is expected to FAIL TO PARSE as committed — that failure
# itself, citing FACT 1b and FACT 2 together, is the defect being
# documented; uncomment the FACT 1a block, on its own, in a scratch copy
# to see that error in isolation instead).

from std.sys import CompilationTarget

# FACT 1a: a bare `comptime if` cannot wrap a TOP-LEVEL (module-scope)
# declaration — "'comptime if' must be contained in a function". Uncomment
# to see this error in isolation (left commented so the rest of this file
# can still be parsed far enough to show FACT 1b and FACT 2's errors too):
#
# comptime if CompilationTarget().is_macos():
#     @extern("__error")
#     def mjo_errno_a() abi("C") -> UInt64: ...
# else:
#     @extern("__errno_location")
#     def mjo_errno_a() abi("C") -> UInt64: ...

# FACT 1b: `@extern` refuses a comptime-computed symbol-name string.
comptime ERRNO_SYM = "__error" if CompilationTarget().is_macos() else "__errno_location"

@extern(ERRNO_SYM)
def mjo_errno_b() abi("C") -> UInt64: ...


# FACT 2: declaring the SAME C symbol under two different arities in one
# module is a "duplicate functions named 'fcntl'" error, even with two
# distinct local Mojo names.
@extern("fcntl")
def mjo_fcntl2(fd: Int32, cmd: Int32) abi("C") -> Int32: ...
@extern("fcntl")
def mjo_fcntl3(fd: Int32, cmd: Int32, arg: Int32) abi("C") -> Int32: ...


def main() raises:
    # Both declarations must actually be CALLED for FACT 2's conflict to
    # surface — an unused @extern declaration is silently dead-code-
    # eliminated before the duplicate-symbol check runs (the same
    # eliminate-if-unused behavior documented in
    # m1-2-write-extern-stdio-conflict.mojo).
    _ = mjo_fcntl2(0, 0)
    _ = mjo_fcntl3(0, 0, 0)
    _ = mjo_errno_b()
    print("if this compiles AND runs, none of the facts above still hold")
