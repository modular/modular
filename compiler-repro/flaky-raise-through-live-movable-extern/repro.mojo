# M1.4 (#128) compiler-defect reproducer: `mojo build` intermittently
# (probabilistically, NOT deterministically) crashes the compiler itself
# when an ordinary Mojo `raise` propagates through several call frames
# while an ancestor frame holds a live value of a Movable struct whose
# construction lowers @extern calls (spike/stack_switch/native_stack.mojo's
# NativeStack, in this repro) -- REPEATED inside a loop in the same
# compiled function. A single, unlooped instance of this shape essentially
# never crashes in measurement; the crash rate rises with how many times
# the shape repeats in one compilation unit.
#
# Bisection history (each row isolates one variable; only the combination
# below reproduces at all):
#   - raise through ONE extra (non-recursive) frame, no loop: never crashed
#     (dozens of trials across several variants).
#   - raise through a SELF-RECURSIVE chain (depth 1 or 5), no loop, called
#     ONCE: never crashed in isolation across 20+ trials.
#   - the SAME recursive-chain-through-a-live-NativeStack call, called ONCE
#     per `mojo build` invocation, REPEATED across many separate `mojo
#     build` invocations: crashed roughly 1 in 20-25 invocations.
#   - the identical call REPEATED 200 times inside a `while` loop in ONE
#     compiled function: crashed roughly 2 of 3 invocations.
#   - the identical call repeated only 20 times inside the loop: 0 crashes
#     in 5 invocations (comfortable margin, not a proof of absence).
#   - a Movable struct with the SAME shape (raising @staticmethod
#     factory, __moveinit__, __del__) but NO @extern calls inside the
#     factory: never crashed, including with the loop. Removing the
#     @extern calls (mmap/munmap) from the factory while keeping
#     everything else identical is what stopped the crash from
#     reproducing, so the @extern-lowering inside the raising factory
#     appears to be a necessary ingredient, not merely a live Movable
#     value.
#   - whether the raise chain is self-recursive or a fixed sequence of
#     DISTINCT nested functions did not change the qualitative outcome:
#     both reproduce under the loop, at similar rates.
#
# This file is the loop shape that reproduces reliably enough to observe
# in a handful of tries (unlike the single-call shape, which needs dozens
# of `mojo build` invocations to see once). It is NOT a minimal single-line
# reproducer -- every attempt to shrink it below "a live NativeStack across
# a multi-frame raise, repeated" stopped reproducing, which is itself part
# of what makes this a genuine crash rather than a simple logic bug: it
# depends on something ambient in the compiler's own state, not purely on
# the source text.
#
# Toolchain: Mojo 1.0.0b2 (2cf4d08a), macOS arm64, `mojo build` (also
# reproduces under `mojo run`, less frequently).
#
# Run (repeat several times -- this is flaky, not deterministic):
#   mojo build -I spike/stack_switch \
#     docs/defects/m1-4-flaky-crash-raise-through-live-movable.mojo \
#     -o /tmp/m1-4-repro
# Expected (bug, on an unlucky run): `mojo build` itself aborts with
#   "Please submit a bug report to https://github.com/modular/modular/issues"
# and a native stack dump, before producing any binary -- this is a
# COMPILER crash, not a runtime failure of the compiled program.

from native_stack import NativeStack, page_size


def raiser_bottom(depth: Int) raises:
    if depth == 0:
        raise Error("boom")
    raiser_bottom(depth - 1)


def guarded(ps: Int) raises:
    var s = NativeStack.create(65536, ps, ps)
    raiser_bottom(5)


def main() raises:
    var ps = page_size()
    var i = 0
    var ok = True
    while i < 200:
        var raised = False
        try:
            guarded(ps)
        except e:
            raised = True
            _ = e
        if not raised:
            ok = False
        i += 1
    print("ok=", ok, "(expect True; the BUG is mojo build itself crashing before this ever runs)")
