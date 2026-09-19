# Mojo compiler SIGSEGV: raising function + two extern calls taking a
# runtime Int and an inline-asm pointer (mojo build, AOT path).
#
# Reproduces the compiler crash described in the linked issue. Draft, not
# intended for merge. The crash is in `mojo build` (AOT), exit code 139
# (SIGSEGV) with a stack dump; `mojo run` fails to lower with a
# `pop.external_call` legalization error instead, which is a different
# failure mode from the same trigger combination.

## Files
#   externs.mojo   - leaf module declaring two @extern C functions and two
#                    tiny non-raising shims that forward to them
#   repro.mojo     - raising function calling both shims, each time passing
#                    a runtime Int var AND an entry_pointer[] (inline asm)
#                    value; main() drives it
#   shim.c         - tiny C backing for the two @extern symbols
#   run.sh         - builds the shim dylib and runs the reproducer
#
## Trigger (minimized)
# A `raises` function that makes TWO extern-reaching calls, where each call
# passes (a) a runtime `Int` variable and (b) an `entry_pointer[]` /
# `inlined_assembly` result as function-pointer-ish arguments. Reducing to a
# single extern call, making the function non-raising, or replacing the
# inline-asm pointer with a plain UnsafePointer all stop the crash.

## Run
#   bash run.sh          # reproduces: mojo build exits 139 (SIGSEGV)
#   bash run.sh --refute # A/B checks that removing `raises` stops the crash
