#!/bin/sh
# Reproduces: a Movable resource's destructor runs (freeing/unmapping its
# backing memory) before the very next statement finishes using plain Int
# addresses derived from it earlier in the same statement.
#
# Expected (bug-free) output: "readback= 90 extra= 3"
# Actual on Mojo 1.0.0b2 (2cf4d08a), macOS arm64: deterministic SIGSEGV
# inside write_it() (5/5 in my testing) -- NativeStack.__del__ (munmap)
# runs before write_it dereferences the address it was just handed.
set -e
cd "$(dirname "$0")"
mojo build -I . repro.mojo -o /tmp/mojo-native-stack-uaf-repro
/tmp/mojo-native-stack-uaf-repro
