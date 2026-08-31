#!/bin/sh
# Reproduces (FLAKY, not deterministic -- rerun a few times): mojo build
# intermittently crashes the compiler itself when a raise propagates
# through frames holding a live Movable value whose construction lowers
# @extern calls, repeated in a loop.
#
# Expected (bug-free): builds and runs clean every time, prints
#   ok= True (expect True; the BUG is mojo build itself crashing before this ever runs)
# Actual on Mojo 1.0.0b2 (2cf4d08a): roughly 2 of 3 `mojo build` invocations
# abort the compiler itself before producing a binary ("Please submit a
# bug report..." + native stack dump). Rerun this script several times if
# the first attempt happens to land in the lucky 1-in-3.
set -e
cd "$(dirname "$0")"
mojo build -I . repro.mojo -o /tmp/mojo-flaky-raise-crash-repro
/tmp/mojo-flaky-raise-crash-repro
