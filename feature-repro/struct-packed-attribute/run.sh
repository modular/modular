#!/bin/sh
# TDD red test for a feature request: `@packed` does not exist yet, so
# this file fails to parse today. It should go GREEN (parse and run,
# printing nothing since main() is empty) once a struct-level
# alignment-override attribute ships (equivalent of C's
# `#pragma pack(N)` / `__attribute__((packed))`).
#
# Current (red) result on Mojo 1.0.0b2 (2cf4d08a):
#   error: use of unknown declaration 'packed'
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
