#!/bin/sh
# Reproduces: __deinit__(deinit self) parses and compiles cleanly but is
# never invoked; __del__(deinit self) on the identical struct shape works.
#
# Expected (bug-free) output:
#   old-style __del__:    counter = 1  (expect 1)
#   new-style __deinit__: counter = 1  (expect 1)
# Actual on Mojo 1.0.0b2 (2cf4d08a):
#   old-style __del__:    counter = 1  (expect 1)
#   new-style __deinit__: counter = 0  (expect 1, BUG: reads 0)
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
