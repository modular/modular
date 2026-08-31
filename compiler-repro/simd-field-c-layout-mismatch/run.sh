#!/bin/sh
# Reproduces: SIMD[DType.uint8, N] as a struct field gets vector alignment
# instead of natural 1-byte alignment, mismatching the equivalent C layout.
#
# Expected (bug-free) output: "size 28 ... offset e 8".
# Actual output on Mojo 1.0.0b2 (2cf4d08a): "size 48 ... offset e 16".
set -e
cd "$(dirname "$0")"
mojo run repro.mojo
