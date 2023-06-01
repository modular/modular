# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo %s 2>&1 | FileCheck %s

from SIMD import Float32

# CHECK: no functions were left in the module after compiling, this usually means that there was no `@export`ed function to use as a root - did you forget an `@export`?
fn main() -> Float32:
    return 0.0
