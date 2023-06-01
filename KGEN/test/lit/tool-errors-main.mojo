# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo %s 2>&1 | FileCheck %s

from SIMD import Float32

# CHECK: could not find 'fn main()'
@export
fn main() -> Float32:
    return 0.0
