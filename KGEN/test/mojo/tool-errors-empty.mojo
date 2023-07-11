# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo %s 2>&1 | FileCheck %s

from SIMD import Float32

# CHECK: module does not `@export` any symbols; nothing to codegen
fn foo() -> Float32:
    return 0.0
