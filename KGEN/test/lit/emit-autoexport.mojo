# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s -emit-header | FileCheck %s

from SIMD import Float32
from IO import print


@export("bar")
# CHECK: extern float bar();
fn main() -> Float32:
    # OK to alias, not proper main
    return 0.0
