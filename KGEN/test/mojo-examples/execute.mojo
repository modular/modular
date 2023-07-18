# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: kgen %s %mojo_cpu_build_arch -emit-llvm -debug-level=full -mlir-print-debuginfo -o /dev/null
# COM: TODO(#13267): compile all the way to object file.

from SIMD import Float32
from IO import print


fn main():
    # CHECK: 2.0
    print(Float32(1.0) + 1.0)
