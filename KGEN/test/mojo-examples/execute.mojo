# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s

from SIMD import Float32
from IO import print


fn main():
    # CHECK: 2.0
    print(Float32(1.0) + 1.0)
