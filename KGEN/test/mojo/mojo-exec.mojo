# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo -debug-level full %s | FileCheck %s

from IO import print


fn main():
    # CHECK: 2.0
    print(Float32(1.0) + 1.0)
