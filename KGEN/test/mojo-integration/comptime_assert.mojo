# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s 2>&1 | FileCheck %s


fn use_positive_int[x: Int]():
    comptime assert x > 0, "expected positive number, got " + String(x)


fn main():
    # CHECK: expected positive number, got -2
    use_positive_int[-2]()
