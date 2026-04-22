# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s 2>&1 | FileCheck %s


def use_positive_int_tstring[x: Int]():
    comptime assert x > 0, t"expected positive number, got {x}"


def main():
    # CHECK: expected positive number, got -3
    use_positive_int_tstring[-3]()
