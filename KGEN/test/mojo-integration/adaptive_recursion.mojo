# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


def foo[axis: Int](i: Int) -> Int:
    comptime if axis < 0:
        return i
    else:
        return foo[axis - 1](i)


def main():
    var x = foo[3](42)
    # CHECK: 42
    print(x)
