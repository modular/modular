# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 3 4 | FileCheck %s

from std.sys import argv


def top_level(x: Int) -> Int:
    return x


def takeIt[T: def(Int) -> Int](cb: T, x: Int):
    print(cb(x))


def main() raises:
    var x = atol(argv()[1])
    var y = atol(argv()[2])

    # CHECK: 3
    takeIt(top_level, x)

    # CHECK: 4
    takeIt(top_level, y)
