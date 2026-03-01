# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 3 4 | FileCheck %s

from std.sys import argv


fn top_level(x: Int) -> Int:
    return x


fn takeIt[T: fn(Int) unified -> Int](cb: T, x: Int):
    print(cb(x))


def main() raises:
    var x = atol(argv()[1])
    var y = atol(argv()[2])

    # CHECK: 3
    takeIt[top_level](top_level, x)

    # CHECK: 4
    takeIt[top_level](top_level, y)
