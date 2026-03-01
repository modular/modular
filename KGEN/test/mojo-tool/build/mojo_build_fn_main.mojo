# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build %s -o %t
# RUN: %t --arg1 | FileCheck %s

from std.sys import argv


fn main():
    # CHECK: This was called inside of `fn` main
    print("This was called inside of `fn` main")

    # CHECK: --arg1
    print(argv()[1])
