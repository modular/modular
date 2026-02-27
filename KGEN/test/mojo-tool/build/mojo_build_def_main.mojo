# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-build %s -o %t
# RUN: not %t --arg1 | FileCheck %s

from sys import argv


def main() raises:
    # CHECK: This was called inside of `def` main
    print("This was called inside of `def` main")

    # CHECK: --arg1
    print(argv()[1])

    # CHECK: Unhandled exception caught during execution: main raised an error
    raise Error("main raised an error")
