# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


def main():
    # CHECK: 0
    # CHECK: 1
    # CHECK: 2
    for x in range(0, 3, 1):
        print(x)

    # CHECK: 9
    # CHECK: 6
    # CHECK: 3
    for y in range(9, 0, -3):
        print(y)

    # CHECK: 42
    for z in range(0, 0, -3):
        print(z)
    else:
        print(42)
