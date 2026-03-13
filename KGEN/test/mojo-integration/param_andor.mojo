# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


def is_positive(a: Int) -> Bool:
    return a > 0


def test_andor[a: Int, b: Int]():
    comptime if is_positive(a) and is_positive(b):
        print(2)
    elif is_positive(a) or is_positive(b):
        print(1)
    else:
        print(0)


def main():
    # CHECK: 2
    test_andor[1, 1]()
    # CHECK: 1
    test_andor[1, -1]()
    # CHECK: 1
    test_andor[-1, 1]()
    # CHECK: 0
    test_andor[-1, -1]()
