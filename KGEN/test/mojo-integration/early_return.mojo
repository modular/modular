# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


def print_or(value: Int, condition: Bool):
    @always_inline
    @parameter
    def do_print(value: Int):
        if condition:
            print(value)
            return
        print("refuse\n")

    do_print(value)


def main():
    # CHECK: 5
    print_or(5, True)
    # CHECK: refuse
    print_or(9, False)
