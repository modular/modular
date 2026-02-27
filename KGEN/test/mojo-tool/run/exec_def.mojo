# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo -debug-level=full %s | FileCheck %s


def main() raises -> None:
    # CHECK: Hello, world!
    print("Hello, world!\n")
