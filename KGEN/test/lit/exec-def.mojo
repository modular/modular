# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s

from IO import print


def main() -> None:
    # CHECK: Hello, world!
    print("Hello, world!\n")
