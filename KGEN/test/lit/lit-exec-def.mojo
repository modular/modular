# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s | FileCheck %s

from IO import print


@register_passable
struct Error:
    pass


def main():
    # CHECK: Hello, world!
    print("Hello, world!\n")
