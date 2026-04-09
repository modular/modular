# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# COM: See MOCO-756

from std.collections import Optional


def print_second_string(first: String, second: String) -> None:
    print("Received", second)


def main():
    var optional_func: Optional[
        def(flags: String, args: String) thin -> None
    ] = print_second_string
    # CHECK: Received second
    optional_func.value()("first", "second")
