# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# COM: See MOCO-756

from collections import Optional


fn print_second_string(first: String, second: String) -> None:
    print("Received", second)


fn main():
    var optional_func: Optional[
        fn (flags: String, args: String) -> None
    ] = print_second_string
    # CHECK: Received second
    optional_func.value()("first", "second")
