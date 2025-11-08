# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This turns on asan itself
# UNSUPPORTED: asan
# RUN: %mojo-build -O0 --sanitize address %s -o %t
# RUN: export ASAN_OPTIONS=abort_on_error=1
# RUN: not not %t 2>&1 | FileCheck %s

from compile import compile_info
from memory import LegacyUnsafePointer as UnsafePointer


def main():
    # CHECK: ERROR: AddressSanitizer: heap-buffer-overflow
    # CHECK: WRITE of size 8
    # CHECK: #0 {{.*}} in sanitize_address::main() {{.*}}sanitize_address.mojo:[[@LINE+3]]
    # CHECK: is located 0 bytes after 8-byte region
    var p: UnsafePointer[Int] = UnsafePointer[Int].alloc(1)
    p[1] = 4
