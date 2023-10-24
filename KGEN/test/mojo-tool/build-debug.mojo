# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin

# RUN: mojo build --debug-level full -O0 %s -o %t
# RUN: lldb %t -o "image lookup -n main" -b | FileCheck %s --check-prefix CHECK-LLDB
# CHECK-LLDB: at build-debug.mojo:11
fn main():
    print("success")
