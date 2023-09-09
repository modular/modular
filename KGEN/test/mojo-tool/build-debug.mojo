# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build --debug-level line-tables -O0 %s -o %t
# RUN: mojo build --debug-level line-tables %s -o %t
# RUN: mojo build --debug-level full %s -o %t


# RUN: mojo build --debug-level full -O0 %s -o %t
# RUN: lldb %t -o "image lookup -n main" -b | FileCheck %s --check-prefix CHECK-LLDB
# CHECK-LLDB: at build-debug.mojo:15
fn main():
    print("success")
