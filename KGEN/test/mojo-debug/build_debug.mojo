# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# LLDB fails with asan because it's built by default with python support in the
# CI, and python fails asan.
# TODO(MOTO-1007): Remove this once we have a fix for asan.
# UNSUPPORTED: asan


# RUN: %mojo-build --debug-level full -O0 %s -o %t
# RUN: mojo debug -X -o -X 'image lookup -r -vn "build_debug::main()"' -X -b %t | FileCheck %s --check-prefix CHECK-LLDB
# CHECK-LLDB: at build_debug.mojo:17
fn main():
    print("success")
