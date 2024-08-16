# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# LLDB fails with asan because it's built by default with python support in the
# CI, and python fails asan.
# UNSUPPORTED: asan


# RUN: mojo build --debug-level full -O0 %s -o %t
# RUN: mojo debug -X -o -X 'image lookup -r -vn "module \`build-debug\`::fn main"' -X -b %t | FileCheck %s --check-prefix CHECK-LLDB
# CHECK-LLDB: at build-debug.mojo:16
fn main():
    print("success")
