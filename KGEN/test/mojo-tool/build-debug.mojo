# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# LLDB fails with asan, see https://github.com/modularml/modular/actions/runs/6748079891/job/18345656726
# UNSUPPORTED: asan


# RUN: mojo build --debug-level full -O0 %s -o %t
# RUN: lldb %t -o 'image lookup -r -vn "module \`build-debug\`::fn main"' -b | FileCheck %s --check-prefix CHECK-LLDB
# CHECK-LLDB: at build-debug.mojo:15
fn main():
    print("success")
