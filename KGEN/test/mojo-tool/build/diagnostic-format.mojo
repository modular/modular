# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not mojo build --diagnostic-format json /does/not.exist 2>&1 | FileCheck %s
# CHECK: {"kind":"error","message":"cannot open '/does/not.exist'{{.*}}"}


# RUN: not mojo build --diagnostic-format json %s 2>&1 | FileCheck %s --check-prefix=CHECK-DIAG
# CHECK-DIAG: "line":[[@LINE+3]]{{.*}}"message":"expression must be mutable in assignment{{.*}}"
# CHECK-DIAG-NEXT: {"kind":"error","message":"failed to parse{{.*}}"}
fn main():
    4 = "hello"
