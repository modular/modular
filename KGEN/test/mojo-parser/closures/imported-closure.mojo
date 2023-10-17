# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo -I %S | FileCheck %s

from test_package.return_closure import pass_int


# CHECK-LABEL: lit.func @"call_it
fn call_it() -> Int:
    # CHECK: call {{.*}}pass_int
    # CHECK-SAME: !kgen.pointer<!escaping> byref_result
    # CHECK-NEXT: call {{.*}}__call__
    # CHECK-SAME: (!kgen.pointer<!escaping> borrow_in_mem, |) -> !Int
    return pass_int(50)()
