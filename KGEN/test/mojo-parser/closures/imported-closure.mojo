# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo -I %S | FileCheck %s

from test_package.return_closure import pass_int


# CHECK-LABEL: lit.func @"call_it
fn call_it() -> Int:
    # CHECK: %0 = kgen.param.constant: !Int = <#lit.struct<{value = 50}>>
    # CHECK: lit.call {{.*}}pass_int{{.*}}(%anonymous2A, %0)
    # CHECK-SAME: !lit.ref<mut !wrapper, {{.*}}> byref_result
    # CHECK-NEXT: %2 = kgen.rebind %anonymous2A
    # CHECK-NEXT: lit.call {{.*}}__call__{{.*}}(%2)
    # CHECK-SAME: (!lit.ref<!wrapper, {{.*}}> borrow_in_mem, |) -> !Int
    return pass_int(50)()
