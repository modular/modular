# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s -I %S | FileCheck %s

from test_package.return_closure import pass_int


# CHECK-LABEL: lit.fn @"call_it
fn call_it() -> Int:
    # CHECK: %0 = kgen.param.constant: !Int = <{50}>
    # CHECK: lit.call {{.*}}pass_int{{.*}}(%0, %anonymous2A)
    # CHECK-SAME: !lit.ref<!Int1, mut {{.*}}> byref_result
    # CHECK-NEXT: %2 = lit.ref.immut %anonymous2A
    # CHECK-NEXT: lit.call {{.*}}__call__{{.*}}(%2)
    # CHECK-SAME: (!lit.ref<!Int1, imm {{.*}}> read_mem, |) -> !Int
    return pass_int(50)()
