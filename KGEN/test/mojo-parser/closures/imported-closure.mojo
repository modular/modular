# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo -I %S | FileCheck %s

from test_package.return_closure import pass_int


# CHECK-LABEL: lit.func @"call_it
fn call_it() -> Int:
    # CHECK: lit.call {{.*}}pass_int
    # CHECK-SAME: !lit.ref<mut !wrapper, {{.*}}> byref_result
    # CHECK-NEXT: %2 = lit.ref.to_pointer %anonymous2A
    # CHECK-NEXT: lit.call {{.*}}__call__
    # CHECK-SAME: (!kgen.pointer<!wrapper> borrow_in_mem, |) -> !Int
    return pass_int(50)()
