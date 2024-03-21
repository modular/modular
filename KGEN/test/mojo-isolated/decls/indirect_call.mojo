# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

fn return_function[T: AnyType]() -> T:
    pass

# CHECK-LABEL: lit.func @"mrvalue_indirect_callee
fn mrvalue_indirect_callee():
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl
    # CHECK-NEXT: call {{.*}}return_function{{.*}}([[RESULT]])
    # CHECK-NEXT: [[CALLEE:%.*]] = lit.load.consume [[RESULT]]
    # CHECK-NEXT: lit.call_indirect [[CALLEE]]()
    return_function[fn() -> None]()()
