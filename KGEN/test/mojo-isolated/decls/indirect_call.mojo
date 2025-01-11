# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

fn return_function[T: AnyType]() -> T:
    pass

# CHECK-LABEL: lit.fn @"mrvalue_indirect_callee
fn mrvalue_indirect_callee():
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl
    # CHECK-NEXT: call {{.*}}return_function{{.*}}([[RESULT]])
    # CHECK-NEXT: [[CALLEE:%.*]] = lit.load.consume [[RESULT]]
    # CHECK-NEXT: lit.call_indirect [[CALLEE]]()
    return_function[fn() -> None]()()

fn indirect_callee() raises -> fn()->None:
    pass

# CHECK-LABEL: lit.fn @"call_it
fn call_it() raises:
    # CHECK-NEXT: [[RESULT:%.*]] = lit.var.decl
    # CHECK-NEXT: call {{.*}}indirect_callee{{.*}}(%__error__, [[RESULT]])
    # CHECK-NEXT: [[CALLEE:%.*]] = lit.load.consume [[RESULT]]
    # CHECK-NEXT: lit.call_indirect [[CALLEE]]()
    indirect_callee()()
