# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: This is to test that the location of escaped identifiers is not causing
# COM: issues when a diagnostic is emitted. We cannot use -verify-diagnostics in
# COM: conjunction with setting -use-mlir-diagnostics=false. The latter is
# COM: needed because mlir diagnostics ignore source ranges.
# RUN: not %parse-mojo-isolated -use-mlir-diagnostics=false %s 2>&1 | FileCheck %s

fn foo(x: __mlir_type.index):
    pass


fn bar():
    # CHECK-NOT: unexpected character
    # CHECK: error: invalid call to 'foo'
    # CHECK-NOT: unexpected character
    var `!` = __mlir_attr.`1 : si32`
    foo(`!`)
