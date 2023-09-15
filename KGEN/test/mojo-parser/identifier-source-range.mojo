# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: This is to test that the location of escaped identifiers is not causing
# COM: issues when a diagnostic is emitted. We cannot use -verify-diagnostics in
# COM: conjunction with setting -use-mlir-diagnostics=false. The latter is
# COM: needed because mlir diagnostics ignore source ranges.
# RUN: not kgen-translate -use-mlir-diagnostics=false -import-mojo %s 2>&1 | FileCheck %s


fn foo(x: Int):
    pass


fn main():
    # CHECK-NOT: unexpected character
    # CHECK: error: invalid call to 'foo'
    # CHECK-NOT: unexpected character
    let `!` = 1.0
    foo(`!`)
