# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


# CHECK: lit.struct.decl @"`_CI_
# CHECK-NEXT: lit.struct.field field0 : !Int
# CHECK-NEXT: lit.func @"__copyinit__
fn foo():
    let w = 5

    fn bar() escaping -> Int:
        let x = w + w
        return x
