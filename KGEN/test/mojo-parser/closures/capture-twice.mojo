# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


# CHECK-LABEL: lit.struct.decl @"`_CI_
# CHECK-NEXT: lit.struct.field field0 : !Int
# CHECK: lit.func @"__copyinit__


# CHECK-LABEL: lit.func @"foo
fn foo():
    let w = 5

    fn bar() escaping -> Int:
        let x = w + w
        return x
