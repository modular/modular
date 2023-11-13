# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | FileCheck %s


# CHECK-LABEL: lit.func @"closure_kw
fn closure_kw(a: Int):
    fn has_kw(b: Int) escaping -> Int:
        return a + b

    var has_kw_ref = has_kw

    # CHECK: kgen.rebind %{{.*}} : !kgen.pointer<!wrapper1> to !kgen.pointer<!wrapper>
    let unbound: fn (Int) escaping -> Int = has_kw
    # CHECK: kgen.rebind %{{.*}} : !lit.ref<mut !wrapper1, *"`has_kw_ref0"> to
    # CHECK-SAME: !lit.ref<mut !wrapper, *"`has_kw_ref0">
    let unbound_ref: fn (Int) escaping -> Int = has_kw_ref
