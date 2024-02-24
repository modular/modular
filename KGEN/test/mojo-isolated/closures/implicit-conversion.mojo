# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.func @"closure_kw
fn closure_kw(a: Int):
    fn has_kw(b: Int) escaping -> Int:
        return a + b

    var has_kw_ref = has_kw

    # CHECK: kgen.rebind %{{.*}} : !lit.ref<!Int2, mut {{.*}}> to !lit.ref<!Int1, mut {{.*}}>
    var unbound: fn (Int) escaping -> Int = has_kw
    # CHECK: kgen.rebind %{{.*}} : !lit.ref<!Int2, mut *"[[LT:.*]]"> to
    # CHECK-SAME: !lit.ref<!Int1, mut *"[[LT]]">
    var unbound_ref: fn (Int) escaping -> Int = has_kw_ref
