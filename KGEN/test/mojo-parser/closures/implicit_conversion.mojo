# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"closure_kw
fn closure_kw(a: Int):
    fn has_kw(b: Int) escaping -> Int:
        return a + b

    var has_kw_ref = has_kw

    # CHECK: kgen.rebind %{{.*}} : !lit.ref<![[INT:.*]], mut {{.*}}> to !lit.ref<![[INT1:.*]], mut {{.*}}>
    var unbound: fn (Int) escaping -> Int = has_kw
    # CHECK: kgen.rebind %{{.*}} : !lit.ref<![[INT]], mut *"[[LT:.*]]"> to
    # CHECK-SAME: !lit.ref<![[INT1]], mut *"[[LT]]">
    var unbound_ref: fn (Int) escaping -> Int = has_kw_ref
