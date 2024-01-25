# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | kgen-opt -verify-parameters | FileCheck %s

# CHECK: lit.struct.decl @"fn{{.*}}"<p0, |>

alias Int = __mlir_type.index


@value
@register_passable
struct Foo[B: Int]:
    pass


# CHECK-LABEL: lit.func @"take_closure{{.*}}<c_type>(
# CHECK-SAME: %arg[closure]: {{.*}}<c_type>
fn take_closure[c_type: Int](closure: fn (z: Foo[c_type]) escaping -> None):
    closure(Foo[c_type]())
    pass
