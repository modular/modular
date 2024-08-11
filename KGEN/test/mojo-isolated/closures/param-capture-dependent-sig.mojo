# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# CHECK: lit.struct.decl @"fn{{.*}}"<p0, |>


@value
@register_passable
struct Foo[B: int]:
    pass


# CHECK-LABEL: lit.func @"take_closure{{.*}}"<c_type>[imm {{.*}}](%closure: {{.*}}<c_type>
fn take_closure[c_type: int](closure: fn (z: Foo[c_type]) escaping -> None):
    closure(Foo[c_type]())
    pass
