# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl @"fn{{.*}}"<p0, |>


@fieldwise_init
@register_passable
struct Foo[B: Index](Copyable):
    pass


# CHECK-LABEL: lit.fn @"take_closure{{.*}}"<c_type>[imm {{.*}}](%closure: {{.*}}<c_type>
fn take_closure[c_type: Index](closure: fn (z: Foo[c_type]) escaping -> None):
    closure(Foo[c_type]())
    pass
