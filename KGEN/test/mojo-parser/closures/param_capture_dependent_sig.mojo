# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl @"fn{{.*}}"<p0: !Int, |>


@fieldwise_init
@register_passable
struct Foo[B: Int](ImplicitlyCopyable):
    pass


# CHECK-LABEL: lit.fn @"take_closure{{.*}}"<c_type: !Int>[imm {{.*}}](%closure: {{.*}}<:!Int c_type>
fn take_closure[c_type: Int](closure: fn (z: Foo[c_type]) escaping -> None):
    closure(Foo[c_type]())
    pass
