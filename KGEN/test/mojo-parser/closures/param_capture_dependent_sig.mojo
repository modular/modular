# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK: lit.struct.decl @"def{{.*}}"<p0: !Int, |>


@fieldwise_init
struct Foo[B: Int](ImplicitlyCopyable, RegisterPassable):
    pass


# CHECK-LABEL: lit.fn @"take_closure{{.*}}"<c_type: !Int>[imm {{.*}}](%closure: {{.*}}<:!Int c_type>
def take_closure[c_type: Int](closure: def (z: Foo[c_type]) escaping -> None):
    closure(Foo[c_type]())
    pass
