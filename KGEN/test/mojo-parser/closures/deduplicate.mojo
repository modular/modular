# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo -split-input-file | FileCheck %s

# COM: Check that closure structs are deduplicated.

# CHECK-COUNT-1: lit.struct.decl @"_CI_
# CHECK-COUNT-1: lit.struct.decl @"_CW_


fn use(a: Int):
    pass


fn makes_escaping_closure(a: Int):
    fn dummy(n: Int) escaping:
        use(a)

    fn duplicate(n: Int) escaping:
        use(a)

# // -----

@register_passable
struct C[B: DType]:
    fn get(self) -> Int:
        return 3

# CHECK-COUNT-1: lit.struct.decl @"_CW_{{.*}}"<p0[p0]: !DType, |>  attributes {closureSignature = !kgen.signature<!lit.signature<<"c_type": !DType,

fn take_closure[
    c_type: DType
](x: C[c_type], closure: fn (z: C[c_type]) escaping -> None):
    closure(x)


fn make_closure[c_type: DType]() -> fn (z: C[c_type]) escaping -> None:
    fn foo(z: C[c_type]) escaping -> None:
        print(z.get())

    return foo
