# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@register_passable
struct C[B: Index]:
    fn get(self) -> Index:
        pass


# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"fn[index](


fn use(a: Index):
    pass


fn take_closure[
    c_type: Index
](x: C[c_type], closure: fn (z: C[c_type]) escaping -> None):
    closure(x)


fn make_closure[c_type: Index]() -> fn (z: C[c_type]) escaping -> None:
    fn foo(z: C[c_type]) escaping -> None:
        use(z.get())

    return foo
