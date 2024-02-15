# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %translate-with-packages %s | kgen-opt -verify-parameters | FileCheck %s


@register_passable
struct C[B: int]:
    fn get(self) -> int:
        pass


# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"fn[index](


fn use(a: int):
    pass


fn take_closure[
    c_type: int
](x: C[c_type], closure: fn (z: C[c_type]) escaping -> None):
    closure(x)


fn make_closure[c_type: int]() -> fn (z: C[c_type]) escaping -> None:
    fn foo(z: C[c_type]) escaping -> None:
        use(z.get())

    return foo
