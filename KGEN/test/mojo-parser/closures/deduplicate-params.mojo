# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo --mojo-disable-builtins | kgen-opt -verify-parameters | FileCheck %s

alias Int = __mlir_type.index


@register_passable
struct C[B: Int]:
    fn get(self) -> Int:
        pass


# CHECK-COUNT-1: lit.struct.decl @"`_CI_
# CHECK-COUNT-1: lit.struct.decl @"_CW_


fn use(a: Int):
    pass


fn take_closure[
    c_type: Int
](x: C[c_type], closure: fn (z: C[c_type]) escaping -> None):
    closure(x)


fn make_closure[c_type: Int]() -> fn (z: C[c_type]) escaping -> None:
    fn foo(z: C[c_type]) escaping -> None:
        use(z.get())

    return foo
