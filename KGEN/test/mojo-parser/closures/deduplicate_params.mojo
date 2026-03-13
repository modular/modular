# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


struct C[B: Int](RegisterPassable):
    def get(self) -> Int:
        pass


# CHECK-COUNT-1: lit.struct.decl @"`_CI_deduplicate_params_escaping0"<c_type: !Int, |>


def use(a: Int):
    pass


def take_closure[
    c_type: Int
](x: C[c_type], closure: def(z: C[c_type]) escaping -> None):
    closure(x)


def make_closure[c_type: Int]() -> def(z: C[c_type]) escaping -> None:
    def foo(z: C[c_type]) escaping -> None:
        use(z.get())

    return foo
