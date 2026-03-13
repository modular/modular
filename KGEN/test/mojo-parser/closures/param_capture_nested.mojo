# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
struct Foo[A: Int, B: Int](ImplicitlyCopyable, RegisterPassable):
    def get(self) -> Int:
        return Self.A


def use(a: Int):
    pass


# COM: Ensure the captured parameter is added to the Closure Impl
# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[C_TYPE:.*]]: !Int, |>

# COM: Ensure the captured parameter is added to the Closure Wrapper
# CHECK: lit.struct.decl @"fn{{.*}}"<p0: !Int, |>


def make_closure[c_type: Int](w: Int) -> def (z: Foo[2, c_type]) escaping -> None:
    def foo(z: Foo[2, c_type]) escaping -> None:
        use(z.get())

    return foo
