# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@fieldwise_init
@register_passable
struct Foo[A: Int, B: Int](Copyable):
    fn get(self) -> Int:
        return A


fn use(a: Int):
    pass


# COM: Ensure the captured parameter is added to the Closure Impl
# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[C_TYPE:.*]]: !Int, |>

# COM: Ensure the captured parameter is added to the Closure Wrapper
# CHECK: lit.struct.decl @"fn{{.*}}"<p0: !Int, |>


fn make_closure[c_type: Int](w: Int) -> fn (z: Foo[2, c_type]) escaping -> None:
    fn foo(z: Foo[2, c_type]) escaping -> None:
        use(z.get())

    return foo
