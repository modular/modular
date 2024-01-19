# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate %s -import-mojo | kgen-opt -verify-parameters | FileCheck %s


@value
@register_passable
struct Foo[A: Int, B: DType]:
    fn get(self) -> Int:
        return A


fn use(a: Int):
    pass


# COM: Ensure the captured parameter is added to the Closure Impl
# CHECK: lit.struct.decl @"`_CI_{{.*}}"<[[C_TYPE:.*]]: !DType, |>

# COM: Ensure the captured parameter is added to the Closure Wrapper
# CHECK: lit.struct.decl @"fn{{.*}}"<p0: !DType, |>


fn make_closure[
    c_type: DType
](w: Int) -> fn (z: Foo[2, c_type]) escaping -> None:
    fn foo(z: Foo[2, c_type]) escaping -> None:
        use(z.get())

    return foo
