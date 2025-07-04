# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


@register_passable("trivial")
struct Thing[a: MutableOrigin]:
    pass


struct Foo:
    pass


fn use(y: Thing):
    pass


# Check that the implicit lifetime of `x` is properly captured when
# referenced through a parameter of `y`.


# CHECK-LABEL: lit.fn @"capture_implicit_origin
fn capture_implicit_origin(var x: Foo, y: Thing[__origin_of(x)]):
    # CHECK: lit.var.decl "anonymous*" synth : !lit.ref<{{.*}}<:origin<1> *"x`">
    fn capture_it():
        use(y)
