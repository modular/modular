# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Thing[a: Origin[mut=True]](TrivialRegisterPassable):
    pass


struct Foo:
    pass


def use(y: Thing):
    pass


# Check that the implicit lifetime of `x` is properly captured when
# referenced through a parameter of `y`.


# CHECK-LABEL: lit.fn @"capture_implicit_origin
def capture_implicit_origin(var x: Foo, y: Thing[origin_of(x)]):
    # CHECK: lit.var.decl "__call_result_tmp__" synth : !lit.ref<{{.*}}<:origin<1> *"x`">
    def capture_it():
        use(y)
