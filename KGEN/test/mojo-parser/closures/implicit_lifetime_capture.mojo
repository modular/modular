# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


struct Thing[a: Origin[mut=True]](TrivialRegisterPassable):
    pass


struct Foo(Movable where False):
    pass


def use(y: Thing):
    pass


# Check that the implicit lifetime of `x` is properly captured when
# referenced through a parameter of `y`.


# CHECK-LABEL: lit.fn @"capture_implicit_origin
def capture_implicit_origin(var x: Foo, y: Thing[origin_of(x)]):
    # COM: The closure captures `y`, whose `Thing` type carries `x`'s implicit
    # COM: origin, so the closure storage struct is parametrized by `x`'s origin
    # COM: and its initializer receives `y`.
    # CHECK: lit.call {{.*}}capture_it::__storage"::@"__init__
    # CHECK-SAME: <:origin<true> *"x
    # CHECK-SAME: "y":
    def capture_it() {read y}:
        use(y)
