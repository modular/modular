# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# MOCO-2327
@register_passable("trivial")
struct Foo:
    # CHECK: lit.fn @"__init__(::Int)"
    fn __init__(out self, *, a: Int):
        pass

    # CHECK: lit.fn @"__init__(::Int)_0"
    fn __init__(out self, *, b: Int):
        pass


def main():
    # CHECK: lit.alias.decl *"{{.*}}": !Foo = <apply(:!lit.generator<(*, "b": !Int) -> !Foo> @decl_name_collision::@Foo::@"__init__(::Int)_0", {42})>
    comptime _foo = Foo(b=42)
