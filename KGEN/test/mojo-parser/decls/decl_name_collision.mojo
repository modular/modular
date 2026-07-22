# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# MOCO-2327
struct Foo(TrivialRegisterPassable):
    # CHECK: lit.fn @"__init__(a:::SIMD[::DType(int), ::SIMDLength(1)])"
    def __init__(out self, *, a: Int):
        pass

    # CHECK: lit.fn @"__init__(b:::SIMD[::DType(int), ::SIMDLength(1)])"
    def __init__(out self, *, b: Int):
        pass


def main() raises:
    # CHECK: lit.alias.decl *"{{.*}}": !Foo = <apply(:!lit.generator<(*, "b": !Int) -> !Foo> @decl_name_collision::@Foo::@"__init__(b:::SIMD[::DType(int), ::SIMDLength(1)])", {:scalar<index> 42})>
    comptime _foo = Foo(b=42)
