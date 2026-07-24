# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a package.

from .test_nested_package.module import nested_function, parametric_fn

comptime top_level_alias = 2


def function():
    call_nested_function()
    return


def call_nested_function():
    nested_function()
    return


@fieldwise_init
struct SomeType(Movable where False):
    var value: Int


@fieldwise_init
struct `weird()struct[]`(Movable where False):
    pass


def `use()weird[]`() -> `weird()struct[]`:
    return `weird()struct[]`()


@fieldwise_init
struct ParameterizedType[value: Int](ImplicitlyCopyable):
    pass


@fieldwise_init
struct Wrapper(Movable where False):
    var data: Int

    comptime MyType = ParameterizedType[42]

    def unused_method(mut self) -> Self.MyType:
        return Self.MyType()
