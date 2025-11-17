# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a package.

from .test_nested_package.module import nested_function

comptime top_level_alias = 2


fn function():
    call_nested_function()
    return


fn call_nested_function():
    nested_function()
    return


@fieldwise_init
struct SomeType:
    var value: Int


@fieldwise_init
struct `weird()struct[]`:
    pass


fn `use()weird[]`() -> `weird()struct[]`:
    return `weird()struct[]`()


@fieldwise_init
struct ParameterizedType[value: Int](ImplicitlyCopyable, Movable):
    pass


@fieldwise_init
struct Wrapper:
    var data: Int

    comptime MyType = ParameterizedType[42]

    fn unused_method(mut self) -> Self.MyType:
        return Self.MyType()
