# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a package.

from .test_nested_package.module import nested_function

alias top_level_alias = __mlir_attr.`2 : index`


fn function():
    call_nested_function()
    return


fn call_nested_function():
    nested_function()
    return


@fieldwise_init
struct SomeType:
    var value: Index


@fieldwise_init
struct `weird()struct[]`:
    pass


fn `use()weird[]`() -> `weird()struct[]`:
    return `weird()struct[]`()


@fieldwise_init
struct ParameterizedType[value: Index](Copyable, Movable):
    pass


@fieldwise_init
struct Wrapper:
    var data: Index

    alias MyType = ParameterizedType[__mlir_attr.`42 : index`]

    fn unused_method(mut self) -> Self.MyType:
        return Self.MyType()
