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


@value
struct SomeType:
    var value: Index


@value
struct `weird()struct[]`:
    pass


fn `use()weird[]`() -> `weird()struct[]`:
    return `weird()struct[]`()


@value
struct ParameterizedType[value: Index]:
    pass


@value
struct Wrapper:
    var data: Index

    alias MyType = ParameterizedType[__mlir_attr.`42 : index`]

    fn unused_method(mut self) -> Self.MyType:
        return Self.MyType()
