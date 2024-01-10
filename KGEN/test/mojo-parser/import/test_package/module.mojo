# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a package.

from .test_nested_package.module import nested_function

alias Int = __mlir_type.index

alias top_level_alias = __mlir_attr.`2 : index`


trait AnyType:
    fn __del__(owned self, /):
        ...


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        ...


trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
        ...


fn function():
    call_nested_function()
    return


fn call_nested_function():
    nested_function()
    return


@value
struct SomeType:
    var value: Int


@value
struct `weird()struct[]`:
    pass


fn `use()weird[]`() -> `weird()struct[]`:
    return `weird()struct[]`()


@value
struct ParameterizedType[value: Int]:
    pass


@value
struct Wrapper:
    var data: Int

    alias MyType = ParameterizedType[__mlir_attr.`42 : index`]

    fn unused_method(inout self) -> Self.MyType:
        return Self.MyType()
