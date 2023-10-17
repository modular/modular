# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# This file is a test input that defines a module within a package.

from .test_nested_package.module import nested_function


alias top_level_alias = 2


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


fn use_weird() -> `weird()struct[]`:
    return `weird()struct[]`()


struct ParameterizedType[value: Int]:
    pass
