# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from String import String


struct SomeStruct:
    fn __init__(inout self):
        pass

    @staticmethod
    fn static_method() -> Int:
        return 420

    fn bar(inout self):
        @noncapturing
        fn non_capturing_nested_function():
            pass

    async fn async_function(inout self):
        @parameter
        fn parameter_nested_function():
            pass

        fn another_nested_function():
            pass

    fn function_that_raises(inout self) raises -> String:
        return "foo"


fn exported_function():
    "This is an exported function."

    fn a_closure():
        pass

    a_closure()


def def_function() -> Int:
    return 120
