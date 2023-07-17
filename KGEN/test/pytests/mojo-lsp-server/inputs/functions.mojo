# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct SomeStruct:
    fn __init__(inout self):
        pass

    @staticmethod
    fn static_method():
        pass

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

    fn function_that_raises(inout self) raises:
        pass


fn exported_function():
    "This is an exported function."

    fn a_closure():
        pass

    a_closure()


def def_function() -> None:
    pass
