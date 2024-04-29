# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct SomeStruct[size: Int, other_param: Bool]:
    """Docstring for SomeStruct.

    More docstring for SomeStruct.

    Constraints:
        The contraints of SomeStruct.

    Parameters:
        size: The size of SomeStruct.
        other_param: Another param.
    """

    fn __init__(
        inout self,
        borrowed borrowed_input: Int,
        init_arg: Int,
        owned owned_input: Int,
        *init_kargs: Int,
    ):
        """Init documentation.

        Args:
            borrowed_input: A borrowed argument.
            init_arg: An Int argument.
            owned_input: An owned argument.
            init_kargs: Multiple arguments.
        """
        _ = init_arg
        _ = init_kargs
        pass

    @staticmethod
    fn static_method() -> Int:
        return 420

    fn bar(inout self):
        fn non_capturing_nested_function():
            pass

    async fn async_function(inout self):
        @parameter
        fn parameter_nested_function():
            pass

        fn another_nested_function():
            pass

    fn function_that_raises(
        inout self, arg_in_function_that_raises: Int
    ) raises -> String:
        """A function that raises.

        Args:
            arg_in_function_that_raises: An arg in a function with by-ref result.
        """
        return "foo"

    fn function_with_param[Param1: Int, Param2: Int](inout self):
        """A function with param.

        Parameters:
          Param1: An Int param.
          Param2: Another Int param.
        """
        pass


fn exported_function():
    "This is an exported function."

    fn a_closure():
        pass

    a_closure()


def def_function() -> Int:
    return 120


fn main():
    print("foo")
