# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -mojo-doc-validate -verify-diagnostics %s


# expected-warning @below {{public symbol 'ArgStruct' is missing a doc string}}
struct ArgStruct:
    pass


struct ParamStruct_Invalid[_type: __mlir_type.`!kgen.dtype`]:
    """This is a class summary.

    # expected-warning @below {{parameter '_type' is not documented}}
    Parameters:
        invalid_param: This is an invalid parameter.
          # expected-warning @above {{unknown parameter 'invalid_param' in doc string}}
    """

    pass


struct ParamStruct_Duplicates[_type: __mlir_type.`!kgen.dtype`]:
    """This is a class summary.

    # expected-note @below {{see previous definition here}}
    Parameters:
        _type: Summary.
        _type: Summary.
          # expected-warning @above {{duplicate parameter '_type' in doc string}}


    # expected-warning @below {{duplicate 'Parameters' section found in doc string}}
    Parameters:
        _type: Summary.
    """

    pass


struct ParamStruct_Order[
    param1: __mlir_type.`!kgen.dtype`, param2: __mlir_type.`!kgen.dtype`
]:
    """This is a class summary.

    Parameters:
        param2: Summary.
          # expected-warning @above {{'param2' is defined at index 1, but specified in doc string at index 0}}
        param1: Summary.
    """

    pass


struct StructWithMissingMethod:
    """This defines methods with missing doc strings."""

    # expected-warning @below {{public symbol 'method_with_missing_doc_string($parser-doc-errors::StructWithMissingMethod)' is missing a doc string}}
    fn method_with_missing_doc_string(self):
        pass

    fn _private_method_with_no_doc_string(self):
        pass

    # expected-warning @below {{public symbol '__init__($parser-doc-errors::StructWithMissingMethod=&)' is missing a doc string}}
    fn __init__(inout self):
        pass


# expected-warning @below {{public symbol 'fn_missing_doc_string()' is missing a doc string}}
fn fn_missing_doc_string():
    pass


fn _fn_private_no_doc_string():
    pass


fn fn_args_invalid(arg: ArgStruct):
    """This is a function summary.

    # expected-warning @below {{argument 'arg' is not documented}}
    Args:
        unknown_arg: This is an argument.
          # expected-warning @above {{unknown argument 'unknown_arg' in doc string}}
    """
    return


fn fn_args_overindent(arg: ArgStruct):
    """This is a function summary.

    Description.

        # expected-warning @below {{section tag 'Args' is overindented}}
        Args:
            arg: This is an argument.
    """
    return


fn fn_args_duplicates(arg: ArgStruct):
    """This is a function summary.

    # expected-note @below {{see previous definition here}}
    Args:
        arg: This is an argument.
        arg: This is an argument.
          # expected-warning @above {{duplicate argument 'arg' in doc string}}

    # expected-warning @below {{duplicate 'Args' section found in doc string}}
    Args:
        arg: This is an argument.
    """
    return


fn fn_args_order(arg: ArgStruct, arg2: ArgStruct):
    """This is a function summary.

    Args:
        arg2: This is an argument.
          # expected-warning @above {{'arg2' is defined at index 1, but specified in doc string at index 0}}
        arg: This is an argument.
    """
    return


fn fn_args_return():
    """This is a function summary.

    # expected-warning @below {{unexpected 'Returns' in doc string for function with no results}}
    # expected-note @below {{see previous definition here}}
    Returns:
      This returns nothing.

    # expected-warning @below {{duplicate 'Returns' section found in doc string}}
    Returns:
      This returns nothing.
    """
    return


fn fn_nested_fn():
    """This is a function that defines a nested struct and function.

    The nested struct and function do not include doc strings, but neither
    should be reported as invalid.
    """

    fn nested_fn():
        pass

    struct NestedStruct:
        pass

    return
