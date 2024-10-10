# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -o /dev/null -mojo-diagnose-missing-doc-strings -verify-diagnostics %s


# expected-warning @below {{public symbol 'ArgStruct' is missing a doc string}}
struct ArgStruct:
    pass


struct _ParamStruct_Private_Missing[_type: __mlir_type.`!kgen.dtype`]:
    """This is a private struct doc string.

    It doesn't need to include a `Parameters:` section.
    """

    pass


# expected-warning @below {{struct takes parameters, but no 'Parameters' in doc string}}
struct ParamStruct_Missing[_type: __mlir_type.`!kgen.dtype`]:
    """This doc string is missing a `Parameters:` section."""

    pass


struct ParamStruct_Invalid[_type: __mlir_type.`!kgen.dtype`]:
    """This is a class summary.

    # expected-warning-re @below {{parameter '{{.*}}_type' is not documented}}
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
        param1: Summary.
    """

    # expected-warning @-4 {{'param2' is defined at index 1, but specified in doc string at index 0}}
    pass


struct StructWithMissingMethod:
    """This defines methods with missing doc strings."""

    # expected-warning @below {{public symbol 'method_with_missing_doc_string' is missing a doc string}}
    fn method_with_missing_doc_string(self):
        pass

    fn _private_method_with_no_doc_string(self):
        pass

    # expected-warning @below {{public symbol '__init__' is missing a doc string}}
    fn __init__(inout self):
        pass


# expected-warning @below {{public symbol 'fn_missing_doc_string' is missing a doc string}}
fn fn_missing_doc_string():
    pass


fn _fn_private_no_doc_string():
    pass


fn fn_poor_style():
    """this summary should be capitalized and end with a period"""
    # expected-warning @above {{doc string summary should begin with a capital letter or non-alpha character, but this begins with 't'}}
    # expected-warning @above {{doc string summary should end with a period '.', but this ends with 'd'}}
    pass


fn _fn_private_args_missing(arg: ArgStruct):
    """This is a private function doc string.

    It doesn't need to include an `Args:` section.
    """
    pass


# expected-warning @below {{function takes arguments, but no 'Args' in doc string}}
fn fn_args_missing(arg: ArgStruct):
    """This doc string is missing an `Args:` section."""
    return


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
        arg: This is an argument.
    """
    # expected-warning @-3 {{'arg2' is defined at index 1, but specified in doc string at index 0}}
    return


fn fn_args_empty(arg: ArgStruct, arg2: ArgStruct):
    """This function contains empty argument descriptions.

    Args:
        arg:
        arg2:
    """
    # expected-warning @-3 {{'arg' does not have a description}}
    # expected-warning @-3 {{'arg2' does not have a description}}
    pass


fn fn_args_poor_style(arg: ArgStruct, arg2: ArgStruct):
    """This function contains arguments with poor style.

    Args:
        arg: `arg` starts with a valid character but doesn't end with a period
        arg2: this should start with a capital letter.
    """
    # expected-warning @-3 {{'arg' description should end with a period '.', but this ends with 'd'}}
    # expected-warning @-3 {{'arg2' description should begin with a capital letter or non-alpha character, but this begins with 't'}}
    pass


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

fn fn_raises():
    """This is a function summary.

    # expected-warning @below {{unexpected 'Raises' in doc string for function that does not throw}}
    # expected-note @below {{see previous definition here}}
    Raises:
      This raises nothing.

    # expected-warning @below {{duplicate 'Raises' section found in doc string}}
    Raises:
      This raises nothing.
    """
    return


# expected-warning @below {{function has results, but no 'Returns' in doc string}}
fn fn_args_missing_return() -> int:
    """This doc string is missing a `Returns:` section."""
    return `0`


fn fn_returns_section_empty() -> int:
    """This doc string includes a `Returns:` section, but it's empty.

    # expected-warning @below {{'Returns' section is empty}}
    Returns:
    """
    return `0`


fn fn_returns_section_poor_style() -> int:
    """This doc string has a `Returns:` section with poor style.

    Returns:
        doesn't start with a capital letter, doesn't end with a period!
    """
    # expected-warning @-2 {{section body should begin with a capital letter or non-alpha character, but this begins with 'd'}}
    # expected-warning @-3 {{section body should end with a period '.', but this ends with '!'}}
    return `0`


fn fn_nested_fn():
    """This is a function that defines a nested function.

    The nested function does not include a doc string, but it should not be
    reported as invalid.
    """

    fn nested_fn():
        pass

    return


struct Error:
    """Error type stub to allow decoupling from the builtins."""

    pass


fn fn_raises_with_return_type(x: int) raises -> int:
    """This is a function that raises, with an explicit return type.

    Because it raises, it implicitly has a memory-only `__result__` argument.
    However, this doc string should not document this hidden argument.

    Args:
        x: An explicit argument.

    Returns:
        `0`.
    """
    return `0`


@value
struct object:
    """Object type stub to allow decoupling from the builtins."""

    pass


def def_implicit_object_return_type(x: int):
    """This is a `def` function with no explicit return type.

    Because it implicitly returns an object, it has a hidden `__result__`
    argument. However, this doc string should not document this hidden argument.

    Args:
        x: An explicit argument.

    Returns:
        Implicitly, this returns a None object.
    """
    pass
