# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s

"""
This is a module summary, that
spills over to the next line."""


# CHECK:  "kind": "module",
# CHECK:  "name": "mojo-doc",
# CHECK:  "summary": "This is a module summary, that spills over to the next line.",
# CHECK:  "description": "",

# CHECK:  "aliases": [
# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_Type",
# CHECK:  "value": "Int",
# CHECK:  "summary": "An example alias of a Type",
alias alias_Type = Int
"""An example alias of a Type"""

# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_Value",
# CHECK:  "value": "10",
# CHECK:  "summary": "An example alias of a Value",
alias alias_Value = 10
"""An example alias of a Value"""


# CHECK:  "functions": [
# CHECK:  "name": "empty_fn",
# CHECK:  "overloads": [
# CHECK:      "description": "The is some kind of description."
# CHECK:      "signature": "empty_fn()"
# CHECK:      "summary": "This is a function summary."
# CHECK:  ]


fn empty_fn():
    """This is a function summary.

    The is some kind of description.
    """
    return


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_that_async",
# CHECK:  "overloads":
# CHECL:      "async": true
# CHECK:      "returns": "an Int."
# CHECK:      "signature": "fn_that_async() -> Int"
# CHECK:      "summary": "This is a function summary."


async fn fn_that_async() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        an Int.
    """
    return 33


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_that_raises",
# CHECK:  "overloads":
# CHECL:      "raises": true
# CHECK:      "returns": "an Int."
# CHECK:      "signature": "fn_that_raises() -> Int"
# CHECK:      "summary": "This is a function summary."


def fn_that_raises() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        an Int.
    """
    return 33


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_args",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "description": "This is an argument."
# CHECK:          "name": "arg"
# CHECK:          "type": "Int"
# CHECK:      "signature": "fn_with_args(arg: Int)",
# CHECK:      "summary": "This is a function summary."


fn fn_with_args(arg: Int):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
    """
    return


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_overload",
# CHECK:  "overloads": [
# CHECK:      "signature": "fn_with_overload()",
# CHECK:      "signature": "fn_with_overload(arg: Int)",


fn fn_with_overload():
    """This is a function summary.

    The is some kind of description.
    """
    return


fn fn_with_overload(arg: Int):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
    """
    return


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_parameter_references",
# CHECK: "overloads":
# CHECK:     "signature": "fn_with_parameter_references[arg1_type: AnyType, arg2_type: AnyType](func: fn (arg1_type, arg2_type) -> (), arg1: arg1_type, arg2: arg2_type)"


fn fn_with_parameter_references[
    arg1_type: AnyType,
    arg2_type: AnyType,
](
    func: __mlir_type[`(`, arg1_type, `,`, arg2_type, `) -> ()`],
    arg1: arg1_type,
    arg2: arg2_type,
):
    pass


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_params",
# CHECK:  "overloads":
# CHECK:      "parameters": [
# CHECK:          "description": "This is a parameter."
# CHECK:          "name": "param"
# CHECK:          "type": "dtype"
# CHECK:      "signature": "fn_with_params[param: dtype]()"


fn fn_with_params[param: __mlir_type.`!kgen.dtype`]():
    """This is a function summary.

    The is some kind of description.

    Parameters:
        param: This is a parameter.
    """
    return


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_params_and_return",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "description": "This is an argument."
# CHECK:         "name": "arg"
# CHECK:         "type": "Int"
# CHECK:     "returns": "This is a return value."
# CHECK:     "signature": "fn_with_params_and_return(arg: Int) -> Int"


fn fn_with_params_and_return(arg: Int) -> Int:
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.

    Returns:
        This is a return value.
    """
    return arg


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_variant",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "description": "This is an argument."
# CHECK:          "name": "arg"
# CHECK:          "type": "Variant[Error, None]"
# CHECK:      "signature": "fn_with_variant(arg: Variant[Error, None])"
# CHECK:      "summary": "This is a function summary."


fn fn_with_variant(
    arg: __mlir_type[`!pop.variant<`, Error, `, `, NoneType, `>`]
):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
    """
    return


# CHECK:  "structs": [
# CHECK:  "kind": "struct",
# CHECK:  "name": "InMemoryStruct",

# Check that special functions are ordered first, and with the correct
# prioritization (i.e. not just name based).
# CHECK:  "kind": "function",
# CHECK:  "name": "__init__",
# CHECK:  "name": "__copyinit__",

# CHECK: "name": "__add__",
# CHECK: "overloads":
# CHECK:      "signature": "__add__(self: Self, other: Self) -> Self"

# CHECK:  "name": "__del___",

# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_by_conventions",
# CHECK:  "overloads"
# CHECK:      "args"
# CHECK:        {
# CHECK:          "description": "This is a by-ref argument."
# CHECK:          "name": "arg"
# CHECK:          "type": "Self"
# CHECK:        },
# CHECK:        {
# CHECK:          "description": "This is a variadic argument."
# CHECK:          "name": "args",
# CHECK:          "type": "*Self"
# CHECK:        }
# CHECK:      "constraints": "This describes the method's constraints.",
# CHECK:      "description": ""
# CHECK:      "returns": "This is a by-ref return value.",
# CHECK:      "signature": "fn_with_by_conventions(inout self: Self, inout arg: Self, inout *args: Self) -> Self",
# CHECK:      "summary": "This is a function summary."


struct InMemoryStruct:
    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __del___(owned self):
        pass

    fn __add__(self, other: Self) -> Self:
        return other

    fn fn_with_by_conventions(
        inout self, inout arg: InMemoryStruct, inout *args: InMemoryStruct
    ) -> InMemoryStruct:
        """This is a function summary.

        Constraints:
            This describes the method's constraints.

        Args:
            arg: This is a by-ref argument.
            args: This is a variadic argument.

        Returns:
            This is a by-ref return value.
        """
        return arg


# CHECK:  "kind": "struct",
# CHECK:  "name": "ParameterClass",
# CHECK:  "summary": "This is a class summary.",
# CHECK:  "constraints": "This describes the struct's constraints.",
# CHECK:  "parameters": [
# CHECK:      "name": "_type"
# CHECK:      "type": "dtype"
# CHECK:      "description": "This is a parameter."
# CHECK:  ],
# CHECK:  "description": "The is some kind of description.\n",

# CHECK:      "kind": "function",
# CHECK:      "name": "fn_with_self_param",
# CHECK:          "parameters": [
# CHECK:              "description": "This is a Self parameter."
# CHECK:              "name": "param"
# CHECK:              "type": "Self"
# CHECK:          "signature": "fn_with_self_param[param: Self](self: Self)"


@register_passable
struct ParameterClass[_type: __mlir_type.`!kgen.dtype`]:
    """This is a class summary.

    The is some kind of description.

    Constraints:
        This describes the struct's constraints.

    Parameters:
        _type: This is a parameter.
    """

    fn fn_with_self_param[param: ParameterClass[_type]](self):
        """A summary.

        Parameters:
            param: This is a Self parameter.
        """
        return
