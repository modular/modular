# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo %s -doc-gen | FileCheck %s

"""
This is a module summary, that
spills over to the next line."""


# CHECK:  "kind": "module",
# CHECK:  "name": "mojo-doc",
# CHECK:  "summary": "This is a module summary, that spills over to the next line.",
# CHECK:  "description": "",
# CHECK:  "children": [

# CHECK:  "kind": "struct",
# CHECK:  "name": "InMemoryStruct",
# CHECK:  "children": [

# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_by_conventions",
# CHECK:  "overloads"
# CHECK:      "signature": "fn_with_by_conventions(self: Self&, arg: Self&, *args: Self&) -> Self&",
# CHECK:      "summary": "This is a function summary.",
# CHECK:      "args"
# CHECK:        {
# CHECK:          "signature": "arg: Self&",
# CHECK:          "description": "This is a by-ref argument."
# CHECK:        },
# CHECK:        {
# CHECK:          "signature": "*args: Self&",
# CHECK:          "description": "This is a variadic argument."
# CHECK:        }
# CHECK:      "returns": "This is a by-ref return value.",
# CHECK:      "description": ""


struct InMemoryStruct:
    fn __copyinit__(self&, existing: Self):
        pass

    fn fn_with_by_conventions(
        self&, arg&: InMemoryStruct, *args&: InMemoryStruct
    ) -> InMemoryStruct:
        """This is a function summary.

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
# CHECK:  "parameters": [
# CHECK:      "signature": "_type: __mlir_type.!kgen.dtype",
# CHECK:      "description": "This is a parameter."
# CHECK:  ],
# CHECK:  "description": "The is some kind of description.\n",

# CHECK:      "kind": "function",
# CHECK:      "name": "fn_with_self_param",
# CHECK:          "signature": "fn_with_self_param(self: Self)",
# CHECK:          "parameters": [
# CHECK:              "signature": "param: Self",
# CHECK:              "description": "This is a Self parameter."


@register_passable
struct ParameterClass[_type: __mlir_type.`!kgen.dtype`]:
    """This is a class summary.

    The is some kind of description.

    Parameters:
        _type: This is a parameter.
    """

    fn fn_with_self_param[param: ParameterClass[_type]](self):
        """A summary.

        Parameters:
            param: This is a Self parameter.
        """
        return


# CHECK:  "name": "empty_fn",
# CHECK:  "overloads": [
# CHECK:      "signature": "empty_fn()",
# CHECK:      "summary": "This is a function summary.",
# CHECK:      "description": "The is some kind of description."
# CHECK:  ]


fn empty_fn():
    """This is a function summary.

    The is some kind of description.
    """
    return


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_args",
# CHECK:  "overloads":
# CHECK:      "signature": "fn_with_args(arg: $Int::Int)",
# CHECK:      "summary": "This is a function summary.",
# CHECK:      "args":
# CHECK:          "signature": "arg: $Int::Int",
# CHECK:          "description": "This is an argument."


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
# CHECK:      "signature": "fn_with_overload(arg: $Int::Int)",


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


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_with_params",
# CHECK:  "overloads":
# CHECK:      "signature": "fn_with_params()",
# CHECK:      "parameters": [
# CHECK:          "signature": "param: __mlir_type.!kgen.dtype",
# CHECK:          "description": "This is a parameter."


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
# CHECK:     "signature": "fn_with_params_and_return(arg: $Int::Int) -> $Int::Int",
# CHECK:     "args":
# CHECK:         "signature": "arg: $Int::Int",
# CHECK:         "description": "This is an argument."
# CHECK:     "returns": "This is a return value."


fn fn_with_params_and_return(arg: Int) -> Int:
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.

    Returns:
        This is a return value.
    """
    return arg
