# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo doc %s | FileCheck %s

# Check that no diagnostics are output:
# RUN: mojo doc %s 2>&1 | FileCheck %s --allow-empty --check-prefix CHECK-DIAG
# CHECK-DIAG-NOT: warning

"""
This is a module summary, that
spills over to the next line."""


# CHECK:  "aliases": [
# CHECK:   "kind": "alias",
# CHECK:   "name": "alias_Type",
# CHECK:   "summary": "An example alias of a Type",
# CHECK:   "value": "Int"
alias alias_Type = Int
"""An example alias of a Type"""

# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_Value",
# CHECK:  "summary": "An example alias of a Value",
# CHECK:  "value": "10"
alias alias_Value = 10
"""An example alias of a Value"""


# CHECK:  "description": "",
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
# CHECK:      "async": true
# CHECK:      "returns": "An Int."
# CHECK:      "signature": "fn_that_async() -> Int"
# CHECK:      "summary": "This is a function summary."


async fn fn_that_async() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        An Int.
    """
    return 33


# CHECK:  "kind": "function",
# CHECK:  "name": "fn_that_raises",
# CHECK:  "overloads":
# CHECK:      "raises": true
# CHECK:      "returns": "An Int."
# CHECK:      "signature": "fn_that_raises() -> Int"
# CHECK:      "summary": "This is a function summary."


def fn_that_raises() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        An Int.
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
# CHECK:     "signature": "fn_with_parameter_references[arg1_type: AnyRegType, arg2_type: AnyRegType](func: fn (arg1_type, arg2_type) -> (), arg1: arg1_type, arg2: arg2_type)"


fn fn_with_parameter_references[
    arg1_type: AnyRegType,
    arg2_type: AnyRegType,
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


@value
struct MyStruct[x: Int]:
    pass


# CHECK: "name": "fn_with_implicit_params",
# CHECK: "parameters":
# CHECK: {
# CHECK:     "description": "Explicitly declared function parameter.",
# CHECK:     "kind": "parameter",
# CHECK:     "name": "p",
# CHECK:     "type": "Int"
# CHECK: },
# CHECK: {
# CHECK:     "description": "",
# CHECK:     "kind": "parameter",
# CHECK:     "name": "x",
# CHECK:     "type": "Int"
# CHECK: }


fn fn_with_implicit_params[p: Int](x: MyStruct):
    """
    An autoparameterized function with documentation.

    Parameters:
      p: Explicitly declared function parameter.

    Args:
      x: An argument whose declared type induces an implicit parameter.
    """
    pass


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
    arg: __mlir_type[`!kgen.variant<`, Error, `, `, NoneType, `>`]
):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
    """
    return


# CHECK:  "name": "pos_only_print",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "name": "obj"
# CHECK:          "passingKind": "pos",
# CHECK:          "type": "object"

# CHECK:          "name": "sep"
# CHECK:          "passingKind": "pos_or_kw",
# CHECK:          "type": "StringLiteral"
# CHECK:      "signature": "pos_only_print(obj: object, /, sep: StringLiteral)",


fn pos_only_print(obj: object, /, sep: StringLiteral):
    """Prints an object type.

    Args:
        obj: The object to print.
        sep: The separator.
    """
    pass


# CHECK:  "name": "keyword_only_prod",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "name": "a"
# CHECK:          "passingKind": "pos",

# CHECK:          "name": "b"
# CHECK:          "passingKind": "pos",

# CHECK:          "name": "offset"
# CHECK:          "passingKind": "kw",
# CHECK:      "signature": "keyword_only_prod(a: Int, b: Int, /, *, offset: Int)",


fn keyword_only_prod(a: Int, b: Int, /, *, offset: Int):
    """Multiply and add an offset.

    Args:
        a: First factor.
        b: Second factor.
        offset: The offset to be added.
    """
    pass


# CHECK:  "kind": "module",
# CHECK:  "name": "mojo-doc",

# CHECK:  "structs": [

# Check that we don't generate any synthesized thunk methods
# from the trait usage.
# CHECK-NOT: "name": "thunk_

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
# CHECK:  "kind": "struct",
# CHECK:  "name": "InMemoryStruct",
# CHECK:  "parentTraits": [
# CHECK-NEXT:   "AnyType"
# CHECK-NEXT:   "Sized"


struct InMemoryStruct(Sized):
    fn __init__(inout self):
        pass

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __del___(owned self):
        pass

    fn __add__(self, other: Self) -> Self:
        return other

    fn __len__(self) -> Int:
        return 0

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


# CHECK:  "constraints": "This describes the struct's constraints.",
# CHECK:  "description": "The is some kind of description.\n",
# CHECK:      "kind": "function",
# CHECK:      "name": "fn_with_self_param",
# CHECK:          "parameters": [
# CHECK:              "description": "This is a Self parameter."
# CHECK:              "name": "param"
# CHECK:              "type": "Self"
# CHECK:          "signature": "fn_with_self_param[param: Self](self: Self)"
# CHECK:  "kind": "struct",
# CHECK:  "name": "ParameterClass",
# CHECK:  "parameters": [
# CHECK:      "description": "This is a parameter."
# CHECK:      "name": "_type"
# CHECK:      "type": "dtype"
# CHECK:  ],
# CHECK:  "summary": "This is a class summary."


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


# CHECK:  "summary": "This is a module summary, that spills over to the next line."

##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

# CHECK:  "traits": [
# CHECK:    "description": "The is some kind of description.",
# CHECK:    "functions":
# CHECK:      "kind": "function",
# CHECK:      "name": "f",
# CHECK:      "summary": "This is a trait function doc."
# CHECK:    "kind": "trait",
# CHECK:    "name": "Trait",
# CHECK:    "summary": "This is a trait doc."


trait Trait:
    """This is a trait doc.

    The is some kind of description.
    """

    fn f(self: Self):
        """This is a trait function doc."""
        ...


# Check that we include version information in the generated JSON.
# CHECK: "version":
