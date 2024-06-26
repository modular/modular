# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
This is a module summary, that
spills over to the next line."""

# RUN: mojo doc %s | FileCheck %s

# Check that no diagnostics are output:
# RUN: mojo doc %s 2>&1 | FileCheck %s --allow-empty --check-prefix CHECK-DIAG
# CHECK-DIAG-NOT: warning

"""
This is a module summary, that
spills over to the next line."""

from layout.int_tuple import *
from sys.info import triple_is_nvidia_cuda


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


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_construct",
# CHECK:  "value": "DynamicTuple(0, 1, 2, 3, 4)"
alias alias_construct = IntTuple(0, 1, 2, 3, 4)


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_cond",
# CHECK:  "value": "2 if triple_is_nvidia_cuda() else 1"
alias alias_cond = 2 if triple_is_nvidia_cuda() else 1

# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_fn",
# CHECK:  "value": "fn(Int, Int) -> None"
alias alias_fn = fn (Int, Int) -> None


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_str",
# CHECK:  "value": "\"\""
alias alias_str = ""


# CHECK:  "deprecated": "deprecated alias",
# CHECK:  "kind": "alias",
# CHECK:  "name": "deprecated_alias",
@deprecated("deprecated alias")
alias deprecated_alias = 1


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


# CHECK-NOT:  fn_hidden


@doc_private
fn fn_hidden() -> Int:
    """This is a function summary.

    The is some kind of description.

    Returns:
        An Int.
    """
    return 33


# CHECK:  "name": "fn_that_async",
# CHECK:  "overloads":
# CHECK:      "async": true
# CHECK:      "returnsDoc": "An Int."
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
# CHECK:      "raisesDoc": "Raises an exeception when it wants to."
# CHECK:      "returnsDoc": "An Int."
# CHECK:      "signature": "fn_that_raises() -> Int"
# CHECK:      "summary": "This is a function summary."


def fn_that_raises() -> Int:
    """This is a function summary.

    The is some kind of description.

    Raises:
        Raises an exeception when it wants to.

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
# CHECK:          "convention": "inout"
# CHECK:          "description": "This is an inout arg."
# CHECK:          "name": "inoutArg"
# CHECK:          "type": "Int"
# CHECK:          "convention": "owned"
# CHECK:          "description": "This is an owned arg."
# CHECK:          "name": "ownedArg"
# CHECK:          "type": "Int"
# CHECK:          "convention": "borrowed"
# CHECK:          "description": "This is a borrowedArg."
# CHECK:          "name": "borrowedArg"
# CHECK:          "type": "Int"
# CHECK:      "signature": "fn_with_args(arg: Int, inout inoutArg: Int, owned ownedArg: Int, borrowedArg: Int)",
# CHECK:      "summary": "This is a function summary."


fn fn_with_args(
    arg: Int,
    inout inoutArg: Int,
    owned ownedArg: Int,
    borrowed borrowedArg: Int,
):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
        inoutArg: This is an inout arg.
        ownedArg: This is an owned arg.
        borrowedArg: This is a borrowedArg.
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
# CHECK:     "signature": "fn_with_parameter_references[arg1_type: AnyTrivialRegType, arg2_type: AnyTrivialRegType](func: fn (arg1_type, arg2_type) -> (), arg1: arg1_type, arg2: arg2_type)"


fn fn_with_parameter_references[
    arg1_type: AnyTrivialRegType,
    arg2_type: AnyTrivialRegType,
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
# CHECK:          "passingKind": "pos",
# CHECK:          "type": "dtype"
# CHECK:          "description": "This is a second parameter."
# CHECK:          "name": "param2"
# CHECK:          "passingKind": "kw",
# CHECK:          "type": "dtype"
# CHECK:      "signature": "fn_with_params[param: dtype, /, *, param2: dtype]()"


fn fn_with_params[
    param: __mlir_type.`!kgen.dtype`, /, *, param2: __mlir_type.`!kgen.dtype`
]():
    """This is a function summary.

    The is some kind of description.

    Parameters:
        param: This is a parameter.
        param2: This is a second parameter.
    """
    return


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_params_and_return",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "description": "This is an argument."
# CHECK:         "name": "arg"
# CHECK:         "type": "Int"
# CHECK:     "returnsDoc": "This is a return value."
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
# CHECK: "args":
# CHECK: {
# CHECK:     "name": "arg",
# CHECK:     "type": "MyStruct[x]"
# CHECK: }
# CHECK: "parameters":
# CHECK: {
# CHECK:     "description": "Explicitly declared function parameter.",
# CHECK:     "kind": "parameter",
# CHECK:     "name": "p",
# CHECK:     "type": "Int"
# CHECK: }
# CHECK: "signature": "fn_with_implicit_params[p: Int](arg: MyStruct[x])"


fn fn_with_implicit_params[p: Int](arg: MyStruct):
    """
    An autoparameterized function with documentation.

    Parameters:
      p: Explicitly declared function parameter.

    Args:
      arg: An argument whose declared type induces an implicit parameter.
    """
    pass


# CHECK:  "name": "pos_only_print",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "name": "obj"
# CHECK:          "passingKind": "pos",
# CHECK:          "type": "object"

# CHECK:          "name": "sep"
# CHECK:          "passingKind": "pos_or_kw",
# CHECK:          "type": "String"
# CHECK:      "signature": "pos_only_print(obj: object, /, sep: String)",


fn pos_only_print(obj: object, /, sep: String):
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


# CHECK:  "name": "default_args_and_params",
# CHECK:  "overloads":
# CHECK:      "args":
# CHECK:          "default": "2",
# CHECK:          "name": "b"

# CHECK:          "default": "3",
# CHECK:          "name": "c"

# CHECK:      "parameters":
# CHECK:          "default": "1",
# CHECK:          "name": "a"
# CHECK:      "signature": "default_args_and_params[a: Int = 1](b: Int = 2, /, *, c: Int = 3)",


fn default_args_and_params[a: Int = 1](b: Int = 2, /, *, c: Int = 3):
    """Test default handling.

    Parameters:
        a: Param.

    Args:
        b: Arg.
        c: Arg.
    """
    pass


# CHECK: "name": "variadic_pack",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "*vals",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "*Ts"

# CHECK:     "parameters":
# CHECK:         "name": "*Ts",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "AnyType"

# CHECK:     "signature": "variadic_pack[*Ts: AnyType](*vals: *Ts)",


fn variadic_pack[*Ts: AnyType](*vals: *Ts):
    """Test variadic pack argument type printing.

    Parameters:
        Ts: Variadic types.

    Args:
        vals: Variadic pack arguments.
    """
    pass


# CHECK: "name": "variadic_arg_hack",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "vals",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "variadic<!lit.ref<:trait<{{.*}}AnyType> element_type, imm #lit.lifetime>, borrow_in_mem>"

# CHECK:     "signature": "variadic_arg_hack[element_type: AnyType](vals: variadic<!lit.ref<:trait<_stdlib::_builtin::_anytype::_AnyType> element_type, imm #lit.lifetime>, borrow_in_mem>)",


fn variadic_arg_hack[
    element_type: AnyType
](
    vals: __mlir_type[
        `!kgen.variadic<!lit.ref<`,
        element_type,
        `, #lit.lifetime<0>: !lit.lifetime<0>, 0>, borrow_in_mem>`,
    ]
):
    """Test hacky use case of `!kgen.variadic` argument type printing.

    Args:
        vals: !kgen.variadic arguments.
    """
    pass


# CHECK: "name": "variadic_params_args",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "*vals",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "Int"

# CHECK:         "name": "**kwargs",
# CHECK:         "passingKind": "kw",
# CHECK:         "type": "object"

# CHECK:     "parameters":
# CHECK:         "name": "*nums",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "Int"

# CHECK:     "signature": "variadic_params_args[*nums: Int](*vals: Int, *, owned **kwargs: object)",


fn variadic_params_args[*nums: Int](*vals: Int, **kwargs: object):
    """Test variadic argument/parameter type printing.

    Parameters:
        nums: Variadic parameters.

    Args:
        vals: Variadic arguments.
        kwargs: Variadic keyword arguments.
    """
    pass


# CHECK: "name": "parameter_with_escaped_mlir_name",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "value",
# CHECK:         "type": "type"

# CHECK:     "parameters":
# CHECK:         "kind": "parameter",
# CHECK:         "name": "type",

# CHECK:     "signature": "parameter_with_escaped_mlir_name[type: AnyType](value: type)",


fn parameter_with_escaped_mlir_name[type: AnyType](value: type):
    pass


# CHECK: "kind": "function"
# CHECK: "overloads":
# CHECK:    "deprecated": "deprecated function"
# CHECK:    "name": "deprecated_function"
@deprecated("deprecated function")
fn deprecated_function():
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
# CHECK:  "name": "__del__",

# CHECK: "name": "__add__",
# CHECK: "overloads":
# CHECK:      "signature": "__add__(self: Self, other: Self) -> Self"

# CHECK:  "name": "__len__",

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
# CHECK:          "name": "*args",
# CHECK:          "type": "Self"
# CHECK:        }
# CHECK:      "constraints": "This describes the method's constraints.",
# CHECK:      "description": ""
# CHECK:      "returnsDoc": "This is a by-ref return value.",
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

    fn __del__(owned self):
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


# CHECK:  "kind": "struct",
# CHECK:  "name": "StructWithDefault",
# CHECK:  "parameters":
# CHECK:      "default": "1",
# CHECK:      "name": "a"


struct StructWithDefault[a: Int = 1]:
    pass


# CHECK: "deprecated": "deprecated struct"
# CHECK: "name": "DeprecatedStruct"
@deprecated("deprecated struct")
struct DeprecatedStruct:
    pass


# CHECK:  "summary": "This is a module summary, that spills over to the next line."

##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

# CHECK:  "traits": [
# CHECK:    "description": "The is some kind of description.",
# CHECK:    "functions":
# CHECK:      "kind": "function",
# Check that we don't generate inherited methods (like __del__ from AnyType).
# CHECK-NOT: "name": "__del__"
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


# CHECK: "deprecated": "deprecated trait"
# CHECK: "name": "DeprecatedTrait"
@deprecated("deprecated trait")
trait DeprecatedTrait:
    pass


# Check that we include version information in the generated JSON.
# CHECK: "version":
