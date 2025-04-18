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

# """
# This is a module summary, that
# spills over to the next line."""

from layout.int_tuple import *
from sys.info import is_nvidia_gpu


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
# CHECK:  "value": "IntTuple(0, 1, 2, 3, 4)"
alias alias_construct = IntTuple(0, 1, 2, 3, 4)


# CHECK:  "kind": "alias",
# CHECK:  "name": "alias_cond",
# CHECK:  "value": "2 if is_nvidia_gpu() else 1"
alias alias_cond = 2 if is_nvidia_gpu() else 1

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


# CHECK-LABEL:  "name": "fn_that_async",
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
# CHECK:          "convention": "mut"
# CHECK:          "description": "This is an mut arg."
# CHECK:          "name": "inoutArg"
# CHECK:          "type": "Int"
# CHECK:          "convention": "owned"
# CHECK:          "description": "This is an owned arg."
# CHECK:          "name": "ownedArg"
# CHECK:          "type": "Int"
# CHECK:          "convention": "read"
# CHECK:          "description": "This is a borrowedArg."
# CHECK:          "name": "borrowedArg"
# CHECK:          "type": "Int"
# CHECK:      "signature": "fn_with_args(arg: Int, mut inoutArg: Int, owned ownedArg: Int, borrowedArg: Int)",
# CHECK:      "summary": "This is a function summary."


fn fn_with_args(
    arg: Int,
    mut inoutArg: Int,
    owned ownedArg: Int,
    read borrowedArg: Int,
):
    """This is a function summary.

    The is some kind of description.

    Args:
        arg: This is an argument.
        inoutArg: This is an mut arg.
        ownedArg: This is an owned arg.
        borrowedArg: This is a borrowedArg.
    """
    return


# CHECK-LABEL:  "name": "fn_with_overload",
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


# CHECK: "kind": "function",
# CHECK: "name": "fn_with_fn_param_and_arg",
# CHECK: "overloads":
# CHECK:     "args":
# CHECK:         "name": "arg_fn"
# CHECK:         "type": "fn(S, S) capturing -> Bool"
# CHECK:     "parameters":
# CHECK:         "name": "T"
# CHECK:         "type": "AnyType"
# CHECK:         "name": "param_fn"
# CHECK:         "type": "fn(T, T) capturing -> Bool"
# CHECK:         "default": "T"
# CHECK:         "name": "S"
# CHECK:         "type": "AnyType"
# CHECK:     "signature": "fn_with_fn_param_and_arg[: origin.set, //, T: AnyType, param_fn: fn(T, T) capturing -> Bool, S: AnyType = T](arg_fn: fn(S, S) capturing -> Bool) -> S"


fn fn_with_fn_param_and_arg[
    T: AnyType,
    param_fn: fn (T, T) capturing [_] -> Bool,
    S: AnyType = T,
](arg_fn: fn (S, S) capturing [_] -> Bool) -> S:
    pass


@value
struct MyStruct[x: Int]:
    pass


# CHECK-LABEL: "name": "fn_with_implicit_params",
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
# CHECK:          "name": "x"
# CHECK:          "passingKind": "pos",
# CHECK:          "type": "String"

# CHECK:          "name": "sep"
# CHECK:          "passingKind": "pos_or_kw",
# CHECK:          "type": "String"
# CHECK:      "signature": "pos_only_print(x: String, /, sep: String)",


fn pos_only_print(x: String, /, sep: String):
    """Prints a String type.

    Args:
        x: The String to print.
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
# CHECK:         "type": "Variadic[ref [ImmutableAnyOrigin] element_type]"

# CHECK:     "signature": "variadic_arg_hack[element_type: AnyType](vals: Variadic[ref [ImmutableAnyOrigin] element_type])",


fn variadic_arg_hack[
    element_type: AnyType
](
    vals: __mlir_type[
        `!kgen.variadic<!lit.ref<`,
        element_type,
        `, #lit.any.origin<0>: !lit.origin<0>, 0>, read_mem>`,
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
# CHECK:         "type": "String"

# CHECK:     "parameters":
# CHECK:         "name": "*nums",
# CHECK:         "passingKind": "pos_or_kw",
# CHECK:         "type": "Int"

# CHECK:     "signature": "variadic_params_args[*nums: Int](*vals: Int, *, owned **kwargs: String)",


fn variadic_params_args[*nums: Int](*vals: Int, **kwargs: String):
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


##===----------------------------------------------------------------------===##
# Ref args and results.
##===----------------------------------------------------------------------===##

# MOTO-516: [doc generation] Print 'ref' arguments and results in docgen

# CHECK-LABEL: "name": "fn_with_anon_refs",
# CHECK: "args": [
# CHECK-NEXT:     {
# CHECK-NEXT:       "convention": "ref",


# CHECK:     "signature": "fn_with_anon_refs(ref ref_arg1: AnyTrivialRegType) -> ref [ref_arg1] AnyTrivialRegType",
fn fn_with_anon_refs(
    ref ref_arg1: AnyTrivialRegType,
) -> ref [ref_arg1] AnyTrivialRegType:
    pass


# CHECK-LABEL: "name": "fn_with_named_refs",
# CHECK:     "signature": "fn_with_named_refs[life: MutableOrigin](ref [life] ref_arg1: AnyTrivialRegType) -> ref [life] AnyTrivialRegType",
fn fn_with_named_refs[
    life: MutableOrigin
](ref [life]ref_arg1: AnyTrivialRegType) -> ref [
    __origin_of(ref_arg1)
] AnyTrivialRegType:
    pass


# MOTO-870: Improve doc gen of struct Origin parameters
# CHECK-LABEL: "name": "fn_with_origins",
# CHECK:     "signature": "fn_with_origins[: Bool, //, o1: Origin[$0], o2: MutableOrigin](ref [o1] arg1: Int, ref [o2] arg2: Int) -> ref [o1] Int",
fn fn_with_origins[
    o1: Origin[_], o2: Origin[True]
](ref [o1]arg1: Int, ref [o2]arg2: Int) -> ref [arg1] Int:
    pass


# MOTO-870: Improve doc gen of struct Origin parameters
# CHECK-LABEL: "name": "fn_with_mult_result_origins",
# CHECK:     "signature": "fn_with_mult_result_origins(ref arg1: Int, ref arg2: Int) -> ref [arg1, arg2] Int",
fn fn_with_mult_result_origins(
    ref arg1: Int, ref arg2: Int
) -> ref [arg1, arg2] Int:
    pass


# CHECK-LABEL: "name": "fn_with_named_result",
# CHECK:     "signature": "fn_with_named_result(a: Int, out res: String)",
fn fn_with_named_result(a: Int, out res: String):
    res = ""


# CHECK: "kind": "function"
# CHECK: "overloads":
# CHECK:    "deprecated": "deprecated function"
# CHECK:    "name": "deprecated_function"
@deprecated("deprecated function")
fn deprecated_function():
    pass


# MOTO-418: Improve AST type printing of `reversed` in API docs
# CHECK-LABEL: "name": "dep_type"
# CHECK: "returnType": "ref [value] UsesParameter[K]",
# CHECK: "signature": "dep_type[K: AnyType](ref value: UsesParameter[K]) -> ref [value] UsesParameter[K]",
struct UsesParameter[A: AnyType]:
    pass


fn dep_type[
    K: AnyType
](ref value: UsesParameter[K]) -> ref [value] UsesParameter[K]:
    return value


# Check that we dump optional default values correctly.
from collections.optional import Optional


# CHECK: "signature": "optional_default_arg_none(input: Optional[SIMD[int64, 1]] = Optional(None))"
fn optional_default_arg_none(input: Optional[Int64] = None):
    pass


# CHECK: "signature": "optional_default_arg_13(input: Optional[SIMD[int64, 1]] = Optional(__init__[__mlir_type.!pop.int_literal](13)))"
fn optional_default_arg_13(input: Optional[Int64] = Int64(13)):
    pass


# ===----------------------------------------------------------------------=== #
# Struct documentation
# ===----------------------------------------------------------------------=== #

# CHECK:  "kind": "module",
# CHECK:  "name": "mojo_doc",

# CHECK-LABEL:  "structs": [


# MOTO-516: [doc generation] Print 'ref' arguments and results in docgen
# CHECK:  "convention": "register_passable_trivial",
@register_passable("trivial")
struct HMyUnsafePointer[
    T: AnyType,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    # CHECK: "signature": "__getitem__(self) -> ref [MutableAnyOrigin, address_space] T",
    fn __getitem__(
        self,
    ) -> ref [MutableAnyOrigin, address_space] T:
        pass

    # CHECK: "signature": "address_of(ref [address_space] arg: T) -> Self",
    @staticmethod
    fn address_of(ref [address_space]arg: T) -> Self:
        pass


# CHECK:  "signature": "struct HMyUnsafePointer[T: AnyType, address_space: AddressSpace = AddressSpace(0)]",


struct HList[T: CollectionElement, hint_trivial_type: Bool = False]:
    # CHECK: "signature": "__getitem__(ref self, idx: Int) -> ref [self] T",
    fn __getitem__(ref self, idx: Int) -> ref [self] T:
        pass


# FIXME(MOTO-692): This should say `T: CollectionElement`.
# CHECK: "signature": "struct HList[T: Copyable & Movable, hint_trivial_type: Bool = False]",


# Check that we don't generate any synthesized thunk methods
# from the trait usage.
# CHECK-NOT: "name": "thunk_

# Check that special functions are ordered first, and with the correct
# prioritization (i.e. not just name based).
# CHECK:  "kind": "function",
# CHECK:  "name": "__init__",
# CHECK:     "signature": "__init__(out self)",
# CHECK:  "signature": "__copyinit__(out self, existing: Self)",
# CHECK:  "name": "__del__",

# CHECK: "name": "__add__",
# CHECK: "overloads":
# CHECK:      "signature": "__add__(self, other: Self) -> Self"

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
# CHECK:      "signature": "fn_with_by_conventions(mut self, mut arg: Self, mut *args: Self) -> Self",
# CHECK:      "summary": "This is a function summary."
# CHECK:  "kind": "struct",
# CHECK:  "name": "InMemoryStruct",
# CHECK:  "parentTraits": [
# CHECK-NEXT:   "AnyType"
# CHECK-NEXT:   "Sized"
# CHECK:  "signature": "struct InMemoryStruct"


struct InMemoryStruct(Sized):
    fn __init__(out self):
        pass

    fn __copyinit__(out self, existing: Self):
        pass

    fn __del__(owned self):
        pass

    fn __add__(self, other: Self) -> Self:
        return other

    fn __len__(self) -> Int:
        return 0

    fn fn_with_by_conventions(
        mut self, mut arg: InMemoryStruct, mut*args: InMemoryStruct
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
# CHECK:  "convention": "register_passable",
# CHECK:  "description": "The is some kind of description.\n",
# CHECK:      "kind": "function",
# CHECK:      "name": "fn_with_self_param",
# CHECK:          "parameters": [
# CHECK:              "description": "This is a Self parameter."
# CHECK:              "name": "param"
# CHECK:              "type": "Self"
# CHECK:          "signature": "fn_with_self_param[param: Self](self)"
# CHECK:  "kind": "struct",
# CHECK:  "name": "ParameterClass",
# CHECK:  "parameters": [
# CHECK:      "description": "This is a parameter."
# CHECK:      "name": "_type"
# CHECK:      "type": "dtype"
# CHECK:  ],
# CHECK:  "signature": "struct ParameterClass[_type: dtype]",
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


# CHECK:  "kind": "struct",
# CHECK:  "name": "StructWithFnParam",
# CHECK:  "parameters":
# CHECK:      "name": "T"
# CHECK:      "type": "AnyType"
# CHECK:      "name": "param_fn"
# CHECK:      "type": "fn(T, T) capturing -> Bool"
# CHECK:      "default": "T",
# CHECK:      "name": "S"
# CHECK:      "type": "AnyType"
# CHECK: "signature": "struct StructWithFnParam[__origins__: origin.set, //, T: AnyType, param_fn: fn(T, T) capturing -> Bool, S: AnyType = T]",


struct StructWithFnParam[
    T: AnyType,
    param_fn: fn (T, T) capturing [_] -> Bool,
    S: AnyType = T,
]:
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

    fn f(self):
        """This is a trait function doc."""
        ...


# CHECK: "deprecated": "deprecated trait"
# CHECK: "name": "DeprecatedTrait"
@deprecated("deprecated trait")
trait DeprecatedTrait:
    pass


# Check that we include version information in the generated JSON.
# CHECK: "version":
