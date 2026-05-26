# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: %parse-mojo-isolated -verify-diagnostics %s


@fieldwise_init
struct SomeNonCopyable:
    pass


@fieldwise_init
struct SomeCopyable(Copyable):
    pass


@fieldwise_init
struct SomeVA[*elt_types: AnyType]:
    pass


# @expected-note @below{{function declared here}}
def all_copyable[*elt_type: Copyable](t: SomeVA[*elt_type]):
    pass


# @expected-note @below{{function declared here}}
def all_int[*elt_type: type_of(Int)](t: SomeVA[*elt_type]):
    pass


def foo():
    # @expected-error @below{{invalid call to 'all_copyable': value passed to 't' cannot be converted from 'SomeVA[SomeCopyable, SomeNonCopyable]' to 'SomeVA[*elt_type.values]'}}
    # @expected-note @below{{.elt_types.values` of the first value is 'SomeCopyable, SomeNonCopyable' but the second value is 'elt_type.values'}}
    all_copyable(SomeVA[SomeCopyable, SomeNonCopyable]())

    # @expected-error @below{{invalid call to 'all_int': value passed to 't' cannot be converted from 'SomeVA[Int, SomeNonCopyable]' to 'SomeVA[*elt_type.values]'}}
    all_int(SomeVA[Int, SomeNonCopyable]())


# expected-note @below {{'ParamSubst' declared here}}
struct ParamSubst[
    T: TrivialRegisterPassable,
    shape: __mlir_type[`!kgen.param_list<`, T, `>`],
]:
    pass


# expected-note @below {{function declared here}}
def more_refined_variadic[*elt_type: Copyable]():
    pass


# TODO: better error message: print out the constraint failure reason.
def less_refined_variadic[*elt_type: AnyType]():
    # expected-error @below {{invalid call to 'more_refined_variadic': value passed to 'elt_type' cannot be converted from 'TypeList[values]' to 'TypeList[elt_type.values]'}}
    more_refined_variadic[*elt_type]()
    pass


def main():
    # We do not handle conversion between variadic of values at the moment (maybe we should?).
    # expected-error @below {{'ParamSubst' parameter 'shape' has 'KGENParamList[Int]' type, but value has type 'KGENParamList[__mlir_type.index]'}}
    var _: ParamSubst[
        Int,
        __mlir_attr.`#kgen.param_list<1, 2> : !kgen.param_list<index>`,
    ]
