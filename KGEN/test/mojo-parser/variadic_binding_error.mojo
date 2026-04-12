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
def all_copyable[*elt_type: Copyable](t: SomeVA[*elt_type.upcast[AnyType]()]):
    pass


# @expected-note @below{{function declared here}}
def all_int[*elt_type: type_of(Int)](t: SomeVA[*elt_type.upcast[AnyType]()]):
    pass


def foo():
    # @expected-error @below{{invalid call to 'all_copyable': value passed to 't' cannot be converted from 'SomeVA[SomeCopyable, SomeNonCopyable]' to 'SomeVA[*elt_type.values]'}}
    # @expected-note @below{{.elt_types.values` of left value is 'SomeCopyable, SomeNonCopyable' but the right value is 'elt_type.values'}}
    all_copyable(SomeVA[SomeCopyable, SomeNonCopyable]())

    # @expected-error @below{{invalid call to 'all_int': value passed to 't' cannot be converted from 'SomeVA[Int, SomeNonCopyable]' to 'SomeVA[*elt_type.values]'}}
    all_int(SomeVA[Int, SomeNonCopyable]())


# expected-note @below {{'ParamSubst' declared here}}
struct ParamSubst[
    T: TrivialRegisterPassable,
    shape: __mlir_type[`!kgen.param_list<`, T, `>`],
]:
    pass


def main():
    # We do not handle conversion between variadic of values at the moment (maybe we should?).
    # expected-error @below {{'ParamSubst' parameter 'shape' has 'KGENParamList[Int]' type, but value has type 'KGENParamList[__mlir_type.index]'}}
    var _: ParamSubst[
        Int,
        __mlir_attr.`#kgen.param_list<1, 2> : !kgen.param_list<index>`,
    ]
