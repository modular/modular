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
fn all_copyable[*elt_type: Copyable](t: SomeVA[*elt_type]):
    pass


# @expected-note @below{{function declared here}}
fn all_int[*elt_type: type_of(Int)](t: SomeVA[*elt_type]):
    pass


fn foo():
    # @expected-error @below{{invalid call to 'all_copyable': value passed to 't' cannot be converted from 'SomeVA[SomeCopyable, SomeNonCopyable]' to 'SomeVA[elt_type]'}}
    # @expected-note @below{{.elt_types of left value is 'SomeCopyable, SomeNonCopyable' but the right value is 'elt_type'}}
    all_copyable(SomeVA[SomeCopyable, SomeNonCopyable]())

    # @expected-error @below{{invalid call to 'all_int': value passed to 't' cannot be converted from 'SomeVA[Int, SomeNonCopyable]' to 'SomeVA[elt_type]'}}
    all_int(SomeVA[Int, SomeNonCopyable]())


struct ParamSubst[
    T: __TypeOfAllTypes,
    shape: __mlir_type[`!kgen.variadic<`, T, `>`],
]:
    pass


fn main():
    # We do not handle conversion between variadic of values at the moment (maybe we should?).
    var _: ParamSubst[
        Int,
        # @expected-error @below{{can not convert 'Variadic[__mlir_type.index]' to 'Variadic[Int]'}}
        __mlir_attr.`#kgen.variadic<1, 2> : !kgen.variadic<index>`,
    ]
