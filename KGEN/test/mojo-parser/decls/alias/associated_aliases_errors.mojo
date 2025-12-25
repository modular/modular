# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics %s

# expected-error @below {{only traits may contain a comptime member without an initializer}}
comptime K: Int

trait MyTrait:  # expected-note {{trait 'MyTrait' declared here}}
    # expected-note @below {{required member 'N' is not specified}}
    # expected-note @below {{comptime member 'N' type 'Bool' does not conform to trait's required type 'Int'}}
    comptime N: Int


# expected-error @below {{'StructConformingExplicitlyWithNoMatchingAlias' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithNoMatchingAlias(MyTrait):
    pass


# expected-error @below {{'StructConformingExplicitlyWithMismatchedAlias' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithMismatchedAlias(MyTrait):
    comptime N: Bool = Bool()


# expected-error @below {{'StructConformingExplicitlyWithMemberSameName' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithMemberSameName(MyTrait):
    var N: Int


@fieldwise_init
struct StructWithNoMatchingAlias:
    pass


@fieldwise_init
struct StructWithMismatchedAlias:
    comptime N: Bool = Bool()


struct StructWithUninitializedAlias:
    # expected-error @below {{only traits may contain a comptime member without an initializer}}
    comptime N: Bool


struct StructWithTypelessUninitializedAlias:
    # This makes sure we print out this error, rather than the also-relevant "alias without initial value must have a type" error
    # expected-error @below {{expected '=' after comptime declaration}}
    comptime N


# expected-note @below {{function declared here}}
fn funcForMyTrait[T: MyTrait](t: T) -> Int:
    comptime X = T.N
    return X


fn testError1():
    # TODO(MOCO-1152): Add more detailed errors for this
    # expected-error @below {{invalid call to 'funcForMyTrait': failed to infer parameter 'T', argument type 'StructWithNoMatchingAlias' does not conform to trait 'MyTrait'}}
    var whatev: Int = funcForMyTrait(StructWithNoMatchingAlias())


fn testError2():
    # TODO(MOCO-1152): Add more detailed errors for this
    # expected-error @below {{invalid call to 'funcForMyTrait': failed to infer parameter 'T', argument type 'StructWithMismatchedAlias' does not conform to trait 'MyTrait'}}
    var whatev: Int = funcForMyTrait(StructWithMismatchedAlias())


# // -----


struct TensorIndex[rank: Int]:
    pass


trait Stencil:
    comptime rank: Int


# // -----

# Tests that we get a nice error when an override alias has an incompatible
# type.


struct ZInt:
    pass


struct ZBool:
    pass


trait TraitWithTypeAlias:
    # expected-note @below {{parent trait's member defined here}}
    comptime T: ZBool


trait TraitWithSameTypeAlias(TraitWithTypeAlias):
    # expected-error @below {{invalid redefinition of 'T': cannot convert 'ZInt' to parent trait's member's type 'ZBool'}}
    comptime T: ZInt


# // -----

# Makes sure that we don't crash if there are multiple overrides.


struct ZInt:
    pass


struct ZBool:
    pass


struct ZFloat:
    pass


trait SuperTrait:
    comptime T: ZFloat


trait TraitWithTooManyAliases(SuperTrait):
    # expected-note @below {{previous definition here}}
    comptime T: ZBool
    # expected-error @below {{invalid redefinition of 'T'}}
    comptime T: ZInt
