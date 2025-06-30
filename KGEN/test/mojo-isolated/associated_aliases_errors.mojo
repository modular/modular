# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -split-input-file -verify-diagnostics %s


trait Copyable:
    fn __copyinit__(out self, other: Self):
        ...


trait Movable:
    fn __moveinit__(out self, owned other: Self):
        ...


trait ExplicitlyCopyable:
    fn copy(self) -> Self:
        ...


# expected-error @below {{only traits may contain an alias without an initializer}}
alias K: Int


struct Int(Copyable, Movable):
    fn __init__(out self):
        pass


struct Bool(Copyable, Movable):
    fn __init__(out self):
        pass


trait MyTrait:  # expected-note {{trait 'MyTrait' declared here}}
    # expected-note @below {{required alias 'N' is not specified}}
    # expected-note @below {{alias 'N' type 'Bool' doesn't conform to trait's alias 'N' type 'Int'}}
    alias N: Int


trait TraitWithInitializedAlias:
    # expected-error @below {{associated alias declarations in a trait shouldn't have an initializer}}
    alias Z: Int = Int()


# expected-error @below {{'StructConformingExplicitlyWithNoMatchingAlias' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithNoMatchingAlias(MyTrait):
    pass


# expected-error @below {{'StructConformingExplicitlyWithMismatchedAlias' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithMismatchedAlias(MyTrait):
    alias N: Bool = Bool()


# expected-error @below {{'StructConformingExplicitlyWithMemberSameName' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithMemberSameName(MyTrait):
    var N: Int


@fieldwise_init
struct StructWithNoMatchingAlias:
    pass


@fieldwise_init
struct StructWithMismatchedAlias:
    alias N: Bool = Bool()


struct StructWithUninitializedAlias:
    # expected-error @below {{only traits may contain an alias without an initializer}}
    alias N: Bool


struct StructWithTypelessUninitializedAlias:
    # This makes sure we print out this error, rather than the also-relevant "alias without initial value must have a type" error
    # expected-error @below {{only traits may contain an alias without an initializer}}
    alias N


# expected-note @below {{function declared here}}
fn funcForMyTrait[T: MyTrait](t: T) -> Int:
    alias X = T.N
    return X


fn testError1():
    # TODO(MOCO-1152): Add more detailed errors for this
    # expected-error @below {{invalid call to 'funcForMyTrait': could not deduce parameter 'T' of callee 'funcForMyTrait'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'StructWithNoMatchingAlias' does not conform to trait 'MyTrait'}}
    var whatev: Int = funcForMyTrait(StructWithNoMatchingAlias())


fn testError2():
    # TODO(MOCO-1152): Add more detailed errors for this
    # expected-error @below {{invalid call to 'funcForMyTrait': could not deduce parameter 'T' of callee 'funcForMyTrait'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'StructWithMismatchedAlias' does not conform to trait 'MyTrait'}}
    var whatev: Int = funcForMyTrait(StructWithMismatchedAlias())


# // -----


struct Int:
    pass


struct TensorIndex[rank: Int]:
    pass


trait Stencil:
    alias rank: Int
    # expected-error @below {{associated alias declarations in a trait shouldn't have an initializer}}
    alias Index = TensorIndex[rank]


# // -----

# Tests that we get a nice error when an override alias has an incompatible
# type.


struct ZInt:
    pass


struct ZBool:
    pass


trait TraitWithTypeAlias:
    # expected-note @below {{parent trait's alias defined here}}
    alias T: ZBool


trait TraitWithSameTypeAlias(TraitWithTypeAlias):
    # expected-error @below {{invalid redefinition of 'T': cannot convert 'ZInt' to parent trait's alias's type 'ZBool'}}
    alias T: ZInt


# // -----

# Makes sure that we don't crash if there are multiple overrides.


struct ZInt:
    pass


struct ZBool:
    pass


struct ZFloat:
    pass


trait SuperTrait:
    alias T: ZFloat


trait TraitWithTooManyAliases(SuperTrait):
    # expected-note @below {{previous definition here}}
    alias T: ZBool
    # expected-error @below {{invalid redefinition of 'T'}}
    alias T: ZInt
