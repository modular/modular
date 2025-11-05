# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -split-input-file -verify-diagnostics %s

trait Movable:
    fn __moveinit__(out self, deinit other: Self):
        ...


trait Copyable:
    fn __copyinit__(out self, other: Self):
        ...

    fn copy(self) -> Self:
        return Self.__copyinit__(self)


trait ImplicitlyCopyable(Copyable):
    pass


# expected-error @below {{only traits may contain an alias without an initializer}}
alias K: Int


struct Int(ImplicitlyCopyable, Movable):
    fn __init__(out self):
        pass


struct Bool(ImplicitlyCopyable, Movable):
    fn __init__(out self):
        pass

    # This is needed by the compiler to synthesize trivial bit.
    @implicit
    fn __init__(out self, value: __mlir_type.i1):
        pass

trait MyTrait:  # expected-note {{trait 'MyTrait' declared here}}
    # expected-note @below {{required alias 'N' is not specified}}
    # expected-note @below {{alias 'N' type 'Bool' doesn't conform to trait's alias 'N' type 'Int'}}
    alias N: Int


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
    # expected-error @below {{expected '=' after alias targets}}
    alias N


# expected-note @below {{function declared here}}
fn funcForMyTrait[T: MyTrait](t: T) -> Int:
    alias X = T.N
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


struct Int:
    pass


struct TensorIndex[rank: Int]:
    pass


trait Stencil:
    alias rank: Int


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
