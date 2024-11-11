# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @below {{trait 'Movable' declared here}}
trait Movable:
    # expected-note @below {{required function '__moveinit__' is not implemented}}
    fn __moveinit__(out self, owned existing: Self, /):
        pass


trait ErroneousTrait:
    # expected-error @+1 {{'self' argument must have type 'Self' in trait method declaration, but actually has type 'index'}}
    fn add(self: int):
        ...


trait FooTrait:
    fn foo(self):
        ...


struct ParamType[x: int](FooTrait):
    fn foo(self):
        pass


fn invalid_trait_bind():
    # expected-error @below {{'ParamType' missing required parameter 'x'}}
    alias Bound: FooTrait = ParamType


fn different_trait_types[T: Copyable, U: Copyable](x: T) -> U:
    # expected-error @below {{cannot implicitly convert 'T' value to 'U' in return value}}
    return x


# expected-note @below {{trait 'SimpleTrait' declared here}}
trait SimpleTrait:
    # expected-note @below {{required function 'some_method' is not implemented}}
    fn some_method(self):
        pass


# expected-note @below {{struct 'TraitStruct' does not implement all requirements for 'Movable'}}
struct TraitStruct(SimpleTrait):
    fn some_method(self):
        pass


fn test_many_things_of_specified_trait[
    element_type: __mlir_type[`!lit.anytrait<`, AnyType, `>`],
    *element_types: element_type,
]():
    pass


# expected-note @below {{'DoesNotConform' does not implement all requirements for 'SimpleTrait'}}
struct DoesNotConform:
    pass


fn call_many_things_of_specified_trait(a: TraitStruct):
    # This is ok!
    test_many_things_of_specified_trait[AnyType, TraitStruct, Int]()

    # expected-error @+1 {{cannot bind type 'TraitStruct' to trait 'Movable'}}
    test_many_things_of_specified_trait[Movable, TraitStruct, Int]()

    test_many_things_of_specified_trait[
        SimpleTrait,
        TraitStruct,
        # expected-error @+1 {{cannot bind type 'DoesNotConform' to trait 'SimpleTrait'}}
        DoesNotConform,
    ]()
