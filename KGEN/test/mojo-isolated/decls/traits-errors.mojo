# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


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
    # expected-error @below {{parametric type 'ParamType[?]' cannot bind to trait with missing parameters}}
    alias Bound: FooTrait = ParamType


fn different_trait_types[T: Copyable, U: Copyable](x: T) -> U:
    # expected-error @below {{cannot implicitly convert 'T' value to 'U' in return value}}
    return x


trait SimpleTrait:
    fn some_method(self):
        pass


struct TraitStruct(SimpleTrait):
    fn some_method(self):
        pass


# expected-note @+1 {{function declared here}}
fn test_many_things_of_specified_trait[
    element_type: __mlir_type[`!lit.anytrait<`, AnyType, `>`],
    *element_types: element_type,
]():
    pass


fn call_many_things_of_specified_trait(a: TraitStruct):
    # This is ok!
    test_many_things_of_specified_trait[AnyType, TraitStruct, Int]()

    # expected-error @+1 {{parameter #1 has 'Movable' type, but value has type 'AnyStruct[TraitStruct]'}}
    test_many_things_of_specified_trait[Movable, TraitStruct, Int]()

    # expected-error @+1 {{parameter #1 has 'SimpleTrait' type, but value has type 'AnyStruct[Int]'}}
    test_many_things_of_specified_trait[SimpleTrait, TraitStruct, Int]()
