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
    fn add(self: Index):
        ...


trait FooTrait:
    fn foo(self):
        ...


struct ParamType[x: Index](FooTrait):
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


fn test_many_things_of_specified_trait[element_type: __type_of(AnyType),
                                       *element_types: element_type]():
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

@register_passable("trivial")
trait TrivialTrait:
    fn doSomething(self):
        ...

trait MemTraitViolation(TrivialTrait):
    fn bar(self):
        ...

@register_passable
trait NonTrivialRGTrait:
    fn bar(self):
        ...

# expected-error @+1 {{a struct must be register passable in order to inherit from a register passable trait}}
struct StructViolation1(NonTrivialRGTrait):
    pass

# expected-error @+1 {{a struct must be register passable in order to inherit from a register passable trait}}
struct StructViolation2(TrivialTrait):
    pass

# expected-error @+1 {{a struct must be register passable in order to inherit from a register passable trait}}
struct StructViolation3(MemTraitViolation):
    fn bar(self):
        pass

@explicit_destroy
trait TFoo():
    # expected-note @+1 {{candidate declared here with type 'fn[TFoo](self: $0) -> None'}}
    fn foo(self):
        ...

@value
struct Bar[T:TFoo]:
    pass

fn bindAnyTraitToTrait():
    # COM: binding the trait type to the parameter T triggers the building of a parameter with a vtable.
    # This vtable is built from the constraints on T. In this case, the constraint is "fn foo(self):"
    # But the trait TFoo is not an implementation of itself so the synthesis fails.
    # expected-error @+1 {{no 'foo' candidates have type 'fn(self: TFoo) -> None'}}
    var _list = Bar[TFoo]()
