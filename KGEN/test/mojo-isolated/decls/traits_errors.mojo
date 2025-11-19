# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @below {{trait 'MyMovable' declared here}}
trait MyMovable:
    # expected-note @below {{required function '__moveinit__' is not implemented}}
    fn __moveinit__(out self, deinit existing: Self, /):
        ...


trait ErroneousTrait:
    # expected-error @+1 {{'self' argument must have type 'Self' in trait method declaration, but actually has type 'Int'}}
    fn add(self: Int):
        ...


trait FooTrait:
    fn foo(self):
        ...


struct ParamType[x: Int](FooTrait):
    fn foo(self):
        pass


fn invalid_trait_bind():
    # expected-error @below {{'ParamType' missing required parameter 'x'}}
    comptime Bound: FooTrait = ParamType


fn different_trait_types[
    T: ImplicitlyCopyable, U: ImplicitlyCopyable
](x: T) -> U:
    # expected-error @below {{cannot implicitly convert 'T' value to 'U' in return value}}
    return x


# expected-note @below {{trait 'SimpleTrait' declared here}}
trait SimpleTrait:
    # expected-note @below {{required function 'some_method' is not implemented}}
    # expected-note @below {{no 'some_method' candidates have type 'fn(self: ParamDoesNotConform[x]) -> None'}}
    fn some_method(self):
        ...


# expected-error @below {{'TraitStruct' does not implement all requirements for 'MyMovable'}}
struct TraitStruct(MyMovable, SimpleTrait):
    fn some_method(self):
        pass


fn test_many_things_of_specified_trait[
    element_type: type_of(AnyType), *element_types: element_type
]():
    pass


# expected-error @below {{'DoesNotConform' does not implement all requirements for 'SimpleTrait'}}
struct DoesNotConform(SimpleTrait):
    pass


# expected-error @below {{'ParamDoesNotConform[x]' does not implement all requirements for 'SimpleTrait'}}
struct ParamDoesNotConform[x: Int](SimpleTrait):
    # expected-note @below {{candidate declared here with type 'fn(self: ParamDoesNotConform[x], y: Int) -> None' (specialized from 'fn[x: Int, /](self: ParamDoesNotConform[x], y: Int) -> None')}}
    fn some_method(self, y: Int):
        pass


fn call_many_things_of_specified_trait(a: TraitStruct):
    # This is ok!
    test_many_things_of_specified_trait[AnyType, TraitStruct, Int]()

    # expected-error @+1 {{cannot bind type 'TraitStruct' to trait 'Movable'}}
    test_many_things_of_specified_trait[Movable, TraitStruct, TraitStruct]()

    test_many_things_of_specified_trait[
        SimpleTrait,
        TraitStruct,
        # This will succeed, the error will be raised when resolving `DoesNotConform`.
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
trait TFoo:
    fn foo(self):
        ...


@fieldwise_init
struct Bar[T: TFoo]:
    pass


fn bindAnyTraitToTrait():
    # expected-error @+1 {{cannot implicitly convert 'TFoo' type as a value to an instance of 'TFoo' in type parameter; did you mean to instantiate 'TFoo'?}}
    var _list = Bar[TFoo]()


fn anytrait_assignment():
    # expected-error @below {{cannot implicitly convert 'AnyTrait[ImplicitlyCopyable]' value to 'AnyTrait[Movable]' in alias initializer}}
    comptime t: type_of(Movable) = ImplicitlyCopyable


trait SomeTrait:
    comptime A: Int


@fieldwise_init
struct TakeInt[A: Int]:
    pass


# expected-note @below {{function declared here}}
fn take_two_inferred_params[Size: Int](x: TakeInt[Size], y: TakeInt[Size]):
    pass


fn call_take_two_inferred_params[T: SomeTrait](x: T):
    # expected-error @below {{invalid call to 'take_two_inferred_params': failed to infer parameter 'Size', it inferred to two different values: 'T.A' and '1'}}
    # expected-note @below {{try `rebind` them to one type if they will be concretized to the same type}}
    take_two_inferred_params(TakeInt[T.A](), TakeInt[1]())


# Check that a trait method with a default implementation returning a non-None
# type may not use 'pass'.
trait TBar:
    # expected-error @+4 {{trait method has results but default implementation returns no value; did you mean '...'?}}
    # expected-note @below {{in 'bar', declared here}}
    # expected-note @below {{original default implementation from trait 'TBar' here}}
    fn bar(self) -> Int:
        pass

trait TBarSub(TBar):
    # expected-note @below {{conflicting implementation from trait 'TBarSub' here}}
    fn bar(self) -> Int:
        return 0

# expected-error @+1 {{trait method requirement 'bar' has conflicting default implementations in 'TBar' and 'TBarSub' you must implement it manually}}
struct TBarActual(TBarSub):
    pass
