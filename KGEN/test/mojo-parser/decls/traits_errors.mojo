# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -verify-diagnostics


# expected-note @below {{trait 'MyMovable' declared here}}
trait MyMovable:
    # expected-note @below {{required function '__init__' is not implemented}}
    def __init__(out self, *, deinit take: Self):
        ...


trait ErroneousTrait:
    # expected-error @+1 {{'self' argument must have type 'Self' in trait method declaration, but actually has type 'Int'}}
    def add(self: Int):
        ...


trait FooTrait:
    def foo(self):
        ...


# expected-error @below {{'where' clauses in conformance lists are only supported on structs}}
trait WhereClauseInTraitConformanceList(FooTrait where True):
    pass


struct ParamType[x: Int](FooTrait):
    def foo(self):
        pass


def invalid_trait_bind():
    # expected-error @below {{'ParamType[_]' is not concrete, use '[]' to bind missing parameters}}
    comptime Bound: FooTrait = ParamType


def different_trait_types[
    T: ImplicitlyCopyable, U: ImplicitlyCopyable
](x: T) -> U:
    # expected-error @below {{cannot implicitly convert 'T' value to 'U'}}
    return x


# expected-note @below {{trait 'SimpleTrait' declared here}}
trait SimpleTrait:
    # expected-note @below {{required function 'some_method' is not implemented}}
    # expected-note @below {{no 'some_method' candidates have type 'def(self: ParamDoesNotConform[x]) -> None'}}
    def some_method(self):
        ...


# expected-error @below {{'TraitStruct' does not implement all requirements for 'MyMovable'}}
struct TraitStruct(MyMovable, SimpleTrait):
    def some_method(self):
        pass


# expected-note @+1 {{function declared here}}
def test_many_things_of_specified_trait[
    element_type: type_of(AnyType), //, *element_types: element_type
]():
    pass


# expected-error @below {{'DoesNotConform' does not implement all requirements for 'SimpleTrait'}}
struct DoesNotConform(SimpleTrait):
    pass


# expected-error @below {{'ParamDoesNotConform[x]' does not implement all requirements for 'SimpleTrait'}}
struct ParamDoesNotConform[x: Int](SimpleTrait):
    # expected-note @below {{candidate declared here with type 'def(self: ParamDoesNotConform[x], y: Int) -> None' (specialized from 'def[x: Int, //](self: ParamDoesNotConform[x], y: Int) -> None')}}
    def some_method(self, y: Int):
        pass


def call_many_things_of_specified_trait(a: TraitStruct):
    # This is ok!
    test_many_things_of_specified_trait[element_type=AnyType, TraitStruct, Int]()

    # expected-error @+1 {{'test_many_things_of_specified_trait' parameter 'element_types' has 'Movable' type, but value has type 'AnyStruct[TraitStruct]'}}
    test_many_things_of_specified_trait[element_type=Movable, TraitStruct, TraitStruct]()

    test_many_things_of_specified_trait[
        element_type=SimpleTrait,
        TraitStruct,
        # This will succeed, the error will be raised when resolving `DoesNotConform`.
        DoesNotConform,
    ]()


# expected-note@+1 {{trait 'TrivialTrait' declared here}}
trait TrivialTrait(TrivialRegisterPassable):
    # expected-note@+1 {{required function 'doSomething' is not implemented}}
    def doSomething(self):
        ...


# expected-note@+1 {{inherited through 'MemTraitViolation' here}}
trait MemTraitViolation(TrivialTrait):
    def bar(self):
        ...


trait NonTrivialRGTrait(RegisterPassable):
    def bar(self):
        ...


# expected-error @+1 {{does not implement all requirements for}}
struct StructViolation2(TrivialTrait):
    pass


# expected-error @+1 {{does not implement all requirements for}}
struct StructViolation3(MemTraitViolation):
    def bar(self):
        pass


@explicit_destroy
trait TFoo:
    def foo(self):
        ...


@fieldwise_init
struct Bar[T: TFoo]:  # expected-note {{'Bar' declared here}}
    pass


def bindAnyTraitToTrait():
    # expected-error @+1 {{'Bar' parameter 'T' has 'TFoo' type, but value has type 'AnyTrait[TFoo]'}}
    var _list = Bar[TFoo]()


def anytrait_assignment():
    # expected-error @below {{cannot implicitly convert 'AnyTrait[ImplicitlyCopyable]' value to 'AnyTrait[FooTrait]' in comptime initializer}}
    comptime t: type_of(FooTrait) = ImplicitlyCopyable


trait SomeTrait:
    comptime A: Int


@fieldwise_init
struct TakeInt[A: Int]:
    pass


# expected-note @below {{function declared here}}
def take_two_inferred_params[Size: Int](x: TakeInt[Size], y: TakeInt[Size]):
    pass


def call_take_two_inferred_params[T: SomeTrait](x: T):
    # expected-error @below {{invalid call to 'take_two_inferred_params': value passed to 'y' cannot be converted from 'TakeInt[1]' to 'TakeInt[T.A]'}}
    take_two_inferred_params(TakeInt[T.A](), TakeInt[1]())


# Check that a trait method with a default implementation returning a non-None
# type may not use 'pass'.
trait TBar:
    # expected-error @+4 {{trait method with a return type must not use 'pass'; use '...' to declare the method as required}}
    # expected-note @below {{in 'bar', declared here}}
    # expected-note @below {{original default implementation from trait 'TBar' here}}
    def bar(self) -> Int:
        pass


trait TBarSub(TBar):
    # expected-note @below {{conflicting implementation from trait 'TBarSub' here}}
    def bar(self) -> Int:
        return 0


# expected-error @+1 {{trait method requirement 'bar' has conflicting default implementations in 'TBar' and 'TBarSub' you must implement it manually}}
struct TBarActual(TBarSub):
    pass


trait WhereClauseOnTraitMethod:
    # expected-error @+1 {{'where' clauses on trait methods are not supported}}
    def guarded_method(self) where Self.x > 10:
        pass


trait ConflictTraitName:
    # expected-note @+1 {{trait method declared here}}
    def test[a: Int](self):
        pass


# expected-error @+1 {{name conflict between parameter 'a' in the default trait method and a parameter in the struct}}
struct ConflictStruct[a: Int](ConflictTraitName):
    pass
