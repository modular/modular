# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


# COM: Just check that conformance checking succeeds.
trait TraitForReg(Copyable):
    @implicit
    def __init__(out self, x: Int):
        ...

    @staticmethod
    def may_throw() raises -> Self:
        ...


struct RegTypeTrivial(TraitForReg, TrivialRegisterPassable):
    @implicit
    def __init__(out self, x: Int):
        pass

    @staticmethod
    def may_throw() raises -> Self:
        pass


trait AsyncTrait:
    async def foo(self) -> Int:
        ...

    async def bar(self) raises -> Int:
        ...


struct AsyncStruct(AsyncTrait):
    async def foo(self) -> Int:
        pass

    async def bar(self) raises -> Int:
        pass


# CHECK-LABEL: lit.struct.decl @AsyncStructReg
struct AsyncStructReg(AsyncTrait, TrivialRegisterPassable):
    async def foo(self) -> Int:
        pass

    async def bar(self) raises -> Int:
        pass


trait Explicit:
    def __int__(self) -> Int:
        ...


trait Implicit:
    def __as_int__(self) -> Int:
        ...


@fieldwise_init
struct Foo(Explicit, Implicit):
    def __int__(self) -> Int:
        return 42

    def __as_int__(self) -> Int:
        return 42


struct Bar:
    @implicit
    def __init__[T: Implicit](out self, value: T):
        pass

    def __init__[T: Explicit](out self, value: T):
        pass


# CHECK-LABEL: lit.fn @"construct_implicit_type_explicitly
def construct_implicit_type_explicitly():
    _ = Bar(Foo())


# CHECK-LABEL: lit.fn @"async_trait
def async_trait[T: AsyncTrait](value: T):
    # CHECK: lit.async.call[!lit.generator<[2]("self": {{.*}} read_mem, ?, "__result__": !lit.ref<!Int, mut *[0,1]> byref_result) async -> !kgen.none>: #kgen.get_witness
    _ = value.foo()


def take_intable[T: Intable](x: T):
    pass


# CHECK-LABEL: lit.fn @"nonmaterializable_trait
def nonmaterializable_trait():
    # CHECK-NEXT: [[SLOT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!Int,
    # CHECK-NEXT: [[VAL:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: store [[VAL]], [[SLOT]]
    # CHECK-NEXT:  = lit.ref.immut [[SLOT]]
    # CHECK-NEXT: call {{.*}}take_intable{{.*}}<:!Intable !Int
    take_intable(1)
