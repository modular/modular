# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo --kgen-print-inline-type-values | FileCheck %s


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    @implicit
    fn __init__(out self, x: Int):
        ...

    fn __copyinit__(out self, existing: Self):
        ...

    @staticmethod
    fn may_throw() raises -> Self:
        ...


@register_passable("trivial")
struct RegTypeTrivial(TraitForReg):
    @implicit
    fn __init__(out self, x: Int):
        pass

    @staticmethod
    fn may_throw() raises -> Self:
        pass


trait AsyncTrait:
    async fn foo(self) -> Int:
        ...

    async fn bar(self) raises -> Int:
        ...


struct AsyncStruct(AsyncTrait):
    async fn foo(self) -> Int:
        pass

    async fn bar(self) raises -> Int:
        pass


# CHECK-LABEL: lit.struct.decl @AsyncStructReg
@register_passable("trivial")
struct AsyncStructReg(AsyncTrait):
    async fn foo(self) -> Int:
        pass

    async fn bar(self) raises -> Int:
        pass


trait Explicit:
    fn __int__(self) -> Int:
        ...


trait Implicit:
    fn __as_int__(self) -> Int:
        ...


@fieldwise_init
struct Foo(Explicit, Implicit):
    fn __int__(self) -> Int:
        return 42

    fn __as_int__(self) -> Int:
        return 42


struct Bar:
    @implicit
    fn __init__[T: Implicit](out self, value: T):
        pass

    fn __init__[T: Explicit](out self, value: T):
        pass


# CHECK-LABEL: lit.fn @"construct_implicit_type_explicitly
fn construct_implicit_type_explicitly():
    _ = Bar(Foo())


# CHECK-LABEL: lit.fn @"async_trait
fn async_trait[T: AsyncTrait](value: T):
    # CHECK: lit.async.call[!lit.generator<[2]("self": {{.*}} read_mem, ?, "__result__": !lit.ref<!Int, mut *[0,1]> byref_result) async -> !kgen.none>: #kgen.get_witness
    _ = value.foo()


fn take_intable[T: Intable](x: T):
    pass


# CHECK-LABEL: lit.fn @"nonmaterializable_trait
fn nonmaterializable_trait():
    # CHECK-NEXT: [[SLOT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!Int,
    # CHECK-NEXT: [[VAL:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: store [[VAL]], [[SLOT]]
    # CHECK-NEXT:  = lit.ref.immut [[SLOT]]
    # CHECK-NEXT: call {{.*}}take_intable{{.*}}<:!Intable !Int
    take_intable(1)
