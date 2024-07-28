# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo --kgen-print-inline-type-values | FileCheck %s


# COM: Just check that conformance checking succeeds.
trait TraitForReg:
    fn __init__(inout self, x: Int):
        ...

    fn __copyinit__(inout self, existing: Self):
        ...

    @staticmethod
    fn may_throw() raises -> Self:
        ...


@register_passable("trivial")
struct RegTypeTrivial(TraitForReg):
    fn __init__(inout self, x: Int):
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
    # CHECK-LABEL: lit.func @"foo{{.*}}_thunk"{{.*}}(%self: !lit.ref<!AsyncStructReg, imm {{.*}}> borrow_in_mem, ?, %__result__: !lit.ref<!Int,
    async fn foo(self) -> Int:
        # CHECK: [[POP_CORO:%.*]] = lit.async.call
        # CHECK: lit.call {{.*}}__init__{{.*}}([[CORO_TMP:%.*]], [[POP_CORO]])
        # CHECK-NEXT: [[CORO:%.*]] = lit.load.consume [[CORO_TMP]]
        # CHECK-NEXT: lit.call {{.*}}@Coroutine::@"__await__{{.*}}([[CORO]], %__result__)
        # CHECK-NEXT: %none =
        # CHECK-NEXT: return %none
        pass

    # CHECK-LABEL: lit.func @"bar{{.*}}_thunk"{{.*}}(%self: !lit.ref<!AsyncStructReg, imm {{.*}}> borrow_in_mem, ?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<!Int,
    async fn bar(self) raises -> Int:
        # CHECK: [[POP_CORO:%.*]] = lit.async.call
        # CHECK: lit.call {{.*}}__init__{{.*}}([[CORO_TMP:%.*]], [[POP_CORO]])
        # CHECK-NEXT: [[CORO:%.*]] = lit.load.consume [[CORO_TMP]]
        # CHECK-NEXT: lit.call {{.*}}@RaisingCoroutine::@"__await__{{.*}}([[CORO]], %__error__, %__result__)
        # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
        # CHECK-NEXT: return [[FALSE]]
        pass


# CHECK-LABEL: lit.func @"async_trait
fn async_trait[T: AsyncTrait](value: T):
    # CHECK: lit.async.call[!lit.signature<[2]("self": {{.*}} borrow_in_mem, ?, "__result__": !lit.ref<!Int, mut *[0,1]> byref_result) async -> !kgen.none>: get_type_method
    _ = value.foo()


fn take_intable[T: Intable](x: T):
    pass


# CHECK-LABEL: lit.func @"nonmaterializable_trait
fn nonmaterializable_trait():
    # CHECK-NEXT: [[SLOT:%.*]] = lit.var.decl {{.*}} : !lit.ref<!Int,
    # CHECK-NEXT: [[VAL:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: store [[VAL]], [[SLOT]]
    # CHECK-NEXT:  = lit.ref.immut [[SLOT]]
    # CHECK-NEXT: call {{.*}}take_intable{{.*}}<:!Intable [!Int, {"__int__"
    take_intable(1)
