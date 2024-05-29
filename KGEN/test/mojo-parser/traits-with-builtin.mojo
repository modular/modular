# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo --kgen-print-inline-vtables | FileCheck %s


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
    async fn foobar(self):
        pass


struct AsyncStruct(AsyncTrait):
    async fn foobar(self):
        pass


# CHECK-LABEL: lit.struct.decl @AsyncStructReg
@register_passable("trivial")
struct AsyncStructReg(AsyncTrait):
    # CHECK-LABEL: lit.func @"foobar{{.*}}_thunk"{{.*}}(%self: !lit.ref<!AsyncStructReg, imm {{.*}}>
    async fn foobar(self):
        # CHECK: [[POP_CORO:%.*]] = lit.async.call
        # CHECK-NEXT: [[CORO:%.*]] = lit.call {{.*}}__init__{{.*}}([[POP_CORO]])
        # CHECK-NEXT: [[RES:%.*]] = lit.call {{.*}}__await__{{.*}}([[CORO]])
        # CHECK-NEXT: return [[RES]]
        pass


# CHECK-LABEL: lit.func @"async_trait
fn async_trait[T: AsyncTrait](value: T):
    # CHECK: lit.async.call[!lit.signature<[1]("self": {{.*}} borrow_in_mem) async -> !kgen.none>: get_type_method
    _ = value.foobar()


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
