# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen -elaborate -O0 %s -S | FileCheck %s


@value
@register_passable
struct RegPassable:
    var v: Float32
    var w: Float32


@value
struct MemOnly:
    var a: Int
    var b: Int


fn owned_generic[T: AnyType](owned x: T):
    pass


fn borrowed_generic[T: AnyType](x: T):
    pass


# CHECK-LABEL: kgen.func export @test_owned(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)> owned,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(index, index) memoryOnly>> owned_in_mem)
@export
fn test_owned(owned x: RegPassable, owned y: MemOnly):
    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg0)
    borrowed_generic(x)

    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # COM: check callsite to inlined RegPassable::__copyinit__
    # NOTE: The copy is optimized away since it is trivial.
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x)

    # CHECK: [[XPTR3:%.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    # COM: check callsite to inlined MemOnly::__copyinit__
    # CHECK: [[V7:%.*]] = kgen.struct.gep [[XPTR3]][0] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V8:%.*]] = kgen.struct.gep %arg1[0] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
    # CHECK: pop.store [[V9]], [[V7]] : !kgen.pointer<index>
    # CHECK: [[V10:%.*]] = kgen.struct.gep [[XPTR3]][1] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V11:%.*]] = kgen.struct.gep %arg1[1] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
    # CHECK: pop.store [[V12]], [[V10]] : !kgen.pointer<index>
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"([[XPTR3]])
    owned_generic(y)

    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x^)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg1)
    owned_generic(y^)


# CHECK: kgen.func export @test_borrowed(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)>,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem)
@export
fn test_borrowed(x: RegPassable, y: MemOnly):
    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg0)
    borrowed_generic(x)

    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # COM: check callsite to inlined RegPassable::__copyinit__
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x)

    # CHECK: [[XPTR3:%.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    # COM: check callsite to inlined MemOnly::__copyinit__
    # CHECK: [[V7:%.*]] = kgen.struct.gep [[XPTR3]][0] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V8:%.*]] = kgen.struct.gep %arg1[0] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V9:%.*]] = pop.load [[V8]] : !kgen.pointer<index>
    # CHECK: pop.store [[V9]], [[V7]] : !kgen.pointer<index>
    # CHECK: [[V10:%.*]] = kgen.struct.gep [[XPTR3]][1] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V11:%.*]] = kgen.struct.gep %arg1[1] : <struct<(index, index) memoryOnly>>
    # CHECK: [[V12:%.*]] = pop.load [[V11]] : !kgen.pointer<index>
    # CHECK: pop.store [[V12]], [[V10]] : !kgen.pointer<index>
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"([[XPTR3]])
    owned_generic(y)
