# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate -O0 %s -S | FileCheck %s


alias AnyType = __mlir_type.`!kgen.anytype`


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


fn borrowed_generic[T: AnyType](borrowed x: T):
    pass


# CHECK-LABEL: kgen.func export @test_owned(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)> owned,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(index, index) memoryOnly>> owned_in_mem)
@export
fn test_owned(owned x: RegPassable, owned y: MemOnly):
    # CHECK: %[[XPTR:.*]] = pop.stack_allocation 1 x struct<(scalar<f32>, scalar<f32>)>
    # CHECK: pop.store %arg0, %[[XPTR]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: %[[XL:.*]] = pop.load %[[XPTR]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%[[XL]])
    borrowed_generic(x)

    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # CHECK: %[[XL2:.*]] = pop.load %[[XPTR]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: %[[XCOPY:.*]] = kgen.call @"{{.*}}RegPassable::__copyinit__{{.*}}"(%[[XL2]])
    # CHECK: %[[XPTR2:.*]] = pop.stack_allocation 1 x struct<(scalar<f32>, scalar<f32>)>
    # CHECK: pop.store %[[XCOPY]], %[[XPTR2]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: %[[XL3:.*]] = pop.load %[[XPTR2]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XL3]])
    owned_generic(x)

    # CHECK: %[[XPTR3:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    # CHECK: kgen.call @"{{.*}}MemOnly::__copyinit__{{.*}}"(%[[XPTR3]], %arg1)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XPTR3]])
    owned_generic(y)

    # CHECK: %[[XL3:.*]] = pop.load %[[XPTR]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XL3]])
    owned_generic(x ^)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg1)
    owned_generic(y ^)


# CHECK: kgen.func export @test_borrowed(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)> borrow,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem)
@export
fn test_borrowed(borrowed x: RegPassable, borrowed y: MemOnly):
    # CHECK: %[[XPTR:.*]] = pop.stack_allocation 1 x struct<(scalar<f32>, scalar<f32>)>
    # CHECK: pop.store %arg0, %[[XPTR]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: %[[XL:.*]] = pop.load %[[XPTR]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%[[XL]])
    borrowed_generic(x)

    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # CHECK: %[[XCOPY:.*]] = kgen.call @"{{.*}}RegPassable::__copyinit__{{.*}}"(%arg0)
    # CHECK: %[[XPTR2:.*]] = pop.stack_allocation 1 x struct<(scalar<f32>, scalar<f32>)>
    # CHECK: pop.store %[[XCOPY]], %[[XPTR2]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: %[[XL2:.*]] = pop.load %[[XPTR2]] : !kgen.pointer<struct<(scalar<f32>, scalar<f32>)>>
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XL2]])
    owned_generic(x)

    # CHECK: %[[XPTR3:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    # CHECK: kgen.call @"{{.*}}MemOnly::__copyinit__{{.*}}"(%[[XPTR3]], %arg1)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XPTR3]])
    owned_generic(y)
