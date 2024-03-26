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


fn borrowed_generic[T: AnyType](borrowed x: T):
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

    # CHECK: %[[XCOPY:.*]] = kgen.call @"{{.*}}RegPassable::__copyinit__{{.*}}"(%arg0)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XCOPY]])
    owned_generic(x)

    # CHECK: %[[XPTR3:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    # CHECK: kgen.call @"{{.*}}MemOnly::__copyinit__{{.*}}"(%[[XPTR3]], %arg1)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XPTR3]])
    owned_generic(y)

    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg0)
    owned_generic(x^)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%arg1)
    owned_generic(y^)


# CHECK: kgen.func export @test_borrowed(
# CHECK-SAME: %arg0: !kgen.struct<(scalar<f32>, scalar<f32>)> borrow,
# CHECK-SAME: %arg1: !kgen.pointer<struct<(index, index) memoryOnly>> borrow_in_mem)
@export
fn test_borrowed(borrowed x: RegPassable, borrowed y: MemOnly):
    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg0)
    borrowed_generic(x)

    # CHECK: kgen.call @"{{.*}}borrowed_generic{{.*}}"(%arg1)
    borrowed_generic(y)

    # CHECK: %[[XCOPY:.*]] = kgen.call @"{{.*}}RegPassable::__copyinit__{{.*}}"(%arg0)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XCOPY]])
    owned_generic(x)

    # CHECK: %[[XPTR3:.*]] = pop.stack_allocation 1 x struct<(index, index) memoryOnly>
    # CHECK: kgen.call @"{{.*}}MemOnly::__copyinit__{{.*}}"(%[[XPTR3]], %arg1)
    # CHECK: kgen.call @"{{.*}}owned_generic{{.*}}"(%[[XPTR3]])
    owned_generic(y)
