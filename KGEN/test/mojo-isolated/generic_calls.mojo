# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@value
@register_passable
struct RegPassable:
    pass


fn owned_generic[T: AnyType](owned x: T):
    pass


fn borrowed_generic[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.func @"test_owned{{.*}}(%x: !lit.ref<!RegPassable, mut *"x`"> owned_in_mem)
fn test_owned(owned x: RegPassable):
    # CHECK: [[XIMUT:%.*]] = lit.ref.immut %x : <!RegPassable, mut *"x`">
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[XIMUT]])
    borrowed_generic(x)

    # CHECK: [[XREF:%.*]] = lit.ref.load %x
    # CHECK: lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}([[XCOPY:%.*]], [[XREF]])
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)

    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>(%x)
    owned_generic(x^)


# CHECK-LABEL: lit.func @"test_borrowed{{.*}}"(%x: !RegPassable)
fn test_borrowed(x: RegPassable):
    # CHECK: [[XREF:%.*]] = lit.var.decl "__sbvalue_tmp__"
    # CHECK-NEXT: [[XPTR:%.*]] = lit.ref.to_pointer [[XREF]]
    # CHECK-NEXT: mark_initialized [[XREF]]
    # CHECK-NEXT: pop.store %x, [[XPTR]]
    # CHECK-NEXT: [[XIMM:%.*]] = lit.ref.immut [[XREF]]
    # CHECK-NEXT: lit.call @{{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[XIMM]])
    borrowed_generic(x)
    # CHECK-NEXT: mark_consumed [[XREF]]
    # CHECK-NEXT: lit.ownership.use %x : !RegPassable

    # CHECK: lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}([[XCOPY:%.*]], %x)
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)


# CHECK-LABEL: lit.func @"function_reference
fn function_reference():
    # CHECK: create_closure[{{.*}}@"function_reference
    borrowed_generic(function_reference)
