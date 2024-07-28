# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

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


# CHECK-LABEL: lit.func @"test_owned{{.*}}"(%x: !RegPassable owned)
fn test_owned(owned x: RegPassable):
    # CHECK: [[XVAR:%.*]] = lit.var.decl "x"
    # CHECK: lit.ref.store %x, [[XVAR]]
    # CHECK: [[XIMUT:%.*]] = lit.ref.immut [[XVAR]] : <!RegPassable, mut *"x`">
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[XIMUT]])
    borrowed_generic(x)

    # CHECK: [[XREF:%.*]] = lit.ref.load [[XVAR]]
    # CHECK: lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}([[XCOPY:%.*]], [[XREF]])
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)

    # CHECK: [[XMOVED:%.*]] = lit.transfer_mem_ownership [[XVAR]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XMOVED]])
    owned_generic(x^)


# CHECK-LABEL: lit.func @"test_borrowed{{.*}}"(%x: !RegPassable)
fn test_borrowed(x: RegPassable):
    # CHECK: [[XSTACK:%.*]] = pop.stack_allocation 1 x !RegPassable
    # CHECK: pop.store %x, [[XSTACK]]
    # CHECK: [[XREF:%.*]] = lit.ref.from_pointer [[XSTACK]] :
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[XREF]])
    borrowed_generic(x)
    # CHECK-NEXT: lit.ownership.use %x : !RegPassable

    # CHECK: lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}([[XCOPY:%.*]], %x)
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)


# CHECK-LABEL: lit.func @"function_reference
fn function_reference():
    # CHECK: create_closure[{{.*}}@"function_reference
    borrowed_generic(function_reference)
