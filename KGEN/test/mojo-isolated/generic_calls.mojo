# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@fieldwise_init
@register_passable
struct RegPassable(Copyable):
    pass


fn owned_generic[T: AnyType](owned x: T):
    pass


fn borrowed_generic[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test_owned{{.*}}(%x: !lit.ref<!RegPassable, mut *"x`"> owned_in_mem)
fn test_owned(owned x: RegPassable):
    # CHECK: [[XIMUT:%.*]] = lit.ref.immut %x : <!RegPassable, mut *"x`">
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[XIMUT]])
    borrowed_generic(x)

    # CHECK: [[XIMM:%.*]] = lit.ref.immut %x
    # CHECK: [[XCOPY:%.*]] = lit.var.decl
    # CHECK: lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}([[XIMM]], [[XCOPY]])
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)

    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>(%x)
    owned_generic(x^)


# CHECK-LABEL: lit.fn @"test_borrowed{{.*}}(%x: !lit.ref<!RegPassable, imm *"x`"> read_mem)
fn test_borrowed(x: RegPassable):
    # CHECK-NEXT: lit.call @{{.*}}::@"borrowed_generic{{.*}}<{{.*}}>(%x)
    borrowed_generic(x)

    # CHECK: [[XCOPY:%.*]] = lit.var.decl
    # CHECK: lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}(%x, [[XCOPY]])
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)


# CHECK-LABEL: lit.fn @"function_reference
fn function_reference():
    # CHECK: create_closure[{{.*}}@"function_reference
    borrowed_generic(function_reference)
