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
struct RegPassable(ImplicitlyCopyable, RegisterPassable):
    pass


@fieldwise_init
struct MemOnly(ImplicitlyCopyable):
    pass


def owned_generic[T: AnyType](var x: T):
    pass


def borrowed_generic[T: AnyType](x: T):
    pass


# CHECK-LABEL: lit.fn @"test_owned{{.*}}(%x: !lit.ref<!RegPassable, mut *"x`"> owned_in_mem, %y: !lit.ref<!MemOnly, mut *"y`1"> owned_in_mem)
def test_owned(var x: RegPassable, var y: MemOnly):
    # CHECK: [[XIMUT:%.*]] = lit.ref.immut %x : <!RegPassable, mut *"x`">
    # CHECK: lit.call {{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[XIMUT]])
    borrowed_generic(x)

    # CHECK: [[YIMUT:%.*]] = lit.ref.immut %y : <!MemOnly, mut *"y`1">
    # CHECK: lit.call {{.*}}::@"borrowed_generic{{.*}}<{{.*}}>([[YIMUT]])
    borrowed_generic(y)

    # CHECK: [[XIMM:%.*]] = lit.ref.immut %x
    # CHECK: [[XCOPY:%.*]] = lit.var.decl
    # CHECK: lit.call {{.*}}::@RegPassable::@"__init__{{.*}}copy"
    # CHECK: lit.call {{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)

    # CHECK: [[YCOPY:%.*]] = lit.var.decl
    # CHECK: lit.memcpy %y, [[YCOPY]]
    # CHECK: lit.call {{.*}}::@"owned_generic{{.*}}<{{.*}}>([[YCOPY]])
    owned_generic(y)

    # CHECK: lit.call {{.*}}::@"owned_generic{{.*}}<{{.*}}>(%x)
    owned_generic(x^)

    # CHECK: lit.call {{.*}}::@"owned_generic{{.*}}<{{.*}}>(%y)
    owned_generic(y^)


# CHECK-LABEL: lit.fn @"test_borrowed{{.*}}(%x: !lit.ref<!RegPassable, imm *"x`"> read_mem, %y: !lit.ref<!MemOnly, imm *"y`1"> read_mem)
def test_borrowed(x: RegPassable, y: MemOnly):
    # CHECK-NEXT: lit.call {{.*}}::@"borrowed_generic{{.*}}<{{.*}}>(%x)
    borrowed_generic(x)

    # CHECK-NEXT: lit.call {{.*}}::@"borrowed_generic{{.*}}<{{.*}}>(%y)
    borrowed_generic(y)

    # CHECK: [[XCOPY:%.*]] = lit.var.decl
    # CHECK: lit.call {{.*}}::@RegPassable::@"__init__{{.*}}copy"
    # CHECK: lit.call {{.*}}::@"owned_generic{{.*}}<{{.*}}>([[XCOPY]])
    owned_generic(x)

    # CHECK: [[YCOPY:%.*]] = lit.var.decl
    # CHECK: lit.memcpy %y, [[YCOPY]]
    # CHECK: lit.call {{.*}}::@"owned_generic{{.*}}<{{.*}}>([[YCOPY]])
    owned_generic(y)


# CHECK-LABEL: lit.fn @"function_reference
def function_reference():
    # CHECK: kgen.param.constant: {{.*}}#kgen.func.symbol<@generic_calls::@"function_reference()"
    borrowed_generic(function_reference)
