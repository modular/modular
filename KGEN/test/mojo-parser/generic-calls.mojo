# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | kgen-opt -verify-parameters | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias AnyType = __mlir_type.`!kgen.anytype`
alias AnyRegType = __mlir_type.`!kgen.anyregtype`

trait Destructable:
    fn __del__(owned self, /):
       ...

trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
       ...

trait Movable:
    fn __moveinit__(inout self, owned existing: Self, /):
       ...

# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #

@value
@register_passable
struct RegPassable:
    pass


fn owned_generic[T: AnyType](owned x: T):
    pass


fn borrowed_generic[T: AnyType](borrowed x: T):
    pass


# CHECK-LABEL: lit.func @"test_owned{{.*}}"(%x[x]: !RegPassable)
fn test_owned(owned x: RegPassable):
    # CHECK: %[[XVAR:.*]] = lit.varlet.decl "x"
    # CHECK: lit.ref.store %x, %[[XVAR]]
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}<:type !RegPassable>(%[[XVAR]])
    borrowed_generic(x)

    # CHECK: %[[XREF:.*]] = lit.ref.load %[[XVAR]]
    # CHECK: %[[XCOPY:.*]] = lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}"(%[[XREF]])
    # CHECK: %[[XVAR2:.*]] = lit.varlet.decl
    # CHECK: lit.ref.store %[[XCOPY]], %[[XVAR2]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<:type !RegPassable>(%[[XVAR2]])
    owned_generic(x)

    # CHECK: %[[XMOVED:.*]] = lit.ownership.end_lifetime %[[XVAR]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<:type !RegPassable>(%[[XMOVED]])
    owned_generic(x ^)


# CHECK-LABEL: lit.func @"test_borrowed{{.*}}"(%x[x]: !RegPassable borrow)
fn test_borrowed(borrowed x: RegPassable):
    # CHECK: [[XSTACK:%.*]] = pop.stack_allocation 1 x !RegPassable
    # CHECK: lit.store.borrow %x, [[XSTACK]] : <!RegPassable>
    # CHECK: [[XPTR:%.*]] = lit.ref.from_pointer [[XSTACK]]
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}<:type !RegPassable>([[XPTR]])
    borrowed_generic(x)
    # CHECK-NEXT: lit.ownership.use %x : !RegPassable

    # CHECK: [[XCOPY:%.*]] = lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}(%x)
    # CHECK: [[XVAR:%.*]] = lit.varlet.decl
    # CHECK: lit.ref.store [[XCOPY]], [[XVAR]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}<:type !RegPassable>([[XVAR]])
    owned_generic(x)


fn generic[T: AnyType](t: T):
    pass


# CHECK-LABEL: lit.func @"test_reg_converts_to_generic
# CHECK-SAME: "<[[T:.*]][T]: regtype>
fn test_reg_converts_to_generic[T: AnyRegType](t: T):
    # CHECK: [[TREB:%.*]] = kgen.rebind %t : !kgen.paramref<[[T]]> to !kgen.paramref<:type rebind(:regtype [[T]])>
    # CHECK: [[TPTR:%.*]] = pop.stack_allocation 1 x :type rebind(:regtype [[T]])
    # CHECK: lit.store.borrow [[TREB]], [[TPTR]]
    # CHECK: [[TREF:%.*]] = lit.ref.from_pointer [[TPTR]]
    # CHECK: lit.call @{{.*}}::@"generic{{.*}}<:type rebind(:regtype [[T]])>([[TREF]])
    generic(t)


struct Foo[T: AnyType]:
    pass


fn generic_foo[T: AnyRegType](t: Foo[T]):
    pass


# CHECK-LABEL: lit.func @"infers_for_generic_foo
# CHECK-SAME: ]<[[T:.*]][T]: regtype>
fn infers_for_generic_foo[T: AnyRegType](t: Foo[T]):
    # CHECKL lit.call @{{.*}}::@"generic_foo{{.*}}<:regtype [[T]]>(%t)
    generic_foo(t)
