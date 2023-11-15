# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen-translate -import-mojo --mojo-disable-builtins %s | kgen-opt -verify-parameters | FileCheck %s

alias AnyType = __mlir_type.`!kgen.anytype`
alias AnyRegType = __mlir_type.`!kgen.anyregtype`


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
    # CHECK: %[[XPTR:.*]] = lit.ref.to_pointer %[[XVAR]]
    # CHECK: %[[XREB:.*]] = kgen.rebind %[[XPTR]]
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}"<:type !RegPassable>(%[[XREB]])
    borrowed_generic(x)

    # CHECK: %[[XREF:.*]] = lit.ref.load %[[XVAR]]
    # CHECK: %[[XCOPY:.*]] = lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}"(%[[XREF]])
    # CHECK: %[[XVAR2:.*]] = lit.varlet.decl
    # CHECK: %[[XPTR2:.*]] = lit.ref.to_pointer %[[XVAR2]]
    # CHECK: pop.store %[[XCOPY]], %[[XPTR2]] : !kgen.pointer<!RegPassable>
    # CHECK: %[[XREB2:.*]] = kgen.rebind %[[XPTR2]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}"<:type !RegPassable>(%[[XREB2]])
    owned_generic(x)

    # CHECK: %[[XPTR3:.*]] = lit.ref.to_pointer %[[XVAR]]
    # CHECK: %[[XMOVED:.*]] = lit.ownership.end_lifetime %[[XPTR3]] : !kgen.pointer<!RegPassable> {isReg = false}
    # CHECK: %[[XREB3:.*]] = kgen.rebind %[[XMOVED]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}"<:type !RegPassable>(%[[XREB3]])
    owned_generic(x ^)


# CHECK-LABEL: lit.func @"test_borrowed{{.*}}"(%x[x]: !RegPassable borrow)
fn test_borrowed(borrowed x: RegPassable):
    # CHECK: %[[XSTACK:.*]] = pop.stack_allocation 1 x !RegPassable
    # CHECK: pop.store %x, %[[XSTACK]] : !kgen.pointer<!RegPassable>
    # CHECK: %[[XREB:.*]] = kgen.rebind %[[XSTACK]]
    # CHECK: lit.call @{{.*}}::@"borrowed_generic{{.*}}"<:type !RegPassable>(%[[XREB]])
    borrowed_generic(x)
    # CHECK-NEXT: lit.ownership.use %x : !RegPassable

    # CHECK: %[[XCOPY:.*]] = lit.call @{{.*}}::@RegPassable::@"__copyinit__{{.*}}"(%x)
    # CHECK: %[[XVAR:.*]] = lit.varlet.decl
    # CHECK: %[[XPTR:.*]] = lit.ref.to_pointer %[[XVAR]]
    # CHECK: pop.store %[[XCOPY]], %[[XPTR]] : !kgen.pointer<!RegPassable>
    # CHECK: %[[XREB2:.*]] = kgen.rebind %[[XPTR]]
    # CHECK: lit.call @{{.*}}::@"owned_generic{{.*}}"<:type !RegPassable>(%[[XREB2]])
    owned_generic(x)


fn generic[T: AnyType](t: T):
    pass


# CHECK-LABEL: lit.func @"test_reg_converts_to_generic
# CHECK-SAME: "<[[T:.*]][T]: regtype>
fn test_reg_converts_to_generic[T: AnyRegType](t: T):
    # CHECK: %[[TREB:.*]] = kgen.rebind %t : !kgen.paramref<[[T]]> to !kgen.paramref<:type rebind(:regtype [[T]])>
    # CHECK: %[[TPTR:.*]] = pop.stack_allocation 1 x :type rebind(:regtype [[T]])
    # CHECK: pop.store %[[TREB]], %[[TPTR]]
    # CHECK: lit.call @{{.*}}::@"generic{{.*}}"<:type rebind(:regtype [[T]])>(%[[TPTR]])
    generic(t)
