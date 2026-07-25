# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Base:
    pass


trait Extra:
    pass


trait Marker:
    pass


# CHECK-LABEL: lit.struct.decl @SingleLineTrailingWhere
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
@fieldwise_init
struct SingleLineTrailingWhere[T: Base] (Movable where False) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @TrailingWhereWithParent
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# CHECK-SAME: (!constrained_AnyType_ImplicitlyDeletable_Movable_Marker)
@fieldwise_init
struct TrailingWhereWithParent[T: Base](Marker, Movable where False) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @MultilineParentTrailingWhere
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# CHECK-SAME: (!constrained_AnyType_ImplicitlyDeletable_Movable_Marker1)
@fieldwise_init
struct MultilineParentTrailingWhere[T: Base](Marker, Movable where False) where conforms_to(
    T, Extra
):
    pass


# CHECK-LABEL: lit.struct.decl @MultipleTrailingWhereClauses
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# CHECK-SAME: conforms_to(:!AnyType_Base T, :meta<!{{.*}}Base> !{{.*}}Base))
@fieldwise_init
struct MultipleTrailingWhereClauses[T: Base] (Movable where False) where conforms_to(
    T, Extra
) where conforms_to(T, Base):
    pass


struct ConditionallyDeletableWrapper[T: AnyType](
    ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable), Movable where False,
):
    var value: Self.T


# CHECK-LABEL: lit.struct.decl @TrailingWhereDeletableField
# CHECK-SAME: <T: !AnyType,
# CHECK-SAME: conforms_to(:!AnyType T,
# CHECK: lit.struct.field value : !lit.struct<#ConditionallyDeletableWrapper
# CHECK-SAME: <:!AnyType T>>
struct TrailingWhereDeletableField[T: AnyType](Movable where False)
    where conforms_to(T, ImplicitlyDeletable):
    var value: ConditionallyDeletableWrapper[Self.T]


# The auto-synthesized fieldwise `__init__` for a struct with a trailing
# `where` clause must be able to use that clause's assumption to prove a
# field's conditional conformance -- mirroring the dtor case above, but
# exercising StructEmitter::synthesizeFieldwiseInit's field-movability check
# instead of synthesizeEmptyDtor's.
struct ConditionallyMovableWrapper[T: ImplicitlyDeletable](
    Movable where conforms_to(T, Movable),
):
    var value: Self.T


# CHECK-LABEL: lit.struct.decl @TrailingWhereMovableField
# CHECK-SAME: <T: !AnyType_ImplicitlyDeletable
# CHECK: lit.struct.field value : !lit.struct<#ConditionallyMovableWrapper
# CHECK-SAME: <:!AnyType_ImplicitlyDeletable T>>
struct TrailingWhereMovableField[T: ImplicitlyDeletable](
    Movable where conforms_to(T, Movable)
):
    var value: ConditionallyMovableWrapper[Self.T]
