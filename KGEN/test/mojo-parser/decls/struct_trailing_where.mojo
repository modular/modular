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
struct SingleLineTrailingWhere[T: Base] where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @TrailingWhereWithParent
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# CHECK-SAME: (!AnyType_ImplicitlyDeletable_Marker)
@fieldwise_init
struct TrailingWhereWithParent[T: Base](Marker) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @MultilineParentTrailingWhere
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# CHECK-SAME: (!AnyType_ImplicitlyDeletable_Marker)
@fieldwise_init
struct MultilineParentTrailingWhere[T: Base](Marker) where conforms_to(
    T, Extra
):
    pass


# CHECK-LABEL: lit.struct.decl @MultipleTrailingWhereClauses
# CHECK-SAME: <T: !AnyType_Base, {{{.*}}conforms_to(:!AnyType_Base T, :meta<!{{.*}}Extra> !{{.*}}Extra))
# CHECK-SAME: conforms_to(:!AnyType_Base T, :meta<!{{.*}}Base> !{{.*}}Base))
@fieldwise_init
struct MultipleTrailingWhereClauses[T: Base] where conforms_to(
    T, Extra
) where conforms_to(T, Base):
    pass


struct ConditionallyDeletableWrapper[T: AnyType](
    ImplicitlyDeletable where conforms_to(T, ImplicitlyDeletable),
):
    var value: Self.T


# CHECK-LABEL: lit.struct.decl @TrailingWhereDeletableField
# CHECK-SAME: <T: !AnyType,
# CHECK-SAME: conforms_to(:!AnyType T,
# CHECK: lit.struct.field value : !lit.struct<#ConditionallyDeletableWrapper
# CHECK-SAME: <:!AnyType T>>
struct TrailingWhereDeletableField[T: AnyType]
    where conforms_to(T, ImplicitlyDeletable):
    var value: ConditionallyDeletableWrapper[Self.T]
