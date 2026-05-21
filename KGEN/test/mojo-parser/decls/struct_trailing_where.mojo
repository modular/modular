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
# CHECK-SAME: <T: !Base, {<conforms_to(:!Base T, [{{[^]]*}}@Extra]),
@fieldwise_init
struct SingleLineTrailingWhere[T: Base] where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @TrailingWhereWithParent
# CHECK-SAME: <T: !Base, {<conforms_to(:!Base T, [{{[^]]*}}@Extra]),
# CHECK-SAME: (!AnyType_ImplicitlyDestructible_Marker)
@fieldwise_init
struct TrailingWhereWithParent[T: Base](Marker) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @MultilineParentTrailingWhere
# CHECK-SAME: <T: !Base, {<conforms_to(:!Base T, [{{[^]]*}}@Extra]),
# CHECK-SAME: (!AnyType_ImplicitlyDestructible_Marker)
@fieldwise_init
struct MultilineParentTrailingWhere[T: Base](
    Marker
) where conforms_to(T, Extra):
    pass


# CHECK-LABEL: lit.struct.decl @MultipleTrailingWhereClauses
# CHECK-SAME: <T: !Base, {<conforms_to(:!Base T, [{{[^]]*}}@Extra]),
# CHECK-SAME: <conforms_to(:!Base T, [{{[^]]*}}@Base]),
@fieldwise_init
struct MultipleTrailingWhereClauses[
    T: Base
] where conforms_to(T, Extra) where conforms_to(T, Base):
    pass
