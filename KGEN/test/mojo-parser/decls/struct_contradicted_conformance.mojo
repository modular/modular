# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# A conformance condition disproven by the struct's own `where` clause leaves the
# facts in scope unsatisfiable, so it must reach the same verdict as the literal
# `where False` spelling -- otherwise the behavior depends on whether the
# condition happens to fold to a literal at parse time. These shapes reproduce
# with no fields at all; the field-carrying ones are covered by
# `movable_where_false_semantic_contradiction.mojo`. See MOCO-4135.

# RUN: %parse-mojo-isolated %s | FileCheck %s

# CHECK-LABEL: lit.struct.decl @ContradictedByOwnClause
struct ContradictedByOwnClause[n: Int](Movable where not (n > 0)) where n > 0:
    pass


# The `conforms_to`-negation spelling, which also pins the order of the relation
# rules: `Movable` canonicalizes to a multi-symbol trait, so the goal decomposes
# into a conjunction that the negated assumption contradicts only as a whole.
# CHECK-LABEL: lit.struct.decl @ContradictedByOwnConformsToClause
struct ContradictedByOwnConformsToClause[T: AnyType](
    Movable where conforms_to(T, Movable)
) where not conforms_to(T, Movable):
    pass
