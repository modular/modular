# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Regression test for a compiler bug where a struct combining its own
# `where <predicate>` clause with `Trait where False` -- especially when
# composed with another such struct as a field -- spuriously failed
# conformance/witness verification ("cannot synthesize move constructor" /
# "does not implement all requirements for 'Movable'"), even though both
# sides had already declared they don't need the trait. See MOCO-4135.
#
# Root cause was two distinct bugs, both in
# KGEN/lib/LITDialect/LITUtils.cpp:
#   1. `LIT::inferConstraintRelation` was missing an "assumption trivially
#      false implies anything" case -- this alone fixes the self-conformance
#      case below.
#   2. `LIT::canDischargeConstraint` (singular) had a separate bug in its
#      per-assumption loop: an early `Contradicts` verdict from one
#      assumption discarded an already-found `Implies` from another -- this
#      is what actually blocked the field-composition case below.
#
# This is deliberately a standalone, minimal test rather than folded into an
# existing large file: this area has a documented history of subtle
# regressions -- a structurally similar (but not identical) fix landed in
# PR #90795 and was reverted 3 days later in PR #90810 (ref MOCO-3942) for
# silently causing CheckLifetimes.cpp to skip a needed destructor call. If
# this test starts failing, read the commits that introduced it (and the
# corresponding KGEN/internal/claude_kb/entries/known-limitations/
# where-false-constraints.md entry) before touching
# inferConstraintRelation/canDischargeConstraint again.

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | FileCheck %s

# Self-conformance case: a struct's own arithmetic where-clause combined with
# `Movable where False` on itself, no field composition needed. Checked via
# Traits.cpp's canDischargeMethodConstraints.
# CHECK-LABEL: lit.struct.decl @SelfConstrained
struct SelfConstrained[value: Int](Movable where False) where value >= 0:
    pass


# Field-composition case: Outer has its own where-clause AND `Movable where
# False`; Inner (used as Outer's field) is also `Movable where False`.
# Checked via ASTType::isMovable -> canDischargeConstraint, where the caller
# assumptions mix Outer's own where-clause with the synthesized function's
# own trivially-false conformance clause.
# CHECK-LABEL: lit.struct.decl @ComposedInner
struct ComposedInner[value: Int](Movable where False):
    pass

# CHECK-LABEL: lit.struct.decl @ComposedOuter
struct ComposedOuter[value: Int](Movable where False) where value >= 0:
    var field: ComposedInner[Self.value]


# Mixed-conformance case: Inner additionally has a genuinely conditional
# (non-False) conformance to a different trait, matching
# explicit_destroy.mojo's PredicateOnStructInner shape.
# CHECK-LABEL: lit.struct.decl @MixedConformanceInner
struct MixedConformanceInner[value: Int](
    ImplicitlyDeletable where value >= 0, Movable where False,
):
    pass

# CHECK-LABEL: lit.struct.decl @MixedConformanceOuter
struct MixedConformanceOuter[value: Int](Movable where False) where value >= 0:
    var field: MixedConformanceInner[Self.value]
