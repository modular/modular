# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Positive controls guarding against over-suppression. Semantic suppression of
# never-conforming slots (see never_conforming_no_synthesis.mojo) must not spill
# onto conformances that genuinely hold. Each case below asserts that a member
# IS synthesized and that the constraint attached to it is exactly right:
#
#   1. an unconditional conformance synthesizes with NO `where True` leak -- the
#      canonical trait marker carries no `constrained_` prefix and the
#      synthesized signature carries no constraint suffix (the Category-B
#      regression that four tests only caught incidentally);
#   2. a satisfiable conditional keeps its constraint verbatim in both the marker
#      and the synthesized signature;
#   3. a condition *implied* by the struct's own clause (`n > 0` under `n > 5`)
#      still synthesizes -- this guards the two-check contradiction logic in
#      getConformanceCondition against collapsing an implied (true) condition to
#      false;
#   4. the MixedConformance shape keeps `ImplicitlyDeletable` and drops only the
#      `Movable where False` slot.

# RUN: %parse-mojo-isolated %s | FileCheck %s


# --- 1. Unconditional: synthesizes, no `where True` leak --------------------

# The marker is the plain `!AnyType_ImplicitlyDeletable_Movable` alias (no
# `constrained_` prefix), and the synthesized ctor name ends `$)` with no
# `{constraint}` suffix. A leaked `where True` would turn both into the
# constrained spelling.
# CHECK-LABEL: lit.struct.decl @UncondMovable(!AnyType_ImplicitlyDeletable_Movable)
# CHECK: lit.fn @"__init__(move:{{.*}}UncondMovable$)"
struct UncondMovable(Movable):
    pass


# --- 2. Satisfiable conditional: constraint preserved -----------------------

# CHECK-LABEL: lit.struct.decl @SatMovable<T: !AnyType>(!constrained_AnyType_ImplicitlyDeletable_Movable)
# CHECK: __init__(move:{{.*}}SatMovable[$0]$){conforms_to($0, ::AnyType & ::Movable)}
struct SatMovable[T: AnyType](Movable where conforms_to(T, Movable)):
    pass


# --- 3. Condition implied by the struct clause: still synthesizes -----------

# `n > 5` implies `n > 0`, so the Movable condition holds rather than being
# contradicted; the move ctor is synthesized. This is the guard that keeps the
# ambient-unsatisfiability check from mistaking an implied condition for a
# contradicted one.
# CHECK-LABEL: lit.struct.decl @ImpliedMovable
# CHECK: __init__(move:{{.*}}ImpliedMovable[$0]$){
struct ImpliedMovable[n: Int](Movable where n > 0) where n > 5:
    pass


# --- 4. MixedConformance: keep ImplicitlyDeletable, drop Movable ------------

# The conditional `ImplicitlyDeletable where value >= 0` slot survives (marker
# keeps `ImplicitlyDeletable`, and the struct is linear pending that condition);
# the `Movable where False` slot is dropped, so no move ctor is synthesized.
# CHECK-LABEL: lit.struct.decl @MixedInner<value: !Int>(!constrained_AnyType_ImplicitlyDeletable)
# CHECK-SAME: does not conditionally conform to 'ImplicitlyDeletable' for these parameters
# CHECK-NOT: __init__(move:
struct MixedInner[value: Int](
    ImplicitlyDeletable where value >= 0, Movable where False,
):
    pass


# Sentinel: bounds the final `CHECK-NOT` before the stdlib decls (which
# legitimately synthesize move ctors) appear in the dump.
# CHECK-LABEL: lit.struct.decl @PCSentinel
struct PCSentinel:
    pass
