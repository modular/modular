# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test errors for duplicate traits with conflicting conditional conformances
# in struct inheritance lists.
#
# When the same trait appears multiple times in a struct's inheritance list
# (either directly or through trait compositions), the conditional conformance
# constraints must agree.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# ===========================================================================
# Duplicate trait: unconditional + conditional
# ===========================================================================


trait DupTraitA:
    pass


struct DupUnconditionalAndConditional[T: Movable](
    DupTraitA,
    # expected-error @below {{trait}}
    DupTraitA where conforms_to(T, Copyable),
    Movable,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^


# ===========================================================================
# Duplicate trait: different conditional constraints
# ===========================================================================


trait DupTraitB:
    pass


struct DupDifferentConstraints[T: Movable](
    DupTraitB where conforms_to(T, Copyable),
    # expected-error @below {{trait}}
    DupTraitB where conforms_to(T, Intable),
    Movable,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^


# ===========================================================================
# Trait composition + standalone with conflicting constraints
# ===========================================================================
# A & B where cond gives both A and B the condition. Listing A again without
# a condition conflicts.


trait CompA:
    pass


trait CompB:
    pass


struct CompositionConflictsWithStandalone[T: Movable](
    CompA & CompB where conforms_to(T, Copyable),
    # expected-error @below {{trait}}
    CompA,
    Movable,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^
