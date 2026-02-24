# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test errors for conditional trait conformance.
# These errors are detected during struct declaration parsing and conformance
# verification, before any instantiation occurs.
#
# NOTE: Declaration-time errors (diamond, ancestor) are tested first because
# they are emitted before method constraint errors during parsing.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# ===========================================================================
# Diamond with different constraints requires explicit Base
# ===========================================================================
# When a struct conditionally conforms to DerivedA and DerivedB with different
# constraints, and both inherit from Base, the user must explicitly list Base.


trait DiamondBase:
    pass


trait DiamondDerivedA(DiamondBase):
    pass


trait DiamondDerivedB(DiamondBase):
    pass


struct DiamondMissingExplicitBase[T: Movable](
    # expected-error @below {{ancestor trait}}
    DiamondDerivedA where conforms_to(T, Copyable),
    DiamondDerivedB where conforms_to(T, Intable),
    Movable,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^


# ===========================================================================
# Derived constraint must imply ancestor constraint
# ===========================================================================
# When both a derived trait and its ancestor are explicitly listed with
# constraints, the derived constraint must logically imply the ancestor's.


trait AncestorImplicationBase:
    pass


trait AncestorImplicationDerived(AncestorImplicationBase):
    pass


struct DerivedDoesNotImplyAncestor[T: Movable](
    # Base requires Intable
    AncestorImplicationBase where conforms_to(T, Intable),
    # But Derived only requires Copyable - this doesn't imply Intable!
    # expected-error @below {{constraint for}}
    AncestorImplicationDerived where conforms_to(T, Copyable),
    Movable,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^


# ===========================================================================
# Unconditional derived with conditional ancestor
# ===========================================================================
# When a derived trait is listed unconditionally but its ancestor is explicitly
# listed conditionally, this is inconsistent: the derived trait always requires
# the ancestor, so the ancestor cannot be conditional.


trait UnconditionalDerivedBase:
    pass


trait UnconditionalDerivedChild(UnconditionalDerivedBase):
    pass


struct UnconditionalDerivedConditionalAncestor[T: Movable](
    # Derived is unconditional - always conforms
    # expected-error @below {{constraint for}}
    UnconditionalDerivedChild,
    # But ancestor is conditional - inconsistent!
    UnconditionalDerivedBase where conforms_to(T, Copyable),
    Movable,
):
    var data: Self.T

    fn __init__(out self, var data: Self.T):
        self.data = data^

    fn __moveinit__(out self, deinit take: Self):
        self.data = take.data^


# ===========================================================================
# Conditional conformance to RegisterPassable is not supported
# ===========================================================================
# RegisterPassable and TrivialRegisterPassable affect the type's ABI/convention
# which is a per-declaration decision that cannot vary per instantiation.

struct ConditionalTrivialRegPassable[T: Movable](
    # expected-error-re @below {{conditional conformance to '{{(Trivial)?}}RegisterPassable' is not supported}}
    TrivialRegisterPassable where conforms_to(T, TrivialRegisterPassable),
    Movable,
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^

struct ConditionalRegPassable[T: Movable](
    # expected-error @below {{conditional conformance to 'RegisterPassable' is not supported}}
    RegisterPassable where conforms_to(T, RegisterPassable),
    Movable,
):
    var value: Self.T

    fn __init__(out self, var value: Self.T):
        self.value = value^


# ===========================================================================
# Unconditional conformance with conditional method
# ===========================================================================
# A struct that unconditionally claims to conform to a trait, but the method
# implementing the trait requirement has a constraint that can't be proven
# from nothing (the unconditional conformance provides no assumptions).


# expected-note @below {{trait 'UnconditionalConformanceTrait' declared here}}
trait UnconditionalConformanceTrait:
    # expected-note @below {{required by trait method here}}
    fn do_something(self):
        ...


# expected-error @below {{does not implement all requirements for 'UnconditionalConformanceTrait'}}
struct UnconditionalWithConditionalMethod[x: Int](
    Movable, UnconditionalConformanceTrait
):
    # expected-note @below {{method 'do_something' has constraints that cannot be proven or disproven from conformance constraint}}
    fn do_something(self) where Self.x > 10:
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ===========================================================================
# Conditional conformance with non-implied method constraint
# ===========================================================================
# A struct that conditionally conforms to a trait (requires T: Intable),
# but the method has a different constraint (requires T: Copyable) that
# cannot be proven from the conformance constraint.


# expected-note @below {{trait 'MismatchedConstraintTrait' declared here}}
trait MismatchedConstraintTrait:
    # expected-note @below {{required by trait method here}}
    fn process(self):
        ...


# expected-error @below {{does not implement all requirements for 'MismatchedConstraintTrait'}}
struct MismatchedConstraints[T: Movable](
    MismatchedConstraintTrait where conforms_to(T, Intable), Movable
):
    # This method requires Copyable, but conformance only guarantees Intable
    # expected-note @below {{method 'process' has constraints that cannot be proven or disproven from conformance constraint}}
    fn process(self) where conforms_to(Self.T, Copyable):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ===========================================================================
# Weaker conformance constraint with stronger method constraint
# ===========================================================================
# A struct with a weaker conformance constraint (T: Copyable) but a method
# that requires a stronger constraint (T: Copyable AND Intable).


# expected-note @below {{trait 'WeakerConformanceTrait' declared here}}
trait WeakerConformanceTrait:
    # expected-note @below {{required by trait method here}}
    fn execute(self):
        ...


# expected-error @below {{does not implement all requirements for 'WeakerConformanceTrait'}}
struct WeakerConformanceStrongerMethod[T: Movable](
    WeakerConformanceTrait where conforms_to(T, Copyable),
    Movable,
):
    # This method requires BOTH Copyable AND Intable, but conformance only guarantees Copyable
    # expected-note @below {{method 'execute' has constraints that cannot be proven or disproven from conformance constraint}}
    fn execute(
        self,
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Intable):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ===========================================================================
# Conditional conformance with both unconditional and conditional methods
# ===========================================================================
# A struct with conditional conformance where both an unconditional method and
# a conditional method (whose constraint is implied) exist. Both are valid
# candidates, causing ambiguity.


# expected-note @below {{trait 'AmbiguousMethodTrait' declared here}}
trait AmbiguousMethodTrait:
    # expected-note @below {{ambiguous use of 'perform'}}
    fn perform(self):
        ...


# expected-error @below {{does not implement all requirements for 'AmbiguousMethodTrait'}}
struct AmbiguousUnconditionalAndConditional[T: Movable](
    AmbiguousMethodTrait where conforms_to(T, Copyable),
    Movable,
):
    # Unconditional method - always valid
    # expected-note @below {{candidate declared here}}
    fn perform(self):
        pass

    # Conditional method - also valid because conformance implies T: Copyable
    # expected-note @below {{candidate declared here}}
    fn perform(self) where conforms_to(Self.T, Copyable):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ===========================================================================
# Multiple conditional methods both implied by stronger conformance
# ===========================================================================
# A struct with a strong conformance constraint (T: Copyable AND T: Intable)
# but two methods with weaker, different constraints. Both are implied by the
# conformance, causing ambiguity.


# expected-note @below {{trait 'MultipleConditionsTrait' declared here}}
trait MultipleConditionsTrait:
    # expected-note @below {{ambiguous use of 'run'}}
    fn run(self):
        ...


# expected-error @below {{does not implement all requirements for 'MultipleConditionsTrait'}}
struct AmbiguousBothConditional[T: Movable](
    MultipleConditionsTrait where conforms_to(T, Copyable) and conforms_to(
        T, Intable
    ),
    Movable,
):
    # Method requiring Copyable - implied by conformance
    # expected-note @below {{candidate declared here}}
    fn run(self) where conforms_to(Self.T, Copyable):
        pass

    # Method requiring Intable - also implied by conformance
    # expected-note @below {{candidate declared here}}
    fn run(self) where conforms_to(Self.T, Intable):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ===========================================================================
# Unprovable overload causes error even with a valid candidate
# ===========================================================================
# Following overload selection rules, ALL candidates' constraints must be
# definitively provable or disproved. If any candidate has unprovable
# constraints (constraints we can neither prove nor disprove from the
# conformance), we error - even if another candidate has provable constraints.
#
# This prevents ambiguity: the user might have intended the unprovable
# candidate to be selected, but our constraint system can't verify that.


# expected-note @below {{trait 'UnprovableCandidateTrait' declared here}}
trait UnprovableCandidateTrait:
    # expected-note @below {{required by trait method here}}
    fn handle(self):
        ...


# expected-error @below {{does not implement all requirements for 'UnprovableCandidateTrait'}}
struct UnprovableWithValidCandidate[T: Movable](
    UnprovableCandidateTrait where conforms_to(T, Copyable),
    Movable,
):
    # Unprovable: Intable is unrelated to Copyable - can't prove or disprove.
    # expected-note @below {{method 'handle' has constraints that cannot be proven or disproven from conformance constraint}}
    fn handle(self) where conforms_to(Self.T, Intable):
        pass

    # Provable: Copyable matches the conformance constraint.
    # But we still error because we can't rule out the above candidate.
    fn handle(self) where conforms_to(Self.T, Copyable):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass


# ===========================================================================
# Method with `where not` is disproved (contradicts conformance)
# ===========================================================================
# When a method has `where not X` and the conformance requires X, the method
# is "disproved" - definitively not a valid candidate. This is different from
# "unprovable" because we CAN make a determination (it's definitely invalid).
#
# With only a disproved candidate and no valid alternative, we get a normal
# "not implemented" error.


# expected-note @below {{trait 'ContradictingConstraintTrait' declared here}}
trait ContradictingConstraintTrait:
    # expected-note @below {{no 'apply' candidates have type}}
    fn apply(self):
        ...


# expected-error @below {{does not implement all requirements for 'ContradictingConstraintTrait'}}
struct DisprovedWithWhereNot[T: Movable](
    ContradictingConstraintTrait where conforms_to(T, Copyable),
    Movable,
):
    # Disproved: `not conforms_to(T, Copyable)` contradicts conformance.
    fn apply(self) where not conforms_to(Self.T, Copyable):
        pass

    fn __moveinit__(out self, deinit take: Self):
        pass
