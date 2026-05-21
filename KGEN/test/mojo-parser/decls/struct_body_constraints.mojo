# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test struct-level trailing `where` body constraints. These constraints are
# checked during parameter inference (`InferenceState::checkBodyConstraints`)
# at every site that binds the struct's parameters, so a violation surfaces
# at instantiation rather than at first use of an instance.
#
# Each section uses its own struct/function decl so per-section diagnostic
# notes don't bleed across test sites (mirrors the convention in
# `constraint_overload_errors.mojo`).

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Satisfied body constraint at instantiation - positive case
##===----------------------------------------------------------------------===##
# Verifies the new `checkBodyConstraints` path doesn't produce false
# positives on the happy path.


struct SatisfiedStruct[N: Int]
    where N > 0:
    pass


def use_satisfied_struct():
    var x: SatisfiedStruct[5]


##===----------------------------------------------------------------------===##
# Violated body constraint at instantiation
##===----------------------------------------------------------------------===##
# Binding `N = -1` violates `N > 0`. The new check folds this into a
# binding-time error rather than letting it leak into downstream uses.


# expected-note @below {{'ViolatedStruct' declared here}}
struct ViolatedStruct[N: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where N > 0:
    pass


def use_violated_struct():
    # expected-error @below {{violated constraint}}
    var x: ViolatedStruct[-1]


##===----------------------------------------------------------------------===##
# Unprovable body constraint without evidence (Issue #1 regression)
##===----------------------------------------------------------------------===##
# When the caller is itself parametric and offers no assumption that
# discharges the body constraint, the binding is unprovable. The fix in
# `ParamInf::inferForStruct` ensures the strict (non-discarding) path
# returns a single "lacking evidence" error and does NOT silently return
# valid bindings while emitting a top-level error against the same site.


# expected-note @below {{cannot prove constraint}}
struct UnprovableStruct[N: Int]
    # expected-note @below {{constraint declared here needs evidence for}}
    where N > 0:
    pass


def use_unprovable_struct[K: Int]():
    # expected-error @below {{lacking evidence to prove correctness}}
    var x: UnprovableStruct[K]


##===----------------------------------------------------------------------===##
# Unprovable body constraint dischargeable from caller's assumptions
##===----------------------------------------------------------------------===##
# The caller's `where K > 0` is threaded through as an additional assumption
# during constraint checking, so the body constraint is provable.


struct DischargeableStruct[N: Int]
    where N > 0:
    pass


def use_dischargeable_struct[K: Int]() where K > 0:
    var x: DischargeableStruct[K]


##===----------------------------------------------------------------------===##
# Body constraint enforced eagerly at function-signature type formation
##===----------------------------------------------------------------------===##
# A function signature that mentions a parameterized struct re-enters
# `inferForStruct` to type-check the parameter type itself. The struct's
# body constraint is checked at that site, *not* deferred to the eventual
# call site. Because Mojo's trailing function `where` clause is parsed
# after the parameter list, it isn't yet visible as an assumption while
# parameter types are being formed — so a parametric binding that needs
# the trailing `where` to be provable will surface as a hard error here.
# (The dischargeable analog inside a function *body* is covered above by
# `use_dischargeable_struct`, where the trailing `where` is in scope.)


# expected-note @below {{cannot prove constraint}}
struct PositiveOnly[N: Int]
    # expected-note @below {{constraint declared here needs evidence for}}
    where N > 0:
    pass


# expected-error @below {{lacking evidence to prove correctness}}
def bad_signature_use[K: Int](x: PositiveOnly[K]):
    pass


def call_bad_signature_use():
    var p: PositiveOnly[7]


##===----------------------------------------------------------------------===##
# Multi-parameter body constraint
##===----------------------------------------------------------------------===##
# A constraint that mentions two parameters must be checked once both are
# bound, and yield a hard binding-time error when violated.


struct OrderedPair[A: Int, B: Int]
    where A < B:
    pass


def use_ordered_pair_satisfied():
    var p: OrderedPair[1, 5]


# expected-note @below {{'ViolatedOrderedPair' declared here}}
struct ViolatedOrderedPair[A: Int, B: Int]
    # expected-note @below {{constraint declared here evaluated to False}}
    where A < B:
    pass


def use_ordered_pair_violated():
    # expected-error @below {{violated constraint}}
    var p: ViolatedOrderedPair[5, 1]


##===----------------------------------------------------------------------===##
# Body constraint references trait conformance
##===----------------------------------------------------------------------===##
# A struct gated on `conforms_to` should only accept type bindings whose
# inferred parameter satisfies the trait. This is the value-level analog of
# the existing trait-conformance verification flow but applied at struct
# instantiation rather than at conformance declaration.


struct OnlyIntableSatisfied[T: AnyType]
    where conforms_to(T, Intable):
    pass


struct ConcreteIntable(Intable):
    def __int__(self) -> Int:
        return 0


def use_intable_satisfied():
    var x: OnlyIntableSatisfied[ConcreteIntable]


# expected-note @below {{'OnlyIntableViolated' declared here}}
struct OnlyIntableViolated[T: AnyType]
    # expected-note @below {{constraint declared here evaluated to False}}
    where conforms_to(T, Intable):
    pass


struct NotIntable:
    pass


def use_intable_violated():
    # expected-error @below {{violated constraint}}
    var x: OnlyIntableViolated[NotIntable]


# expected-note @below {{cannot prove constraint}}
struct OnlyIntableUnprovable[T: AnyType]
    # expected-note @below {{constraint declared here needs evidence for}}
    where conforms_to(T, Intable):
    pass


def use_intable_dischargeable[T: AnyType]() where conforms_to(T, Intable):
    var x: OnlyIntableUnprovable[T]


def use_intable_unprovable[T: AnyType]():
    # expected-error @below {{lacking evidence to prove correctness}}
    var x: OnlyIntableUnprovable[T]
