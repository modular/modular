# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Auto-Predication During Signature Emission
#
# When a parametric struct appears in a function or struct signature
# (parameter declaration, function argument, return type, or thrown type),
# binding the struct's parameters can produce unprovable body constraints:
# the trailing `where` clauses are parsed AFTER the signature, so their
# assumptions aren't yet in scope at the binding site.
#
# The auto-predication feature collects these unprovable body constraints into a
# deferral context during signature emission and ensures that each one is
# implied by any body constraints on the signature.
#
# Wiring covered by these tests:
#   - Function parameter declaration types (`TypeCheckedParamList::create`)
#   - Function argument types         (`typeCheckOneArgument`)
#   - Function return types           (`typeCheckResult`)
#   - Function `raises T` thrown types (`typeCheckResult` errorType branch)
#   - Struct parameter declaration types (`TypeCheckedParamList::create`)
#
# Per-parameter `where` constraints are intentionally NOT deferred (they
# would change candidate viability during overload resolution); the cases
# below only cover body (trailing) constraints.
##===----------------------------------------------------------------------===##


##===----------------------------------------------------------------------===##
# Discharged via trailing `where` on a function
##===----------------------------------------------------------------------===##
# A parametric binding whose body constraint is unprovable at the binding
# site is deferred and successfully discharged once the trailing `where`
# clause has been folded into the decl scope's known assumptions. The
# cases below should type-check without diagnostics.


struct PositiveOnly[N: Int] where N > 0:
    pass


struct OnlyIntable[T: AnyType] where conforms_to(T, Intable):
    pass


# Parameter declaration position.
def discharged_from_param_int[K: Int, X: PositiveOnly[K]]() where K > 0:
    pass


def discharged_from_param_trait[T: AnyType, X: OnlyIntable[T]]()
    where conforms_to(T, Intable):
    pass


# Argument type position.
def discharged_from_arg_int[K: Int](x: PositiveOnly[K]) where K > 0:
    pass


def discharged_from_arg_trait[T: AnyType](x: OnlyIntable[T])
    where conforms_to(T, Intable):
    pass


# Return type position.
def discharged_from_ret[K: Int]() -> PositiveOnly[K] where K > 0:
    pass


# Thrown type position (`raises T`).
def discharged_from_throws[K: Int]() raises PositiveOnly[K] where K > 0:
    pass


##===----------------------------------------------------------------------===##
# Discharged via trailing `where` on a struct
##===----------------------------------------------------------------------===##
# Structs go through the same flow as functions, so deferral and discharge apply
# identically to struct parameter declaration positions.


struct DischargedStruct[K: Int, X: PositiveOnly[K]] where K > 0:
    pass


struct DischargedStructTrait[T: AnyType, X: OnlyIntable[T]]
    where conforms_to(T, Intable):
    pass


##===----------------------------------------------------------------------===##
# Multiple deferrals — all discharged
##===----------------------------------------------------------------------===##
# When a signature introduces multiple deferred body constraints, the
# discharge loop iterates over each independently and discharges those
# implied by the trailing `where`.


def all_discharged[A: Int, B: Int,
                   X: PositiveOnly[A], Y: PositiveOnly[B]]()
    where A > 0 where B > 0:
    pass


struct AllDischargedStruct[A: Int, B: Int,
                            X: PositiveOnly[A], Y: PositiveOnly[B]]
    where A > 0 where B > 0:
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function parameter declaration position
##===----------------------------------------------------------------------===##


struct PosForParam[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > 0)'}}
    where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(K > 0)'}}
def undischarged_param[K: Int,
                       # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                       X: PosForParam[K]]():
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function argument position
##===----------------------------------------------------------------------===##


struct PosForArg[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > 0)'}}
    where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(K > 0)'}}
def undischarged_arg[K: Int](
        # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
        x: PosForArg[K]):
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function return type position
##===----------------------------------------------------------------------===##


struct PosForRet[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > 0)'}}
    where N > 0:
    pass


# expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
# expected-note @below {{add a trailing 'where' clause that requires '(K > 0)'}}
def undischarged_ret[K: Int]() -> PosForRet[K]:
    pass


##===----------------------------------------------------------------------===##
# Unprovable in function `raises T` thrown type position
##===----------------------------------------------------------------------===##


struct PosForThrows[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > 0)'}}
    where N > 0:
    pass


# expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
# expected-note @below {{add a trailing 'where' clause that requires '(K > 0)'}}
def undischarged_throws[K: Int]() raises PosForThrows[K]:
    pass


##===----------------------------------------------------------------------===##
# Unprovable in struct parameter declaration position
##===----------------------------------------------------------------------===##


struct PosForStruct[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(K > 0)'}}
    where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(K > 0)'}}
struct UndischargedStruct[K: Int,
                          # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                          X: PosForStruct[K]]:
    pass


##===----------------------------------------------------------------------===##
# Multiple deferrals — partial discharge
##===----------------------------------------------------------------------===##
# Each deferred constraint is re-checked independently; the discharge loop
# reports an error per still-unprovable constraint while silently
# discharging the rest. Here only `A > 0` is in the where clause, so the
# `B > 0` constraint from `PosForPartial[B]` remains unprovable.


struct PosForPartial[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(B > 0)'}}
    where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(B > 0)'}}
def partial_discharge[A: Int, B: Int, X: PosForPartial[A],
                      # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                      Y: PosForPartial[B]]() where A > 0:
    pass


##===----------------------------------------------------------------------===##
# Unprovable trait conformance — function parameter declaration
##===----------------------------------------------------------------------===##


struct IntableForParam[T: AnyType]
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, AnyType & ImplicitlyDeletable & Intable)'}}
    where conforms_to(T, Intable):
    pass


# expected-note @below {{add a trailing 'where' clause that requires 'conforms_to(T, AnyType & ImplicitlyDeletable & Intable)'}}
def undischarged_trait_param[T: AnyType,
                             # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                             X: IntableForParam[T]]():
    pass


##===----------------------------------------------------------------------===##
# Unprovable trait conformance — function argument
##===----------------------------------------------------------------------===##


struct IntableForArg[T: AnyType]
    # expected-note @below {{constraint declared here needs evidence for 'conforms_to(T, AnyType & ImplicitlyDeletable & Intable)'}}
    where conforms_to(T, Intable):
    pass


# expected-note @below {{add a trailing 'where' clause that requires 'conforms_to(T, AnyType & ImplicitlyDeletable & Intable)'}}
def undischarged_trait_arg[T: AnyType](
        # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
        x: IntableForArg[T]):
    pass


##===----------------------------------------------------------------------===##
# Multiple deferrals — none discharged
##===----------------------------------------------------------------------===##
# When no trailing `where` clause is present, every deferred body constraint
# remains unprovable and the discharge loop emits one error per constraint.


struct PosForNoneA[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(A > 0)'}}
    where N > 0:
    pass


struct PosForNoneB[N: Int]
    # expected-note @below {{constraint declared here needs evidence for '(B > 0)'}}
    where N > 0:
    pass


# expected-note @below {{add a trailing 'where' clause that requires '(A > 0)'}}
# expected-note @below {{add a trailing 'where' clause that requires '(B > 0)'}}
def all_undischarged[A: Int,
                     # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                     X: PosForNoneA[A],
                     B: Int,
                     # expected-error @below {{invalid bindings in signature: lacking evidence to prove correctness}}
                     Y: PosForNoneB[B]]():
    pass


##===----------------------------------------------------------------------===##
# Multi-candidate deferral rejection
##===----------------------------------------------------------------------===##
# When the deferral sink is installed but overload resolution leaves more
# than one body-constraint-inconclusive candidate, deferral is rejected
# (we cannot commit to a single candidate). The normal inconclusiveness
# error fires, plus an extra note explaining why deferral did not apply.


# expected-note @below {{cannot prove constraint for candidate}}
def mc_target[K: Int](dummy: Int) -> Int
    # expected-note @below {{constraint declared here}}
    where K > 0:
    return dummy


# expected-note @below {{cannot prove constraint for candidate}}
def mc_target[K: Int](dummy: Int) -> Int
    # expected-note @below {{constraint declared here}}
    where K < 100:
    return dummy


struct MCReceiver[i: Int]:
    pass


def multi_cand_deferral_rejected[K: Int](
    # expected-error @below {{ambiguous call to 'mc_target': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    # expected-note @below {{body constraints cannot be deferred because more than one candidate is inconclusive}}
    x: MCReceiver[mc_target[K](0)]):
    pass
