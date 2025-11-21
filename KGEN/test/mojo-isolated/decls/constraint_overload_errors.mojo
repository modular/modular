# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Constraint Overload Error Reporting Tests
# Tests for error messages when overload resolution fails due to unprovable
# or inconclusive "where" constraints.
##===----------------------------------------------------------------------===##

# Helper function that represent unprovable constraints since they are not
# always_inline("builtin").
# expected-note @below {{cannot evaluate call to non-builtin function declared here}}
fn is_prime(x: Int) -> Bool:
    return x > 1

# expected-note @below {{cannot evaluate call to non-builtin function declared here}}
fn is_square(x: Int) -> Bool:
    return x > 0

##===----------------------------------------------------------------------===##
# Inconclusive Param Constraints - Single Candidate
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
fn single_param_constraint[
    # expected-note-re @below {{constraint declared here needs evidence for 'is_prime({{[0-9]+}})'}}
    x: Int where is_prime(x),
]():
    pass

fn test_single_param_constraint():
    # expected-error @below {{invalid call to 'single_param_constraint': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    single_param_constraint[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Param Constraints - Multi Candidate, One Inconclusive
##===----------------------------------------------------------------------===##

@always_inline("builtin")
fn is_natural_number(x: Int) -> Bool:
    return x >= 0

# expected-note @below {{cannot prove constraint for candidate}}
fn multi_param_one_inconclusive[
    # expected-note @below {{constraint declared here}}
    x: Int where is_prime(x),
]():
    pass

fn multi_param_one_inconclusive[
    x: Int where is_natural_number(x),
]():
    pass

fn test_multi_param_one_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_param_one_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_param_one_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Param Constraints - Multi Candidate, Multi Inconclusive
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
fn multi_param_multi_inconclusive[
    # expected-note @below {{constraint declared here}}
    x: Int where is_prime(x),
]():
    pass

# expected-note @below {{cannot prove constraint for candidate}}
fn multi_param_multi_inconclusive[
    # expected-note @below {{constraint declared here}}
    x: Int where is_square(x),
]():
    pass

fn test_multi_param_multi_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_param_multi_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_param_multi_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Fn Constraints - Single Candidate
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
fn single_fn_constraint[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

fn test_single_fn_constraint():
    # expected-error @below {{invalid call to 'single_fn_constraint': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    single_fn_constraint[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Fn Constraints - Multi Candidate, One Inconclusive, Equal Fitness
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
fn multi_fn_one_inconclusive[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

# expected-note @below {{candidate is valid but cannot be selected until other candidates are disproved}}
fn multi_fn_one_inconclusive[x: Int]()
    where x >= 0:
    pass

fn test_multi_fn_one_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_fn_one_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_fn_one_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Fn Constraints - Multi Candidate, Multi Inconclusive, Equal Fitness
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
fn multi_fn_multi_inconclusive[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

# expected-note @below {{cannot prove constraint for candidate}}
fn multi_fn_multi_inconclusive[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_square(x):
    pass

fn test_multi_fn_multi_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_fn_multi_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_fn_multi_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Fn Constraints - Multi Candidate, Best Fitness Selects Non-Inconclusive
##===----------------------------------------------------------------------===##

@fieldwise_init
struct SourceStruct:
    pass

struct TargetStruct:
    @implicit
    fn __init__(out self, s: SourceStruct):
        pass

# This candidate will never work, even if we prove the inconclusive constraint.
# It should be skipped in favor of the better candidate.
fn multi_fn_best_fitness_ok[x: Int](s: TargetStruct)
    where is_prime(x):
    pass

# This candidate is selected due to better fitness (IntLiteral is better match).
fn multi_fn_best_fitness_ok[x: Int](s: SourceStruct):
    pass

fn test_multi_fn_best_fitness_ok():
    # No error expected - the IntLiteral overload should be selected
    multi_fn_best_fitness_ok[2](SourceStruct())


##===----------------------------------------------------------------------===##
# Inconclusive Fn Constraints - Multi Candidate, Inconclusive Has Best Fitness
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
fn multi_fn_best_fitness_inconclusive[x: IntLiteral]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

fn multi_fn_best_fitness_inconclusive[x: Int]()
    where x >= 0:
    pass

fn test_multi_fn_best_fitness_inconclusive():
    # expected-error @below {{invalid call to 'multi_fn_best_fitness_inconclusive': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    multi_fn_best_fitness_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Mixed Param and Fn Constraints
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
fn mixed_constraints[
    # expected-note @below {{constraint declared here}}
    x: Int where is_prime(x),
    y: Int
]()
    # No errors expected here since the earlier param constraint failed.
    where y > 0:
    pass

fn mixed_constraints[
    x: Int where x >= 0,
    y: Int
]()
    where y >= 0:
    pass

fn test_mixed_constraints():
    # expected-error @below {{ambiguous call to 'mixed_constraints': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    mixed_constraints[2, 1]()


##===----------------------------------------------------------------------===##
# Reference to Overloaded Declaration (not a call)
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
fn ref_to_overload[
    # expected-note @below {{constraint declared here}}
    x: Int where is_prime(x),
]():
    pass

fn ref_to_overload[
    x: Int where x >= 0
]():
    pass

fn test_ref_to_overload():
    # expected-error @below {{ambiguous reference to 'ref_to_overload': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    ref_to_overload[2]


##===----------------------------------------------------------------------===##
# Multiple Inconclusive Constraints on Same Candidate
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
fn multiple_inconclusive_same[
    # expected-note @below {{constraint declared here}}
    x: Int where is_prime(x),
    y: Int where is_square(y),
]():
    pass

fn test_multiple_inconclusive_same():
    # expected-error @below {{invalid call to 'multiple_inconclusive_same': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    multiple_inconclusive_same[2, 4]()


##===----------------------------------------------------------------------===##
# Combination of Provable and Inconclusive Constraints
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
fn provable_and_inconclusive[
    x: Int where x > 0,  # This is provable with literal 2
    # expected-note @below {{constraint declared here}}
    y: Int where is_prime(y),  # This is not provable
]():
    pass

fn provable_and_inconclusive[
    x: Int where x < 0,
]():
    pass

fn test_provable_and_inconclusive():
    # expected-error @below {{invalid call to 'provable_and_inconclusive': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraints here to aid in candidate selection}}
    provable_and_inconclusive[2, 2]()


##===----------------------------------------------------------------------===##
# Violated vs Inconclusive Constraints
##===----------------------------------------------------------------------===##

# This candidate has a violated constraint (should be rejected)
fn violated_vs_inconclusive[
    x: Int where x > 10,  # Violated by x=2
]():
    pass

# This candidate has an inconclusive constraint
# expected-note @below {{cannot prove constraint}}
fn violated_vs_inconclusive[
    # expected-note @below {{constraint declared here}}
    x: Int where is_prime(x),  # Inconclusive for x=2
]():
    pass

fn test_violated_vs_inconclusive():
    # The first candidate should be eliminated due to violated constraint,
    # leaving only the inconclusive one which should error
    # expected-error @below {{invalid call to 'violated_vs_inconclusive': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraints here to aid in candidate selection}}
    violated_vs_inconclusive[2]()
