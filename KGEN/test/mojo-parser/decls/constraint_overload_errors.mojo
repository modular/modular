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
def is_prime(x: Int) -> Bool:
    return x > 1

# expected-note @below {{cannot evaluate call to non-builtin function declared here}}
def is_square(x: Int) -> Bool:
    return x > 0

##===----------------------------------------------------------------------===##
# Inconclusive Param Constraints - Single Candidate
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
def single_param_constraint[
    # expected-note-re @below {{constraint declared here needs evidence for 'is_prime({{[0-9]+}})'}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),
]():
    pass

def test_single_param_constraint():
    # expected-error @below {{invalid call to 'single_param_constraint': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    single_param_constraint[2]()


##===----------------------------------------------------------------------===##
# Inconclusive Param Constraints - Multi Candidate, One Inconclusive
##===----------------------------------------------------------------------===##

@always_inline("builtin")
def is_natural_number(x: Int) -> Bool:
    return x >= 0

# expected-note @below {{cannot prove constraint for candidate}}
def multi_param_one_inconclusive[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),
](a: UInt32):
    pass

def multi_param_one_inconclusive[
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_natural_number(x),
](a: Int32):
    pass

def test_multi_param_one_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_param_one_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_param_one_inconclusive[2](1)


##===----------------------------------------------------------------------===##
# Inconclusive Param Constraints - Multi Candidate, Multi Inconclusive
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
def multi_param_multi_inconclusive[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),
](a: Int32):
    pass

# expected-note @below {{cannot prove constraint for candidate}}
def multi_param_multi_inconclusive[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_square(x),
](a: UInt32):
    pass

def test_multi_param_multi_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_param_multi_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_param_multi_inconclusive[2](2)


##===----------------------------------------------------------------------===##
# Inconclusive def Constraints - Single Candidate
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
def single_fn_constraint[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

def test_single_fn_constraint():
    # expected-error @below {{invalid call to 'single_fn_constraint': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    single_fn_constraint[2]()


##===----------------------------------------------------------------------===##
# Inconclusive def Constraints - Multi Candidate, One Inconclusive, Equal Fitness
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
def multi_fn_one_inconclusive[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

# expected-note @below {{candidate is valid but cannot be selected until other candidates are disproved}}
def multi_fn_one_inconclusive[x: Int]()
    where x >= 0:
    pass

def test_multi_fn_one_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_fn_one_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_fn_one_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Inconclusive def Constraints - Multi Candidate, Multi Inconclusive, Equal Fitness
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
def multi_fn_multi_inconclusive[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

# expected-note @below {{cannot prove constraint for candidate}}
def multi_fn_multi_inconclusive[x: Int]()
    # expected-note @below {{constraint declared here}}
    where is_square(x):
    pass

def test_multi_fn_multi_inconclusive():
    # expected-error @below {{ambiguous call to 'multi_fn_multi_inconclusive': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    multi_fn_multi_inconclusive[2]()


##===----------------------------------------------------------------------===##
# Inconclusive def Constraints - Multi Candidate, Best Fitness Selects Non-Inconclusive
##===----------------------------------------------------------------------===##

@fieldwise_init
struct SourceStruct:
    pass

struct TargetStruct:
    @implicit
    def __init__(out self, s: SourceStruct):
        pass

# This candidate will never work, even if we prove the inconclusive constraint.
# It should be skipped in favor of the better candidate.
def multi_fn_best_fitness_ok[x: Int](s: TargetStruct)
    where is_prime(x):
    pass

# This candidate is selected due to better fitness (IntLiteral is better match).
def multi_fn_best_fitness_ok[x: Int](s: SourceStruct):
    pass

def test_multi_fn_best_fitness_ok():
    # No error expected - the IntLiteral overload should be selected
    multi_fn_best_fitness_ok[2](SourceStruct())


##===----------------------------------------------------------------------===##
# Inconclusive def Constraints - Multi Candidate, Inconclusive Has Best Fitness
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
def multi_fn_best_fitness_inconclusive[x: Int](a: IntLiteral)
    # expected-note @below {{constraint declared here}}
    where is_prime(x):
    pass

def multi_fn_best_fitness_inconclusive[x: Int](a: Int)
    where x >= 0:
    pass

def test_multi_fn_best_fitness_inconclusive():
    # expected-error @below {{invalid call to 'multi_fn_best_fitness_inconclusive': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    multi_fn_best_fitness_inconclusive[2](2)


##===----------------------------------------------------------------------===##
# Mixed Param and def Constraints
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
def mixed_constraints[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),
    y: Int
]()
    # No errors expected here since the earlier param constraint failed.
    where y > 0:
    pass

def mixed_constraints[
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where x >= 0,
    y: Int
]()
    where y >= 0:
    pass

def test_mixed_constraints():
    # expected-error @below {{ambiguous call to 'mixed_constraints': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    mixed_constraints[2, 1]()


##===----------------------------------------------------------------------===##
# Reference to Overloaded Declaration (not a call)
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint for candidate}}
def ref_to_overload[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),
](a: Int32):
    pass

def ref_to_overload[
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where x >= 0
](a: UInt32):
    pass

def test_ref_to_overload():
    # expected-error @below {{ambiguous reference to 'ref_to_overload': lacking evidence to select candidate}}
    # expected-note @below {{provide evidence for or against the constraints here to aid in candidate selection}}
    ref_to_overload[2]


##===----------------------------------------------------------------------===##
# Multiple Inconclusive Constraints on Same Candidate
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
def multiple_inconclusive_same[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    y: Int where is_square(y),
]():
    pass

def test_multiple_inconclusive_same():
    # expected-error @below {{invalid call to 'multiple_inconclusive_same': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
    multiple_inconclusive_same[2, 4]()


##===----------------------------------------------------------------------===##
# Combination of Provable and Inconclusive Constraints
##===----------------------------------------------------------------------===##

# expected-note @below {{cannot prove constraint}}
def provable_and_inconclusive[
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where x > 0,  # This is provable with literal 2
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    y: Int where is_prime(y),  # This is not provable
](a: Int32):
    pass

def provable_and_inconclusive[
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where x < 0,
](a: UInt32):
    pass

def test_provable_and_inconclusive():
    # expected-error @below {{invalid call to 'provable_and_inconclusive': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraints here to aid in candidate selection}}
    provable_and_inconclusive[2, 2](4)


##===----------------------------------------------------------------------===##
# Violated vs Inconclusive Constraints
##===----------------------------------------------------------------------===##

# This candidate has a violated constraint (should be rejected)
def violated_vs_inconclusive[
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where x > 10,  # Violated by x=2
](a: Int32):
    pass

# This candidate has an inconclusive constraint
# expected-note @below {{cannot prove constraint}}
def violated_vs_inconclusive[
    # expected-note @below {{constraint declared here}}
    # expected-warning @below {{'where' clauses inside parameter lists are deprecated}}
    # expected-note @below {{use a trailing 'where' clause after the signature instead}}
    x: Int where is_prime(x),  # Inconclusive for x=2
](a: UInt32):
    pass

def test_violated_vs_inconclusive():
    # The first candidate should be eliminated due to violated constraint,
    # leaving only the inconclusive one which should error
    # expected-error @below {{invalid call to 'violated_vs_inconclusive': lacking evidence to prove correctness}}
    # expected-note @below {{provide evidence for the constraints here to aid in candidate selection}}
    violated_vs_inconclusive[2](4)
