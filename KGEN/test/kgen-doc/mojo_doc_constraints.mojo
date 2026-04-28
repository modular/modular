# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Tests for where clause (constraints) documentation generation."""

# RUN: kgen-doc %s | FileCheck %s

# Check that no diagnostics are output:
# RUN: kgen-doc %s 2>&1 | FileCheck %s --allow-empty --check-prefix CHECK-DIAG
# CHECK-DIAG-NOT: warning

##===----------------------------------------------------------------------===##
# Helper predicates for constraint testing
##===----------------------------------------------------------------------===##


def is_positive(x: Int) -> Bool:
    return x > 0


def is_even(x: Int) -> Bool:
    return x % 2 == 0


def complex_pred(a: Int, b: Int) -> Bool:
    return a + b > 0


##===----------------------------------------------------------------------===##
# Function-level where clauses
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_single_where_clause",
# CHECK: "signature": "fn_with_single_where_clause[N: Int]() where is_positive(N)"
def fn_with_single_where_clause[N: Int]() where is_positive(N):
    """Function with a single where clause constraint.

    Parameters:
        N: A positive integer parameter.
    """
    pass


# CHECK-LABEL: "name": "fn_with_compound_where_clause",
# CHECK: "signature": "fn_with_compound_where_clause[N: Int, M: Int]() where is_positive(N) and is_even(M)"
def fn_with_compound_where_clause[
    N: Int, M: Int
]() where is_positive(N) and is_even(M):
    """Function with compound where clause using 'and'.

    Parameters:
        N: A positive integer parameter.
        M: An even integer parameter.
    """
    pass


# CHECK-LABEL: "name": "fn_with_where_and_args",
# CHECK: "signature": "fn_with_where_and_args[N: Int](value: Int) where is_positive(N)"
def fn_with_where_and_args[N: Int](value: Int) where is_positive(N):
    """Function with where clause and regular arguments.

    Parameters:
        N: A positive integer parameter.

    Args:
        value: An integer value.
    """
    pass


# CHECK-LABEL: "name": "fn_with_where_and_return",
# CHECK: "signature": "fn_with_where_and_return[N: Int]() -> Int where is_positive(N)"
def fn_with_where_and_return[N: Int]() -> Int where is_positive(N):
    """Function with where clause and return type.

    Parameters:
        N: A positive integer parameter.

    Returns:
        An integer value.
    """
    return N


##===----------------------------------------------------------------------===##
# Parameter-level constraints (inline where clauses)
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_parameter_constraint",
# CHECK: "constraints": " where is_positive(N)"
# CHECK: "name": "N",
# CHECK: "signature": "fn_with_parameter_constraint[N: Int where is_positive(N)]()"
def fn_with_parameter_constraint[N: Int where is_positive(N)]():
    """Function with parameter-level constraint.

    Parameters:
        N: A positive integer parameter with inline constraint.
    """
    pass


# CHECK-LABEL: "name": "fn_with_multiple_param_constraints",
# CHECK: "constraints": " where is_positive(A)"
# CHECK: "name": "A",
# CHECK: "constraints": " where is_even(B)"
# CHECK: "name": "B",
# CHECK: "signature": "fn_with_multiple_param_constraints[A: Int where is_positive(A), B: Int where is_even(B)]()"
def fn_with_multiple_param_constraints[
    A: Int where is_positive(A), B: Int where is_even(B)
]():
    """Function with multiple parameter-level constraints.

    Parameters:
        A: A positive integer.
        B: An even integer.
    """
    pass


##===----------------------------------------------------------------------===##
# Complex constraint expressions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_multi_param_pred",
# CHECK: "signature": "fn_with_multi_param_pred[A: Int, B: Int]() where complex_pred(A, B)"
def fn_with_multi_param_pred[A: Int, B: Int]() where complex_pred(A, B):
    """Function with a constraint that uses multiple parameters.

    Parameters:
        A: First integer.
        B: Second integer.
    """
    pass


# CHECK-LABEL: "name": "fn_with_or_constraint",
# CHECK: "signature": "fn_with_or_constraint[N: Int]() where is_positive(N) or is_even(N)"
def fn_with_or_constraint[N: Int]() where is_positive(N) or is_even(N):
    """Function with an 'or' constraint.

    Parameters:
        N: An integer that is either positive or even.
    """
    pass


##===----------------------------------------------------------------------===##
# Ternary 'if' vs recovered 'and'/'or'
# Verifies that a genuine ternary 'if' expression (where all three operands
# differ) is preserved, while lowered 'and'/'or' patterns are reconstructed.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_ternary_if_constraint",
# CHECK: "signature": "fn_with_ternary_if_constraint[N: Int, M: Int]() where is_even(M) if is_positive(N) else is_even(N)"
def fn_with_ternary_if_constraint[
    N: Int, M: Int
]() where is_even(M) if is_positive(N) else is_even(N):
    """Function with a genuine ternary 'if' in the where clause.

    The condition, then-branch, and else-branch are all distinct, so this
    must NOT be collapsed into 'and' or 'or'.

    Parameters:
        N: Controls which branch is taken.
        M: Used in the then-branch constraint.
    """
    pass


# CHECK-LABEL: "name": "fn_with_recovered_and",
# CHECK: "signature": "fn_with_recovered_and[N: Int, M: Int]() where is_positive(N) and is_even(M)"
def fn_with_recovered_and[N: Int, M: Int]() where is_positive(N) and is_even(M):
    """Function whose 'and' is lowered to a ternary by the compiler.

    The doc printer must reconstruct 'and' from the lowered form.

    Parameters:
        N: A positive integer parameter.
        M: An even integer parameter.
    """
    pass


##===----------------------------------------------------------------------===##
# Default values with constraints
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_default_and_constraint",
# CHECK: "default": "10",
# CHECK: "name": "N",
# CHECK: "signature": "fn_with_default_and_constraint[N: Int = 10]() where is_positive(N)"
def fn_with_default_and_constraint[N: Int = 10]() where is_positive(N):
    """Function with default parameter value and where clause.

    Parameters:
        N: A positive integer with default value 10.
    """
    pass


# CHECK-LABEL: "name": "fn_with_default_and_param_constraint",
# CHECK: "constraints": " where is_even(M)"
# CHECK: "default": "4",
# CHECK: "name": "M",
# CHECK: "signature": "fn_with_default_and_param_constraint[M: Int where is_even(M) = 4]()"
def fn_with_default_and_param_constraint[M: Int where is_even(M) = 4]():
    """Function with default value and parameter-level constraint.

    Parameters:
        M: An even integer with default value 4.
    """
    pass


##===----------------------------------------------------------------------===##
# Binary operator sugar in constraints
# Tests that binary dunder calls in where clauses are printed using their
# operator syntax (via the existing binaryOpNames map in ASTPrinter.cpp).
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_eq_constraint",
# CHECK: "signature": "fn_with_eq_constraint[N: Int]() where (N == 0)"
def fn_with_eq_constraint[N: Int]() where N == 0:
    """Function with an equality constraint.

    Parameters:
        N: An integer parameter constrained to be zero.
    """
    pass


# CHECK-LABEL: "name": "fn_with_ne_constraint",
# CHECK: "signature": "fn_with_ne_constraint[N: Int]() where (N != 0)"
def fn_with_ne_constraint[N: Int]() where N != 0:
    """Function with an inequality constraint.

    Parameters:
        N: An integer parameter constrained to be nonzero.
    """
    pass


# CHECK-LABEL: "name": "fn_with_lt_constraint",
# CHECK: "signature": "fn_with_lt_constraint[N: Int]() where (N < 10)"
def fn_with_lt_constraint[N: Int]() where N < 10:
    """Function with a less-than constraint.

    Parameters:
        N: An integer parameter constrained to be less than 10.
    """
    pass


##===----------------------------------------------------------------------===##
# Unary operator sugar in constraints
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_not_constraint",
# CHECK: "signature": "fn_with_not_constraint[x: Bool]() where not x"
def fn_with_not_constraint[x: Bool]() where not x:
    """Function with a 'not' constraint.

    Parameters:
        x: A boolean parameter constrained to be false.
    """
    pass


# CHECK-LABEL: "name": "fn_with_neg_constraint",
# CHECK: "signature": "fn_with_neg_constraint[N: Int]() where (-N == 1)"
def fn_with_neg_constraint[N: Int]() where -N == 1:
    """Function with a negation constraint.

    Parameters:
        N: An integer parameter constrained so its negation equals one.
    """
    pass


##===----------------------------------------------------------------------===##
# Combined function and parameter constraints
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_both_constraint_types",
# CHECK: "constraints": " where is_positive(X)"
# CHECK: "name": "X",
# CHECK: "signature": "fn_with_both_constraint_types[X: Int where is_positive(X), Y: Int]() where is_even(Y)"
def fn_with_both_constraint_types[
    X: Int where is_positive(X), Y: Int
]() where is_even(Y):
    """Function with both parameter-level and function-level constraints.

    Parameters:
        X: A positive integer (parameter-level constraint).
        Y: An even integer (function-level constraint).
    """
    pass


##===----------------------------------------------------------------------===##
# Identity type reconstruction simplification
# Tests that when a constraint references a computed property of a function
# argument (e.g. arg.property), the output uses the argument name instead of
# re-expanding the full parameterized type.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_identity_reconstruction",
# CHECK: "signature": "fn_with_identity_reconstruction(c: Container) where (c.inner_size == 2)"
def fn_with_identity_reconstruction(c: Container) where c.inner_size == 2:
    """Function where a constraint references a computed property of an argument.

    The constraint 'c.inner_size' should be printed using the argument name 'c'
    rather than re-expanding the full Container type with all its parameters.

    Args:
        c: A container with inner_size constrained to 2.
    """
    pass


# Negative test: AltContainer has different declared param names than Container,
# so AltContainer[c.T] should NOT simplify to just 'c'.
# CHECK-LABEL: "name": "fn_no_false_positive_different_struct",
# CHECK: "signature": "fn_no_false_positive_different_struct(c: Container) where (AltContainer[c.T].alt_size == 4)"
def fn_no_false_positive_different_struct(
    c: Container,
) where AltContainer[c.T].alt_size == 4:
    """Constraint references a DIFFERENT struct built from the same auto-params.

    AltContainer[c.T] is NOT an identity reconstruction of c (which is a
    Container), so the output must NOT simplify to 'c.alt_size'.

    Args:
        c: A container argument.
    """
    pass


##===----------------------------------------------------------------------===##
# Conditional (if/else) constraints — _mlir_value must not appear
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "name": "fn_with_cond_where",
# CHECK: "constraints": " where (C <= B) if (A <= B) else (C <= 0)"
# CHECK: "name": "C",
# CHECK: "signature": "fn_with_cond_where[A: Int, B: Int, C: Int where (C <= B) if (A <= B) else (C <= 0)]()"
def fn_with_cond_where[
    A: Int, B: Int, C: Int where (C <= B) if (A <= B) else (C <= 0)
]():
    """Function with a conditional inline constraint.

    Parameters:
        A: First integer.
        B: Upper bound.
        C: An integer constrained to be <= B when A <= B, else <= 0.
    """
    pass


##===----------------------------------------------------------------------===##
# Trait conformance constraints (conforms_to)
# Tests that simple conforms_to(T, Trait) constraints are merged into the
# parameter's type bounds (e.g. T: Serializable & Printable) rather than
# shown as a separate where clause. Associated type paths (e.g. T.Element)
# are non-mergeable and remain as where clauses with unqualified trait names.
# Note: Module-path stripping (e.g. std::builtin::bool::Boolable -> Boolable)
# cannot be tested here because locally-defined traits have no module prefix.
# That code path is validated via stdlib APIs (all, any, take_while, drop_while).
##===----------------------------------------------------------------------===##


trait Serializable:
    def serialize(self) -> Int:
        ...


trait Printable:
    def print_it(self):
        ...


# CHECK-LABEL: "name": "fn_with_single_trait_conformance",
# CHECK: "signature": "fn_with_single_trait_conformance[T: Serializable & Printable]()"
def fn_with_single_trait_conformance[
    T: Serializable
]() where conforms_to(T, Printable):
    """Function with a single trait conformance constraint.

    Parameters:
        T: A type that must conform to both Serializable and Printable.
    """
    pass


# CHECK-LABEL: "name": "fn_with_compound_trait_conformance",
# CHECK: "signature": "fn_with_compound_trait_conformance[T: Serializable & Printable & Serializable]()"
def fn_with_compound_trait_conformance[
    T: Serializable
]() where conforms_to(T, Serializable & Printable):
    """Function with a compound trait conformance constraint.

    Parameters:
        T: A type that must conform to both Serializable and Printable.
    """
    pass


# CHECK-LABEL: "name": "fn_with_param_trait_constraint",
# CHECK: "signature": "fn_with_param_trait_constraint[T: Serializable & Printable]()"
def fn_with_param_trait_constraint[
    T: Serializable where conforms_to(T, Printable)
]():
    """Function with parameter-level trait conformance constraint.

    Parameters:
        T: A type that must conform to both Serializable and Printable.
    """
    pass


##===----------------------------------------------------------------------===##
# Struct with method-level constraints
# Note: In JSON output, struct fields are alphabetical, so 'functions' comes
# before 'name' and 'signature'.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: "structs":
# CHECK: "functions":
# CHECK: "name": "method_with_where",
# CHECK: "signature": "method_with_where[M: Int](self) where is_positive(M)"
# CHECK: "name": "method_with_param_constraint",
# CHECK: "constraints": " where is_even(K)"
# CHECK: "signature": "method_with_param_constraint[K: Int where is_even(K)](self)"
# CHECK: "name": "MethodConstraints"
# CHECK: "signature": "struct MethodConstraints"
@fieldwise_init
struct MethodConstraints:
    """A struct to test method-level constraints."""

    def method_with_where[M: Int](self) where is_positive(M):
        """Method with a where clause.

        Parameters:
            M: A positive integer parameter.
        """
        pass

    def method_with_param_constraint[K: Int where is_even(K)](self):
        """Method with parameter-level constraint.

        Parameters:
            K: An even integer parameter.
        """
        pass


##===----------------------------------------------------------------------===##
# Struct with parameter constraints
##===----------------------------------------------------------------------===##


# CHECK: "name": "StructWithParamConstraint",
# CHECK: "constraints": " where is_positive(N)"
# CHECK: "name": "N",
# CHECK: "signature": "struct StructWithParamConstraint[N: Int where is_positive(N)]"
@fieldwise_init
struct StructWithParamConstraint[N: Int where is_positive(N)]:
    """A struct with a parameter-level constraint.

    Parameters:
        N: A positive integer with inline constraint.
    """

    pass


##===----------------------------------------------------------------------===##
# Struct with multi-parameter constraints where a later parameter's constraint
# references an earlier parameter. Verifies that constraint strings and
# signatures are generated correctly for dependent parameter constraints.
##===----------------------------------------------------------------------===##


# CHECK: "name": "StructWithDependentParamConstraint",
# CHECK: "constraints": " where is_positive(N)"
# CHECK: "name": "M",
# CHECK: "signature": "struct StructWithDependentParamConstraint[N: Int, M: Int where is_positive(N)]"
@fieldwise_init
struct StructWithDependentParamConstraint[N: Int, M: Int where is_positive(N)]:
    """A struct where a later parameter's constraint references an earlier one.

    Parameters:
        N: An unconstrained integer.
        M: An integer whose constraint references N.
    """

    pass


# CHECK: "name": "StructWithMultiParamPredConstraint",
# CHECK: "constraints": " where complex_pred(N, M)"
# CHECK: "name": "M",
# CHECK: "signature": "struct StructWithMultiParamPredConstraint[N: Int, M: Int where complex_pred(N, M)]"
@fieldwise_init
struct StructWithMultiParamPredConstraint[
    N: Int, M: Int where complex_pred(N, M)
]:
    """A struct where a parameter's constraint uses multiple sibling parameters.

    Parameters:
        N: First integer parameter.
        M: Second integer, constrained by a predicate involving both N and M.
    """

    pass


##===----------------------------------------------------------------------===##
# Helper types for identity type reconstruction tests (defined after use
# because kgen-doc processes all declarations in a module).
##===----------------------------------------------------------------------===##


trait HasSize(TrivialRegisterPassable):
    comptime size: Int


struct Container[T: HasSize](TrivialRegisterPassable):
    """A generic container parameterized on a sized type."""

    comptime inner_size = Self.T.size


struct AltContainer[U: HasSize](TrivialRegisterPassable):
    """A different single-param container (param name 'U', not 'T')."""

    comptime alt_size = Self.U.size * 2
