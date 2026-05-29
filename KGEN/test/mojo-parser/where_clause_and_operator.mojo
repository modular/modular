# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s | FileCheck %s

##===----------------------------------------------------------------------===##
# Tests that when using `and` in where clauses, the compiler properly
# extracts individual propositions for constraint checking.
##===----------------------------------------------------------------------===##


def bool_pred(x: Int) -> Bool:
    return True


def need_bool_pred[x: Int]() where bool_pred(x):
    pass


def i1_pred(x: Int) -> __mlir_type.i1:
    return __mlir_attr.`1: i1`


def need_i1_pred[x: Int]() where i1_pred(x):
    pass


# We should see an 'and' operator, instead of 'cond'.
# CHECK-LABEL: lit.fn @"test_and_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool
# CHECK-SAME: and(#lit.struct.extract<:!Bool apply(:!lit.generator<("x": !Int) -> !Bool> @where_clause_and_operator::@"bool_pred(::Int)"
def test_and_bool[
    x: Int, y: Int, z: Int
]() where bool_pred(x) and bool_pred(y) and bool_pred(z):
    # These calls should succeed because the compiler can now extract
    # component propositions from the compound `and` expression.
    need_bool_pred[x]()
    need_bool_pred[y]()
    need_bool_pred[z]()


# CHECK-LABEL: lit.fn @"call_test_and_bool()"
def call_test_and_bool():
    # There should still be a `cond` here.
    # CHECK-NEXT: kgen.param.if <to_builtin(:scalar<bool> #lit.struct.extract<:!Bool cond
    comptime if bool_pred(1) and bool_pred(2) and bool_pred(3):
        test_and_bool[1, 2, 3]()


def call_test_and_bool_nested():
    comptime if bool_pred(1):
        comptime if bool_pred(3):
            comptime if bool_pred(2):
                test_and_bool[1, 2, 3]()


# CHECK-LABEL: lit.fn @"test_and_i1[
# CHECK-SAME: {<sugar_preserved(cond(from_builtin(:i1
# CHECK-SAME: and(from_builtin(:i1 apply(:!lit.generator<("x": !Int) -> i1> @where_clause_and_operator::@"i1_pred(::Int)"
def test_and_i1[x: Int, y: Int]() where i1_pred(x) and i1_pred(y):
    need_i1_pred[x]()
    need_i1_pred[y]()


# CHECK-LABEL: lit.fn @"test_and_i1_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool cond(from_builtin(:i1
# CHECK-SAME: and(from_builtin(:i1 apply(:!lit.generator<("x": !Int) -> i1> @where_clause_and_operator::@"i1_pred(::Int)"
def test_and_i1_bool[x: Int, y: Int]() where i1_pred(x) and bool_pred(y):
    need_i1_pred[x]()
    need_bool_pred[y]()


# CHECK-LABEL: lit.fn @"test_and_bool_i1[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool
# CHECK-SAME: and(from_builtin(:i1 apply(:!lit.generator<("x": !Int) -> i1> @where_clause_and_operator::@"i1_pred(::Int)"
def test_and_bool_i1[x: Int, y: Int]() where bool_pred(x) and i1_pred(y):
    need_bool_pred[x]()
    need_i1_pred[y]()


# Test with `or` operator as well.
# CHECK-LABEL: lit.fn @"test_or_bool[
# CHECK-SAME: {<sugar_preserved(#lit.struct.extract<:!Bool
# CHECK-SAME: or(#lit.struct.extract<:!Bool apply(:!lit.generator<("x": !Int) -> !Bool> @where_clause_and_operator::@"bool_pred(::Int)"
def test_or_bool[x: Int, y: Int]() where bool_pred(x) or bool_pred(y):
    # With `or`, we can't unconditionally call need_one or need_two
    # but we can call them under appropriate parametric conditions
    comptime if bool_pred(x):
        need_bool_pred[x]()

    comptime if bool_pred(y):
        need_bool_pred[y]()
