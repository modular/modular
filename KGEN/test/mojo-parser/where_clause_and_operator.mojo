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

fn bool_pred(x: Int) -> Bool:
    return True

fn need_bool_pred[x: Int]() where bool_pred(x):
    pass

fn i1_pred(x: Int) -> __mlir_type.i1:
    return __mlir_attr.`1: i1`

fn need_i1_pred[x: Int]() where i1_pred(x):
    pass

# We should see an 'and' operator, instead of 'cond'.
# CHECK-LABEL: lit.fn @"test_and_bool[
# CHECK-SAME: where {<sugar_alias(#lit.struct.extract<:!Bool cond(
# CHECK-SAME: , and(
fn test_and_bool[x: Int, y: Int, z: Int]() where bool_pred(x) and bool_pred(y) and bool_pred(z):
    # These calls should succeed because the compiler can now extract
    # component propositions from the compound `and` expression.
    need_bool_pred[x]()
    need_bool_pred[y]()
    need_bool_pred[z]()

# CHECK-LABEL: lit.fn @"call_test_and_bool()"
fn call_test_and_bool():
    # There should still be a `cond` here.
    # CHECK-NEXT: kgen.param.if <#lit.struct.extract<:!Bool cond
    @parameter
    if bool_pred(1) and bool_pred(2) and bool_pred(3):
        test_and_bool[1, 2, 3]()

fn call_test_and_bool_nested():
    @parameter
    if bool_pred(1):
        @parameter
        if bool_pred(3):
            @parameter
            if bool_pred(2):
                test_and_bool[1, 2, 3]()

# CHECK-LABEL: lit.fn @"test_and_i1[
# CHECK-SAME: where {<sugar_alias(cond(
# CHECK-SAME: , and(
fn test_and_i1[x: Int, y: Int]() where i1_pred(x) and i1_pred(y):
    need_i1_pred[x]()
    need_i1_pred[y]()

# CHECK-LABEL: lit.fn @"test_and_i1_bool[
# CHECK-SAME: where {<sugar_alias(#lit.struct.extract<:!Bool cond(
# CHECK-SAME: , and(
fn test_and_i1_bool[x: Int, y: Int]() where i1_pred(x) and bool_pred(y):
    need_i1_pred[x]()
    need_bool_pred[y]()

# CHECK-LABEL: lit.fn @"test_and_bool_i1[
# CHECK-SAME: where {<sugar_alias(#lit.struct.extract<:!Bool cond(
# CHECK-SAME: , and(
fn test_and_bool_i1[x: Int, y: Int]() where bool_pred(x) and i1_pred(y):
    need_bool_pred[x]()
    need_i1_pred[y]()

# Test with `or` operator as well.
# CHECK-LABEL: lit.fn @"test_or_bool[
# CHECK-SAME: where {<sugar_alias(#lit.struct.extract<:!Bool cond(
# CHECK-SAME: , or(
fn test_or_bool[x: Int, y: Int]() where bool_pred(x) or bool_pred(y):
    # With `or`, we can't unconditionally call need_one or need_two
    # but we can call them under appropriate parametric conditions
    @parameter
    if bool_pred(x):
        need_bool_pred[x]()

    @parameter
    if bool_pred(y):
        need_bool_pred[y]()
