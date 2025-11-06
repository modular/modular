# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s \
# RUN:   | kgen-opt --kgen-print-inline-type-values | FileCheck %s


@always_inline("builtin")
fn dtype_where_clause_eq_ne[d: DType]() -> Int where d == DType.int32:
    return 42


@always_inline("builtin")
fn dtype_where_clause_eq_ne[d: DType]() -> Int where d != DType.int32:
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_isnot[d: DType]() -> Int where d is DType.float32:
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_isnot[d: DType]() -> Int where d is not DType.float32:
    return 2


# CHECK-LABEL: lit.fn @"foo
fn foo():
    # CHECK: lit.alias.decl *"x`": !Int = <{42}>
    alias x = dtype_where_clause_eq_ne[DType.int32]()
    # CHECK: lit.alias.decl *"y`1": !Int = <{0}>
    alias y = dtype_where_clause_eq_ne[DType.int64]()
    # CHECK: lit.alias.decl *"a`2": !Int = <{1}>
    alias a = dtype_where_clause_is_isnot[DType.float32]()
    # CHECK: lit.alias.decl *"b`3": !Int = <{2}>
    alias b = dtype_where_clause_is_isnot[DType.float64]()
