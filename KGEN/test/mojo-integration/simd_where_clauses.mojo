# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s \
# RUN:   | kgen-opt --kgen-print-inline-type-values | FileCheck %s


alias SIMDInt = Scalar[DType.int32]


@always_inline("builtin")
fn simd_where_clause_bool_default_init() -> Int where SIMDInt():
    return 1


@always_inline("builtin")
fn simd_where_clause_bool_default_init() -> Int where not SIMDInt():
    return 0


@always_inline("builtin")
fn simd_where_clause_bool_int_init[x: Int]() -> Int where SIMDInt(x):
    return 1


@always_inline("builtin")
fn simd_where_clause_bool_int_init[x: Int]() -> Int where not SIMDInt(x):
    return 0


# CHECK-LABEL: lit.fn @"use_them
fn use_them():
    # CHECK: lit.alias.decl *"x`": !Int = <{0}>
    comptime x = simd_where_clause_bool_default_init()

    # CHECK: lit.alias.decl *"y`1": !Int = <{1}>
    comptime y = simd_where_clause_bool_int_init[1]()
    # CHECK: lit.alias.decl *"z`2": !Int = <{0}>
    comptime z = simd_where_clause_bool_int_init[0]()
