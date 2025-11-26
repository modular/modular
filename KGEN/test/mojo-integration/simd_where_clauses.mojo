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


@always_inline("builtin")
fn simd_where_clause_int_gt[x: Int]() -> Int where SIMDInt(x) > SIMDInt(4):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_gt[x: Int]() -> Int where SIMDInt(x) <= SIMDInt(4):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_ge[x: Int]() -> Int where SIMDInt(x) >= SIMDInt(4):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_ge[x: Int]() -> Int where SIMDInt(x) < SIMDInt(4):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_lt[x: Int]() -> Int where SIMDInt(x) < SIMDInt(4):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_lt[x: Int]() -> Int where SIMDInt(x) >= SIMDInt(4):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_le[x: Int]() -> Int where SIMDInt(x) <= SIMDInt(4):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_le[x: Int]() -> Int where SIMDInt(x) > SIMDInt(4):
    return 0


# CHECK-LABEL: lit.fn @"use_them
fn use_them():
    # CHECK: lit.alias.decl *"x`": !Int = <{0}>
    comptime x = simd_where_clause_bool_default_init()

    # CHECK: lit.alias.decl *"y`1": !Int = <{1}>
    comptime y = simd_where_clause_bool_int_init[1]()
    # CHECK: lit.alias.decl *"z`2": !Int = <{0}>
    comptime z = simd_where_clause_bool_int_init[0]()

    # CHECK: lit.alias.decl *"a`3": !Int = <{1}>
    comptime a = simd_where_clause_int_gt[9]()
    # CHECK: lit.alias.decl *"b`4": !Int = <{0}>
    comptime b = simd_where_clause_int_gt[4]()

    # CHECK: lit.alias.decl *"c`5": !Int = <{1}>
    comptime c = simd_where_clause_int_ge[4]()
    # CHECK: lit.alias.decl *"d`6": !Int = <{0}>
    comptime d = simd_where_clause_int_ge[-1]()

    # CHECK: lit.alias.decl *"e`7": !Int = <{1}>
    comptime e = simd_where_clause_int_lt[-1]()
    # CHECK: lit.alias.decl *"f`8": !Int = <{0}>
    comptime f = simd_where_clause_int_lt[4]()

    # CHECK: lit.alias.decl *"g`9": !Int = <{1}>
    comptime g = simd_where_clause_int_le[4]()
    # CHECK: lit.alias.decl *"h`10": !Int = <{0}>
    comptime h = simd_where_clause_int_le[5]()
