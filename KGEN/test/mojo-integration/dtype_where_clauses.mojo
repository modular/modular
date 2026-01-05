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
fn dtype_where_clause_is_signed[d: DType]() -> Int where d.is_signed():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_signed[d: DType]() -> Int where not d.is_signed():
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_unsigned[d: DType]() -> Int where d.is_unsigned():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_unsigned[d: DType]() -> Int where not d.is_unsigned():
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_integral[d: DType]() -> Int where d.is_integral():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_integral[d: DType]() -> Int where not d.is_integral():
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_floating_point[
    d: DType
]() -> Int where d.is_floating_point():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_floating_point[
    d: DType
]() -> Int where not d.is_floating_point():
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_half_float[d: DType]() -> Int where d.is_half_float():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_half_float[
    d: DType
]() -> Int where not d.is_half_float():
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_float8[d: DType]() -> Int where d.is_float8():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_float8[d: DType]() -> Int where not d.is_float8():
    return 0


@always_inline("builtin")
fn dtype_where_clause_is_numeric[d: DType]() -> Int where d.is_numeric():
    return 1


@always_inline("builtin")
fn dtype_where_clause_is_numeric[d: DType]() -> Int where not d.is_numeric():
    return 0


# CHECK-LABEL: lit.fn @"foo
fn foo():
    # CHECK: lit.alias.decl *"x`": !Int = <{42}>
    comptime x = dtype_where_clause_eq_ne[DType.int32]()
    # CHECK: lit.alias.decl *"y`1": !Int = <{0}>
    comptime y = dtype_where_clause_eq_ne[DType.int64]()

    # CHECK: lit.alias.decl *"c`{{.*}}": !Int = <{1}>
    comptime c = dtype_where_clause_is_signed[DType.int32]()
    # CHECK: lit.alias.decl *"d`{{.*}}": !Int = <{0}>
    comptime d = dtype_where_clause_is_signed[DType.uint32]()
    # CHECK: lit.alias.decl *"e`{{.*}}": !Int = <{1}>
    comptime e = dtype_where_clause_is_signed[DType.float64]()

    # CHECK: lit.alias.decl *"f`{{.*}}": !Int = <{1}>
    comptime f = dtype_where_clause_is_unsigned[DType.uint32]()
    # CHECK: lit.alias.decl *"g`{{.*}}": !Int = <{0}>
    comptime g = dtype_where_clause_is_unsigned[DType.int32]()
    # CHECK: lit.alias.decl *"h`{{.*}}": !Int = <{0}>
    comptime h = dtype_where_clause_is_unsigned[DType.float64]()

    # CHECK: lit.alias.decl *"i`{{.*}}": !Int = <{1}>
    comptime i = dtype_where_clause_is_integral[DType.uint32]()
    # CHECK: lit.alias.decl *"j`{{.*}}": !Int = <{1}>
    comptime j = dtype_where_clause_is_integral[DType.int32]()
    # CHECK: lit.alias.decl *"k`{{.*}}": !Int = <{0}>
    comptime k = dtype_where_clause_is_integral[DType.float64]()

    # CHECK: lit.alias.decl *"l`{{.*}}": !Int = <{0}>
    comptime l = dtype_where_clause_is_floating_point[DType.int32]()
    # CHECK: lit.alias.decl *"m`{{.*}}": !Int = <{1}>
    comptime m = dtype_where_clause_is_floating_point[DType.float32]()

    # CHECK: lit.alias.decl *"n`{{.*}}": !Int = <{0}>
    comptime n = dtype_where_clause_is_half_float[DType.int32]()
    # CHECK: lit.alias.decl *"o`{{.*}}": !Int = <{0}>
    comptime o = dtype_where_clause_is_half_float[DType.float32]()
    # CHECK: lit.alias.decl *"p`{{.*}}": !Int = <{1}>
    comptime p = dtype_where_clause_is_half_float[DType.float16]()
    # CHECK: lit.alias.decl *"q`{{.*}}": !Int = <{1}>
    comptime q = dtype_where_clause_is_half_float[DType.bfloat16]()

    # CHECK: lit.alias.decl *"r0`{{.*}}": !Int = <{1}>
    comptime r0 = dtype_where_clause_is_float8[DType.float8_e3m4]()
    # CHECK: lit.alias.decl *"r1`{{.*}}": !Int = <{1}>
    comptime r1 = dtype_where_clause_is_float8[DType.float8_e5m2]()
    # CHECK: lit.alias.decl *"r2`{{.*}}": !Int = <{0}>
    comptime r2 = dtype_where_clause_is_float8[DType.float32]()

    # CHECK: lit.alias.decl *"s0`{{.*}}": !Int = <{1}>
    comptime s0 = dtype_where_clause_is_numeric[DType.int32]()
    # CHECK: lit.alias.decl *"s1`{{.*}}": !Int = <{1}>
    comptime s1 = dtype_where_clause_is_numeric[DType.float32]()
    # CHECK: lit.alias.decl *"s2`{{.*}}": !Int = <{0}>
    comptime s2 = dtype_where_clause_is_numeric[DType.bool]()
