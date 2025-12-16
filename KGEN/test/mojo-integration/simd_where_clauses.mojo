# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate --mojo-enable-prebuilt-packages -import-mojo %s \
# RUN:   | kgen-opt --kgen-print-inline-type-values | FileCheck %s


comptime SIMDInt = Scalar[DType.int32]


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
fn simd_where_clause_int_eq[x: Int]() -> Int where SIMDInt(x) == SIMDInt(4):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_eq[x: Int]() -> Int where SIMDInt(x) != SIMDInt(4):
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


@always_inline("builtin")
fn simd_where_clause_int_add[
    x: Int, y: Int
]() -> Int where SIMDInt(x) + SIMDInt(y):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_add[
    x: Int, y: Int
]() -> Int where not (SIMDInt(x) + SIMDInt(y)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_sub[
    x: Int, y: Int
]() -> Int where SIMDInt(x) - SIMDInt(y):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_sub[
    x: Int, y: Int
]() -> Int where not (SIMDInt(x) - SIMDInt(y)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_mul[
    x: Int, y: Int
]() -> Int where SIMDInt(x) * SIMDInt(y):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_mul[
    x: Int, y: Int
]() -> Int where not (SIMDInt(x) * SIMDInt(y)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_and[
    x: Int, y: Int
]() -> Int where SIMDInt(x) & SIMDInt(y):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_and[
    x: Int, y: Int
]() -> Int where not (SIMDInt(x) & SIMDInt(y)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_xor[
    x: Int, y: Int
]() -> Int where SIMDInt(x) ^ SIMDInt(y):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_xor[
    x: Int, y: Int
]() -> Int where not (SIMDInt(x) ^ SIMDInt(y)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_or[
    x: Int, y: Int
]() -> Int where SIMDInt(x) | SIMDInt(y):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_or[
    x: Int, y: Int
]() -> Int where not (SIMDInt(x) | SIMDInt(y)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_pos[x: Int]() -> Int where +SIMDInt(x):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_pos[x: Int]() -> Int where not (+SIMDInt(x)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_neg[x: Int]() -> Int where -SIMDInt(x):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_neg[x: Int]() -> Int where not (-SIMDInt(x)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_invert[x: Int]() -> Int where ~SIMDInt(x):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_invert[x: Int]() -> Int where not (~SIMDInt(x)):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_shl[
    x: Int
]() -> Int where SIMDInt(x) << SIMDInt(1) == SIMDInt(4):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_shl[
    x: Int
]() -> Int where SIMDInt(x) << SIMDInt(1) != SIMDInt(4):
    return 0


@always_inline("builtin")
fn simd_where_clause_int_shr[
    x: Int
]() -> Int where SIMDInt(x) >> SIMDInt(1) == SIMDInt(3):
    return 1


@always_inline("builtin")
fn simd_where_clause_int_shr[
    x: Int
]() -> Int where SIMDInt(x) >> SIMDInt(1) != SIMDInt(3):
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

    # CHECK: lit.alias.decl *"add0`11": !Int = <{1}>
    comptime add0 = simd_where_clause_int_add[1, 1]()
    # CHECK: lit.alias.decl *"add1`12": !Int = <{0}>
    comptime add1 = simd_where_clause_int_add[1, -1]()

    # CHECK: lit.alias.decl *"sub0`13": !Int = <{1}>
    comptime sub0 = simd_where_clause_int_sub[1, -1]()
    # CHECK: lit.alias.decl *"sub1`14": !Int = <{0}>
    comptime sub1 = simd_where_clause_int_sub[1, 1]()

    # CHECK: lit.alias.decl *"mul0`15": !Int = <{1}>
    comptime mul0 = simd_where_clause_int_mul[1, -1]()
    # CHECK: lit.alias.decl *"mul1`16": !Int = <{0}>
    comptime mul1 = simd_where_clause_int_mul[1, 0]()

    # CHECK: lit.alias.decl *"and0`17": !Int = <{1}>
    comptime and0 = simd_where_clause_int_and[1, 3]()
    # CHECK: lit.alias.decl *"and1`18": !Int = <{0}>
    comptime and1 = simd_where_clause_int_and[1, 2]()

    # CHECK: lit.alias.decl *"xor0`19": !Int = <{1}>
    comptime xor0 = simd_where_clause_int_xor[1, 3]()
    # CHECK: lit.alias.decl *"xor1`20": !Int = <{0}>
    comptime xor1 = simd_where_clause_int_xor[1, 1]()

    # CHECK: lit.alias.decl *"or0`21": !Int = <{1}>
    comptime or0 = simd_where_clause_int_or[1, 3]()
    # CHECK: lit.alias.decl *"or1`22": !Int = <{0}>
    comptime or1 = simd_where_clause_int_or[0, 0]()

    # CHECK: lit.alias.decl *"pos0`23": !Int = <{1}>
    comptime pos0 = simd_where_clause_int_pos[1]()
    # CHECK: lit.alias.decl *"pos1`24": !Int = <{0}>
    comptime pos1 = simd_where_clause_int_pos[0]()

    # CHECK: lit.alias.decl *"neg0`25": !Int = <{1}>
    comptime neg0 = simd_where_clause_int_neg[-1]()
    # CHECK: lit.alias.decl *"neg1`26": !Int = <{0}>
    comptime neg1 = simd_where_clause_int_neg[0]()

    # CHECK: lit.alias.decl *"inv0`27": !Int = <{1}>
    comptime inv0 = simd_where_clause_int_invert[0]()
    # CHECK: lit.alias.decl *"inv1`28": !Int = <{0}>
    comptime inv1 = simd_where_clause_int_invert[-1]()

    # CHECK: lit.alias.decl *"i0`{{.*}}": !Int = <{1}>
    comptime i0 = simd_where_clause_int_eq[4]()
    # CHECK: lit.alias.decl *"i1`{{.*}}": !Int = <{0}>
    comptime i1 = simd_where_clause_int_eq[9]()

    # CHECK: lit.alias.decl *"shl0`{{.*}}": !Int = <{1}>
    comptime shl0 = simd_where_clause_int_shl[2]()
    # CHECK: lit.alias.decl *"shl1`{{.*}}": !Int = <{0}>
    comptime shl1 = simd_where_clause_int_shl[1]()

    # CHECK: lit.alias.decl *"shr0`{{.*}}": !Int = <{1}>
    comptime shr0 = simd_where_clause_int_shr[6]()
    # CHECK: lit.alias.decl *"shr1`{{.*}}": !Int = <{0}>
    comptime shr1 = simd_where_clause_int_shr[2]()
