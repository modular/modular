# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


def test0(a: Int, b: Int) -> Bool:
    alias pred_attr = __mlir_attr.`#index<cmp_predicate sle>`

    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


def test1[cmp: Bool](a: Int, b: Int) -> Bool:
    fn select_pred[cmp: Bool]() -> __mlir_type.`!kgen.deferred`:
        @parameter
        if cmp:
            return __mlir_attr.`#index<cmp_predicate sle>`
        else:
            return __mlir_attr.`#index<cmp_predicate sgt>`

    alias pred_attr = select_pred[cmp]()

    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


@always_inline("nodebug")
fn to_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return to_string[string, extra]()


@always_inline("nodebug")
fn to_string[
    string: StaticString, extra: VariadicList[StaticString]
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.value,
        `> : !kgen.string`,
    ]


fn test2[pred: StaticString](x: Int, y: Int) -> Bool:
    fn get_pred[pred: StaticString]() -> __mlir_type.`!kgen.deferred`:
        return __mlir_deferred_attr[
            `#index<cmp_predicate `, +to_string[pred](), `>`
        ]

    var z = __mlir_op.`index.cmp`[pred = get_pred[pred]()](x, y)

    return z


alias dtype_to_llvm_type_i32[
    dtype: DType
] = __mlir_type.`i32` if dtype is DType.int32 or dtype is DType.uint32 else __mlir_type.`!kgen.none`


# Helper function to convert a SIMD to a kgen struct
# It's intentionally marked with @no_inline to make sure elaborator correctly handles callsite
@no_inline
fn simd_to_kgen_struct[
    n: Int, dtype: DType
](simd: SIMD[dtype, n]) -> __mlir_type[
    `!kgen.struct<(`,
    __mlir_type[
        `!kgen.variadic_splat<`,
        __mlir_type[`!pop.scalar<`, dtype.value, `>`],
        `, `,
        n.value,
        `>`,
    ],
    `)>`,
]:
    var llvmst = __mlir_op.`llvm.mlir.undef`[
        _type = __mlir_type[
            `!llvm.struct<(`,
            __mlir_type[
                `!kgen.variadic_splat<`,
                dtype_to_llvm_type_i32[dtype],
                `, `,
                n.value,
                `>`,
            ],
            `)>`,
        ]
    ]()

    var st = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type[
            `!kgen.struct<(`,
            __mlir_type[
                `!kgen.variadic_splat<`,
                __mlir_type[`!pop.scalar<`, dtype.value, `>`],
                `, `,
                n.value,
                `>`,
            ],
            `)>`,
        ]
    ](llvmst)

    @parameter
    for i in range(n):
        var e = simd[i]
        st = __mlir_op.`kgen.struct.replace`[
            _type = __mlir_type[
                `!kgen.struct<(`,
                __mlir_type[
                    `!kgen.variadic_splat<`,
                    __mlir_type[`!pop.scalar<`, dtype.value, `>`],
                    `, `,
                    n.value,
                    `>`,
                ],
                `)>`,
            ],
            index = __mlir_attr[i.value, `:index`],
        ](e, st)

    return st


# Helper function to convert a kgen struct to a SIMD
# It's intentionally marked with @no_inline to make sure elaborator correctly handles callsite
@no_inline
fn kgen_struct_to_simd_reverse[
    n: Int, dtype: DType
](
    st: __mlir_type[
        `!kgen.struct<(`,
        __mlir_type[
            `!kgen.variadic_splat<`,
            __mlir_type[`!pop.scalar<`, dtype.value, `>`],
            `, `,
            n.value,
            `>`,
        ],
        `)>`,
    ]
) -> SIMD[dtype, n]:
    var simd = SIMD[dtype, n]()

    @parameter
    for i in range(n):
        var e = __mlir_op.`kgen.struct.extract`[
            _type = __mlir_type[`!pop.scalar<`, dtype.value, `>`],
            index = __mlir_attr[i.value, `:index`],
        ](st)

        simd[n - i - 1] = e
    return simd


fn test3[n: Int, dtype: DType](vec: SIMD[dtype, n]) -> SIMD[dtype, n]:
    var st = simd_to_kgen_struct[n, dtype](vec)
    var simd = kgen_struct_to_simd_reverse[n, dtype](st)

    return simd


def main():
    # CHECK: test0 = True
    print("test0 = ", test0(1, 2))

    # CHECK: test1[True] = True
    print("test1[True] = ", test1[True](1, 2))

    # CHECK: test1[False] = False
    print("test1[False] = ", test1[False](1, 2))

    # CHECK: test2["sle"] = True
    print('test2["sle"] = ', test2["sle"](1, 2))

    # CHECK: test2["sge"] = False
    print('test2["sge"] = ', test2["sge"](1, 2))

    # CHECK: test3[1, 2, 3, 4] = [4, 3, 2, 1]
    print("test3[1, 2, 3, 4] = ", test3(SIMD[DType.int32, 4](1, 2, 3, 4)))

    # CHECK: test3[1, 2, 3, 4, 5, 6, 7, 8] = [8, 7, 6, 5, 4, 3, 2, 1]
    print(
        "test3[1, 2, 3, 4, 5, 6, 7, 8] = ",
        test3(SIMD[DType.int32, 8](1, 2, 3, 4, 5, 6, 7, 8)),
    )
