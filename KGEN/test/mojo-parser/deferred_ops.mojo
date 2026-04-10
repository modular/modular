# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"test0(::Int,::Int)"
def test0(a: Int, b: Int) raises -> Bool:
    comptime pred_attr = __mlir_attr.`#index<cmp_predicate sle>`

    # CHECK: kgen.deferred "index.cmp"(%{{.*}}, %{{.*}} : !Int, !Int) {pred = #kgen<deferred #index<cmp_predicate sle>> : !kgen.deferred} : i1
    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


# CHECK-LABEL: lit.fn @"test1[::Bool](::Int,::Int)"
def test1[cmp: Bool](a: Int, b: Int) raises -> Bool:
    # CHECK: lit.fn *"select_pred[::Bool]()"<*"cmp`2x": !Bool>() -> !kgen.deferred
    def select_pred[cmp: Bool]() -> __mlir_type.`!kgen.deferred`:
        comptime if cmp:
            return __mlir_attr.`#index<cmp_predicate sle>`
        else:
            return __mlir_attr.`#index<cmp_predicate sgt>`

    comptime pred_attr = select_pred[cmp]()

    # CHECK: kgen.deferred "index.cmp"(%{{.}}, %{{.*}} : !Int, !Int) {pred = #kgen.param.expr<apply, #kgen.bind_params<:!lit.generator<<"cmp": !Bool>() -> !kgen.deferred> *"select_pred[::Bool]()", :!Bool cmp> : !kgen.generator<!lit.generator<() -> !kgen.deferred>>> : !kgen.deferred} : i1
    var res = __mlir_op.`index.cmp`[pred=pred_attr](a, b)
    return res


@always_inline("nodebug")
def to_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.values,
        `> : !kgen.string`,
    ]


# CHECK-LABEL: lit.fn @"test2[::StringSlice[::Bool(False)
def test2[pred: StaticString](x: Int, y: Int) -> Bool:
    def get_pred[pred: StaticString]() -> __mlir_type.`!kgen.deferred`:
        # CHECK: kgen.param.constant: !kgen.deferred =  <#kgen<attr_ctor_deferred("#index<cmp_predicate ", {{.*}} elide_type unit
        return __mlir_deferred_attr[
            `#index<cmp_predicate `, +to_string[pred](), `>`
        ]

    var z = __mlir_op.`index.cmp`[pred = get_pred[pred]()](x, y)
    return z


struct DType:
    comptime type = __mlir_type.`!kgen.dtype`
    var value: Self.type


# CHECK-LABEL: lit.fn @"test3[::Int,deferred_ops::DType]
def test3[
    n: Int, dtype: DType
](
    x: __mlir_type[
        `!kgen.struct<(`,
        __mlir_type[
            `!kgen.param_list_splat<`,
            __mlir_type[`!pop.scalar<`, dtype.value, `>`],
            `, `,
            n._mlir_value,
            `>`,
        ],
        `)>`,
    ]
):
    # CHECK: kgen.deferred "kgen.struct.extract"(%x : !kgen.struct<(!kgen.param_list_splat<!pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, #lit.struct.extract<:!Int n, "_mlir_value"> : index>)>) {index = 0 : index} : !pop.scalar<#lit.struct.extract<:!DType dtype, "value">>
    var e0 = __mlir_op.`kgen.struct.extract`[
        _type=__mlir_type[`!pop.scalar<`, dtype.value, `>`],
        index=__mlir_attr.`0:index`,
    ](x)

    # CHECK: kgen.deferred "kgen.struct.replace"(%1, %x : !pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, !kgen.struct<(!kgen.param_list_splat<!pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, #lit.struct.extract<:!Int n, "_mlir_value"> : index>)>) {index = 0 : index} : !kgen.struct<(!kgen.param_list_splat<!pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, #lit.struct.extract<:!Int n, "_mlir_value"> : index>)>
    _ = __mlir_op.`kgen.struct.replace`[index=__mlir_attr.`0:index`](e0, x)

    # CHECK: kgen.deferred "kgen.struct.replace"(%3, %x : !pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, !kgen.struct<(!kgen.param_list_splat<!pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, #lit.struct.extract<:!Int n, "_mlir_value"> : index>)>) {index = 1 : index} : !kgen.struct<(!kgen.param_list_splat<!pop.scalar<#lit.struct.extract<:!DType dtype, "value">>, #lit.struct.extract<:!Int n, "_mlir_value"> : index>)>
    _ = __mlir_op.`kgen.struct.replace`[
        _type=__mlir_type[
            `!kgen.struct<(`,
            __mlir_type[
                `!kgen.param_list_splat<`,
                __mlir_type[`!pop.scalar<`, dtype.value, `>`],
                `, `,
                n._mlir_value,
                `>`,
            ],
            `)>`,
        ],
        index=__mlir_attr.`1:index`,
    ](e0, x)
