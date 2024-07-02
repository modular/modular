# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

alias `1` = __mlir_attr.`1 : index`
alias `42` = __mlir_attr.`42 : index`


# CHECK: lit.func @"mlirMagicTest{{.*}}(%x: bf16, %y: f8E5M2)
fn mlirMagicTest(
    x: __mlir_type.bf16, y: __mlir_type.f8E5M2
) -> __mlir_type.index:
    # CHECK: lit.alias.decl [[A:.*]] = <1>
    alias a: __mlir_type.index = `1`
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<f64, mut
    var b: __mlir_type.f64
    # CHECK: %c = lit.var.decl "c" var : !lit.ref<pointer<pointer<float32>>, mut
    var c: __mlir_type.`!kgen.pointer<!kgen.pointer<float32>>`

    # CHECK: %d = lit.var.decl
    # CHECK: [[TMP:%.*]] = kgen.param.constant: i17 = <4>
    # CHECK: lit.ref.store [[TMP]], %d
    var d = __mlir_attr.`4: i17`

    # CHECK: %dt = lit.var.decl
    # CHECK: [[TMP:%.*]] = kgen.param.constant: dtype = <f32>
    # CHECK: lit.ref.store [[TMP]], %dt
    var dt = __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype `

    # CHECK-NEXT: %idxConstant = lit.var.decl
    # CHECK: kgen.param.constant = <42>
    var idxConstant = __mlir_op.`index.constant`[value=`42`]()

    # CHECK: [[TMP:%.*]] = lit.ref.load %idxConstant
    # CHECK: [[TMP2:%.*]] = index.castu [[TMP:%.*]] : index to i1
    var i1Cast = __mlir_op.`index.castu`[_type = __mlir_type.i1](idxConstant)

    # CHECK: lit.alias.decl [[NEW_LOWER:.*]] = <max([[A]], 42)>
    alias new_lower = __mlir_attr[
        `#kgen.param.expr<max, `, a, `, `, (`42`), `> : index`
    ]

    # CHECK: [[TMP1:%.*]] = kgen.param.constant = <[[NEW_LOWER]]>
    # CHECK: [[TMP2:%.*]] = kgen.param.constant = <1>
    # CHECK: [[SHRU:%.*]] = index.shru [[TMP1]], [[TMP2]]
    # CHECK: lit.return [[SHRU]] : index
    return __mlir_op.`index.shru`(new_lower, `1`)


# CHECK-LABEL: lit.func @"mlirTypesAndAttrs{{.*}}()"<dtype: dtype>()
fn mlirTypesAndAttrs[dtype: __mlir_type.`!kgen.dtype`]():
    # CHECK: %a = lit.var.decl "a" var : !lit.ref<scalar<dtype>, mut
    var a: __mlir_type[`!pop.scalar<`, dtype, `>`]
    # CHECK: %b = lit.var.decl "b" var : !lit.ref<simd<4, dtype>,
    var b: __mlir_type[`!pop.simd<4, `, dtype, `>`]


# Issue #6282: [Lit] Placeholder substitution does not work on nested types
# CHECK-LABEL: lit.struct.decl @ComplexSubstitution<T: dtype>
struct ComplexSubstitution[T: __mlir_type.`!kgen.dtype`]:
    # CHECK: lit.struct.field pointer : !kgen.pointer<scalar<T>>
    var pointer: __mlir_type[`!kgen.pointer<!pop.scalar<`, T, `>>`]


# Issue #6374: [Lit] Add support for type placeholder
# CHECK-LABEL: typePlaceholder
fn typePlaceholder():
    # CHECK: %x = lit.var.decl {{.*}} : !lit.ref<variadic<i32>,
    var x: __mlir_type[`!kgen.variadic<`, __mlir_type.i32, `>`]


# CHECK-LABEL: lit.func @"fancierSubstitutions
fn fancierSubstitutions():
    # CHECK: = lit.var.decl {{.*}} : !lit.ref<complex<i32>,
    var complexInt: __mlir_type[`complex<`, __mlir_type.i32, `>`]

    # CHECK: lit.alias.decl [[A:.*]] = <1>
    alias a: __mlir_type.index = `1`
    # CHECK: lit.alias.decl *"new_lower{{.*}}" = <max([[A]], 42)>
    alias new_lower = __mlir_attr[
        `#kgen.param.expr<max,`, a, `, `, (`42`), `> : index`
    ]


# This shows that we can use unary+ to make the printer avoid printing types.
# See Issue #6468: [Lit] __mlir_attr construction fails for !kgen.list
# CHECK-LABEL: @"testAttrConcatWithoutType{{.*}}()"<length>() ->
fn testAttrConcatWithoutType[
    length: __mlir_type.index,
]():
    # CHECK: lit.alias.decl *"x{{.*}}": variadic<index> = <[1, length]>
    alias x = __mlir_attr[
        `#kgen.variadic<`, +`1`, `,`, length, `> : !kgen.variadic<index>`
    ]


# Show conversion of lvalue address into a pointer.
# Issue #6825: Expose a way to get the address of an lvalue


# CHECK-LABEL: lit.struct.decl @MyPointer<elType: type>
@register_passable
struct MyPointer[elType: __mlir_type.`!kgen.type`]:
    alias StorageTy = __mlir_type[`!kgen.pointer<`, elType, `>`]
    # CHECK: lit.struct.field value : !kgen.pointer<elType>
    var value: Self.StorageTy

    fn __init__(value: Self.StorageTy) -> MyPointer[elType]:
        return MyPointer[elType] {value: value}


# CHECK-LABEL: lit.func @"structured_for_loop()"
fn structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 = %index0 : index) -> index {
    __mlir_region loop_body(i: __mlir_type.index):
        # CHECK-NEXT: %index1 = kgen.param.constant = <1>
        # CHECK-NEXT: %1 = index.add %arg0, %index1
        # CHECK-NEXT: hlcf.continue %1 : index
        __mlir_op.`hlcf.continue`(__mlir_op.`index.add`(i, `1`))

    # CHECK: lit.return %0 : index
    return __mlir_op.`hlcf.loop`[
        _type = __mlir_type.index, _region = __mlir_attr.`"loop_body"`
    ](__mlir_attr.`0 : index`)


# CHECK-LABEL: lit.func @"mlir_properties()"
fn mlir_properties():
    # CHECK: kgen.source_loc[1]
    _ = __mlir_op.`kgen.source_loc`[
        _type = (
            __mlir_type.index,
            __mlir_type.index,
            __mlir_type.`!kgen.string`,
        ),
        _properties = __mlir_attr.`{inlineCount = 1 : i64}`,
    ]()
