# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics | FileCheck %s

# CHECK: lit.func @"mlirMagicTest{{.*}}(%x: bf16 borrow, %y: f8E5M2 borrow)
fn mlirMagicTest(
    x: __mlir_type.bf16, y: __mlir_type.f8E5M2
) -> __mlir_type.index:
    # CHECK: kgen.param.declare a = <1>
    alias a: __mlir_type.index = (1).__as_mlir_index()
    # CHECK: %b = lit.varlet.decl "b", var = true, synth = false : <f64>
    var b: __mlir_type.f64
    # CHECK: %c = lit.varlet.decl "c", var = true, synth = false : <pointer<pointer<f32>>>
    var c: __mlir_type.`!pop.pointer<!pop.pointer<f32>>`

    # CHECK: [[TMP:%.*]] = kgen.param.constant: i17 = <4>
    # CHECK: %d = lit.varlet.decl
    # CHECK: pop.store [[TMP]], %d : !pop.pointer<i17>
    var d = __mlir_attr.`4: i17`

    # CHECK: [[TMP:%.*]] = kgen.param.constant: dtype = <f32>
    # CHECK: %dt = lit.varlet.decl
    # CHECK: pop.store [[TMP]], %dt  : !pop.pointer<dtype>
    var dt = __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype `

    # CHECK: index.constant 42
    # CHECK-NEXT: %idxConstant = lit.varlet.decl
    var idxConstant = __mlir_op.`index.constant`[
        value : (42).__as_mlir_index()
    ]()

    # CHECK: [[TMP:%.*]] = pop.load %idxConstant
    # CHECK: [[TMP2:%.*]] = index.castu [[TMP:%.*]] : index to i1
    var i1Cast = __mlir_op.`index.castu`[_type : __mlir_type.i1](idxConstant)

    # CHECK: kgen.param.declare new_lower = <max(a, 42)>
    alias new_lower = __mlir_attr[
        `#kgen.param.expr<max, `, a, `, `, (42).__as_mlir_index(), `> : index`
    ]

    # CHECK: [[TMP1:%.*]] = kgen.param.constant = <new_lower>
    # CHECK: [[TMP2:%.*]] = index.constant 1
    # CHECK: [[SHRU:%.*]] = index.shru [[TMP1]], [[TMP2]]
    # CHECK: lit.return [[SHRU]] : index
    return __mlir_op.`index.shru`(new_lower, (1).__as_mlir_index())


# CHECK-LABEL: lit.func @"mlirTypesAndAttrs
fn mlirTypesAndAttrs[dtype: __mlir_type.`!kgen.dtype`]():
    # CHECK: %a = lit.varlet.decl "a", var = true, synth = false : <scalar<dtype>>
    var a: __mlir_type[`!pop.scalar<`, dtype, `>`]
    # CHECK: %b = lit.varlet.decl "b", var = true, synth = false : <simd<4, dtype>>
    var b: __mlir_type[`!pop.simd<4, `, dtype, `>`]


# Issue #6282: [Lit] Placeholder substitution does not work on nested types
struct ComplexSubstitution[type: __mlir_type.`!kgen.dtype`]:
    # CHECK: lit.struct.field pointer : !pop.pointer<scalar<type>>
    var pointer: __mlir_type[`!pop.pointer<!pop.scalar<`, type, `>>`]


# Issue #6374: [Lit] Add support for type placeholder
fn typePlaceholder():
    # CHECK: %x = lit.varlet.decl {{.*}} : <!kgen.variadic<i32>>
    var x: __mlir_type[`!kgen.variadic<`, __mlir_type.i32, `>`]


# CHECK-LABEL: lit.func @"fancierSubstitutions
fn fancierSubstitutions():
    # CHECK: = lit.varlet.decl {{.*}} : <complex<i32>>
    var complexInt: __mlir_type[`complex<`, __mlir_type.i32, `>`]

    alias a: __mlir_type.index = (1).__as_mlir_index()
    # CHECK: kgen.param.declare new_lower = <max(a, 42)>
    alias new_lower = __mlir_attr[
        `#kgen.param.expr<max,`, a, `, `, (42).__as_mlir_index(), `> : index`
    ]


# This shows that we can use unary+ to make the printer avoid printing types.
# See Issue #6468: [Lit] __mlir_attr construction fails for !kgen.list
# CHECK-LABEL: testAttrConcatWithoutType
fn testAttrConcatWithoutType[
    length: __mlir_type.index,
]():
    # CHECK: kgen.param.declare x: variadic<index> = <[1, length]>
    alias x = __mlir_attr[
        `#kgen.variadic<`,
        +(1).__as_mlir_index(),
        `,`,
        length,
        `> : !kgen.variadic<index>`,
    ]


# Show conversion of lvalue address into a pointer.
# Issue #6825: Expose a way to get the address of an lvalue

# CHECK-LABEL: lit.struct.decl @MyPointer
@register_passable
struct MyPointer[eltType: __mlir_type.`!kgen.mlirtype`]:
    alias StorageTy = __mlir_type[`!pop.pointer<`, eltType, `>`]
    # CHECK: lit.struct.field value : !pop.pointer<eltType>
    var value: StorageTy

    fn __init__(value: StorageTy) -> MyPointer[eltType]:
        return MyPointer[eltType] {value: value}


# CHECK-LABEL: getAddressOf{{.*}}"<T: type>(%arg: !pop.pointer<T> byref)
fn getAddressOf[T: __mlir_type.`!kgen.mlirtype`](arg&: T) -> MyPointer[T]:
    return __mlir_op.`pop.pointer.bitcast`[_type : MyPointer[T].StorageTy](
        __get_lvalue_as_address(arg)
    )
    # CHECK-NEXT: lit.ownership.def.lvalue %arg
    # CHECK-NEXT: %0 = kgen.call @"{{.*}}@MyPointer::@"__init__(__mlir_type.!pop.pointer<eltType>)"<:type T>(%arg)
    # CHECK-NEXT: lit.return %0


# CHECK-LABEL: lit.func @"structured_for_loop()"
fn structured_for_loop() -> __mlir_type.index:
    # CHECK: %0 = hlcf.loop (%arg0 = %idx0 : index) -> index {
    __mlir_region loop_body(i: __mlir_type.index):
        # CHECK-NEXT: %idx1 = index.constant 1
        # CHECK-NEXT: %1 = index.add %arg0, %idx1
        # CHECK-NEXT: hlcf.continue %1 : index
        __mlir_op.`hlcf.continue`(
            __mlir_op.`index.add`(i, (1).__as_mlir_index())
        )

    # CHECK: lit.return %0 : index
    return __mlir_op.`hlcf.loop`[
        _type : __mlir_type.index, _region:"loop_body".value
    ]((0).__as_mlir_index())
