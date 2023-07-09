// RUN: kgen-opt %s -lower-structs -allow-unregistered-dialect -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-NOT: lit.struct.decl
lit.struct.decl @SmallVector<N, T: type> {
  lit.struct.field data: !pop.array<N, T>
}

!size2 = !kgen.declref<@SmallVector<N = 2, T:type = !pop.simd<4, f32>>>
!size4 = !kgen.declref<@SmallVector<N = 4, T:type = !pop.simd<1, f64>>>

// CHECK-LABEL: @two_vectors
kgen.func @two_vectors(
  %arg0: !pop.array<2, simd<4, f32>>,
  %arg1: !pop.array<4, simd<1, f64>>
) -> (!size2, !size4) {
  %0 = lit.struct.create(data=%arg0) : (!pop.array<2, simd<4, f32>>) -> !size2
  %1 = lit.struct.create(data=%arg1) : (!pop.array<4, simd<1, f64>>) -> !size4

  // CHECK: kgen.return %arg0, %arg1
  kgen.return %0, %1 : !size2, !size4
}

// CHECK-NOT: lit.struct.decl
lit.struct.decl @Box<T: type> {
  lit.struct.field value: !kgen.paramref<T>
}

// CHECK-NOT: lit.struct.decl
lit.struct.decl @Pair<T1: type, T2: type> {
  lit.struct.field first: !kgen.paramref<T1>
  lit.struct.field second: !kgen.paramref<T2>
}

// CHECK-LABEL: @make_box
kgen.func @make_box(%v: f32) -> !kgen.declref<@Box<T:type = f32>> {
  // CHECK: kgen.return %arg0 : f32
  %0 = lit.struct.create(value=%v) : (f32) -> !kgen.declref<@Box<T:type = f32>>
  kgen.return %0 : !kgen.declref<@Box<T:type = f32>>
}

!i8Pair = !kgen.declref<@Pair<T1:type = i8, T2:type = i8>>

// CHECK-LABEL: @make_pair
// CHECK: %[[A:.*]]: i8, %[[B:.*]]: i8
kgen.func @make_pair(%a: i8, %b: i8) -> !i8Pair {
  // CHECK: pop.struct.create(%arg1, %arg0) : !pop.struct<i8, i8>
  %0 = lit.struct.create(first=%b, second=%a) : (i8, i8) -> !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(%pair: !i8Pair) -> !i8Pair {
  %c1 = llvm.mlir.constant(2 : i8) : i8
  // CHECK: pop.struct.replace %{{.*}}, %{{.*}}[1]
  %0 = lit.struct.insert %c1, %pair[second] : i8 into !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%pair: !i8Pair) -> i8 {
  // CHECK: pop.struct.extract %{{.*}}[1]
  %0 = lit.struct.extract %pair[second] : i8 from !i8Pair
  kgen.return %0 : i8
}

// CHECK-LABEL: @struct_gep
kgen.func @struct_gep(%pair: !pop.pointer<!i8Pair>) -> !pop.pointer<i8> {
  // CHECK: pop.struct.gep %{{.*}}[1]
  %0 = lit.struct.gep %pair[second] : <i8> from <!i8Pair>
  kgen.return %0 : !pop.pointer<i8>
}

lit.struct.decl @NestedA<T: type> {
  lit.struct.field v: !kgen.paramref<T>
}
lit.struct.decl @NestedB<t: dtype> {
  lit.struct.field a: !kgen.declref<@NestedA<T:type = !pop.simd<1, t>>>
}
lit.struct.decl @NestedC {
  lit.struct.field b: !kgen.declref<@NestedB<t:dtype = f32>>
}

// CHECK-LABEL: @use_nested(%arg0: !pop.scalar<f32>)
kgen.func @use_nested(%a: !kgen.declref<@NestedC>) {
  kgen.return
}

// CHECK-LABEL: @struct_element(%arg0: !pop.pointer<simd<2, f32>>
kgen.func @struct_element(%a: !pop.pointer<!kgen.declref<@NestedA<T:type = !pop.simd<2, f32>>>>) {
  kgen.return
}


lit.struct.decl @IndexStruct {
  lit.struct.field value : index
}

lit.struct.decl @StructInsideStruct {
  lit.struct.field x : !kgen.declref<@IndexStruct>
}

// CHECK-LABEL: @passStructAsLValue
kgen.func @passStructAsLValue(%s: !pop.pointer<@StructInsideStruct>) ->
!pop.pointer<index> {
  %0 = lit.struct.gep %s[x] : <@IndexStruct> from <@StructInsideStruct>
  %1 = lit.struct.gep %0[value] : <index> from <@IndexStruct>
  // CHECK: kgen.return %arg0
  kgen.return %1 : !pop.pointer<index>
}

lit.struct.decl @IndexField {
  lit.struct.field first: index
  lit.struct.field second: index
}

// CHECK-LABEL: @structExtract
kgen.generator @structExtract<p: !kgen.declref<@IndexField> -> res: index>() {
  // CHECK: kgen.param.result_bind<#pop.struct.extract<:struct<index, index> p, 1 : index>>
  kgen.param.result_bind<#lit.struct.extract<:!kgen.declref<@IndexField> p, "second">>
  kgen.return
}

kgen.generator @structExtractInsideStruct<p: @IndexField>(
    %arg0: !kgen.declref<@SmallVector<N = #lit.struct.extract<:@IndexField p, "second">, T: type = index>>) {
  %0 = lit.struct.extract %arg0[data] : !pop.array<#lit.struct.extract<:@IndexField p, "second">, index> from
    !kgen.declref<@SmallVector<N = #lit.struct.extract<:@IndexField p, "second">, T: type = index>>
  kgen.return
}

lit.struct.decl @Struct {}

lit.struct.decl @StructParam<param: @Struct> {
  lit.struct.field value : !pop.array<apply(:(!kgen.declref<@Struct>) -> index @return_one, param), index>
}

kgen.generator @return_one(%arg0: !kgen.declref<@Struct>) -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: @use_struct_param
// CHECK-SAME: !pop.array<apply(:(!pop.struct<>) -> index @return_one, { }), index>
kgen.generator @use_struct_param(%arg0: !kgen.declref<@StructParam<param: @Struct = #lit.struct<{}>>>) {
  lit.struct.extract %arg0[value] : !pop.array<apply(:(!kgen.declref<@Struct>) -> index @return_one, #lit.struct<{}>), index>
    from !kgen.declref<@StructParam<param: @Struct = #lit.struct<{}>>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @lifetime_lower
// CHECK-SAME: (%arg0: !pop.struct<>) {
kgen.generator @lifetime_lower<p: !lit.lifetime>(%a: !lit.lifetime) {

  // CHECK: kgen.param.declare A: struct<> = <{ }>
  kgen.param.declare A : !lit.lifetime = <#lit.lifetime>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_lifetime_lower
kgen.generator @call_lifetime_lower() {
  // CHECK: %struct = kgen.param.constant: struct<> = <{ }>
  %cst = kgen.param.constant: lifetime = <#lit.lifetime>
  // CHECK: kgen.call @lifetime_lower(%struct) : (!pop.struct<>) -> ()
  kgen.call @lifetime_lower<:lifetime #lit.lifetime>(%cst) : (!lit.lifetime) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type(
// CHECK-SAME: %arg0: !pop.pointer<@Foo>
// CHECK-SAME: %arg1: !pop.pointer<@Foo>)
kgen.generator @ref_type<p: !lit.lifetime>(%a: !lit.ref<@Foo, p>,
                                           %b: !lit.ref<mut @Foo, p>) {
  // Random use of a parameter that goes away should be updated.
  // CHECK: kgen.param.declare A: struct<> = <{ }>
  kgen.param.declare A : !lit.lifetime = <p>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_ref_type
kgen.generator @call_ref_type<q: !lit.lifetime>(%a: !lit.ref<@Foo, p>,
                                                %b: !lit.ref<mut @Foo, p>) {
  // CHECK-NEXT: kgen.call @ref_type(%arg0, %arg1) : (!pop.pointer<@Foo>, !pop.pointer<@Foo>)
  kgen.call @ref_type<:lifetime q>(%a, %b): (!lit.ref<@Foo, p>, !lit.ref<mut @Foo, p>) -> ()
  kgen.return
}


// -----

// CHECK-LABEL: kgen.generator @parameterized_declref_type
kgen.generator @parameterized_declref_type() {
  // CHECK-NEXT: !pop.array<2, simd<apply(:(!pop.struct<>) -> index @unbox, { }), f32>>
  %3 = pop.stack_allocation 1 x !kgen.declref<@StaticTuple<size = 2,
    type: type = !kgen.declref<@SIMD<size: @Int = #lit.struct<{}>, type: dtype = f32>>>>
  kgen.return
}

lit.struct.decl @SIMD<size: @Int, type: dtype> {
  lit.struct.field value : !pop.simd<apply(:(!kgen.declref<@Int>) -> index @unbox, size), type>
}

lit.struct.decl @Int {}

lit.struct.decl @StaticTuple<size, type: type> {
  lit.struct.field array : !pop.array<size, type>
}

// -----

// CHECK-LABEL: kgen.generator @nested_declref_type
// CHECK-SAME: !pop.closure<(!pop.simd<apply(:(index) -> index @pass, 1), si32>
kgen.generator @nested_declref_type(
    %arg1: !kgen.declref<@UnaryClosure<input_type: type = !kgen.declref<@SIMD<size = 1>>>>) {
  kgen.return
}

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

lit.struct.decl @SIMD<size> {
  lit.struct.field value : !pop.simd<apply(:(index) -> index @pass, size), si32>
}

lit.struct.decl @UnaryClosure<input_type: type> {
  lit.struct.field value : !pop.closure<(!kgen.paramref<input_type>) -> ()>
}

// -----

kgen.generator @not_a_struct() {
  kgen.return
}

// expected-error @below {{operation contains a declref type that does not refer to a struct}}
kgen.generator @bad_declref(%arg0: !kgen.declref<@not_a_struct<a = 1>>) {
  kgen.return
}
