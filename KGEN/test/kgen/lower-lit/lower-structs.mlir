// RUN: kgen-opt %s -lower-lit-types -allow-unregistered-dialect -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Parametric Structs
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.struct.decl
lit.struct.decl @SmallVector<N, T: regtype> register_passable {
  lit.struct.field data: !pop.array<N, T>
}

!size2 = !kgen.declref<@SmallVector<2, :regtype !pop.simd<4, f32>>>
!size4 = !kgen.declref<@SmallVector<4, :regtype !pop.simd<1, f64>>>

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
lit.struct.decl @Box<T: regtype> register_passable {
  lit.struct.field value: !kgen.paramref<T>
}

// CHECK-NOT: lit.struct.decl
lit.struct.decl @Pair<T1: regtype, T2: regtype> {
  lit.struct.field first: !kgen.paramref<T1>
  lit.struct.field second: !kgen.paramref<T2>
}

// CHECK-LABEL: @make_box
kgen.func @make_box(%v: f32) -> !kgen.declref<@Box<:regtype f32>> {
  // CHECK: kgen.return %arg0 : f32
  %0 = lit.struct.create(value=%v) : (f32) -> !kgen.declref<@Box<:regtype f32>>
  kgen.return %0 : !kgen.declref<@Box<:regtype f32>>
}

!i8Pair = !kgen.declref<@Pair<:regtype i8, :regtype i8>>

// CHECK-LABEL: @make_pair
// CHECK: %[[A:.*]]: i8, %[[B:.*]]: i8
kgen.func @make_pair(%a: i8, %b: i8) -> !i8Pair {
  // CHECK: kgen.struct.create(%arg1, %arg0) : !kgen.struct<(i8, i8) memoryOnly>
  %0 = lit.struct.create(first=%b, second=%a) : (i8, i8) -> !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(%pair: !i8Pair) -> !i8Pair {
  %c1 = llvm.mlir.constant(2 : i8) : i8
  // CHECK: kgen.struct.replace %{{.*}}, %{{.*}}[1]
  %0 = lit.struct.insert %c1, %pair[second] : i8 into !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%pair: !i8Pair) -> i8 {
  // CHECK: kgen.struct.extract %{{.*}}[1]
  %0 = lit.struct.extract %pair[second] : i8 from !i8Pair
  kgen.return %0 : i8
}

lit.struct.decl @NestedA<T: regtype> register_passable {
  lit.struct.field v: !kgen.paramref<T>
}
lit.struct.decl @NestedB<t: dtype> register_passable {
  lit.struct.field a: !kgen.declref<@NestedA<:regtype !pop.simd<1, t>>>
}
lit.struct.decl @NestedC register_passable {
  lit.struct.field b: !kgen.declref<@NestedB<:dtype f32>>
}

// CHECK-LABEL: @use_nested(%arg0: !pop.scalar<f32>)
kgen.func @use_nested(%a: !kgen.declref<@NestedC>) {
  kgen.return
}

// CHECK-LABEL: @struct_element(%arg0: !kgen.pointer<simd<2, f32>>
kgen.func @struct_element(%a: !kgen.pointer<!kgen.declref<@NestedA<:regtype !pop.simd<2, f32>>>>) {
  kgen.return
}


lit.struct.decl @IndexStruct register_passable {
  lit.struct.field value : index
}

lit.struct.decl @StructInsideStruct register_passable {
  lit.struct.field x : !kgen.declref<@IndexStruct>
}

lit.struct.decl @IndexField {
  lit.struct.field first: index
  lit.struct.field second: index
}

// CHECK-LABEL: @structExtract
kgen.generator @structExtract<p: !kgen.declref<@IndexField> -> res: index>() {
  // CHECK: kgen.param.result_bind<#kgen.struct.extract<:struct<(index, index) memoryOnly> p, 1>>
  kgen.param.result_bind<#lit.struct.extract<:!kgen.declref<@IndexField> p, "second">>
  kgen.return
}

kgen.generator @structExtractInsideStruct<p: @IndexField>(
    %arg0: !kgen.declref<@SmallVector<#lit.struct.extract<:@IndexField p, "second">, :regtype index>>) {
  %0 = lit.struct.extract %arg0[data] : !pop.array<#lit.struct.extract<:@IndexField p, "second">, index> from
    !kgen.declref<@SmallVector<#lit.struct.extract<:@IndexField p, "second">, :regtype index>>
  kgen.return
}

lit.struct.decl @Struct register_passable {}

lit.struct.decl @StructParam<param: @Struct> register_passable {
  lit.struct.field value : !pop.array<apply(:(!kgen.declref<@Struct>) -> index @return_one, param), index>
}

kgen.generator @return_one(%arg0: !kgen.declref<@Struct>) -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: @use_struct_param
// CHECK-SAME: !pop.array<apply(:(!kgen.struct<()>) -> index @return_one, { }), index>
kgen.generator @use_struct_param(%arg0: !kgen.declref<@StructParam<:@Struct #lit.struct<{}>>>) {
  lit.struct.extract %arg0[value] : !pop.array<apply(:(!kgen.declref<@Struct>) -> index @return_one, #lit.struct<{}>), index>
    from !kgen.declref<@StructParam<:@Struct #lit.struct<{}>>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @lifetime_lower
// CHECK-SAME: (%arg0: !kgen.struct<()>) {
kgen.generator @lifetime_lower<p: !lit.lifetime>(%a: !lit.lifetime) {

  // CHECK: kgen.param.declare A: struct<()> = <{ }>
  kgen.param.declare A : !lit.lifetime = <#lit.lifetime>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_lifetime_lower
kgen.generator @call_lifetime_lower() {
  // CHECK: %struct = kgen.param.constant: struct<()> = <{ }>
  %cst = kgen.param.constant: lifetime = <#lit.lifetime>
  // CHECK: kgen.call @lifetime_lower(%struct) : (!kgen.struct<()>) -> ()
  kgen.call @lifetime_lower<:lifetime #lit.lifetime>(%cst) : (!lit.lifetime) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type(
// CHECK-SAME: %arg0: !kgen.pointer<struct<()>>
// CHECK-SAME: %arg1: !kgen.pointer<struct<()>>)
kgen.generator @ref_type<p: !lit.lifetime>(%a: !lit.ref<@Struct, p>,
                                           %b: !lit.ref<mut @Struct, p>) {
  // Random use of a parameter that goes away should be updated.
  // CHECK: kgen.param.declare A: struct<()> = <{ }>
  kgen.param.declare A : !lit.lifetime = <p>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_ref_type
kgen.generator @call_ref_type<q: !lit.lifetime>(%a: !lit.ref<@Struct, p>,
                                                %b: !lit.ref<mut @Struct, p>) {
  // CHECK-NEXT: kgen.call @ref_type(%arg0, %arg1) : (!kgen.pointer<struct<()>>, !kgen.pointer<struct<()>>)
  kgen.call @ref_type<:lifetime q>(%a, %b): (!lit.ref<@Struct, p>, !lit.ref<mut @Struct, p>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @raw_pointer_from_ref_type
// CHECK-SAME: (%arg0: !kgen.pointer<struct<()>>) -> !kgen.pointer<struct<()>>
kgen.generator @raw_pointer_from_ref_type<q: !lit.lifetime>(%a: !lit.ref<@Struct, p>)
  -> !kgen.pointer<@Struct> {
  // CHECK-NEXT: kgen.return %a
  %ptr = lit.ref.to_pointer %a: !lit.ref<@Struct, p>
  kgen.return %ptr: !kgen.pointer<@Struct>
}

// CHECK: kgen.extern.generator @empty_region_dont_crash
kgen.extern.generator @empty_region_dont_crash()



//===----------------------------------------------------------------------===//
// Reference Lowering
//===----------------------------------------------------------------------===//

lit.struct.decl @PairStruct {
  lit.struct.field x : si32
  lit.struct.field y : ui32
}

// CHECK-LABEL: @gerToGEPFooFromBar
kgen.generator @gerToGEPFooFromBar<l: !lit.lifetime>
  (%arg0: !lit.ref<mut @PairStruct, l>, %arg1: si32) -> si32 {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(si32, ui32) memoryOnly>>
  %0 = lit.ref.struct.ger %arg0[x] : <mut si32, l> from @PairStruct

  // CHECK-NEXT: pop.store %arg1, %0
  lit.ref.store %arg1, %0 : !lit.ref<mut si32, l>

  // CHECK-NEXT: %1 = pop.load %0 : !kgen.pointer<si32>
  %a = lit.ref.load %0 : !lit.ref<mut si32, l>
  // CHECK-NEXT: kgen.return %1
  kgen.return %a : si32
}


// -----

// CHECK-LABEL: kgen.generator @parameterized_declref_type
kgen.generator @parameterized_declref_type() {
  // CHECK-NEXT: array<2, simd<apply(:(!kgen.struct<()>) -> index @unbox, { }), f32>>
  %3 = pop.stack_allocation 1 x @StaticTuple<2,
    :regtype !kgen.declref<@SIMD<:@Int #lit.struct<{}>, :dtype f32>>>
  kgen.return
}

lit.struct.decl @SIMD<size: @Int, type: dtype> register_passable {
  lit.struct.field value : !pop.simd<apply(:(!kgen.declref<@Int>) -> index @unbox, size), type>
}

lit.struct.decl @Int register_passable {}

lit.struct.decl @StaticTuple<size, ty: regtype> register_passable {
  lit.struct.field array : !pop.array<size, ty>
}

// -----

// CHECK-LABEL: kgen.generator @nested_declref_type
// CHECK-SAME: !kgen.signature<(!pop.simd<apply(:(index) -> index @pass, 1), si32>
kgen.generator @nested_declref_type(
    %arg1: !kgen.declref<@UnaryClosure<:regtype !kgen.declref<@SIMD<1>>>>) {
  kgen.return
}

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

lit.struct.decl @SIMD<size> register_passable {
  lit.struct.field value : !pop.simd<apply(:(index) -> index @pass, size), si32>
}

lit.struct.decl @UnaryClosure<input_type: regtype> register_passable {
  lit.struct.field value : !kgen.signature<(!kgen.paramref<input_type>) -> ()>
}

//===----------------------------------------------------------------------===//
// Recursive Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Bar {
  lit.struct.field x : !kgen.declref<@Pointer<:regtype !kgen.declref<@Foo>>>
  lit.struct.field y : ui32
}

lit.struct.decl @Foo {
  lit.struct.field x : !kgen.declref<@Bar>
  lit.struct.field y : f32
}

lit.struct.decl @Pointer<ty: regtype> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

!bar_ref = !kgen.declref<@Bar>
!foo_ref = !kgen.declref<@Foo>
!foo_ptr_ref = !kgen.declref<@Pointer<:regtype !foo_ref>>
!null_ptr = !kgen.pointer<scalar<invalid>>

// CHECK-LABEL: @makeBar
kgen.func @makeBar(%arg0: !foo_ptr_ref, %arg1: ui32) -> !bar_ref {
  // CHECK: %0 = kgen.struct.create(%arg0, %arg1) : !kgen.struct<(pointer<struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>>, ui32) memoryOnly>
  // CHECK: kgen.return %1 : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  %0 = lit.struct.create(x=%arg0, y=%arg1) : (!foo_ptr_ref, ui32) -> !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @makeFoo
kgen.func @makeFoo(%arg0: !bar_ref, %arg1: f32) -> !foo_ref {
  // CHECK: %0 = kgen.struct.create(%arg0, %arg1) : !kgen.struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>
  %0 = lit.struct.create(x=%arg0, y=%arg1) : (!bar_ref, f32) -> !foo_ref
  kgen.return %0 : !foo_ref
}

// CHECK-LABEL: @structInsertUIntToBar
kgen.func @structInsertUIntToBar(%arg0: ui32, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: %0 = kgen.struct.replace %arg0, %arg1[1] : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  %0 = lit.struct.insert %arg0, %arg1[y] : ui32 into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertFooPtrToBar
kgen.func @structInsertFooPtrToBar(%arg0: !foo_ptr_ref, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>> to !kgen.pointer<scalar<invalid>>
  // CHECK: %1 = kgen.struct.replace %0, %arg1[0] : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  // CHECK: kgen.return %1 : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  %0 = lit.struct.insert %arg0, %arg1[x] : !foo_ptr_ref into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertBarToFoo
kgen.func @structInsertBarToFoo(%arg0: !foo_ptr_ref, %arg1: ui32,  %arg2: !foo_ref) -> !foo_ref {
  // CHECK: %0 = kgen.call @makeBar(%arg0, %arg1) : (!kgen.pointer<struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>>, ui32) -> !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  // CHECK: %1 = kgen.struct.replace %0, %arg2[0] : !kgen.struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return %1 : !kgen.struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>
  %0 = kgen.call @makeBar(%arg0, %arg1): (!foo_ptr_ref, ui32) -> !bar_ref
  %1 = lit.struct.insert %0, %arg2[x] : !bar_ref into !foo_ref
  kgen.return %1 : !foo_ref
}

// CHECK-LABEL: @structExtractFooFromBar
kgen.func @structExtractFooFromBar(%arg0: !bar_ref) -> !foo_ptr_ref {
  // CHECK: %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  // CHECK: %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<invalid>> to !kgen.pointer<struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>>
  // CHECK: kgen.return %1 : !kgen.pointer<struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>>
  %0 = lit.struct.extract %arg0[x] : !foo_ptr_ref from !bar_ref
  kgen.return %0 : !foo_ptr_ref
}

// CHECK-LABEL: @structExtractBarFromFoo
kgen.func @structExtractBarFromFoo(%arg0: !foo_ref) -> !bar_ref {
  // CHECK: %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(struct<(pointer<scalar<invalid>>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(pointer<scalar<invalid>>, ui32) memoryOnly>
  %0 = lit.struct.extract %arg0[x] : !bar_ref from !foo_ref
  kgen.return %0 : !bar_ref
}

lit.struct.decl @Recursive register_passable {
  lit.struct.field x : !kgen.pointer<@Recursive>
}

// CHECK-LABEL: @thing
// CHECK: -> !kgen.declref<@Recursive>
kgen.generator @thing() -> !kgen.declref<@Recursive> {
  // CHECK: kgen.unreachable
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @foo<T: regtype>()
kgen.generator @foo<T: type>() {
  kgen.return
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.trait.decl
lit.trait.decl @Trait {
}

// CHECK: kgen.generator @trait_fn<T: regtype>()
kgen.generator @trait_fn<T: trait<@Trait>>() {
  kgen.return
}
