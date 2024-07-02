// RUN: kgen-opt %s -lower-lit-types -allow-unregistered-dialect -split-input-file | kgen-opt -verify-parameters | FileCheck %s

//===----------------------------------------------------------------------===//
// Parametric Structs
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.struct.decl
lit.struct.decl @SmallVector<N, T: type> register_passable {
  lit.struct.field data: !pop.array<N, T>
}

!size2 = !lit.declref<@SmallVector<2, :type !pop.simd<4, f32>>>
!size4 = !lit.declref<@SmallVector<4, :type !pop.simd<1, f64>>>

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
lit.struct.decl @Box<T: type> register_passable {
  lit.struct.field value: !kgen.paramref<T>
}

// CHECK-NOT: lit.struct.decl
lit.struct.decl @Pair<T1: type, T2: type> {
  lit.struct.field first: !kgen.paramref<T1>
  lit.struct.field second: !kgen.paramref<T2>
}

// CHECK-LABEL: @make_box
kgen.func @make_box(%v: f32) -> !lit.declref<@Box<:type f32>> {
  // CHECK: kgen.return %arg0 : f32
  %0 = lit.struct.create(value=%v) : (f32) -> !lit.declref<@Box<:type f32>>
  kgen.return %0 : !lit.declref<@Box<:type f32>>
}

!i8Pair = !lit.declref<@Pair<:type i8, :type i8>>

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

lit.struct.decl @NestedA<T: type> register_passable {
  lit.struct.field v: !kgen.paramref<T>
}
lit.struct.decl @NestedB<t: dtype> register_passable {
  lit.struct.field a: !lit.declref<@NestedA<:type !pop.simd<1, t>>>
}
lit.struct.decl @NestedC register_passable {
  lit.struct.field b: !lit.declref<@NestedB<:dtype f32>>
}

// CHECK-LABEL: @use_nested(%arg0: !pop.scalar<f32>)
kgen.func @use_nested(%a: !lit.declref<@NestedC>) {
  kgen.return
}

// CHECK-LABEL: @struct_element(%arg0: !kgen.pointer<simd<2, f32>>
kgen.func @struct_element(%a: !kgen.pointer<!lit.declref<@NestedA<:type !pop.simd<2, f32>>>>) {
  kgen.return
}


lit.struct.decl @IndexStruct register_passable {
  lit.struct.field value : index
}

lit.struct.decl @StructInsideStruct register_passable {
  lit.struct.field x : !lit.declref<@IndexStruct>
}

lit.struct.decl @IndexField {
  lit.struct.field first: index
  lit.struct.field second: index
}

// CHECK-LABEL: @structExtract
kgen.generator @structExtract<p: !lit.declref<@IndexField>>() {
  kgen.param.constant = <#lit.struct.extract<:!lit.declref<@IndexField> p, "second">>
  kgen.return
}

kgen.generator @structExtractInsideStruct<p: @IndexField>(
    %arg0: !lit.declref<@SmallVector<#lit.struct.extract<:@IndexField p, "second">, :type index>>) {
  %0 = lit.struct.extract %arg0[data] : !pop.array<#lit.struct.extract<:@IndexField p, "second">, index> from
    !lit.declref<@SmallVector<#lit.struct.extract<:@IndexField p, "second">, :type index>>
  kgen.return
}

lit.struct.decl @Struct register_passable {}

lit.struct.decl @StructParam<param: @Struct> register_passable {
  lit.struct.field value : !pop.array<apply(:(!lit.declref<@Struct>) -> index @return_one, param), index>
}

kgen.generator @return_one(%arg0: !lit.declref<@Struct>) -> index {
  %0 = index.constant 0
  kgen.return %0 : index
}

// CHECK-LABEL: @use_struct_param
// CHECK-SAME: !pop.array<apply(:(!kgen.struct<()>) -> index @return_one, { }), index>
kgen.generator @use_struct_param(%arg0: !lit.declref<@StructParam<:@Struct #lit.struct<{}>>>) {
  lit.struct.extract %arg0[value] : !pop.array<apply(:(!lit.declref<@Struct>) -> index @return_one, #lit.struct<{}>), index>
    from !lit.declref<@StructParam<:@Struct #lit.struct<{}>>>
  kgen.return
}

// CHECK-LABEL: kgen.generator @lifetime_lower
// CHECK-SAME: (%arg0: !kgen.struct<()>) {
kgen.generator @lifetime_lower<p: !lit.lifetime<0>>(%a: !lit.lifetime<1>) {

  // CHECK: kgen.param.declare A: struct<()> = <{ }>
  kgen.param.declare A: !lit.lifetime<1> = <#lit.lifetime>

  // CHECK: kgen.param.declare B: struct<()> = <{ }>
  kgen.param.declare B: lifetime.set = <{imm p}>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_lifetime_lower
kgen.generator @call_lifetime_lower() {
  // CHECK: %struct = kgen.param.constant: struct<()> = <{ }>
  %cst = kgen.param.constant: lifetime<1> = <#lit.lifetime>
  // CHECK: kgen.call @lifetime_lower<:struct<()> { }>(%struct) : (!kgen.struct<()>) -> ()
  kgen.call @lifetime_lower<:lifetime<0> #lit.lifetime>(%cst) : (!lit.lifetime<1>) -> ()
  kgen.return
}

kgen.generator @take_lifetime<lt: lifetime<0>>() {
  kgen.return
}

// CHECK-LABEL: kgen.generator @implicit_lifetime_as_param
kgen.generator @implicit_lifetime_as_param() {
  // CHECK-NEXT: @take_lifetime<:struct<()> { }>()
  kgen.call @take_lifetime<:lifetime<0> *[0,0]>() : () -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @ref_type<p: struct<()>, q: struct<()>>(
// CHECK-SAME: %arg0: !kgen.pointer<struct<()>>
// CHECK-SAME: %arg1: !kgen.pointer<struct<()>>)
kgen.generator @ref_type<p: !lit.lifetime<0>, q: !lit.lifetime<1>>
    (%a: !lit.ref<@Struct, imm p>, %b: !lit.ref<@Struct, mut p>) {
  // Random use of a parameter that goes away should be updated.
  // CHECK: kgen.param.declare A: struct<()> = <p>
  kgen.param.declare A : !lit.lifetime<0> = <p>
  kgen.return
}

// CHECK-LABEL: kgen.generator @call_ref_type
kgen.generator @call_ref_type<a: !lit.lifetime<0>, b: !lit.lifetime<1>>
    (%a: !lit.ref<@Struct, imm a>, %b: !lit.ref<@Struct, mut b>) {
  // CHECK-NEXT: kgen.call @ref_type<:struct<()> a, :struct<()> b>(%arg0, %arg1)
  // CHECK-SAME: : (!kgen.pointer<struct<()>>, !kgen.pointer<struct<()>>)
  kgen.call @ref_type<:lifetime<0> a, :lifetime<1> b>(%a, %b): (!lit.ref<@Struct, imm a>, !lit.ref<@Struct, mut b>) -> ()
  kgen.return
}

// CHECK-LABEL: kgen.generator @raw_pointer_from_ref_type
// CHECK-SAME: (%arg0: !kgen.pointer<struct<()>>) -> !kgen.pointer<struct<()>>
kgen.generator @raw_pointer_from_ref_type<q: !lit.lifetime<0>>(%a: !lit.ref<@Struct, imm q>)
  -> !kgen.pointer<@Struct> {
  // CHECK-NEXT: kgen.return %a
  %ptr = lit.ref.to_pointer %a: !lit.ref<@Struct, imm q>
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

// CHECK-LABEL: kgen.generator @gerToGEPFooFromBar
kgen.generator @gerToGEPFooFromBar<l: !lit.lifetime<1>, l2: !lit.lifetime<1>>
  (%arg0: !lit.ref<@PairStruct, mut l>, %arg1: si32) -> si32 {
  // CHECK-NEXT: %0 = kgen.struct.gep %arg0[0] : <struct<(si32, ui32) memoryOnly>>
  %0 = lit.ref.struct.ger %arg0[x] : <si32, mut l> from @PairStruct

  // This rebind should be removed entirely by lower types.
  %rb = kgen.rebind %0 : !lit.ref<si32, mut l> to !lit.ref<si32, mut l2>

  // CHECK-NEXT: pop.store %arg1, %0
  lit.ref.store %arg1, %rb : <si32, mut l2>

  // CHECK-NEXT: %1 = pop.load %0 : !kgen.pointer<si32>
  %a = lit.ref.load %0 : !lit.ref<si32, mut l>
  // CHECK-NEXT: kgen.return %1
  kgen.return %a : si32
}

// Issue #29038 - lower lit can't change positions of parameters.
// CHECK-LABEL: kgen.generator @takes_val_after_lifetime
// CHECK-SAME: <life: struct<()>, type: type>(%arg0: !kgen.pointer<type>)
kgen.generator @takes_val_after_lifetime<life: lifetime<1>, type: type>(%a: !lit.ref<type, mut life>) {
  kgen.return
}

//===----------------------------------------------------------------------===//
// Reference Pack Lowering
//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.generator @takes_pack<life: struct<()>, types: variadic<type>>
kgen.generator @takes_pack
<life: !lit.lifetime<1>, types: !kgen.variadic<!kgen.type>>
// CHECK-SAME: (%arg0: !kgen.pack<variadic_ptr_map(:variadic<type> types, 42)>) {
(%args: !lit.ref.pack<:variadic<!kgen.type> types, mut life, 42>) {

  // CHECK-NEXT: = kgen.pack.extract %arg0[0] : <variadic_ptr_map(:variadic<type> types, 42)>
  %v1 = lit.ref.pack.extract %args[0]: !lit.ref.pack<:variadic<!kgen.type> types, mut life, 42>

  // CHECK-NEXT: = kgen.pack.extract %arg0[1] : <variadic_ptr_map(:variadic<type> types, 42)>
  %v2 = lit.ref.pack.extract %args[1]: !lit.ref.pack<:variadic<!kgen.type> types, mut life, 42>

  kgen.return
}

// CHECK-LABEL: kgen.generator @pass_pack
kgen.generator @pass_pack<life: !lit.lifetime<1>>
  (%index: !lit.ref<index, mut life>,
   %float: !lit.ref<f32, mut life>) {

  // CHECK-NEXT: kgen.pack.create(%arg0, %arg1) : !kgen.pack<[pointer<index>, pointer<f32>]>
  %pack = lit.ref.pack.create(%index, %float) :
    !lit.ref.pack<:variadic<!kgen.type> [index, f32], mut life, 0>
  // CHECK-NEXT: kgen.call @takes_pack<:struct<()> life, :variadic<type> [index, f32]>(%0)
  kgen.call @takes_pack<:lifetime<1> life, :variadic<!kgen.type> [index, f32]>(%pack)
     : (!lit.ref.pack<:variadic<!kgen.type> [index, f32], mut life, 0>) -> ()

  // CHECK-NEXT: kgen.param.constant: !kgen.pack<[pointer<i8>, pointer<ui4>, pointer<i32>]> = <<store_to_mem(3), store_to_mem(1), store_to_mem(4)>>
  %3 = kgen.param.constant: !lit.ref.pack<:variadic<!kgen.type> [i8, ui4, i32], mut life, 0>
     = <<store_to_mem(3), store_to_mem(1), store_to_mem(4)>>

  kgen.return
}

// -----

// CHECK-LABEL: kgen.generator @parameterized_declref_type
kgen.generator @parameterized_declref_type() {
  // CHECK-NEXT: array<2, simd<apply(:(!kgen.struct<()>) -> index @unbox, { }), f32>>
  %3 = pop.stack_allocation 1 x @StaticTuple<2,
    :type !lit.declref<@SIMD<:@Int #lit.struct<{}>, :dtype f32>>>
  kgen.return
}

lit.struct.decl @SIMD<size: @Int, type: dtype> register_passable {
  lit.struct.field value : !pop.simd<apply(:(!lit.declref<@Int>) -> index @unbox, size), type>
}

lit.struct.decl @Int register_passable {}

lit.struct.decl @StaticTuple<size, ty: type> register_passable {
  lit.struct.field array : !pop.array<size, ty>
}

// -----

// CHECK-LABEL: kgen.generator @nested_declref_type
// CHECK-SAME: !kgen.signature<(!pop.simd<apply(:(index) -> index @pass, 1), si32>
kgen.generator @nested_declref_type(
    %arg1: !lit.declref<@UnaryClosure<:type !lit.declref<@SIMD<1>>>>) {
  kgen.return
}

kgen.generator @pass(%arg0: index) -> index {
  kgen.return %arg0 : index
}

lit.struct.decl @SIMD<size> register_passable {
  lit.struct.field value : !pop.simd<apply(:(index) -> index @pass, size), si32>
}

lit.struct.decl @UnaryClosure<input_type: type> register_passable {
  lit.struct.field value : !kgen.signature<(!kgen.paramref<input_type>) -> ()>
}

//===----------------------------------------------------------------------===//
// Recursive Structs
//===----------------------------------------------------------------------===//

// -----

lit.struct.decl @Bar {
  lit.struct.field x : !lit.declref<@Pointer<:type !lit.declref<@Foo>>>
  lit.struct.field y : ui32
}

lit.struct.decl @Foo {
  lit.struct.field x : !lit.declref<@Bar>
  lit.struct.field y : f32
}

lit.struct.decl @Pointer<ty: type> register_passable {
  lit.struct.field address : !kgen.pointer<ty>
}

!bar_ref = !lit.declref<@Bar>
!foo_ref = !lit.declref<@Foo>
!foo_ptr_ref = !lit.declref<@Pointer<:type !foo_ref>>
!null_ptr = !kgen.pointer<none>

// CHECK-LABEL: @makeFoo
kgen.func @makeFoo(%arg0: !bar_ref, %arg1: f32) -> !foo_ref {
  // CHECK: %0 = kgen.struct.create(%arg0, %arg1) : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>
  %0 = lit.struct.create(x=%arg0, y=%arg1) : (!bar_ref, f32) -> !foo_ref
  kgen.return %0 : !foo_ref
}

// CHECK-LABEL: @makeBar
kgen.func @makeBar(%arg0: !foo_ptr_ref, %arg1: ui32) -> !bar_ref {
  // CHECK: [[V0:%.*]] = kgen.struct.create(%arg0, %arg1) : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return [[V0]] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.create(x=%arg0, y=%arg1) : (!foo_ptr_ref, ui32) -> !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertUIntToBar
kgen.func @structInsertUIntToBar(%arg0: ui32, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: %0 = kgen.struct.replace %arg0, %arg1[1] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.insert %arg0, %arg1[y] : ui32 into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertFooPtrToBar
kgen.func @structInsertFooPtrToBar(%arg0: !foo_ptr_ref, %arg1: !bar_ref) -> !bar_ref {
  // CHECK: [[V0:%.*]] = kgen.struct.replace %arg0, %arg1[0] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return [[V0]] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.insert %arg0, %arg1[x] : !foo_ptr_ref into !bar_ref
  kgen.return %0 : !bar_ref
}

// CHECK-LABEL: @structInsertBarToFoo
kgen.func @structInsertBarToFoo(%arg0: !foo_ptr_ref, %arg1: ui32,  %arg2: !foo_ref) -> !foo_ref {
  // CHECK: [[V0:%.*]] = kgen.call @makeBar(%arg0, %arg1) : (!kgen.pointer<none>, ui32) -> !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: [[V1:%.*]] = kgen.struct.replace [[V0]], %arg2[0] : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return [[V1]] : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>

  %0 = kgen.call @makeBar(%arg0, %arg1): (!foo_ptr_ref, ui32) -> !bar_ref
  %1 = lit.struct.insert %0, %arg2[x] : !bar_ref into !foo_ref
  kgen.return %1 : !foo_ref
}

// CHECK-LABEL: @structExtractFooFromBar
kgen.func @structExtractFooFromBar(%arg0: !bar_ref) -> !foo_ptr_ref {
  // CHECK: [[V0:%.*]] = kgen.struct.extract %arg0[0] : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  // CHECK: kgen.return [[V0]] : !kgen.pointer<none>
  %0 = lit.struct.extract %arg0[x] : !foo_ptr_ref from !bar_ref
  kgen.return %0 : !foo_ptr_ref
}

// CHECK-LABEL: @structExtractBarFromFoo
kgen.func @structExtractBarFromFoo(%arg0: !foo_ref) -> !bar_ref {
  // CHECK: %0 = kgen.struct.extract %arg0[0] : !kgen.struct<(struct<(pointer<none>, ui32) memoryOnly>, f32) memoryOnly>
  // CHECK: kgen.return %0 : !kgen.struct<(pointer<none>, ui32) memoryOnly>
  %0 = lit.struct.extract %arg0[x] : !bar_ref from !foo_ref
  kgen.return %0 : !bar_ref
}

lit.struct.decl @Recursive register_passable {
  lit.struct.field x : !kgen.pointer<@Recursive>
}

// CHECK-LABEL: @thing
// CHECK: -> !kgen.pointer<none>
kgen.generator @thing() -> !lit.declref<@Recursive> {
  // CHECK: kgen.unreachable
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @foo<T: type>()
kgen.generator @foo<T: type>() {
  kgen.return
}

//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

// CHECK-NOT: lit.trait.decl
lit.trait.decl @Trait {
}

// CHECK: kgen.generator @trait_fn<T: type>()
kgen.generator @trait_fn<T: trait<@Trait>>() {
  kgen.return
}

// -----

//===----------------------------------------------------------------------===//
// Erase pointer types in struct
//===----------------------------------------------------------------------===//

!ptr = !lit.declref<@Ptr, !lit.anystruct<@Ptr>>

lit.struct.decl @Ptr register_passable {
  lit.struct.field ptr: !kgen.pointer<index>
}

// CHECK-LABEL:  kgen.func @foo(%arg0: !kgen.pointer<index>)
kgen.func @foo(%x: !kgen.pointer<index>) {
    kgen.return
}

// CHECK-LABEL: kgen.func @pass_it(%arg0: !kgen.pointer<none>)
kgen.func @pass_it(%y: !ptr) {
    // CHECK: [[V0:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<index>
    // CHECK: kgen.call @f([[V0]]) : (!kgen.pointer<index>) -> ()
    %0 = lit.struct.extract %y[ptr]: !kgen.pointer<index> from !ptr
    kgen.call @f(%0): (!kgen.pointer<index>)->()
    kgen.return
}


// -----

//===----------------------------------------------------------------------===//
// More Recursive Structs
//===----------------------------------------------------------------------===//

!bar = !lit.declref<@Bar, !lit.anystruct<@Bar>>
!foo = !lit.declref<@Foo, !lit.anystruct<@Foo>>

lit.struct.decl @Bar register_passable {
  lit.struct.field foo: !foo
}

lit.struct.decl @Foo register_passable {
  lit.struct.field bar_ptr: !kgen.pointer<@Bar>
}

// CHECK-LABEL:  kgen.func @f(%arg0: !kgen.pointer<none>)
kgen.func @f(%bar: !bar) {
    // CHECK: [[V0:%.*]] = pop.pointer.bitcast %arg0 : !kgen.pointer<none> to !kgen.pointer<pointer<none>>
    // CHECK: kgen.call @g([[V0]]) : (!kgen.pointer<pointer<none>>) -> ()

    %foo = lit.struct.extract %bar[foo]: !foo from !bar
    %bar_ptr = lit.struct.extract %foo[bar_ptr]: !kgen.pointer<@Bar> from !foo
    kgen.call @g(%bar_ptr): (!kgen.pointer<@Bar>)->()
    kgen.return
}

// CHECK-LABEL: kgen.func @g(%arg0: !kgen.pointer<pointer<none>>)
kgen.func @g(%arg0: !kgen.pointer<@Bar>) {
    kgen.return
}

// -----

lit.struct.decl @Pointer<T: type, as, exclusive: i1> register_passable_trivial {
  lit.struct.field value: !kgen.pointer<T, as exclusive(exclusive)>
}

kgen.generator @make_ptr<T: type>() -> !kgen.pointer<T> {
  kgen.unreachable
}

// CHECK-LABEL: kgen.generator @pointer_const
kgen.generator @pointer_const<T: type>() {
  // CHECK-NEXT: constant: pointer<none> = <ptr_bitcast(:pointer<T> apply(:() -> !kgen.pointer<T> @make_ptr<:type T>))>
  kgen.param.constant: @Pointer<:type T, 0, :i1 0> = <{value: pointer<T> = apply(:() -> !kgen.pointer<T> @make_ptr<:type T>)}>
  // CHECK-NEXT: constant: pointer<none, 1 exclusive(1)> = <0>
  kgen.param.constant: @Pointer<:type T, 1, :i1 1> = <{value: pointer<T, 1 exclusive(1)> = 0}>
  kgen.return
}
