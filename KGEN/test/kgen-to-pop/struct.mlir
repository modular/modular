// RUN: kgen-opt %s -lower-structs | FileCheck %s

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
  // CHECK: pop.struct.construct(%arg0) : !pop.struct<array<2, simd<4, f32>>>
  %0 = lit.struct.create(data=%arg0) : (!pop.array<2, simd<4, f32>>) -> !size2
  // CHECK: pop.struct.construct(%arg1) : !pop.struct<array<4, scalar<f64>>>
  %1 = lit.struct.create(data=%arg1) : (!pop.array<4, simd<1, f64>>) -> !size4
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
  // CHECK: pop.struct.construct(%arg0) : !pop.struct<f32>
  %0 = lit.struct.create(value=%v) : (f32) -> !kgen.declref<@Box<T:type = f32>>
  kgen.return %0 : !kgen.declref<@Box<T:type = f32>>
}

!i8Pair = !kgen.declref<@Pair<T1:type = i8, T2:type = i8>>

// CHECK-LABEL: @make_pair
// CHECK: %[[A:.*]]: i8, %[[B:.*]]: i8
kgen.func @make_pair(%a: i8, %b: i8) -> !i8Pair {
  // CHECK: pop.struct.construct(%arg1, %arg0) : !pop.struct<i8, i8>
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
  // CHECK: pop.struct.get %{{.*}}[1]
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

// CHECK-LABEL: @use_nested
// CHECK-SAME: !pop.struct<struct<struct<scalar<f32>>>>
kgen.func @use_nested(%a: !kgen.declref<@NestedC>) {
  kgen.return
}

// CHECK-LABEL: @struct_element
// CHECK-SAME: !pop.pointer<struct<simd<2, f32>>>
kgen.func @struct_element(%a: !pop.pointer<!kgen.declref<@NestedA<T:type = !pop.simd<2, f32>>>>) {
  kgen.return
}
