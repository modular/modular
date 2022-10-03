// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s

// CHECK-NOT: kgen.struct.decl
kgen.struct.decl @SmallVector<N, T: type> {
  data: !pop.array<N, T>
}

!size2 = !kgen.typedef<@SmallVector<N = 2, T:type = !pop.simd<4, f32>>>
!size4 = !kgen.typedef<@SmallVector<N = 4, T:type = !pop.scalar<f64>>>

// CHECK-LABEL: @two_vectors
kgen.func @two_vectors(
  %arg0: !pop.array<2, !pop.simd<4, f32>>,
  %arg1: !pop.array<4, !pop.scalar<f64>>
) -> (!size2, !size4) {
  // CHECK: llvm.mlir.undef : !llvm.struct<(array<2 x vector<4xf32>>)>
  // CHECK: llvm.mlir.undef : !llvm.struct<(array<4 x f64>)>
  %0 = kgen.struct.create(%arg0) : (!pop.array<2, !pop.simd<4, f32>>) -> !size2
  %1 = kgen.struct.create(%arg1) : (!pop.array<4, !pop.scalar<f64>>) -> !size4
  kgen.return %0, %1 : !size2, !size4
}

// CHECK-NOT: kgen.struct.decl
kgen.struct.decl @Box<T: type> {
  value: !kgen.paramref<T>
}

// CHECK-NOT: kgen.struct.decl
kgen.struct.decl @Pair<T1: type, T2: type> {
  first: !kgen.paramref<T1>
  second: !kgen.paramref<T2>
}

// CHECK-LABEL: @make_box
kgen.func @make_box(%v: f32) -> !kgen.typedef<@Box<T:type = f32>> {
  // CHECK: %[[BOX:.*]] = llvm.mlir.undef
  // CHECK: llvm.insertvalue %{{.*}}, %[[BOX]][0]
  %0 = kgen.struct.create(%v) : (f32) -> !kgen.typedef<@Box<T:type = f32>>
  kgen.return %0 : !kgen.typedef<@Box<T:type = f32>>
}

!i8Pair = !kgen.typedef<@Pair<T1:type = i8, T2:type = i8>>

// CHECK-LABEL: @make_pair
// CHECK: %[[A:.*]]: i8, %[[B:.*]]: i8
kgen.func @make_pair(%a: i8, %b: i8) -> !i8Pair {
  // CHECK: llvm.insertvalue %[[B]], %{{.*}}[0]
  // CHECK: llvm.insertvalue %[[A]], %{{.*}}[1]
  %0 = kgen.struct.create(%b, %a) : (i8, i8) -> !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_insert
kgen.func @struct_insert(%pair: !i8Pair) -> !i8Pair {
  %c1 = llvm.mlir.constant(2 : i8) : i8
  // CHECK: llvm.insertvalue %{{.*}}, %{{.*}}[1]
  %0 = kgen.struct.insert %c1, %pair[second] : i8 into !i8Pair
  kgen.return %0 : !i8Pair
}

// CHECK-LABEL: @struct_extract
kgen.func @struct_extract(%pair: !i8Pair) -> i8 {
  // CHECK: llvm.extractvalue %{{.*}}[1]
  %0 = kgen.struct.extract %pair[second] : i8 from !i8Pair
  kgen.return %0 : i8
}

kgen.struct.decl @NestedA<T: type> {
  v: !kgen.paramref<T>
}
kgen.struct.decl @NestedB<t: dtype> {
  a: !kgen.typedef<@NestedA<T:type = !pop.scalar<t>>>
}
kgen.struct.decl @NestedC {
  b: !kgen.typedef<@NestedB<t:dtype = f32>>
}

// CHECK-LABEL: @use_nested
// CHECK-SAME: !llvm.struct<(struct<(struct<(f32)>)>)>
kgen.func @use_nested(%a: !kgen.typedef<@NestedC>) {
  kgen.return
}

// CHECK-LABEL: @struct_element
// CHECK-SAME: !llvm.ptr<struct<(vector<2xf32>)>>
kgen.func @struct_element(%a: !pop.pointer<!kgen.typedef<@NestedA<T:type = !pop.simd<2, f32>>>>) {
  kgen.return
}
