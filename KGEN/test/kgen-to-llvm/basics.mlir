// RUN: kgen-opt -split-input-file -lower-kgen-to-llvm="index-bitwidth=64" %s | FileCheck %s

// CHECK-LABEL: llvm.func internal @trivial
// CHECK-SAME: (%[[ARG0:.*]]: i32)
// CHECK-NEXT: llvm.return %[[ARG0]] : i32
kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// -----

// CHECK-LABEL: llvm.func internal @convert_pop_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr<f32>
// CHECK-SAME: %{{.*}}: vector<4xf32>

kgen.func @convert_pop_types(
    %arg0: !pop.simd<1, f32>,
    %arg1: !pop.pointer<simd<1, f32>>,
    %arg2: !pop.simd<4, f32>) {
  kgen.return
}

// -----

kgen.func @trivial(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}

kgen.func @no_result(%arg0: !pop.simd<1, f32>) {
  kgen.return
}

kgen.func @two_results(%arg0: !pop.simd<1, f32>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>) {
  kgen.return %arg0, %arg0 : !pop.simd<1, f32>, !pop.simd<1, f32>
}

// CHECK-LABEL: llvm.func internal @convert_call
// CHECK-SAME: %[[ARG0:.*]]: f32
kgen.func @convert_call(%arg0: !pop.simd<1, f32>) {
  // CHECK: llvm.call @trivial(%[[ARG0]]) : (f32) -> f32
  %0 = kgen.call @trivial(%arg0) : (!pop.simd<1, f32>) -> !pop.simd<1, f32>
  // CHECK: llvm.call @no_result(%[[ARG0]]) : (f32) -> ()
  kgen.call @no_result(%arg0) : (!pop.simd<1, f32>) -> ()
  // CHECK: %[[PACK:.*]] = llvm.call @two_results(%[[ARG0]]) : (f32) -> !llvm.struct<(f32, f32)>
  %1:2 = kgen.call @two_results(%arg0) : (!pop.simd<1, f32>) -> (!pop.simd<1, f32>, !pop.simd<1, f32>)
  // CHECK: llvm.extractvalue %[[PACK]][0]
  // CHECK: llvm.extractvalue %[[PACK]][1]
  kgen.return
}

// -----

kgen.func @reference_me(%a: i64) -> i64 {
  kgen.return %a : i64
}

// CHECK-LABEL: llvm.func internal @addressof
// CHECK-SAME: -> !llvm.ptr<func<i64 (i64)>>
kgen.func @addressof() -> ((i64) -> i64) {
  // CHECK: llvm.mlir.addressof @reference_me : !llvm.ptr<func<i64 (i64)>>
  %0 = kgen.addressof @reference_me : (i64) -> i64
  kgen.return %0 : (i64) -> i64
}

// -----

// CHECK-LABEL: @address_dtype
// CHECK-SAME: %[[ARG0:.*]]: !llvm.ptr,
// CHECK-SAME: %[[ARG1:.*]]: !llvm.vec<4 x ptr>
kgen.func @address_dtype(%arg0 : !pop.simd<1, address>, %arg1 : !pop.simd<4, address>) {
  kgen.return
}

// -----

// CHECK-LABEL: llvm.func @an_extern_func
// CHECK-SAME:  (i32, vector<4xf32>) -> vector<8xf32>
// COM: Check that the next line closes the module - we don't want a body for this!
// CHECK-NEXT: }

kgen.extern.func @an_extern_func(si32, !pop.simd<4, f32>) -> !pop.simd<8, f32>

// -----

// CHECK-LABEL: llvm.mlir.global external @foo
// CHECK-SAME: {addr_space = 0 : i32} : f64
kgen.extern.variable @foo : f64

// -----

kgen.func @constant_str() -> !kgen.string {
  // CHECK: %[[GLOBAL_STR:.*]] = pop.global_constant: !pop.array<3, scalar<si8>> = <[65, 66, 0]>
  // CHECK: %[[BITCAST:.*]] = pop.pointer.bitcast %[[GLOBAL_STR]] : !pop.pointer<array<3, scalar<si8>>> to !pop.pointer<i8>
  // CHECK: %[[CONV_CAST:.*]] = builtin.unrealized_conversion_cast %[[BITCAST]] : !pop.pointer<i8> to !llvm.ptr<i8>
  // CHECK: %[[LENGTH:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, i64)>
  // CHECK: %[[VAL0:.*]] = llvm.insertvalue %[[CONV_CAST]], %[[STRUCT]][0] : !llvm.struct<(ptr<i8>, i64)>
  // CHECK: %[[VAL1:.*]] = llvm.insertvalue %[[LENGTH]], %[[VAL0]][1] : !llvm.struct<(ptr<i8>, i64)>
  %0 = kgen.param.constant: string = <"AB">
  // CHECK: llvm.return %[[VAL1]] : !llvm.struct<(ptr<i8>, i64)>
  kgen.return %0 : !kgen.string
}
