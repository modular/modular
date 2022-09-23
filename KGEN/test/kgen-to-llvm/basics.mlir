// RUN: kgen-opt -split-input-file -lower-kgen-to-llvm="index-bitwidth=64" %s | FileCheck %s

// CHECK-LABEL: llvm.func private @trivial
// CHECK-SAME: (%[[ARG0:.*]]: i32)
// CHECK-NEXT: llvm.return %[[ARG0]] : i32
kgen.func @trivial(%arg0: si32) -> si32 {
  kgen.return %arg0 : si32
}

// -----

// CHECK-LABEL: llvm.func private @produces_result
kgen.func @produces_result<() -> index>() {
  // CHECK: llvm.return
  kgen.return<42>
}

// -----

// CHECK-LABEL: llvm.func private @convert_meta_types
// CHECK-SAME: %{{.*}}: f32
// CHECK-SAME: %{{.*}}: !llvm.ptr<f32>
// CHECK-SAME: %{{.*}}: vector<4xf32>

kgen.func @convert_meta_types(
    %arg0: !meta.scalar<f32>,
    %arg1: !meta.pointer<!meta.scalar<f32>>,
    %arg2: !meta.simd<4, f32>) {
  kgen.return
}

// -----

kgen.func @trivial(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
  kgen.return %arg0 : !meta.scalar<f32>
}

kgen.func @no_result(%arg0: !meta.scalar<f32>) {
  kgen.return
}

kgen.func @two_results(%arg0: !meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>) {
  kgen.return %arg0, %arg0 : !meta.scalar<f32>, !meta.scalar<f32>
}

// CHECK-LABEL: llvm.func private @convert_call
// CHECK-SAME: %[[ARG0:.*]]: f32
kgen.func @convert_call(%arg0: !meta.scalar<f32>) {
  // CHECK: llvm.call @trivial(%[[ARG0]]) : (f32) -> f32
  %0 = kgen.call @trivial(%arg0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  // CHECK: llvm.call @no_result(%[[ARG0]]) : (f32) -> ()
  kgen.call @no_result(%arg0) : (!meta.scalar<f32>) -> ()
  // CHECK: %[[PACK:.*]] = llvm.call @two_results(%[[ARG0]]) : (f32) -> !llvm.struct<(f32, f32)>
  %1:2 = kgen.call @two_results(%arg0) : (!meta.scalar<f32>) -> (!meta.scalar<f32>, !meta.scalar<f32>)
  // CHECK: llvm.extractvalue %[[PACK]][0]
  // CHECK: llvm.extractvalue %[[PACK]][1]
  kgen.return
}
