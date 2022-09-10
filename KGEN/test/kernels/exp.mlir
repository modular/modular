// RUN: kgen-opt %s -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(%x: !meta.scalar<type>) -> !meta.scalar<type>

//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.func @exp_f32(%arg0: f32) -> f32
// CHECK: kgen.call @"exp_intrinsic_f32,type=f32"(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
kgen.generator @exp_f32(%arg0: f32) -> f32 {
  %0 = meta.cast_from_builtin %arg0 : f32 to !meta.scalar<f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!meta.scalar<f32>) -> !meta.scalar<f32>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f32> to f32
  kgen.return %2 : f32
}

// CHECK-LABEL: kgen.func @exp_f64(%arg0: f64) -> f64
// CHECK: kgen.call @"exp_intrinsic_f64,type=f64"(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
kgen.generator @exp_f64(%arg0: f64) -> f64 {
  %0 = meta.cast_from_builtin %arg0 : f64 to !meta.scalar<f64>
  %1 = kgen.call @exp<type: dtype = f64>(%0) : (!meta.scalar<f64>) -> !meta.scalar<f64>
  %2 = meta.cast_to_builtin %1 : !meta.scalar<f64> to f64
  kgen.return %2 : f64
}
