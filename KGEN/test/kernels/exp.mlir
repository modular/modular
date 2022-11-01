// RUN: kgen-opt %s -elaborate-generators="search-path=%S" | FileCheck %s

kgen.include "library.mlir"

kgen.generator.interface @exp<type: dtype>(!pop.scalar<type>) -> !pop.scalar<type>

//===----------------------------------------------------------------------===//

// CHECK-LABEL: kgen.func public @exp_f32
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32
// CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG0]]
// CHECK: kgen.call @"exp_intrinsic_f32,type=f32"(%[[V0]]) : (!pop.scalar<f32>) -> !pop.scalar<f32>
kgen.generator public @exp_f32(%arg0: f32) -> f32 {
  %0 = pop.cast_from_builtin %arg0 : f32 to !pop.scalar<f32>
  %1 = kgen.call @exp<type: dtype = f32>(%0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<f32> to f32
  kgen.return %2 : f32
}

// CHECK-LABEL: kgen.func public @exp_f64
// CHECK-SAME: (%[[ARG0:.*]]: f64) -> f64
// CHECK: %[[V0:.*]] = pop.cast_from_builtin %[[ARG0]]
// CHECK: kgen.call @"exp_intrinsic_f64,type=f64"(%[[V0]]) : (!pop.scalar<f64>) -> !pop.scalar<f64>
kgen.generator public @exp_f64(%arg0: f64) -> f64 {
  %0 = pop.cast_from_builtin %arg0 : f64 to !pop.scalar<f64>
  %1 = kgen.call @exp<type: dtype = f64>(%0) : (!pop.scalar<f64>) -> !pop.scalar<f64>
  %2 = pop.cast_to_builtin %1 : !pop.scalar<f64> to f64
  kgen.return %2 : f64
}
