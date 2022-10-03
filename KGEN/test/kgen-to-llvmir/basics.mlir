// RUN: kgen-opt -split-input-file -emit-llvm %s | FileCheck %s

// CHECK-LABEL: define private float @trivial
kgen.func @trivial(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg0 : !pop.scalar<f32>
}

// CHECK-LABEL: define private void @no_result
kgen.func @no_result(%arg0: !pop.scalar<f32>) {
  kgen.return
}

// CHECK-LABEL: define private { float, float } @two_results
kgen.func @two_results(%arg0: !pop.scalar<f32>) -> (!pop.scalar<f32>, !pop.scalar<f32>) {
  kgen.return %arg0, %arg0 : !pop.scalar<f32>, !pop.scalar<f32>
}

// CHECK-LABEL: define private void @convert_call
kgen.func @convert_call(%arg0: !pop.scalar<f32>) {
  %0 = kgen.call @trivial(%arg0) : (!pop.scalar<f32>) -> !pop.scalar<f32>
  kgen.call @no_result(%arg0) : (!pop.scalar<f32>) -> ()
  %1:2 = kgen.call @two_results(%arg0) : (!pop.scalar<f32>) -> (!pop.scalar<f32>, !pop.scalar<f32>)
  kgen.return
}


