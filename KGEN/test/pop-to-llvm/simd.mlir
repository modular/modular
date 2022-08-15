// RUN: kgen-opt -split-input-file -convert-pop-to-llvm %s | FileCheck %s

// CHECK-LABEL: @constant_simd
kgen.kernel @constant_simd() -> !meta.simd<2, si32> {
  // CHECK: llvm.mlir.constant(dense<0>
  %0 = pop.constant(dense<0> : vector<2xi32>) : !meta.simd<2, si32>
  kgen.return %0 : !meta.simd<2, si32>
}

// -----

// CHECK-LABEL: @constant_simd
kgen.kernel @constant_simd() -> !meta.simd<2, f32> {
  // CHECK: llvm.mlir.constant(dense<[1.{{0*}}e+00, 2.{{0*}}e+00]>
  %0 = pop.constant(dense<[1., 2.]> : vector<2xf32>) : !meta.simd<2, f32>
  kgen.return %0 : !meta.simd<2, f32>
}
