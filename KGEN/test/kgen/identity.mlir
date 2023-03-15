// RUN: kgen %s -emit-header | FileCheck %s

kgen.func @identity(%arg0: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  kgen.return %arg0 : !pop.simd<4, f32>
}

kgen.export @identity to C

// CHECK: extern float[4] identity(float[4]);
