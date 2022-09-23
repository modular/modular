// RUN: kgen %s -emit -func="identity:%t.o"
// RUN: FileCheck %s --input-file=%t.h

kgen.func public @identity(%arg0: !pop.simd<4, f32>) -> !pop.simd<4, f32> {
  kgen.return %arg0 : !pop.simd<4, f32>
}

// CHECK: extern float[4] identity(float[4]);
