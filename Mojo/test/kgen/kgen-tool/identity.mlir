// RUN: kgen %s -emit=header | FileCheck %s

kgen.func export @identity(%arg0: !kgen.simd<4, f32>) cabi -> !kgen.simd<4, f32> {
  hlcf.return %arg0 : !kgen.simd<4, f32>
}


// CHECK: extern float[4] identity(float[4]);
