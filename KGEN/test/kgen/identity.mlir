// RUN: kgen %s -emit -kernel="identity:%t.o"
// RUN: FileCheck %s --input-file=%t.h

kgen.kernel @identity(%arg0: !meta.simd<4, f32>) -> !meta.simd<4, f32> {
  kgen.return %arg0 : !meta.simd<4, f32>
}

// CHECK: extern float __attribute__ ((vector_size(16))) identity(float __attribute__ ((vector_size(16))));
