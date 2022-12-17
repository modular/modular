// RUN: kgen %s -emit -func="someKernel:f32(f32,index)" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALAR

// RUN: kgen %s -emit -func="someBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=BUFFER

// RUN: kgen %s -emit -func="someNDBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=NDBUFFER

// RUN: kgen %s -emit -func="someMetaScalarKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALARMETA

// RUN: kgen %s -emit -func="nestedParametricStruct" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=STRUCT

kgen.func @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel_c(float, ssize_t);

kgen.func @someBufferKernel(%a: !pop.struct<pointer<simd<1, invalid>>, index, !kgen.dtype>) -> index {
  %size = pop.struct.get %a[1] : !pop.struct<pointer<simd<1, invalid>>, index, !kgen.dtype>
  kgen.return %size : index
}
// BUFFER: extern ssize_t someBufferKernel_c(void *, ssize_t, uint8_t);


kgen.func @someNDBufferKernel(%a: !pop.struct<pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype>) -> index {
  %size = pop.struct.get %a[1] : !pop.struct<pointer<simd<1, invalid>>, index, array<5, index>, !kgen.dtype>
  kgen.return %size : index
}
// NDBUFFER: extern ssize_t someNDBufferKernel_c(void *, ssize_t, ssize_t[5], uint8_t);

kgen.func @someMetaScalarKernel(%arg0: !pop.simd<1, f32>) -> !pop.simd<1, f32> {
  kgen.return %arg0 : !pop.simd<1, f32>
}
// SCALARMETA: extern float someMetaScalarKernel_c(float);

kgen.struct.decl @Foo<DT:dtype> {
  kgen.struct.field value : !pop.scalar<DT>
}

kgen.struct.decl @Bar {
  kgen.struct.field a : !kgen.declref<@Foo<DT:dtype=f32>>
  kgen.struct.field b : !pop.scalar<f64>
}

kgen.func @nestedParametricStruct(%a: !kgen.declref<@Bar>) {
  kgen.return
}
// STRUCT: extern void nestedParametricStruct_c(float, double)

kgen.export [@someKernel, @someBufferKernel, @someNDBufferKernel, @someMetaScalarKernel, @nestedParametricStruct]
