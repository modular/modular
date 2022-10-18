// RUN: kgen %s -emit -func="someKernel:f32(f32,index)" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALAR

// RUN: kgen %s -emit -func="someBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=BUFFER

// RUN: kgen %s -emit -func="someNDBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=NDBUFFER

// RUN: kgen %s -emit -func="someMetaScalarKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALARMETA

kgen.func public @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel(float, ssize_t);

kgen.func public @someBufferKernel(%a: !pop.struct<pointer<scalar<invalid>>, index, !kgen.dtype>) -> index {
  %size = pop.struct.get %a[1] : !pop.struct<pointer<scalar<invalid>>, index, !kgen.dtype>
  kgen.return %size : index
}
// BUFFER: extern ssize_t someBufferKernel(void *, ssize_t, uint8_t);


kgen.func public @someNDBufferKernel(%a: !pop.struct<pointer<scalar<invalid>>, index, array<5, index>, !kgen.dtype>) -> index {
  %size = pop.struct.get %a[1] : !pop.struct<pointer<scalar<invalid>>, index, array<5, index>, !kgen.dtype>
  kgen.return %size : index
}
// NDBUFFER: extern ssize_t someNDBufferKernel(void *, ssize_t, ssize_t[5], uint8_t);

// https://github.com/modularml/modular/issues/2636
kgen.func public @someMetaScalarKernel(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg0 : !pop.scalar<f32>
}

// SCALARMETA: extern float someMetaScalarKernel(float);
