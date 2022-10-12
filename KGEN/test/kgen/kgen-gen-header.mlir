// RUN: kgen %s -emit -func="someKernel:f32(f32,index)" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALAR

// RUN: kgen %s -emit -func="someBufferKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=BUFFER

// RUN: kgen %s -emit -func="someMetaScalarKernel" -o %t.o
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALARMETA

kgen.func public @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}
// SCALAR: extern float someKernel(float, ssize_t);

kgen.func public @someBufferKernel(%a: !pop.struct<index, !pop.pointer<!pop.scalar<invalid>>, !kgen.dtype>) -> index {
  %size = pop.get_element %a[0] : !pop.struct<index, !pop.pointer<!pop.scalar<invalid>>, !kgen.dtype>
  kgen.return %size : index
}
// BUFFER: extern ssize_t someBufferKernel(ssize_t, void *, uint8_t);



// https://github.com/modularml/modular/issues/2636
kgen.func public @someMetaScalarKernel(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
  kgen.return %arg0 : !pop.scalar<f32>
}

// SCALARMETA: extern float someMetaScalarKernel(float);
