// RUN: kgen %s -emit -kernel="someKernel:f32(f32,index):%t.o"
// RUN: cat %t.h | FileCheck %s --check-prefixes=SCALAR

// RUN: kgen %s -emit -kernel="someBufferKernel:%t.o"
// RUN: cat %t.h | FileCheck %s --check-prefixes=BUFFER

kgen.kernel @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}

// SCALAR: extern float someKernel(float, intptr_t);

kgen.kernel @someBufferKernel(%a: !meta.buffer<?, ?>) -> index {
  %size = meta.buffer.size %a : !meta.buffer<?, ?>
  kgen.return %size : index
}

// BUFFER: extern intptr_t someBufferKernel(intptr_t, void *, int8_t);
