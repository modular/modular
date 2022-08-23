// RUN: kgen %s -emit -kernel="someKernel:f32(f32,index):%t.o"
// RUN: cat %t.h | FileCheck %s

kgen.kernel @someKernel(%arg1: f32, %arg2: index) -> f32 {
  kgen.return %arg1 : f32
}

// CHECK: extern float someKernel(float, intptr_t);
