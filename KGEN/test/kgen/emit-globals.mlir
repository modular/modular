// RUN: kgen -emit-llvm %s -o - | FileCheck %s

// CHECK: @global_var = internal global float
// CHECK: @llvm.global_ctors = appending global {{.*}} @global_init
// CHECK: @llvm.global_dtors = appending global {{.*}} @global_dtor

kgen.func @global_init() {
  kgen.return
}

kgen.func @global_dtor() {
  kgen.return
}

kgen.global @global_var : f32 (0, @global_init, @global_dtor)

kgen.func @kernel() -> f32 {
  %0 = pop.global.address @global_var : <f32>
  %1 = pop.load %0 : !pop.pointer<f32>
  kgen.return %1 : f32
}

kgen.export @kernel to C as @kernel
