// RUN: kgen -emit-llvm %s -o - | FileCheck %s

// CHECK-DAG: @global_var = internal global float
// CHECK-DAG: @llvm.global_ctors = appending global {{.*}} @global_init
// CHECK-DAG: @llvm.global_dtors = appending global {{.*}} @global_dtor

kgen.func @global_init() {
  kgen.return
}

kgen.func @global_dtor() {
  kgen.return
}

kgen.global @global_var : f32 [@global_init, @global_dtor](0)

kgen.func export C @kernel() -> f32 {
  %0 = pop.global.address @global_var : <f32>
  %1 = pop.load %0 : !kgen.pointer<f32>
  kgen.return %1 : f32
}

// CHECK-DAG: global_internal_var = internal global float undef
kgen.global @global_internal_var : f32

kgen.func export C @kernel2(%arg0: f32) {
  %0 = pop.global.address @global_internal_var : <f32>
  pop.store %arg0, %0: !kgen.pointer<f32>
  kgen.return
}
