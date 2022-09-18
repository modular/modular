// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s

// CHECK-LABEL: @nested_ops
kgen.func @nested_ops(%cond: i1, %a: !meta.buffer<?, f32>, %v: !meta.scalar<f32>) {
  // CHECK: scf.if
  scf.if %cond {
    // CHECK-NOT: meta.buffer.address
    %0 = meta.buffer.address %a : !meta.buffer<?, f32>
    pop.store %v, %0 : !meta.pointer<!meta.scalar<f32>>
  }
  kgen.return
}
