// RUN: kgen-opt -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func private @e2e_lower
kgen.func @e2e_lower(%a: !meta.scalar<f32>, %b: !meta.scalar<f32>, %cond: i1) -> !meta.scalar<f32> {
  // CHECK: llvm.cond_br
  %r = scf.if %cond -> (!meta.scalar<f32>) {
    // CHECK: llvm.fadd
    %0 = pop.add %a, %b : !meta.scalar<f32>
    scf.yield %0 : !meta.scalar<f32>
  } else {
    scf.yield %a : !meta.scalar<f32>
  }
  kgen.return %r : !meta.scalar<f32>
}
