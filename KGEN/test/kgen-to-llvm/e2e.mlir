// RUN: kgen-opt -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func private @e2e_lower
// CHECK-NOT: unrealized_conversion_cast
kgen.func @e2e_lower(%a: !pop.scalar<f32>, %b: !pop.scalar<f32>, %cond: i1) -> !pop.scalar<f32> {
  pop.external_call @foo() : () -> ()
  // CHECK: llvm.cond_br
  %r = scf.if %cond -> (!pop.scalar<f32>) {
    // CHECK: llvm.fadd
    %0 = pop.add %a, %b : !pop.scalar<f32>
    scf.yield %0 : !pop.scalar<f32>
  } else {
    scf.yield %a : !pop.scalar<f32>
  }
  kgen.return %r : !pop.scalar<f32>
}

// CHECK: llvm.func @foo
