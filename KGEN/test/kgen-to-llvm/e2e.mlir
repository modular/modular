// RUN: kgen-opt -lower-to-llvm %s | FileCheck %s

// CHECK-LABEL: llvm.func linkonce @e2e_lower
// CHECK-NOT: unrealized_conversion_cast
kgen.func @e2e_lower(%a: !pop.simd<1, f32>, %b: !pop.simd<1, f32>, %cond: i1) -> !pop.simd<1, f32> {
  pop.external_call @foo() : () -> ()
  // CHECK: llvm.cond_br
  %r = scf.if %cond -> (!pop.simd<1, f32>) {
    // CHECK: llvm.fadd
    %0 = pop.add %a, %b : !pop.simd<1, f32>
    scf.yield %0 : !pop.simd<1, f32>
  } else {
    scf.yield %a : !pop.simd<1, f32>
  }
  kgen.return %r : !pop.simd<1, f32>
}

// CHECK: llvm.func @foo
