// RUN: kgen-opt -split-input-file -pass-pipeline='lower-kgen-to-llvm,lower-scf-to-llvm,llvm.func(lower-pop-to-llvm)' %s -verify-diagnostics

// expected-error @below {{cannot run on operations with CFG regions}}
// expected-note @below {{try running it before lower-scf-to-llvm}}
kgen.func @stack_allocation(%cond: i1) {
  scf.if %cond {
    %0 = pop.stack_allocation 4 x !pop.simd<4, f32>
  }
  kgen.return
}

// -----

kgen.func @call_intrinsic(%inp: !pop.scalar<f32>) -> () {
  // expected-error @below {{'pop.call_llvm_intrinsic' op expected 0 or 1 results, but got 2 results.}}
  pop.call_llvm_intrinsic "llvm.round"(%inp) : (!pop.scalar<f32>) -> (!pop.scalar<f32>, !pop.scalar<f32>)
  kgen.return
}
