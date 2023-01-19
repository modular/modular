// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,lower-scf-to-llvm,lower-pop-closures-to-llvm,llvm.func(lower-pop-to-llvm))' %s -verify-diagnostics

// expected-error @below {{cannot run on operations with CFG regions}}
// expected-note @below {{try running it before lower-scf-to-llvm}}
kgen.func @stack_allocation(%cond: i1) {
  scf.if %cond {
    %0 = pop.stack_allocation 4 x !pop.simd<4, f32>
  }
  kgen.return
}
// -----

kgen.func @no_nested_closures(%c0: !pop.closure<(f32, f32) -> f32>, %arg1: f32) -> () {
  // expected-error @below {{failed to legalize operation 'pop.partial_apply' that was explicitly marked illegal}}
  // expected-error @below {{nested closures are not supported}}
  %0 = pop.partial_apply %c0(?, %arg1) : !pop.closure<(f32, f32) -> f32>
  kgen.return
}
