// RUN: kgen-opt -split-input-file -pass-pipeline='convert-kgen-to-llvm,convert-scf-to-llvm,llvm.func(convert-pop-to-llvm)' %s -verify-diagnostics

// expected-error @below {{cannot run on operations with CFG regions}}
// expected-note @below {{try running it before convert-scf-to-llvm}}
kgen.func @stack_allocation(%cond: i1) {
  scf.if %cond {
    %0 = pop.stack_allocation 4 : !meta.simd<4, f32>
  }
  kgen.return
}
