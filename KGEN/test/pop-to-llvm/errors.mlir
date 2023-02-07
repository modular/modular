// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(lower-kgen-to-llvm,lower-scf-to-llvm,lower-pop-closures-to-llvm,llvm.func(lower-pop-to-llvm))' %s -verify-diagnostics

module attributes {M.target_info = #M.target<triple="", cpu="", features="", data_layout="", simd_bit_width=128>} {
  // expected-error @below {{cannot run on operations with CFG regions}}
  // expected-note @below {{try running it before lower-scf-to-llvm}}
  kgen.func @stack_allocation(%cond: i1) {
    scf.if %cond {
      %0 = pop.stack_allocation 4 x !pop.simd<4, f32>
    }
    kgen.return
  }
}
