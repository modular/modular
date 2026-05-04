// RUN: not kgen-opt -lower-global-pop-to-llvm -verify-diagnostics %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

kgen.func @external_call(%a: !kgen.simd<1, ui32>) {
  pop.external_call @foo(%a) : (!kgen.simd<1, ui32>) -> ()
  // expected-error @below {{existing function with conflicting signature}}
  // expected-error @below {{failed to legalize}}
  %0 = pop.external_call @foo(%a) : (!kgen.simd<1, ui32>) -> !kgen.simd<4, f64>
  kgen.return
}

}
