// RUN: kgen-opt %s -lower-kgen-to-llvm -verify-diagnostics

module attributes {M.target_info = #M.target<triple="", cpu="skylake-avx512", features="+fma", data_layout="", simd_bit_width=128, tune_cpu="skylake-avx512">} {
kgen.func @rebind(%arg0: index) -> f32 {
  // expected-error @below {{invalid rebind between two unequal, unparametric types}}
  // expected-error @below {{failed to legalize operation}}
  %0 = kgen.rebind %arg0 : index to f32
  kgen.return %0 : f32
}
}
