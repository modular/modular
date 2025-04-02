// RUN: kgen-opt %s -elaborate-generators -verify-diagnostics -split-input-file

 module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// expected-note @below {{failed to interpret function @minus}}
kgen.generator @minus(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{failed to interpret operation index.sub(4294967295 : index, -4294967295 : index)}}
  // expected-note @below {{`index.sub` failed due to overflow}}
  %0 = index.sub %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @callIt
kgen.generator export @callIt() -> index {
  // expected-error @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @minus, 4294967295, -4294967295)>
  %0 = kgen.param.constant: index = <value>
  kgen.return %0 : index
}
}
