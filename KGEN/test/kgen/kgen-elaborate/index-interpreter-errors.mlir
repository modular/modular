// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" -verify-diagnostics -split-input-file
// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=true" -split-input-file 2>&1 | FileCheck %s --check-prefix=CHECK-PARAM

// COM: use-parametric-interpret=true has slight difference from =false for error messages. 
//      Using FileCheck instead to check those with CHECK-PRAMA prefix.

 module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 32>, kgen.env = #kgen.env<{}>} {
// CHECK-PARAM: failed to interpret function @minus
// expected-note @below {{failed to interpret function @minus}}
kgen.generator @minus(%arg0: index, %arg1: index) -> index {
  // CHECK-PARAM: failed to interpret operation index.sub(4294967295 : index, -4294967295 : index)
  // CHECK-PARAM: `index.sub` failed due to overflow
  // expected-note @below {{failed to interpret operation index.sub(4294967295 : index, -4294967295 : index)}}
  // expected-note @below {{`index.sub` failed due to overflow}}
  %0 = index.sub %arg0, %arg1
  kgen.return %0 : index
}

// CHECK-LABEL: kgen.func @callIt
// expected-error @+1 {{function instantiation failed}}
kgen.generator export @callIt() -> index {
  // CHECK-PARAM: failed to compile-time evaluate function call
  // expected-note @below {{failed to compile-time evaluate function call}}
  kgen.param.declare value : index = <apply(:(index, index) -> index @minus, 4294967295, -4294967295)>
  %0 = kgen.param.constant: index = <value>
  kgen.return %0 : index
}
}
