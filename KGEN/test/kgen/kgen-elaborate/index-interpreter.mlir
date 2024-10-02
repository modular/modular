// RUN: kgen-opt %s -split-input-file -elaborate-generators -allow-unregistered-dialect | FileCheck %s

module attributes {M.target_info = #M.target<triple = "", arch = "", features = "", data_layout = "",  simd_bit_width = 128, index_bit_width = 64>, kgen.env = #kgen.env<{}>} {
kgen.generator @compare(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp sgt(%arg0, %arg1)
  kgen.return %0 : i1
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> i1 {
  kgen.param.declare value : i1 = <apply(:(index, index) -> i1 @compare, 4294967295, 5)>
  // CHECK-NEXT:  kgen.param.constant: i1 = <1>
  %0 = kgen.param.constant: i1 = <value>
  kgen.return %0 : i1
}
}

// -----

// COM: Cmp falls back to folder when target is not specified

kgen.generator @compare(%arg0: index, %arg1: index) -> i1 {
  %0 = index.cmp sgt(%arg0, %arg1)
  kgen.return %0 : i1
}

// CHECK-LABEL: kgen.func export @main
kgen.generator export @main() -> i1 {
  kgen.param.declare value : i1 = <apply(:(index, index) -> i1 @compare, 4294967294, 5)>
  // CHECK-NEXT:  kgen.param.constant: i1 = <1>
  %0 = kgen.param.constant: i1 = <value>
  kgen.return %0 : i1
}