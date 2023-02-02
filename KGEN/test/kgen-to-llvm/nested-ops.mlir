// RUN: kgen-opt %s -lower-kgen-to-llvm | FileCheck %s

module attributes {M.target_info = #M.target<triple="", cpu="", features="", pointer_size=8, simd_bit_width=128>} {

// CHECK-LABEL: @nested_ops
kgen.func @nested_ops(%cond: i1) -> index {
  // CHECK: scf.if
  %2 = scf.if %cond -> index {
    // CHECK-NOT: kgen.param.constant
    %0 = kgen.param.constant = <4>
    scf.yield %0 : index
  } else {
    // CHECK-NOT: kgen.param.constant
    %1 = kgen.param.constant = <1>
    scf.yield %1 : index
  }
  kgen.return %2 : index
}

}
