// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// Regression test for MOCO-3220: LowerPOPToLLVM was expanding struct
// operands into individual fields, breaking C ABI compatibility.

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @extern_c_struct
kgen.func @extern_c_struct() {
  %s = kgen.param.constant: struct<(scalar<si8>, scalar<si8>, scalar<si8>, scalar<si8>)> = <{ 0, 1, 2, 3 }>

  // CHECK: llvm.call @c_func(%{{.*}}) : (!llvm.struct<(i8, i8, i8, i8)>) -> ()
  pop.external_call @c_func(%s)
    : (!kgen.struct<(scalar<si8>, scalar<si8>, scalar<si8>, scalar<si8>)>) -> ()
  kgen.return
}
}
