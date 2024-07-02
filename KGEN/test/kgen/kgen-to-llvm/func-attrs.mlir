// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK-LABEL: @exclusive_ptr(ptr noalias %0)
kgen.func @exclusive_ptr(%arg0: !kgen.pointer<i32 exclusive(1)>) {
  kgen.return
}

// CHECK-LABEL: @exclusive_ptr_addrspace(ptr addrspace(7) noalias %0)
kgen.func @exclusive_ptr_addrspace(%arg0: !kgen.pointer<i32, 7 exclusive(1)>) {
  kgen.return
}

}
