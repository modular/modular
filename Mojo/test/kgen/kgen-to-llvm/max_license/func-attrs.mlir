// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {

// CHECK: @convergent_call() #[[X:.*]] {
kgen.func @convergent_call() attributes {convergent} {
  nvvm.barrier
  hlcf.return
}

// CHECK: attributes #[[X]] = { convergent "target-cpu" "target-features" }
}
