// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
// CHECK: define void @kernel()
kgen.func export @kernel() {
  kgen.return
}

// CHECK: !nvvm.annotations = !{!1}
// CHECK: !1 = !{ptr @kernel, !"kernel", i32 1}

}
