// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
// CHECK: void @kernel()
kgen.func export @kernel() {
  kgen.return
}

// CHECK: void @kernel_grid_constant(ptr noundef nonnull byval({}) %0, ptr noundef nonnull byval({}) align 64 "nvvm.grid_constant" %1)
kgen.func export @kernel_grid_constant(%0: !kgen.pointer<none> read_mem, %1: !kgen.pointer<none> read_mem) attributes {
  LLVMArgMetadata = [{}, {nvvm.grid_constant = unit}]
} {
  kgen.return
}

// CHECK: !llvm.module.flags = !{![[LLVM_MODULE_FLAGS:.+]]}
// CHECK: ![[LLVM_MODULE_FLAGS]] = !{i32 2, !"Debug Info Version", i32 3}


}
