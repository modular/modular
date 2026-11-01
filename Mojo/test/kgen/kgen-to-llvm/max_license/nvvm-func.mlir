// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

 module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
    // CHECK-LABEL: @byval_with_structs(ptr noalias noundef nonnull byval({ i64, i64 }) %0, ptr noundef nonnull byval({ i64, i64 }) %1)
    kgen.func export @byval_with_structs(%arg0: !kgen.pointer<struct<(index, index)>> owned_in_mem, %arg1: !kgen.pointer<struct<(index, index)>> imm_mem) {
      hlcf.return
    }
}
