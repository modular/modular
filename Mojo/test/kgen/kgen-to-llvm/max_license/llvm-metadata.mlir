// RUN: kgen-opt %s -lower-kgen-to-llvm -split-input-file | kgen-translate -split-input-file -mlir-to-llvmir | FileCheck %s

module attributes {M.target_info = #M.target<triple = "nvptx64-nvidia-cuda", arch = "sm_75", data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64", simd_bit_width = 128>} {
// CHECK: void @kernel()
kgen.func export @kernel() {
  hlcf.return
}

// CHECK: void @kernel_grid_constant(ptr noundef nonnull byval({}) %0, ptr noundef nonnull byval({}) align 64 "nvvm.grid_constant" %1)
kgen.func export @kernel_grid_constant(%0: !kgen.pointer<none> imm_mem, %1: !kgen.pointer<none> imm_mem) attributes {
  fnArgAttrs = [{}, {nvvm.grid_constant = unit}]
} {
  hlcf.return
}

// CHECK: !llvm.module.flags = !{![[LLVM_MODULE_FLAGS:.+]]}
// CHECK: ![[LLVM_MODULE_FLAGS]] = !{i32 2, !"Debug Info Version", i32 3}


}

// -----

module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch = "gfx942", data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", simd_bit_width = 128, index_bit_width = 64>} {

// CHECK: void @kernel()
kgen.func export @kernel() {
  hlcf.return
}

// COM: AMDGPU kernel byref args are emitted in the kernarg/constant address
// COM: space with an `addrspacecast` to flat at the entry block.

// CHECK: void @kernel_grid_constant(ptr addrspace(4) noundef nonnull byref({}) %0, ptr addrspace(4) noundef nonnull byref({}) %1)
// CHECK: addrspacecast ptr addrspace(4) %{{[0-9]+}} to ptr
// CHECK: addrspacecast ptr addrspace(4) %{{[0-9]+}} to ptr
kgen.func export @kernel_grid_constant(%0: !kgen.pointer<none> imm_mem, %1: !kgen.pointer<none> imm_mem) attributes {
  fnArgAttrs = [{}, {nvvm.grid_constant = unit}]
} {
  hlcf.return
}

// CHECK: !llvm.module.flags = !{![[LLVM_MODULE_FLAGS:.+]]}
// CHECK: ![[LLVM_MODULE_FLAGS]] = !{i32 2, !"Debug Info Version", i32 3}


}

// -----

// The `target-abi` module flag survives translation to real LLVM IR,
// matching what RISC-V/LoongArch read via `Module::getModuleFlag`.
module attributes {M.target_info = #M.target<triple = "unknown", arch = "", data_layout = "", simd_bit_width = 128, abi = "lp64d">} {

// CHECK: void @kernel()
kgen.func export @kernel() {
  hlcf.return
}

// CHECK: !llvm.module.flags
// CHECK-DAG: !{{[0-9]+}} = !{i32 1, !"target-abi", !"lp64d"}

}
