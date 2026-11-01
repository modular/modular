// RUN: kgen-opt %s -lower-kgen-to-llvm | kgen-translate -mlir-to-llvmir | FileCheck %s

 module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch = "gfx942", data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", simd_bit_width = 128>} {
   // COM: AMDGPU kernel byref args are emitted in the constant address space
   // COM: (kernarg segment, addrspace 4) with an `addrspacecast` to flat at the
   // COM: entry block; this lets `AMDGPUPromoteKernelArguments` and
   // COM: `InferAddressSpaces` recognize loaded pointers as global.

   // CHECK-LABEL: byval_with_structs(ptr addrspace(4) noalias noundef nonnull byref({ i64, i64 }) %0, ptr addrspace(4) noundef nonnull byref({ i64, i64 })
   // CHECK: addrspacecast ptr addrspace(4) %{{[0-9]+}} to ptr
   // CHECK: addrspacecast ptr addrspace(4) %{{[0-9]+}} to ptr
   kgen.func export @byval_with_structs(%arg0: !kgen.pointer<struct<(index, index)>> owned_in_mem, %arg1: !kgen.pointer<struct<(index, index)>> imm_mem) {
     hlcf.return
   }
}
