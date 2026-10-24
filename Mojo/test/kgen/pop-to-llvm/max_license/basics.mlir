// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

module attributes {M.target_info = #M.target<triple = "amdgpu9.42-amd-amdhsa", arch = "gfx942", data_layout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9", simd_bit_width = 128, index_bit_width = 64>} {

// CHECK-LABEL: kgen.func @multi_lifetimes
kgen.func @multi_lifetimes() {
  // CHECK: %[[A0:.*]] = llvm.alloca {{.*}} x i64
  // CHECK-NEXT: llvm.intr.lifetime.end %[[A0]]
  // CHECK-NEXT: %[[A0_CAST:.*]] = llvm.addrspacecast %[[A0]] : !llvm.ptr<5> to !llvm.ptr
  // CHECK: %[[A1:.*]] = llvm.alloca {{.*}} x i32
  // CHECK-NEXT: llvm.intr.lifetime.end %[[A1]]
  // CHECK-NEXT: %[[A1_CAST:.*]] = llvm.addrspacecast %[[A1]] : !llvm.ptr<5> to !llvm.ptr
  %0 = pop.stack_allocation 1 x i64 marked
  %1 = pop.stack_allocation 1 x i32 marked
  // CHECK-NEXT: lifetime.start %[[A0]]
  // CHECK-NEXT: lifetime.start %[[A1]]
  pop.stack_alloc.lifetime.start(%0, %1) : !kgen.pointer<i64>, !kgen.pointer<i32>
  // CHECK-NEXT: lifetime.end %[[A0]]
  // CHECK-NEXT: lifetime.end %[[A1]]
  pop.stack_alloc.lifetime.end(%0, %1) : !kgen.pointer<i64>, !kgen.pointer<i32>
  // CHECK-NEXT: return
  hlcf.return
}

}
