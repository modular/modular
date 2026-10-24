// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// pop.fence on Metal must lower to air.atomic.fence, never to a bare LLVM
// fence, which crashes the Apple backend compiler (MOCO-4550).
// Constants: mem_flags 3 = device|threadgroup; order 5 = seq_cst;
// scope 0/1/2/4 = thread/threadgroup/device/simdgroup.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  // CHECK-LABEL: @fence_default_scope
  kgen.func @fence_default_scope() {
    // CHECK-DAG: %[[FLAGS:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK-DAG: %[[ORDER:.*]] = llvm.mlir.constant(5 : i32)
    // CHECK-DAG: %[[SCOPE:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.atomic.fence(%[[FLAGS]], %[[ORDER]], %[[SCOPE]])
    // CHECK-NOT: llvm.fence
    pop.fence seq_cst
    hlcf.return
  }

  // CHECK-LABEL: @fence_scopes
  kgen.func @fence_scopes() {
    // CHECK: llvm.mlir.constant(0 : i32)
    // CHECK: llvm.call @air.atomic.fence
    pop.fence syncscope("thread") seq_cst
    // CHECK: llvm.mlir.constant(4 : i32)
    // CHECK: llvm.call @air.atomic.fence
    pop.fence syncscope("warp") seq_cst
    // CHECK: llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.atomic.fence
    pop.fence syncscope("block") seq_cst
    // CHECK: llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.atomic.fence
    pop.fence syncscope("gpu") seq_cst
    hlcf.return
  }
}

// -----

// With the metal4_0 deployment-floor feature, directional orderings emit natively.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal4_0", data_layout = "">} {
  // The order operand is pinned per function at the "block" scope (1) so the
  // ordering constant can never alias the mem_flags (3) or scope constant.
  // CHECK-LABEL: @fence_metal4_acquire
  kgen.func @fence_metal4_acquire() {
    // CHECK-DAG: %[[FLAGS:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK-DAG: %[[ACQ:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK-DAG: %[[SCOPE:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.atomic.fence(%[[FLAGS]], %[[ACQ]], %[[SCOPE]])
    pop.fence syncscope("block") acquire
    hlcf.return
  }

  // CHECK-LABEL: @fence_metal4_release
  kgen.func @fence_metal4_release() {
    // CHECK-DAG: %[[FLAGS:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK-DAG: %[[REL:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK-DAG: %[[SCOPE:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.atomic.fence(%[[FLAGS]], %[[REL]], %[[SCOPE]])
    pop.fence syncscope("block") release
    hlcf.return
  }

  // CHECK-LABEL: @fence_metal4_acq_rel
  kgen.func @fence_metal4_acq_rel() {
    // CHECK-DAG: %[[FLAGS:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK-DAG: %[[AR:.*]] = llvm.mlir.constant(4 : i32)
    // CHECK-DAG: %[[SCOPE:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @air.atomic.fence(%[[FLAGS]], %[[AR]], %[[SCOPE]])
    pop.fence syncscope("block") acq_rel
    hlcf.return
  }
}
