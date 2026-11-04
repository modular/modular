// RUN: kgen-opt -split-input-file -pass-pipeline='builtin.module(kgen.func(lower-pop-to-llvm))' %s | FileCheck %s

// Atomic load/store skip the RMW/cmpxchg metal4_0 gate: kernels rely on their
// seq_cst default on metal3_2 targets.
module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  // CHECK-LABEL: @load_seq_cst
  kgen.func @load_seq_cst(%ptr: !kgen.pointer<scalar<si32>>,
                          %out: !kgen.pointer<scalar<si32>>) {
    // CHECK: llvm.load %{{.*}} atomic seq_cst
    %0 = pop.load atomic seq_cst %ptr : !kgen.pointer<scalar<si32>>
    pop.store %0, %out : !kgen.pointer<scalar<si32>>
    hlcf.return
  }

  // CHECK-LABEL: @store_seq_cst
  kgen.func @store_seq_cst(%ptr: !kgen.pointer<scalar<si32>>,
                           %val: !kgen.scalar<si32>) {
    // CHECK: llvm.store %{{.*}}, %{{.*}} atomic seq_cst
    pop.store atomic seq_cst %val, %ptr : !kgen.pointer<scalar<si32>>
    hlcf.return
  }
}
