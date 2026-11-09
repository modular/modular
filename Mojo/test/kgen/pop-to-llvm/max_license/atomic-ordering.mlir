// RUN: kgen-opt -split-input-file -lower-global-pop-to-llvm %s | FileCheck %s

// AIR order: 0 relaxed, 2 acquire, 3 release, 4 acq_rel, 5 seq_cst.

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal3_2,+air2_7_0", data_layout = "">} {
  // CHECK-LABEL: @rmw_relaxed
  kgen.func @rmw_relaxed(%ptr: !kgen.pointer<scalar<ui32>>,
                         %val: !kgen.scalar<ui32>) {
    // CHECK: %[[ORDER:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: %[[SCOPE:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.atomic.global.add.u.i32(%{{.*}}, %{{.*}}, %[[ORDER]], %[[SCOPE]], %{{.*}})
    %0 = pop.atomic.rmw add(%ptr, %val) monotonic : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @cmpxchg_relaxed
  kgen.func @cmpxchg_relaxed(%ptr: !kgen.pointer<scalar<ui32>>,
                             %cmp: !kgen.scalar<ui32>,
                             %val: !kgen.scalar<ui32>) {
    // CHECK: %[[SUCC:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: %[[FAIL:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32(%{{.*}}, %{{.*}}, %{{.*}}, %[[SUCC]], %[[FAIL]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val monotonic unordered :
                      !kgen.pointer<scalar<ui32>>
    hlcf.return
  }
}

// -----

module attributes {M.target_info = #M.target<triple = "air64-unknown-unknown", arch = "", features = "+metal4_0,+air2_8_0", data_layout = "">} {
  // CHECK-LABEL: @rmw_acquire
  kgen.func @rmw_acquire(%ptr: !kgen.pointer<scalar<ui32>>,
                         %val: !kgen.scalar<ui32>) {
    // CHECK: %[[ORDER:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.atomic.global.add.u.i32(%{{.*}}, %{{.*}}, %[[ORDER]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.rmw add(%ptr, %val) acquire : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @rmw_release
  kgen.func @rmw_release(%ptr: !kgen.pointer<scalar<ui32>>,
                         %val: !kgen.scalar<ui32>) {
    // CHECK: %[[ORDER:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK: llvm.call @air.atomic.global.add.u.i32(%{{.*}}, %{{.*}}, %[[ORDER]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.rmw add(%ptr, %val) release : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @rmw_acq_rel
  kgen.func @rmw_acq_rel(%ptr: !kgen.pointer<scalar<ui32>>,
                         %val: !kgen.scalar<ui32>) {
    // CHECK: %[[ORDER:.*]] = llvm.mlir.constant(4 : i32)
    // CHECK: llvm.call @air.atomic.global.add.u.i32(%{{.*}}, %{{.*}}, %[[ORDER]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.rmw add(%ptr, %val) acq_rel : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @rmw_seq_cst
  kgen.func @rmw_seq_cst(%ptr: !kgen.pointer<scalar<ui32>>,
                         %val: !kgen.scalar<ui32>) {
    // CHECK: %[[ORDER:.*]] = llvm.mlir.constant(5 : i32)
    // CHECK: llvm.call @air.atomic.global.add.u.i32(%{{.*}}, %{{.*}}, %[[ORDER]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.rmw add(%ptr, %val) seq_cst : !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @rmw_xchg_local_acq_rel
  kgen.func @rmw_xchg_local_acq_rel(%ptr: !kgen.pointer<scalar<f32>, 3>,
                                    %val: !kgen.scalar<f32>) {
    // CHECK: %[[ORDER:.*]] = llvm.mlir.constant(4 : i32)
    // CHECK: llvm.call @air.atomic.local.xchg.f32(%{{.*}}, %{{.*}}, %[[ORDER]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.rmw xchg(%ptr, %val) acq_rel : !kgen.pointer<scalar<f32>, 3>
    hlcf.return
  }

  // CHECK-LABEL: @cmpxchg_acq_rel_acquire
  kgen.func @cmpxchg_acq_rel_acquire(%ptr: !kgen.pointer<scalar<ui32>>,
                                     %cmp: !kgen.scalar<ui32>,
                                     %val: !kgen.scalar<ui32>) {
    // CHECK-DAG: %[[SUCC:.*]] = llvm.mlir.constant(4 : i32)
    // CHECK-DAG: %[[FAIL:.*]] = llvm.mlir.constant(2 : i32)
    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32(%{{.*}}, %{{.*}}, %{{.*}}, %[[SUCC]], %[[FAIL]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val acq_rel acquire :
                      !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @cmpxchg_release_monotonic
  kgen.func @cmpxchg_release_monotonic(%ptr: !kgen.pointer<scalar<ui32>>,
                                       %cmp: !kgen.scalar<ui32>,
                                       %val: !kgen.scalar<ui32>) {
    // CHECK-DAG: %[[SUCC:.*]] = llvm.mlir.constant(3 : i32)
    // CHECK-DAG: %[[FAIL:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK: llvm.call @air.atomic.global.cmpxchg.weak.i32(%{{.*}}, %{{.*}}, %{{.*}}, %[[SUCC]], %[[FAIL]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val release monotonic :
                      !kgen.pointer<scalar<ui32>>
    hlcf.return
  }

  // CHECK-LABEL: @cmpxchg_local_seq_cst
  kgen.func @cmpxchg_local_seq_cst(%ptr: !kgen.pointer<scalar<ui32>, 3>,
                                   %cmp: !kgen.scalar<ui32>,
                                   %val: !kgen.scalar<ui32>) {
    // CHECK: %[[SUCC:.*]] = llvm.mlir.constant(5 : i32)
    // CHECK: %[[FAIL:.*]] = llvm.mlir.constant(5 : i32)
    // CHECK: llvm.call @air.atomic.local.cmpxchg.weak.i32(%{{.*}}, %{{.*}}, %{{.*}}, %[[SUCC]], %[[FAIL]], %{{.*}}, %{{.*}})
    %0 = pop.atomic.cmpxchg weak %ptr, %cmp, %val seq_cst seq_cst :
                      !kgen.pointer<scalar<ui32>, 3>
    hlcf.return
  }
}
