// RUN: kgen-opt %s -infer-function-attrs -allow-unregistered-dialect | FileCheck %s

// AIR barriers are external calls; only isConvergentOp marks them convergent.
module attributes {M.target_info = #M.target<triple = "air64-apple-macosx", arch = "apple-m3", data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32", simd_bit_width = 128>} {

  // CHECK: kgen.func @wg_barrier() attributes {convergent}
  kgen.func @wg_barrier() {
    %struct = kgen.param.constant: struct<(scalar<si32>, scalar<si32>)> = <{ 2, 1 }>
    pop.call_llvm_intrinsic side_effecting<1> "llvm.air.wg.barrier", (%struct) : (!kgen.struct<(scalar<si32>, scalar<si32>)>) -> ()
    hlcf.return
  }

  // CHECK: kgen.func @simdgroup_barrier() attributes {convergent}
  kgen.func @simdgroup_barrier() {
    %struct = kgen.param.constant: struct<(scalar<si32>, scalar<si32>)> = <{ 0, 4 }>
    pop.call_llvm_intrinsic side_effecting<1> "llvm.air.simdgroup.barrier", (%struct) : (!kgen.struct<(scalar<si32>, scalar<si32>)>) -> ()
    hlcf.return
  }

  // CHECK: kgen.func @caller() attributes {convergent}
  kgen.func @caller() {
    kgen.call @wg_barrier() : () -> ()
    hlcf.return
  }
}
