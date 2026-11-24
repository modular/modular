// RUN: kgen-opt %s -pass-pipeline='builtin.module(lower-kgen-to-llvm,llvm.func(lower-pop-to-llvm))' | FileCheck %s

module attributes {M.target_info = #M.target<triple="", arch="", features="", data_layout="", simd_bit_width=128>} {
  // A pointer parameter has no defining op to infer an alignment from.
  // CHECK-LABEL: llvm.func internal @cmpxchg_arg_ptr
  kgen.func @cmpxchg_arg_ptr(%ptr: !kgen.pointer<scalar<index>>,
                             %cmp: !kgen.scalar<index>,
                             %new: !kgen.scalar<index>) {
    // CHECK: llvm.cmpxchg %arg0, {{.*}} monotonic monotonic :
    %0 = pop.atomic.cmpxchg %ptr, %cmp, %new monotonic monotonic :
                      !kgen.pointer<scalar<index>>
    hlcf.return
  }
}
