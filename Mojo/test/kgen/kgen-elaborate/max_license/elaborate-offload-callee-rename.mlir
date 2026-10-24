// RUN: kgen-opt %s -elaborate-generators="use-parametric-interpret=false" \
// RUN:   -allow-unregistered-dialect -mlir-print-ir-before=elaborate-generators \
// RUN:   2>&1 | FileCheck %s

// A parametric kernel that another kernel of the same offload group calls is
// emitted twice: as the kernel entry, and as a callee renamed with a `_callee`
// suffix so the two bodies do not share a name.

// CHECK-DAG: kgen.generator @leaf_callee<n>(
// CHECK-DAG: kgen.call @leaf_callee<n>(

kgen.generator @leaf<n: index>(%p: !kgen.pointer<index>) no_inline {
  %0 = kgen.param.constant = <n>
  pop.store %0, %p : !kgen.pointer<index>
  hlcf.return
}

kgen.generator @root<n: index>(%p: !kgen.pointer<index>) {
  kgen.call @leaf<n>(%p) : (!kgen.pointer<index>) -> ()
  hlcf.return
}

// CHECK-LABEL: kgen.func export @offload_entry
kgen.generator export @offload_entry() {
  kgen.param.declare nvptx: target = <#kgen.target<triple = "nvptx64-nvidia-cuda",
                                         arch = "sm_80",
                                         simd_bit_width = 128,
                                         index_bit_width = 64,
                                         tune_cpu = "sm_80",
                                         data_layout = "e-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64">>
  // CHECK: kgen.param.constant: string = <"; ModuleID
  // CHECK-SAME: define {{.*}}@leaf_n_7
  // CHECK: kgen.param.constant: string = <"; ModuleID
  // CHECK-SAME: define {{.*}}@root_n_7
  %0 = kgen.compile_offload<nvptx, 2, "", "",
                            :(!kgen.pointer<index>) -> () @leaf<7>>
                            : !kgen.struct<(string, index)>
  %1 = kgen.compile_offload<nvptx, 2, "", "",
                            :(!kgen.pointer<index>) -> () @root<7>>
                            : !kgen.struct<(string, index)>
  hlcf.return
}
