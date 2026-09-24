// RUN: kgen-opt -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters -kgen-print-inline-type-values | FileCheck %s
// RUN: kgen-opt -emit-bytecode -allow-unregistered-dialect %s | kgen-opt -allow-unregistered-dialect -verify-parameters -kgen-print-inline-type-values | FileCheck %s

kgen.generator @kernel() {
  hlcf.return
}

// `target_attrs` is opaque to the core dialect: any target-prefixed key
// round-trips through both the textual and the bytecode form, and the keys
// below are deliberately not a real backend's.

// CHECK-LABEL: kgen.generator @compile_offload_with_target_attrs
kgen.generator @compile_offload_with_target_attrs() {
  kgen.param.declare dummy: target = <#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>>

  // CHECK: kgen.compile_offload<dummy, 0, "", "", :() -> () @kernel {target_attrs = {backend.flag = true, backend.shape = [2, 4]}}> : !kgen.none
  %0 = kgen.compile_offload<dummy, 0, "", "", : ()->() @kernel {target_attrs = {"backend.shape" = [2, 4], "backend.flag" = true}}> : !kgen.none

  hlcf.return
}

// An offload without `target_attrs` keeps printing without the dictionary.

// CHECK-LABEL: kgen.generator @compile_offload_without_target_attrs
kgen.generator @compile_offload_without_target_attrs() {
  kgen.param.declare dummy: target = <#kgen.target<triple = "unknown", arch = "", simd_bit_width = 128>>

  // CHECK: kgen.compile_offload<dummy, 0, "", "", :() -> () @kernel> : !kgen.none
  // CHECK-NOT: target_attrs
  %0 = kgen.compile_offload<dummy, 0, "", "", : ()->() @kernel> : !kgen.none

  hlcf.return
}
