// RUN: mo-opt %s --mogg-canonicalize-funcs-for-merge --allow-unregistered-dialect | FileCheck %s
// RUN: mo-opt %s --mogg-canonicalize-funcs-for-merge --mogg-merge-identical-funcs --allow-unregistered-dialect | FileCheck %s --check-prefix=MERGE

//===----------------------------------------------------------------------===//
// The pass rewrites every device id reachable inside a kernel stub -- in op
// attributes and in the `!mgp.tensor` types it operates on -- to the canonical
// id 0, keeping the label. Per-device copies of a kernel (e.g. across EP shards)
// then hash equal and merge. Ops and types outside a stub keep their real
// device ids. See GEX-4239. It also rewrites every dynamic dim of a
// `#mosh.ape` op attribute inside a stub to `dyn`.
//===----------------------------------------------------------------------===//

// Two stubs identical but for the device id (gpu:3 vs gpu:5), in both an op
// attribute and a result type.

// CHECK-LABEL: mogg.func @k0 private
// CHECK: "test.op"() {dev = #M.device_ref<"gpu", 0>}
// CHECK: -> !mgp.tensor<4xf32, <"gpu", 0>>
mogg.func @k0 private dispatch(deferred)() attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"k">>} {
  "test.op"() {dev = #M.device_ref<"gpu", 3>} : () -> ()
  %0 = "test.produce"() : () -> !mgp.tensor<4xf32, <"gpu", 3>>
  mogg.output
}

// CHECK-LABEL: mogg.func @k1 private
// CHECK: "test.op"() {dev = #M.device_ref<"gpu", 0>}
// CHECK: -> !mgp.tensor<4xf32, <"gpu", 0>>
mogg.func @k1 private dispatch(deferred)() attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"k">>} {
  "test.op"() {dev = #M.device_ref<"gpu", 5>} : () -> ()
  %0 = "test.produce"() : () -> !mgp.tensor<4xf32, <"gpu", 5>>
  mogg.output
}

// After canonicalization the two stubs are identical, so both call sites in the
// caller collapse onto a single survivor.
// MERGE-LABEL: mogg.func @caller private
// MERGE:      mogg.call @[[K:k[01]]]
// MERGE-NEXT: mogg.call @[[K]]
// MERGE-NOT:  mogg.call @k
mogg.func @caller private dispatch(deferred)() attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"caller">>} {
  mogg.call @k0
  mogg.call @k1
  mogg.output
}

// A device id OUTSIDE any kernel stub is left untouched.
// CHECK-LABEL: func.func @outside
// CHECK: #M.device_ref<"gpu", 3>
func.func @outside() attributes {dev = #M.device_ref<"gpu", 3>} {
  return
}

// Two stubs identical but for a symbolic dim name in an op's shape attribute.
// Dynamic dims become `dyn`; static dims stay.

// CHECK-LABEL: mogg.func @s0 private
// CHECK: "test.op"() {shape = #mosh<ape[dyn, 4]> : !mosh.ape}
mogg.func @s0 private dispatch(deferred)() attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"s">>} {
  "test.op"() {shape = #mosh<ape[batch, 4]> : !mosh.ape} : () -> ()
  mogg.output
}

// CHECK-LABEL: mogg.func @s1 private
// CHECK: "test.op"() {shape = #mosh<ape[dyn, 4]> : !mosh.ape}
mogg.func @s1 private dispatch(deferred)() attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"s">>} {
  "test.op"() {shape = #mosh<ape[seq, 4]> : !mosh.ape} : () -> ()
  mogg.output
}

// MERGE-LABEL: mogg.func @shape_caller private
// MERGE:      mogg.call @[[S:s[01]]]
// MERGE-NEXT: mogg.call @[[S]]
mogg.func @shape_caller private dispatch(deferred)() attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"shape_caller">>} {
  mogg.call @s0
  mogg.call @s1
  mogg.output
}

// A dim name OUTSIDE any kernel stub is left untouched.
// CHECK-LABEL: func.func @outside_shape
// CHECK: #mosh<ape[batch, 4]>
func.func @outside_shape() attributes {shape = #mosh<ape[batch, 4]> : !mosh.ape} {
  return
}
