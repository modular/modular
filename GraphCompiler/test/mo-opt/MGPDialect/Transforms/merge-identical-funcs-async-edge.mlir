// RUN: mo-opt %s -mogg-merge-identical-funcs -symbol-dce | FileCheck %s

// Regression test for GEX-3774: two structurally-identical region stubs each
// call a differently-named-but-equivalent inner stub via mogg.call.async.
// Because the async call is a call-graph edge, the inner-stub name is
// canonicalized before the region is hashed, so the inner stubs and the
// regions each collapse to one in a single merge + symbol-dce run.

mogg.func @inner_1 private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"inner_1">>} {
  mogg.output
}

mogg.func @inner_2 private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"inner_2">>} {
  mogg.output
}

mogg.func @region_a private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"region_a">>} {
  %t = mogg.call.async @inner_1
  mogg.async.join(%t)
  mogg.output
}

mogg.func @region_b private dispatch(deferred)()
  attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"region_b">>} {
  %t = mogg.call.async @inner_2
  mogg.async.join(%t)
  mogg.output
}

// Exactly one inner stub and one region stub survive, and the surviving
// region's async callsite points at the canonical inner stub.
// CHECK:       mogg.func @inner_1 private
// CHECK-NOT:   mogg.func @inner_2 private
// CHECK:       mogg.func @region_a private
// CHECK-NOT:   mogg.func @region_b private
// CHECK:         mogg.call.async @inner_1

// Both model executions target the single merged region.
// CHECK-LABEL: mgp.model @top_level
// CHECK:       mogg.compiled.call @region_a
// CHECK:       mogg.compiled.call @region_a
// CHECK-NOT:   @region_b
// CHECK-NOT:   @inner_2
mgp.model @top_level subgraph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %0, %1, %arg0, %arg1, %arg2 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %arg4: !mgp.device_context<<"gpu", 0>>, %input : !mgp.tensor<4x16xf32>, %idx: index) -> (!mef.chain) {
    mogg.compiled.call @region_a() fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"gpu", 0>} : () -> ()
    mogg.compiled.call @region_b() fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"gpu", 0>} : () -> ()
    mef.output %arg0 : !mef.chain
}
