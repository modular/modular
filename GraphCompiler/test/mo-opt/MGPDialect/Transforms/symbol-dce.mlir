// RUN: mo-opt -symbol-dce

mogg.func @callee_1()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", dispatch = #mogg.dispatch<deferred>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"callee_1">>} {
  mogg.output
}

mogg.func @callee_2()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", dispatch = #mogg.dispatch<deferred>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"callee_2">>} {
  mogg.output
}

mogg.func @caller()
  attributes {abi = 2 : i32, constantResources = [], device = "cpu", dispatch = #mogg.dispatch<deferred>, fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"caller">>} {
  mogg.call @callee_1
  mogg.call @callee_2
  mogg.output
}

// CHECK-NOT: mogg.func @callee_2
// CHECK-NOT: mogg.func @caller
// CHECK: mogg.call @callee_1
// CHECK-NOT: mogg.call
mgp.model @post_abi_fusion : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %3 = mgp.runtime.create %0, %1[1]
  mef.output %3, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  mogg.call @callee_1
  mef.output %arg0 : !mef.chain
}
