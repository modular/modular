// RUN: mo-opt -allow-unregistered-dialect --mogg-lower-compiled-routines %s | FileCheck %s

// An init-block execute with no data operands still reaches its slot through
// the context, so it must be ordered by guarding the context

// CHECK-LABEL: mgp.model @init_execute_guard
// CHECK: %[[CTX:.*]] = mgp.context.create
// CHECK: %[[RUNTIME_CHAIN:.*]] = mgp.runtime.create
// CHECK: %[[MODULE_LOAD_CHAIN:.*]] = mgp.module.load %[[RUNTIME_CHAIN]], %[[CTX]][{{.*}}], dense_resource<fake_binary> : tensor<1xui8>, ["__gen_0"]
// CHECK: %[[GUARDED_CTX:.*]] = mgp.guard %[[MODULE_LOAD_CHAIN]], %[[CTX]] : !mgp.context
// CHECK: "mgp.generic.execute"(%[[GUARDED_CTX]]) <{aRuntimeSlot

mgp.model @init_execute_guard : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  "mogg.compiled.load"() {symbols = ["__gen_0"], value = dense_resource<fake_binary> : tensor<1xui8>} : () -> ()
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %out_chain = mgp.runtime.create %0, %1[1]

  %3 = mogg.compiled.call @__gen_0() fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : () -> index
  %4 = "test.sink"(%3, %0) : (index, !mef.chain) -> !mef.chain

  mef.output %4, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  mef.output %arg0 : !mef.chain
}

// An execute-block execute must NOT be guarded: its slot is populated during
// init, which always completes before the execute phase runs.

// CHECK-LABEL: mgp.model @execute_block_no_guard
// CHECK: } execute
// CHECK-NOT: mgp.guard
// CHECK: "mgp.generic.execute"

mgp.model @execute_block_no_guard : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  "mogg.compiled.load"() {symbols = ["__gen_0"], value = dense_resource<fake_binary> : tensor<1xui8>} : () -> ()
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[0]
  mef.output %2, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %0 = grt.chain.create ()
  %1 = index.constant 42
  %2 = index.constant -20
  %3 = mogg.compiled.call @__gen_0(%1, %2) fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : (index, index) -> index
  %4 = "test.sink"(%3, %0) : (index, !mef.chain) -> !mef.chain
  mef.output %4 : !mef.chain
}
