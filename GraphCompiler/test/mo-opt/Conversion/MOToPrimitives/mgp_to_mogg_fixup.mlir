// RUN: mo-opt -allow-unregistered-dialect --mogg-lower-compiled-routines %s | FileCheck %s

// Jit compile kernels will output the function execution in a particular way,
// we need to run a fixup pass over it to convert to the form MGP expects.

mgp.model @basic : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
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

// CHECK-LABEL: mgp.model @basic
// CHECK: %[[CTX:.*]] = mgp.context.create 2
// CHECK: %[[KERNEL_CTX:.*]] = mgp.runtime.create
// CHECK-NEXT: mgp.module.load %[[KERNEL_CTX]], %[[CTX]][{{.*}}], dense_resource<fake_binary> : tensor<1xui8>, ["__gen_0"]

// CHECK: } execute
// CHECK: "mgp.generic.execute"(%{{.*}}, dSummary = {{.*}}}

mgp.model @basic_control_flow : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  "mogg.compiled.load"() {symbols = ["__gen_0"], value = dense_resource<fake_binary> : tensor<1xui8>} : () -> ()
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[0]
  mef.output %2, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.buffer) -> !mef.chain {
  %lhs = index.constant 42
  %rhs = index.constant -20
  %1 = mgp.buffer.to_bool %ch0, %arg2 : !mgp.buffer
  %2 = grt.if %1 (%ch0: !mef.chain, %lhs: index, %rhs: index, %arg1 as %new: !mgp.context) -> !mef.chain {
    %3 = mogg.compiled.call @__gen_0(%lhs, %rhs) fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : (index, index) -> index
    mef.output %ch0 : !mef.chain
  } else {
    mef.output %ch0 : !mef.chain
  }
  mef.output %2 : !mef.chain
}

// CHECK-LABEL: mgp.model @basic_control_flow
// CHECK: %[[CTX:.*]] = mgp.context.create 2
// CHECK: %[[KERNEL_CTX:.*]] = mgp.runtime.create
// CHECK-NEXT: mgp.module.load %[[KERNEL_CTX]], %[[CTX]][{{.*}}], dense_resource<fake_binary> : tensor<1xui8>, ["__gen_0"]
// CHECK: } execute
// CHECK: "mgp.generic.execute"(%{{.*}}, dSummary = {{.*}}}


// CHECK-LABEL: mgp.model @basic_init

// CHECK: %[[CTX:.*]] = mgp.context.create
// CHECK: %[[CHAIN:.*]] = mgp.runtime.create
// CHECK: "mgp.generic.execute"(%[[CTX]], {{.*}}, %[[CHAIN]]) <{aRuntimeSlot

mgp.model @basic_init : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %out_chain = mgp.runtime.create %0, %1[1]

  %ch = "mgp.module.load.placeholder"() : () -> !mef.chain

  %cst = index.constant 42
  %3 = mogg.compiled.call @__gen_0(%cst, %ch) fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"mogg.routine.async">> constantResources = [], usedParams = [] {device = #M.device_ref<"cpu", 0>} : (index, !mef.chain) -> index
  %4 = "test.sink"(%3, %0) : (index, !mef.chain) -> !mef.chain

  mef.output %4, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  mef.output %arg0 : !mef.chain
}
