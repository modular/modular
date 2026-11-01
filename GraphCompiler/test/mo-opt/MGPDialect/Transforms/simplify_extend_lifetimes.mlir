// RUN: mo-opt --mgp-simplify-extend-lifetimes %s | FileCheck %s

// CHECK-LABEL: mgp.model @extend_lifetimes
// CHECK: execute (%[[CH:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context{{.*}}, %[[T1:.*]]: !mgp.tensor<2x4xf32>, %[[T2:.*]]: !mgp.tensor<2x4xf32>

// CHECK: %[[OUT_CH:.*]] = mogg.routine.async (%[[CTX]], %[[T1]], %[[T2]]
// CHECK: inOutSignature = <"...">
// CHECK: mef.output %[[OUT_CH]]

mgp.model @extend_lifetimes : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %arg2: !mgp.tensor<2x4xf32>, %arg3: !mgp.tensor<2x4xf32>) -> (!mef.chain) {

  %3 = mogg.routine.async () : ()  -> !mef.chain{
    %ch = mogg.async.ready
    mogg.output %ch : !mef.chain
  } inOutSignature = <""> {device = #M.device_ref<"cpu", 0>}

  %5 = mgp.extend.lifetime %3, %arg1, %arg2, %arg3 : (!mef.chain, !mgp.context, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) -> !mef.chain

  mef.output %5: !mef.chain
}
