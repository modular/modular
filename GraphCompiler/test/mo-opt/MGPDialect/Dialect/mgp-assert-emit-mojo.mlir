
// RUN: mo-opt --mgp-lower-primitives %s | FileCheck %s

mgp.model @fold_true_assert : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque<"weights_registry">, %arg1: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %2 = mgp.runtime.create %0, %1[1]
  mef.output %2, %1, %arg0, %arg1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.opaque<"weights_registry">, %arg3: !mgp.device_context<<"cpu", 0>>, %cond : i1) -> (!mef.chain) {
  // Make sure that error message was turned into dynamic value.
  //
  // CHECK: mogg.routine.async
  // CHECK:  mgp.assert %{{.*}}, "Tensor type mismatch for argument 0: Expected shape: [2,4]"
  // CHECK:  constantResources = ["Tensor type mismatch for argument 0: Expected shape: [2,4]"]
  %assert = mgp.assert %cond, "Tensor type mismatch for argument 0: Expected shape: [2,4]"
  %chain = grt.chain.create (%arg0, %assert)
  mef.output %chain : !mef.chain
}
