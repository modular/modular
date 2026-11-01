// RUN: mo-opt %s --mgp-lower-primitives --allow-unregistered-dialect | FileCheck %s

// The waits that fork a side stream off its base stream and join it back stay
// out of sequence fusion. Fused, a fork becomes a deferred task behind
// main-stream launches on the device's worker, so the side stream branches off
// late; a plain cross-stream wait still fuses.

module {
  // CHECK-LABEL: mgp.model @waits
  mgp.model @waits : mof devices [#M.device_ref<"gpu", 0>] init (%wr: !mgp.opaque<"weights_registry">, %dc: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %2, %1, %wr, %dc : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
  } execute (%ch: !mef.chain, %ctx: !mgp.context, %wr: !mgp.opaque<"weights_registry">, %dc: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
    %view = mgp.device_context.select_stream[1] %dc : !mgp.device_context<<"gpu", 0>>
    // CHECK: mgp.device_wait({{.*}}) device <"gpu", 0> isSideStreamFork
    // CHECK: fusionInfo = #device_wait{{[0-9]*}} inOutSignature
    %fork = mgp.device_wait(%ch, %ctx, %view, %dc) device <"gpu", 0> isSideStreamFork : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
    // CHECK: mgp.device_wait({{.*}}) device <"gpu", 0> isSideStreamJoin
    // CHECK: fusionInfo = #device_wait{{[0-9]*}} inOutSignature
    %join = mgp.device_wait(%fork, %ctx, %dc, %view) device <"gpu", 0> isSideStreamJoin : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
    // CHECK: mgp.device_wait({{.*}}) device <"gpu", 0> :
    // CHECK: fusionInfo = #device_wait{{[0-9]*}} canBeFusedInto = true canFuseIntoOther = true
    %plain = mgp.device_wait(%join, %ctx, %view, %dc) device <"gpu", 0> : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
    mef.output %plain : !mef.chain
  }
}
