// XFAIL: *

// RUN: mo-opt --mgp-lower-primitives --mogg-sequence-fusion --jit --mogg-lower-compiled-routines %s -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck %s

// CHECK-LABEL:--- Running 'mgp_assert_false':
// expected-error @+1 {{'mgp_assert_false' returned <error: ASSERT>}}
mgp.model @mgp_assert_false : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
    %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  %ch2 = mgp.runtime.create %ch0, %ctx[0]
  mef.output %ch2, %ctx : !mef.chain, %weights, !mgp.opaque<"weights_registry">, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> (!mef.chain, i1) {
  %ch0 = grt.chain.create ()
  %cond = index.bool.constant false
  %ch = mgp.assert %cond, "ASSERT"
  mef.output %ch, %cond: !mef.chain, i1
}
