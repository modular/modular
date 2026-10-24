// RUN: mo-opt %s --mgp-lower-primitives --allow-unregistered-dialect | FileCheck %s

// Regression guard: `mgp.device_context.select_stream` is a runtime-executed
// primitive (it implements no `EmitsMojoInterface`), so `mgp-lower-primitives`
// must leave it intact rather than wrapping it in a compiled kernel region.
// Lowering it into an async compiled region that produces a `DeviceContext`
// value previously asserted and hung in `Model.execute`; keeping it a bare op
// lets the MEF executor dispatch it by name to its C++ primitive.

module {
  // CHECK-LABEL: mgp.model @sel
  mgp.model @sel : mof devices [#M.device_ref<"gpu", 0>] init (%wr: !mgp.opaque<"weights_registry">, %dc: !mgp.device_context<<"gpu", 0>>) {
    %0 = grt.chain.create ()
    %1 = mgp.context.create 2
    %2 = mgp.runtime.create %0, %1[1]
    mef.output %2, %1, %wr, %dc : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
  } execute (%ch: !mef.chain, %ctx: !mgp.context, %wr: !mgp.opaque<"weights_registry">, %dc: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain, !mgp.device_context<<"gpu", 0>>) {
    // CHECK: mgp.device_context.select_stream[1]
    // CHECK-NOT: mogg.routine.async
    // CHECK-NOT: mgp.generic.execute
    %view = mgp.device_context.select_stream[1] %dc : !mgp.device_context<<"gpu", 0>>
    mef.output %ch, %view : !mef.chain, !mgp.device_context<<"gpu", 0>>
  }
}
