// RUN: mo-opt %s --mgp-finalize-device-graphs --allow-unregistered-dialect --split-input-file --verify-diagnostics

// Finalization owns the invariant that no `builtin.unrealized_conversion_cast`
// survives device-graph synthesis, so it diagnoses the cases it cannot resolve
// instead of letting them fail later in EmitMojo.

// The builder's backing device context dispatches eagerly, so it may only be
// handed to ops audited to use it for host-side buffer views. `mgp.sync` is
// `Pure` yet records a device event, so it must not silently get the backing
// context -- effect-freedom is not the criterion.

mgp.model @dg_unsafe_context_user device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %out = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    // expected-error @+1 {{'mgp.sync' is not known to be safe against the graph builder's backing device context}}
    %dev = builtin.unrealized_conversion_cast %b : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
    %bch = "test.make_chain"() : () -> !mef.chain
    %synced = mgp.sync (%bch, %ctx, %dev) : (!mef.chain, !mgp.context, !mgp.device_context<<"gpu", 0>>) -> !mef.chain
    mgp.device_graph.collective.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %out : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// The collective's `devices` attribute selects which execute-block device
// context backs each per-device graph recorder. A device reference that does
// not appear in the model's `devices` list leaves `collectDeviceContexts`
// nothing to match against.

mgp.model @dg_no_exec_device device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  // expected-error @+1 {{device-graph finalization: no execute device context matches device}}
  mgp.device_graph.collective builder(%b) stable() <"gpu", 1> {
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}
