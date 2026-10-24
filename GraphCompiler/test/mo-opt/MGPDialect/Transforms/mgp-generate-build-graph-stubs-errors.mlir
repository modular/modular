// RUN: mo-opt %s --mgp-generate-build-graph-stubs --allow-unregistered-dialect --split-input-file --verify-diagnostics

// The producer's launch context must be an argument of the kernel stub (in any
// position) -- a context materialized inside the body leaves no argument slot
// for the builder to take over.

mogg.primitive.decl @foreach_impl registration = "mogg.call.foreach" function = #mogg.kernel_function<!lit.generator<() -> i1>, [], [], []>

// expected-error @below {{device-graph recording stub generation: expected the producer launch context to be an argument of the kernel stub}}
mogg.func @stub_1 private dispatch(deferred) (%p: !mogg.pointer) attributes {abi = #mogg.abi<MOGG>, constantResources = [], device = "gpu", fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.k">>} {
  %ctx = "test.make_ctx"() : () -> !mogg.context
  %shape = "test.shape"() : () -> !mogg.shape
  %t = "test.tensor"() : () -> !mogg.tensor
  mogg.call.foreach @foreach_impl shape(%shape) context(%ctx) (%i : !map.index<2>) capturing {%a = %t : !mogg.tensor} device = "gpu" {
  }
  mogg.output
}

mgp.model @dg device_graph : mof devices [#M.device_ref<"gpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      %p = "test.pointer"() : () -> !mogg.pointer
      mogg.call @stub_1 %p : !mogg.pointer
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}
