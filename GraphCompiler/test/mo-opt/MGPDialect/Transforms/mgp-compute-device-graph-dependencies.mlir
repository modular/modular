// RUN: mo-opt %s --mgp-compute-device-graph-dependencies --allow-unregistered-dialect --split-input-file | FileCheck %s

// Each `mogg.routine.async` staged by `mgp-stage-device-graphs` is
// replaced 1:1 by an `mgp.device_graph.dependencies` op: the region body is
// dissolved into the new op's body (referencing enclosing SSA directly), the
// inter-region ordering becomes explicit `!mgp.device_graph.node` edges, and
// `mogg.async` pack/unpack ops are inserted to bridge the dissolved boundary.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute ({{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// The collective body's block arg is just the builder; the body references the
// outer input directly (the op is not `IsolatedFromAbove`).
// CHECK: mgp.device_graph.collective builder(%[[B:[a-zA-Z0-9_]+]]) stable() <"gpu", 0> {

// Region A: no producer node -> empty `dependencies()`. Each dependencies op
// takes the collective builder as its `builder(...)` operand and carries a
// `!mgp.device_graph.builder` block arg. Its body packs the bare outer input
// so the verbatim `async.unpack` lines up; no device context is used, so no
// `unrealized_conversion_cast` is emitted.
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mogg.async.pack(%[[IN]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
// CHECK: } : !mgp.buffer<<"gpu", 0>>

// Region B: depends on A -> `dependencies(%[[NODEA]])`. Its body references A's
// new body result directly (boundary dissolved).
// CHECK: %[[NODEB:.*]], %[[B0:.*]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies(%[[NODEA]]) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : !mgp.tensor<2x3xf32, <"gpu", 0>>

// The collective yield carries B's body result directly as bare `T`.
// CHECK: mgp.device_graph.collective.yield %[[B0]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mogg.routine.async

mgp.model @dg device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %a = mogg.routine.async (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%bb) : !mgp.buffer<<"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    %b_ = mogg.routine.async (%a) : (%arg: !mgp.buffer<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.buffer<<"gpu", 0>>
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy()
      %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.b">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.collective.yield %b_ : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A collective region that uses a device context: the conversion re-sources the
// context from the dependencies op's `!mgp.device_graph.builder` block arg via
// a `builtin.unrealized_conversion_cast`, so the body no longer references the
// outer device-context operand. The region's ABI wraps the context as
// `!mogg.async<!mgp.device_context>`, so the cast is re-packed at the boundary
// and the body's verbatim `mogg.async.unpack` consumes it (the round-trip is
// folded later, in Stage 3).

// CHECK-LABEL: mgp.model @dg_ctx
// CHECK: mgp.device_graph.collective builder(%[[B:[a-zA-Z0-9_]+]]) stable() <"gpu", 0> {
// The dependencies op takes the collective builder as its `builder(...)` operand.
// CHECK: mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK-NEXT: ^bb0(%[[BUILDER:[a-zA-Z0-9_]+]]: !mgp.device_graph.builder):
// The device context is re-sourced from the body's builder block arg, not the
// outer context.
// CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %[[BUILDER]] : !mgp.device_graph.builder to !mgp.device_context<<"gpu", 0>>
// CHECK: %[[CTXP:.*]] = mogg.async.pack(%[[CAST]])
// CHECK: %[[CTXU:.*]] = mogg.async.unpack(%[[CTXP]])
// CHECK: "test.use_ctx"(%[[CTXU]])
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK-NOT: mogg.routine.async

mgp.model @dg_ctx device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %a = mogg.routine.async (%in, %gpu) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>, %ctxarg: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 0>>
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy()
      %t = mgp.tensor.create %ch, %bb, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.ctx">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.collective.yield %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A collective region that uses a HOST (`cpu`) device context. Unlike the
// device context above, a host context must NOT be re-sourced from the builder:
// the builder yields the graph's device context (gpu here), so recording
// host-side work against it would run on the device (and fault on devices
// whose memory is not host-addressable). The host context threads through
// directly from the outer execute-block context operand instead.

// CHECK-LABEL: mgp.model @dg_host_ctx
// CHECK: mgp.device_graph.collective builder(%[[B:[a-zA-Z0-9_]+]]) stable() <"gpu", 0> {
// CHECK: mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// No builder->cpu-context cast: the host context comes straight from the outer
// execute-block context operand.
// CHECK-NOT: builtin.unrealized_conversion_cast %{{.*}} : !mgp.device_graph.builder to !mgp.device_context<<"cpu", 0>>
// CHECK: mogg.async.pack(%[[CPU:[a-zA-Z0-9_]+]]) : !mgp.device_context<<"cpu", 0>>

mgp.model @dg_host_ctx device_graph : mof devices [#M.device_ref<"cpu", 0>, #M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %cpu, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %staging = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %a = mogg.routine.async (%in, %cpu) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>, %ctxarg: !mgp.device_context<<"cpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"cpu", 0>>
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"cpu", 0>>) -> ()
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy()
      %t = mgp.tensor.create %ch, %bb, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.host_ctx">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.collective.yield %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no conversion happens.

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mogg.routine.async
// CHECK-NOT: mgp.device_graph.dependencies

mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>) {
  %staging = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %a = mogg.routine.async (%in_t) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%bb) : !mgp.buffer<<"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.staging.yield %a : !mgp.buffer<<"gpu", 0>>
  } -> !mgp.buffer<<"gpu", 0>>
  mef.output %ch0, %staging : !mef.chain, !mgp.buffer<<"gpu", 0>>
}

// -----

// Each `mogg.routine.async` inside an `mgp.device_graph.collective`
// is replaced 1:1 by an `mgp.device_graph.dependencies` op. Each device region
// is processed separately, but dependency edges are derived globally across all
// converted regions.

// CHECK-LABEL: mgp.model @dg_collective
// CHECK: execute
// CHECK: mgp.device_graph.collective builder(%[[B:[a-zA-Z0-9_]+]]) stable() <"gpu", 0> {
// Device 0, region A is a leaf: no dependencies.
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mogg.async.pack
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 0>>
// CHECK: } : !mgp.buffer<<"gpu", 0>>

// Device 0, region B depends on A and reuses its body result directly.
// CHECK: %[[NODEB:.*]], %[[B0:.*]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies(%[[NODEA]]) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } : !mgp.tensor<2x3xf32, <"gpu", 0>>

// Device 0 yields B's body result.
// CHECK: mgp.device_graph.collective.yield %[[B0]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: }, <"gpu", 1> {

// Device 1, region C is a leaf.
// CHECK: %[[NODEC:.*]], %[[C:.*]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK: mogg.async.pack
// CHECK: mgp.device_graph.dependencies.yield %{{.*}} : !mgp.buffer<<"gpu", 1>>
// CHECK: } : !mgp.buffer<<"gpu", 1>>
// CHECK: mgp.device_graph.collective.yield %[[C]] : !mgp.buffer<<"gpu", 1>>
// CHECK: } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.buffer<<"gpu", 1>>
// CHECK-NOT: mogg.routine.async

mgp.model @dg_collective device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>, %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.buffer<<"gpu", 1>>) {
  %outs:2 = mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %a = mogg.routine.async (%in0, %gpu0) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>, %ctxarg: !mgp.device_context<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%bb) : !mgp.buffer<<"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    %b_ = mogg.routine.async (%a, %gpu0) : (%arg: !mgp.buffer<<"gpu", 0>>, %ctxarg: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.buffer<<"gpu", 0>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
      %ch = mogg.chain.dummy()
      %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.b">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.collective.yield %b_ : !mgp.tensor<2x3xf32, <"gpu", 0>>
  }, <"gpu", 1> {
    %c = mogg.routine.async (%in1, %gpu1) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>, %ctxarg: !mgp.device_context<<"gpu", 1>>) -> !mgp.buffer<<"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %bb = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 1>>
      %p = mogg.async.pack(%bb) : !mgp.buffer<<"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.c">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}
    mgp.device_graph.collective.yield %c : !mgp.buffer<<"gpu", 1>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.buffer<<"gpu", 1>>
  mef.output %ch0, %outs#0, %outs#1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.buffer<<"gpu", 1>>
}

// -----

// A consumer that reads a tensor through a stable slot depends on the region
// that produced the published tensor: the edge looks through `stable.tensor`
// and `stable.place`. The peer's `stable.get` adds no edge.

// CHECK-LABEL: mgp.model @dg_stable_edge
// CHECK: mgp.device_graph.collective builder(%[[B:[a-zA-Z0-9_]+]]) stable(
// CHECK: %[[NODEA:.*]], %[[A:.*]] = mgp.device_graph.dependencies builder(%[[B]]) dependencies() {
// CHECK: %[[P:.*]] = mgp.device_graph.stable.place %[[B]][0], %[[A]]
// CHECK: %[[T:.*]] = mgp.device_graph.stable.tensor %[[P]]
// CHECK: mgp.device_graph.stable.get %[[B]][1]
// CHECK: %[[NODEC:.*]], %{{.*}} = mgp.device_graph.dependencies builder(%[[B]]) dependencies(%[[NODEA]]) {
// CHECK: mgp.device_graph.collective.yield

mgp.model @dg_stable_edge device_graph : mof devices [#M.device_ref<"gpu", 0>, #M.device_ref<"gpu", 1>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu0, %gpu1 : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>, !mgp.device_context<<"gpu", 1>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu0: !mgp.device_context<<"gpu", 0>>, %gpu1: !mgp.device_context<<"gpu", 1>>, %in0: !mgp.tensor<2x3xf32, <"gpu", 0>>, %in1: !mgp.tensor<2x3xf32, <"gpu", 1>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>) {
  %outs:2 = mgp.device_graph.collective builder(%b) stable(!mgp.stable_addr<<2x3xf32, <"gpu", 0>>>, !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>) <"gpu", 0> {
    %a = mogg.routine.async (%in0, %gpu0) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>, %ctxarg: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    %p0 = mgp.device_graph.stable.place %b[0], %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %s0 = mgp.tensor.extract.tensor_spec %a : <2x3xf32, <"gpu", 0>>
    %t0 = mgp.device_graph.stable.tensor %p0, %s0 : !mgp.stable_addr<<2x3xf32, <"gpu", 0>>>
    %g1 = mgp.device_graph.stable.get %b[1] : !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>
    %s1 = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %t1 = mgp.device_graph.stable.tensor %g1, %s1 : !mgp.stable_addr<<2x3xf32, <"gpu", 1>>>
    %c0 = mogg.routine.async (%t0, %t1, %gpu0) : (%arg0: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>, %arg1: !mgp.tensor<2x3xf32, <"gpu", 1>, stable>, %ctxarg: !mgp.device_context<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
      %u0 = mogg.async.unpack(%arg0) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
      %u1 = mogg.async.unpack(%arg1) : !mgp.tensor<2x3xf32, <"gpu", 1>, stable>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 0>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 0>>) -> ()
      %v = builtin.unrealized_conversion_cast %u0 : !mgp.tensor<2x3xf32, <"gpu", 0>, stable> to !mgp.tensor<2x3xf32, <"gpu", 0>>
      %p = mogg.async.pack(%v) : !mgp.tensor<2x3xf32, <"gpu", 0>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.c0">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
    mgp.device_graph.collective.yield %c0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
  }, <"gpu", 1> {
    %bb = mogg.routine.async (%in1, %gpu1) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 1>>, %ctxarg: !mgp.device_context<<"gpu", 1>>) -> !mgp.tensor<2x3xf32, <"gpu", 1>> {
      %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      %ctx_in = mogg.async.unpack(%ctxarg) : !mgp.device_context<<"gpu", 1>>
      "test.use_ctx"(%ctx_in) : (!mgp.device_context<<"gpu", 1>>) -> ()
      %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 1>>
      mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 1>>>
    } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.b">> inOutSignature = <""> {device = #M.device_ref<"gpu", 1>}
    %p1 = mgp.device_graph.stable.place %b[1], %bb : !mgp.tensor<2x3xf32, <"gpu", 1>>
    mgp.device_graph.collective.yield %bb : !mgp.tensor<2x3xf32, <"gpu", 1>>
  } : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
  mef.output %ch0, %outs#0, %outs#1 : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 1>>
}
