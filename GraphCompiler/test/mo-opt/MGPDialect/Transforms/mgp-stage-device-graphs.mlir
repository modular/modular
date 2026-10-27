// RUN: mo-opt %s --mgp-stage-device-graphs --allow-unregistered-dialect --split-input-file | FileCheck %s

// An `isDeviceGraph` model: every `mogg.routine.async` in the execute
// block is moved into a single `IsolatedFromAbove` `mgp.device_graph.staging`
// op. Free values the regions reference (here the model input) become operands
// passed in as bare-`T` block args; values consumed outside become the op's
// results (type `T`), yielded as bare `T`.

// CHECK-LABEL: mgp.model @dg
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// The staging op takes the free value as an operand and exposes it to the body
// as a bare-`T` block arg alongside the device-graph builder.
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// The bare input feeds region A directly; region B consumes A's result. No
// async pack/unpack is introduced at the staging boundary.
// CHECK-NOT: mogg.async.unpack
// CHECK: %[[A:.*]] = mogg.routine.async (%[[INT]])
// CHECK: %[[B:.*]] = mogg.routine.async (%[[A]])
// The escaping value is yielded directly as bare `T`.
// CHECK: mgp.device_graph.staging.yield %[[B]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: mef.output %{{.*}}, %[[STAGING]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>

mgp.model @dg device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %a = mogg.routine.async (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%b) : !mgp.buffer<<"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  %b = mogg.routine.async (%a) : (%arg: !mgp.buffer<<"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.buffer<<"gpu", 0>>
    %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %ch = mogg.chain.dummy()
    %t = mgp.tensor.create %ch, %u, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.b">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %b : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A non-`mogg.routine.async` op interleaved between two staged regions
// (here an `mgp.guard`) must be staged together with them. `mgp.guard` is
// lowered to an equivalent `mogg.routine.async` and then staged as
// other compiled regions.

// CHECK-LABEL: mgp.model @dg_guard
// No `mgp.guard` op is stranded ahead of the staging op (that stranding is the
// cycle); the guard has been rewritten to a compiled region.
// CHECK: execute (
// CHECK-NOT: mgp.guard
// The staging op takes only the model input -- its `inputs` must not reference a
// value the staging op itself produces.
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN:[a-zA-Z0-9_]+]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// Region A (the assert) produces the chain.
// CHECK: %[[A:[a-zA-Z0-9_]+]] = mogg.routine.async (%[[INT]])
// CHECK: fusionInfo = #{{assert[0-9]*}}
// The guard becomes a forwarding compiled region: it takes region A's chain and
// the input, forwards the input through, and carries the "mgp.guard" label.
// CHECK: %[[G:[a-zA-Z0-9_]+]] = mogg.routine.async (%[[A]], %[[INT]]) : (%{{.*}}: !mef.chain, %{{.*}}: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
// CHECK: fusionInfo = #{{guard[0-9]*}}
// Region B consumes the guard region's forwarded result.
// CHECK: mogg.routine.async (%[[G]])
// CHECK: fusionInfo = #{{b[0-9]*}}
// CHECK: mgp.device_graph.staging.yield

mgp.model @dg_guard device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // Region A: produces a chain (mirrors the `mgp.assert` region that feeds a
  // guard in the real graph).
  %ca = mogg.routine.async (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mef.chain {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %r = mogg.async.ready
    mogg.output %r : !mef.chain
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.assert">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  // Guard: gates the model input on region A's chain. Not a compiled region, so
  // the current pass leaves it behind and forms the cycle.
  %g = mgp.guard %ca, %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // Region B: consumes the guarded value.
  %b = mogg.routine.async (%g) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %s = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %ch = mogg.chain.dummy()
    %bf = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %t = mgp.tensor.create %ch, %bf, %s : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%t) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.b">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %b : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// A model WITHOUT `isDeviceGraph` is left untouched: no staging op is created
// and the compiled region stays directly in the execute block.

// CHECK-LABEL: mgp.model @not_dg
// CHECK: mogg.routine.async
// CHECK-NOT: mgp.device_graph.staging

mgp.model @not_dg : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.buffer<<"gpu", 0>>) {
  %a = mogg.routine.async (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = mgp.tensor.extract.buffer %u : <2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%b) : !mgp.buffer<<"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.buffer<<"gpu", 0>>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %a : !mef.chain, !mgp.buffer<<"gpu", 0>>
}

// -----

// A mut-marked input (the lowered form of !mo.buffer, e.g. a KV cache) crosses
// the staging boundary with its type spelled verbatim: staging operand, body
// block arg, yield, and result all carry the `mut` marker.

// CHECK-LABEL: mgp.model @dg_mut_input
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) ->
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>):
// CHECK: %[[A:.*]] = mogg.routine.async (%[[INT]])
// CHECK: mgp.device_graph.staging.yield %[[A]] : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
// CHECK: mef.output %{{.*}}, %[[STAGING]] : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, mut>

mgp.model @dg_mut_input device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) {
  %a = mogg.routine.async (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>, mut>) -> !mgp.tensor<2x3xf32, <"gpu", 0>, mut> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
    %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>, mut>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %a : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, mut>
}

// -----

// A host-placed tensor input (dispatch metadata) crosses the staging boundary
// with its cpu placement spelled verbatim, alongside the gpu data tensor.

// CHECK-LABEL: mgp.model @dg_host_input
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>, %[[H:[a-zA-Z0-9_]+]]: !mgp.tensor<4xsi64>) ->
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]], %[[H]] : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<4xsi64>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>, %[[HT:[a-zA-Z0-9_]+]]: !mgp.tensor<4xsi64>):
// CHECK: %[[A:.*]] = mogg.routine.async (%[[INT]], %[[HT]])
// CHECK: mgp.device_graph.staging.yield %[[A]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>

mgp.model @dg_host_input device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>, %h: !mgp.tensor<4xsi64, <"cpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  %a = mogg.routine.async (%in, %h) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>, %meta: !mgp.tensor<4xsi64, <"cpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %m = mogg.async.unpack(%meta) : !mgp.tensor<4xsi64, <"cpu", 0>>
    %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.a">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %a : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// `mgp.device_graph.stable.create` ops are staged with the regions; one that
// promotes a staged producer result stays with that producer rather than
// forming a dataflow cycle with the staging op.

// CHECK-LABEL: mgp.model @dg_stable_create
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// CHECK: %[[A:[a-zA-Z0-9_]+]] = mogg.routine.async (%[[INT]])
// CHECK: %[[S:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.create %[[A]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: %[[B:[a-zA-Z0-9_]+]] = mogg.routine.async (%[[S]])
// CHECK: mgp.device_graph.staging.yield %[[B]] : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>, stable>

mgp.model @dg_stable_create device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>) {
  %a = mogg.routine.async (%in) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.producer">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  %s = mgp.device_graph.stable.create %a : !mgp.tensor<2x3xf32, <"gpu", 0>>
  %b = mogg.routine.async (%s) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>) -> !mgp.tensor<2x3xf32, <"gpu", 0>, stable> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
    %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>, stable>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %b : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
}

// -----

// A `mgp.device_graph.stable.create` of a model input is staged too; the
// input enters as a staging input, so the partitioner can publish it from its
// device's region.

// CHECK-LABEL: mgp.model @dg_stable_create_input
// CHECK: execute (%{{.*}}: !mef.chain{{.*}}%[[IN:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>) ->
// CHECK: %[[STAGING:.*]] = mgp.device_graph.staging inputs(%[[IN]] : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder, %[[INT:[a-zA-Z0-9_]+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
// CHECK: %[[S:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.create %[[INT]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
// CHECK: %[[B:[a-zA-Z0-9_]+]] = mogg.routine.async (%[[S]])
// CHECK: mgp.device_graph.staging.yield %[[B]] : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
// CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>, stable>

mgp.model @dg_stable_create_input device_graph : mof devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>) {
  %s = mgp.device_graph.stable.create %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
  %b = mogg.routine.async (%s) : (%arg: !mgp.tensor<2x3xf32, <"gpu", 0>, stable>) -> !mgp.tensor<2x3xf32, <"gpu", 0>, stable> {
    %u = mogg.async.unpack(%arg) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
    %p = mogg.async.pack(%u) : !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
    mogg.output %p : !mogg.async<!mgp.tensor<2x3xf32, <"gpu", 0>, stable>>
  } fusionInfo = #mogg.fusion<#mogg.fusion_leaf<"test.consumer">> inOutSignature = <""> {device = #M.device_ref<"gpu", 0>}
  mef.output %ch0, %b : !mef.chain, !mgp.tensor<2x3xf32, <"gpu", 0>, stable>
}
