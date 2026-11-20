// RUN: mo-opt %s --mgp-sink-into-device-graph-bodies \
// RUN:     --allow-unregistered-dialect --split-input-file | FileCheck %s

// `mgp-sink-into-device-graph-bodies` moves the ops canonicalization hoisted
// out of a `mgp.device_graph.dependencies` body back into it, when every use of
// their results is inside that one body. A value the body shares with another
// body or with the rest of the graph stays outside and is read from above.

// CHECK-LABEL: mgp.model @sink
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// The shared spec stays outside, so both bodies can read it.
// CHECK-NEXT: %[[SHARED:.*]] = mgp.tensor.extract.tensor_spec
// CHECK-NEXT: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[B:.*]]: !mgp.device_graph.builder):
// The body's own spec and dimension sank in, ahead of their user.
// CHECK-NEXT: %[[SPEC:.*]] = mgp.tensor_spec.create
// CHECK-NEXT: %[[DIM:.*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]] : <2x3xf32>
// CHECK-NEXT: "test.use"(%[[DIM]]) : (index) -> ()
// CHECK-NEXT: "test.use_common"(%[[SHARED]]) : (!mgp.tensor_spec<2x3xf32>) -> ()
// CHECK-NEXT: }
// The second body still reads the shared spec from above.
// CHECK: mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%{{.*}}) {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: "test.use_common2"(%[[SHARED]]) : (!mgp.tensor_spec<2x3xf32>) -> ()
mgp.model @sink device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %dim = mgp.tensor_spec.get_dim[0] %spec : !mgp.tensor_spec<2x3xf32>
    %shared = mgp.tensor.extract.tensor_spec %in : <2x3xf32, <"gpu", 0>>
    %node0 = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%db0: !mgp.device_graph.builder):
      "test.use"(%dim) : (index) -> ()
      "test.use_common"(%shared) : (!mgp.tensor_spec<2x3xf32>) -> ()
      mgp.device_graph.dependencies.yield
    }
    %node1 = mgp.device_graph.dependencies builder(%b) dependencies(%node0) {
    ^bb0(%db1: !mgp.device_graph.builder):
      "test.use_common2"(%shared) : (!mgp.tensor_spec<2x3xf32>) -> ()
      "test.use_other"(%shared) : (!mgp.tensor_spec<2x3xf32>) -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// A graph input registration stays where it is, even when only one body reads
// its stable value: the inputs register in graph-signature order, so sinking
// one past another would reorder them.

// CHECK-LABEL: mgp.model @sink_add_input
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK-NEXT: %[[STABLE:.*]] = mgp.device_graph.add_input
// CHECK-NEXT: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: "test.use"(%[[STABLE]])
mgp.model @sink_add_input device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %stable = mgp.device_graph.add_input %b, %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%db: !mgp.device_graph.builder):
      "test.use"(%stable) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
      "test.use_2"(%stable) : (!mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// An op that names the enclosing builder may still sink: at run time that
// builder is the body's own, so the operand is rebound to the body argument
// rather than captured. A second `mut` builder parameter would alias.

// CHECK-LABEL: mgp.model @sink_rebinds_builder
// CHECK: mgp.device_graph.collective builder(%[[OUTER:.*]]) stable() <"gpu", 0> {
// CHECK-NEXT: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%[[B:.*]]: !mgp.device_graph.builder):
// CHECK-NEXT: %[[CTX:.*]] = mgp.device_graph.recording_context %[[B]] : !mogg.context
// CHECK-NEXT: "test.use"(%[[CTX]]) : (!mogg.context) -> ()
mgp.model @sink_rebinds_builder device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %rctx = mgp.device_graph.recording_context %b : !mogg.context
    %node = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%db: !mgp.device_graph.builder):
      "test.use"(%rctx) : (!mogg.context) -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}

// -----

// Negative case: a body that yields values stays inline, so nothing sinks into
// it and the op it reads stays outside.

// CHECK-LABEL: mgp.model @no_sink_into_yielding_body
// CHECK: mgp.device_graph.collective builder(%{{.*}}) stable() <"gpu", 0> {
// CHECK-NEXT: %[[SPEC:.*]] = mgp.tensor_spec.create
// CHECK-NEXT: mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
// CHECK-NEXT: ^bb0(%{{.*}}: !mgp.device_graph.builder):
// CHECK-NEXT: "test.use"(%[[SPEC]])
mgp.model @no_sink_into_yielding_body device_graph devices [#M.device_ref<"gpu", 0>]
  init (%weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx, %weights, %gpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"gpu", 0>>
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %gpu: !mgp.device_context<<"gpu", 0>>, %in: !mgp.tensor<2x3xf32, <"gpu", 0>>) -> (!mef.chain) {
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<2x3xf32>
    %node, %out = mgp.device_graph.dependencies builder(%b) dependencies() {
    ^bb0(%db: !mgp.device_graph.builder):
      "test.use"(%spec) : (!mgp.tensor_spec<2x3xf32>) -> ()
      mgp.device_graph.dependencies.yield %in : !mgp.tensor<2x3xf32, <"gpu", 0>>
    } : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.collective.yield
  }
  mef.output %ch0 : !mef.chain
}
