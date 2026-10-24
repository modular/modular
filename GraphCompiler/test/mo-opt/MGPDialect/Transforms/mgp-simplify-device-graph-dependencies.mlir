// RUN: mo-opt %s --allow-unregistered-dialect --split-input-file --mgp-simplify-device-graph-dependencies | FileCheck %s

// A direct edge implied one hop away: %na -> %nb -> %nc, so %na's direct edge
// to %nc is redundant.
// CHECK-LABEL: mgp.model @triangle
mgp.model @triangle : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %b = "test.builder"() : () -> !mgp.device_graph.builder
  // CHECK: %[[NC:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
  %nc = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NB:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NC]]) {
  %nb = mgp.device_graph.dependencies builder(%b) dependencies(%nc) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NA:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NB]]) {
  %na = mgp.device_graph.dependencies builder(%b) dependencies(%nb, %nc) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[NA]])
  "test.keep"(%na) : (!mgp.device_graph.node) -> ()
  mef.output %ch0 : !mef.chain
}

// -----

// Reachability is transitive across multiple hops: %nd is implied through
// %nb -> %nc -> %nd.
// CHECK-LABEL: mgp.model @multi_hop
mgp.model @multi_hop : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %b = "test.builder"() : () -> !mgp.device_graph.builder
  // CHECK: %[[ND:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
  %nd = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NC:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[ND]]) {
  %nc = mgp.device_graph.dependencies builder(%b) dependencies(%nd) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NB:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NC]]) {
  %nb = mgp.device_graph.dependencies builder(%b) dependencies(%nc) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NA:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NB]]) {
  %na = mgp.device_graph.dependencies builder(%b) dependencies(%nb, %nd) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[NA]])
  "test.keep"(%na) : (!mgp.device_graph.node) -> ()
  mef.output %ch0 : !mef.chain
}

// -----

// Diamond: %nd is implied through both %nb and %nc; both of those edges
// survive because neither reaches the other.
// CHECK-LABEL: mgp.model @diamond
mgp.model @diamond : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %b = "test.builder"() : () -> !mgp.device_graph.builder
  // CHECK: %[[ND:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
  %nd = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NB:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[ND]]) {
  %nb = mgp.device_graph.dependencies builder(%b) dependencies(%nd) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NC:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[ND]]) {
  %nc = mgp.device_graph.dependencies builder(%b) dependencies(%nd) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NA:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NB]], %[[NC]]) {
  %na = mgp.device_graph.dependencies builder(%b) dependencies(%nb, %nc, %nd) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[NA]])
  "test.keep"(%na) : (!mgp.device_graph.node) -> ()
  mef.output %ch0 : !mef.chain
}

// -----

// Negative case: independent producers imply nothing about each other, so
// every edge survives.
// CHECK-LABEL: mgp.model @independent
mgp.model @independent : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %b = "test.builder"() : () -> !mgp.device_graph.builder
  // CHECK: %[[NB:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
  %nb = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NC:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
  %nc = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NA:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NB]], %[[NC]]) {
  %na = mgp.device_graph.dependencies builder(%b) dependencies(%nb, %nc) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[NA]])
  "test.keep"(%na) : (!mgp.device_graph.node) -> ()
  mef.output %ch0 : !mef.chain
}

// -----

// The walk crosses an empty-bodied producer: %nx records no nodes, but its
// (empty) node still carries its dependency edges at runtime, so %ny remains
// implied through %nx.
// CHECK-LABEL: mgp.model @through_empty
mgp.model @through_empty : mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %0 = grt.chain.create ()
  %1 = mgp.context.create 0
  mef.output %0, %1, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%ch0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %b = "test.builder"() : () -> !mgp.device_graph.builder
  // CHECK: %[[NY:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies() {
  %ny = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NX:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NY]]) {
  %nx = mgp.device_graph.dependencies builder(%b) dependencies(%ny) {
  ^bb0(%builder: !mgp.device_graph.builder):
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[NA:.*]] = mgp.device_graph.dependencies builder(%{{.*}}) dependencies(%[[NX]]) {
  %na = mgp.device_graph.dependencies builder(%b) dependencies(%nx, %ny) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[NA]])
  "test.keep"(%na) : (!mgp.device_graph.node) -> ()
  mef.output %ch0 : !mef.chain
}
