// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s
// RUN: mo-opt %s --allow-unregistered-dialect | mo-opt --allow-unregistered-dialect

// Round-trip the transient device-graph staging scaffold ops:
// mgp.device_graph.dependencies (+ .yield) and mgp.device_graph.staging
// (+ .yield).

// CHECK-LABEL: func.func @dependencies_round_trip
func.func @dependencies_round_trip(%b: !mgp.device_graph.builder, %in: !mgp.buffer<<"gpu", 0>>)
    -> !mgp.device_graph.node {
  // The builder operand is required; no predecessors, one body result; the body
  // references the func arg directly -- the op is not IsolatedFromAbove. The
  // sole block arg is the `!mgp.device_graph.builder`.
  // CHECK: %[[N0:.+]], %[[R0:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[V:.+]] = "test.work"(%{{.+}})
  // CHECK: mgp.device_graph.dependencies.yield %[[V]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  %n0, %r0 = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.work"(%in) : (!mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>

  // A real dependency edge on %n0, and no body results (the operandless yield
  // is elided on print).
  // CHECK: %[[N1:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[N0]]) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: }
  %n1 = mgp.device_graph.dependencies builder(%b) dependencies(%n0) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.work2"(%r0) : (!mgp.buffer<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  }

  return %n1 : !mgp.device_graph.node
}

// CHECK-LABEL: func.func @staging_round_trip
func.func @staging_round_trip(%in: !mgp.tensor<2x3xf32, <"gpu", 0>>)
    -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
  // IsolatedFromAbove: the free value enters as an operand and a bare-`T` block
  // arg; the body yields bare `T` and the op result is that same `T`.
  // CHECK: %{{.+}} = mgp.device_graph.staging inputs(%{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder, %[[INT:.+]]: !mgp.tensor<2x3xf32, <"gpu", 0>>):
  // CHECK: mgp.device_graph.staging.yield %{{.+}} : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  %out = mgp.device_graph.staging inputs(%in : !mgp.tensor<2x3xf32, <"gpu", 0>>) {
  ^bb0(%builder: !mgp.device_graph.builder,
       %in_t: !mgp.tensor<2x3xf32, <"gpu", 0>>):
    %n, %t = mgp.device_graph.dependencies builder(%builder) dependencies() {
    ^bb0(%dep_builder: !mgp.device_graph.builder):
      mgp.device_graph.dependencies.yield %in_t : !mgp.tensor<2x3xf32, <"gpu", 0>>
    } : !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %t : !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  return %out : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// A `dependencies` op may reference a node defined LATER in the staging body:
// the body is a graph region (`HasOnlyGraphRegion`), so block order does not
// imply dominance. The `| mo-opt` round-trip RUN line is what enforces this --
// an SSACFG region would reject the use-before-def.

// CHECK-LABEL: func.func @forward_reference
func.func @forward_reference() {
  // CHECK: mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // The consumer references %[[NA]], which is only defined by the next op.
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[NA:.+]]) {
  // CHECK: %[[NA]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %nb = mgp.device_graph.dependencies builder(%builder) dependencies(%na) {
    ^bb0(%nb_builder: !mgp.device_graph.builder):
      mgp.device_graph.dependencies.yield
    }
    %na = mgp.device_graph.dependencies builder(%builder) dependencies() {
    ^bb0(%na_builder: !mgp.device_graph.builder):
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.staging.yield
  }
  return
}
