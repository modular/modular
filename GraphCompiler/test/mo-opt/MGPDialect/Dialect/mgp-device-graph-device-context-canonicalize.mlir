// RUN: mo-opt %s --split-input-file --allow-unregistered-dialect --canonicalize | FileCheck %s

// Canonicalization for `mgp.device_graph.device_context`: a read of a build
// region's own builder records no node, so the read is hoisted in front of the
// enclosing `mgp.device_graph.dependencies` op and rebound to the builder that
// op records into.

// The read leaves the body and keeps feeding its in-body user.
// CHECK-LABEL: func.func @hoist_read_out_of_body
func.func @hoist_read_out_of_body(%b: !mgp.device_graph.builder) {
  // CHECK: %[[CTX:.+]] = mgp.device_graph.device_context %{{.+}} : !mgp.device_context<<"gpu", 0>>
  // CHECK: %[[NODE:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%[[BB:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: "test.record"(%[[BB]], %[[CTX]])
  // CHECK-NEXT: }
  // CHECK: "test.keep"(%[[NODE]])
  %node = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %ctx = mgp.device_graph.device_context %builder : !mgp.device_context<<"gpu", 0>>
    "test.record"(%builder, %ctx) : (!mgp.device_graph.builder, !mgp.device_context<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  }
  "test.keep"(%node) : (!mgp.device_graph.node) -> ()
  return
}

// -----

// A nested body's read composes: one hoist per enclosing body, ending in front
// of the outermost op with the outermost builder.
// CHECK-LABEL: func.func @hoist_read_through_nested_bodies
func.func @hoist_read_through_nested_bodies(%b: !mgp.device_graph.builder) {
  // CHECK: %[[CTX:.+]] = mgp.device_graph.device_context %{{.+}}
  // CHECK: %[[OUTER:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%[[OB:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: %[[INNER:.+]] = mgp.device_graph.dependencies builder(%[[OB]]) dependencies() {
  // CHECK-NEXT: ^bb0(%[[IB:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: "test.record"(%[[IB]], %[[CTX]])
  %outer = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%outerBuilder: !mgp.device_graph.builder):
    %inner = mgp.device_graph.dependencies builder(%outerBuilder) dependencies() {
    ^bb0(%innerBuilder: !mgp.device_graph.builder):
      %ctx = mgp.device_graph.device_context %innerBuilder : !mgp.device_context<<"gpu", 0>>
      "test.record"(%innerBuilder, %ctx) : (!mgp.device_graph.builder, !mgp.device_context<<"gpu", 0>>) -> ()
      mgp.device_graph.dependencies.yield
    }
    mgp.device_graph.dependencies.yield
  }
  "test.keep"(%outer) : (!mgp.device_graph.node) -> ()
  return
}

// -----

// A read of a builder the body captured from further out names a different
// builder from the one the op records into, so its operand is left alone.
// CHECK-LABEL: func.func @no_rebind_captured_builder
func.func @no_rebind_captured_builder(%b: !mgp.device_graph.builder, %other: !mgp.device_graph.builder) {
  // CHECK: %[[CTX:.+]] = mgp.device_graph.device_context %[[OTHER:.+]]
  // CHECK: %[[NODE:.+]] = mgp.device_graph.dependencies builder(%[[B:.+]]) dependencies() {
  // CHECK: "test.record"(%{{.+}}, %[[CTX]])
  %node = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %ctx = mgp.device_graph.device_context %other : !mgp.device_context<<"gpu", 0>>
    "test.record"(%builder, %ctx) : (!mgp.device_graph.builder, !mgp.device_context<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  }
  "test.keep"(%node) : (!mgp.device_graph.node) -> ()
  return
}
