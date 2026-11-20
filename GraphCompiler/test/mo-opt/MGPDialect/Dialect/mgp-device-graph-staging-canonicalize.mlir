// RUN: mo-opt %s --split-input-file --allow-unregistered-dialect --canonicalize | FileCheck %s

// Canonicalizations for `mgp.device_graph.staging`:
//   1. unused results are dropped (along with the matching `staging.yield`
//      operands), and
//   2. duplicated `staging.yield` operands are de-duplicated, collapsing the
//      duplicate results onto a single one.
//
// The bodies are self-contained (no `inputs`) so the cases focus on the
// result/yield rewrites; the op stays `IsolatedFromAbove`.

// An unused staged result is removed; only the used value is yielded.
// CHECK-LABEL: func.func @remove_unused_result
func.func @remove_unused_result() -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
  // CHECK: %[[OUT:.+]] = mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: %[[DEAD:.+]] = "test.dead"
  // CHECK: mgp.device_graph.staging.yield %[[USED]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: return %[[OUT]]
  %used, %unused = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = "test.dead"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %b : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  return %used : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// The same value yielded twice collapses to a single result; both uses point
// at it.
// CHECK-LABEL: func.func @deduplicate_results
func.func @deduplicate_results() {
  // CHECK: %[[OUT:.+]] = mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[V:.+]] = "test.work"
  // CHECK: mgp.device_graph.staging.yield %[[V]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: "test.use"(%[[OUT]], %[[OUT]])
  %first, %second = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %a : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  "test.use"(%first, %second) : (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
  return
}

// -----

// Both patterns compose: a duplicated-and-unused result set reduces to the one
// distinct used value.
// CHECK-LABEL: func.func @dedup_and_remove_unused
func.func @dedup_and_remove_unused() -> !mgp.tensor<2x3xf32, <"gpu", 0>> {
  // CHECK: %[[OUT:.+]] = mgp.device_graph.staging inputs() {
  // CHECK: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: mgp.device_graph.staging.yield %[[USED]] : !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: return %[[OUT]]
  %a0, %a1, %dead = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = "test.dead"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %a, %b : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  return %a0 : !mgp.tensor<2x3xf32, <"gpu", 0>>
}

// -----

// Negative case: all results used and distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change
func.func @no_change() {
  // CHECK: %[[A:.+]]:2 = mgp.device_graph.staging inputs() {
  // CHECK: %[[X:.+]] = "test.work"
  // CHECK: %[[Y:.+]] = "test.work2"
  // CHECK: mgp.device_graph.staging.yield %[[X]], %[[Y]] : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  // CHECK: "test.use"(%[[A]]#0, %[[A]]#1)
  %x, %y = mgp.device_graph.staging inputs() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    %b = "test.work2"() : () -> !mgp.tensor<2x3xf32, <"gpu", 0>>
    mgp.device_graph.staging.yield %a, %b : !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  } -> !mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>
  "test.use"(%x, %y) : (!mgp.tensor<2x3xf32, <"gpu", 0>>, !mgp.tensor<2x3xf32, <"gpu", 0>>) -> ()
  return
}

// -----

// Folder for `mgp.device_graph.dependencies`: duplicated dependency edges are
// redundant (listing a predecessor twice does not change ordering), so the
// repeats are dropped, keeping the first occurrence of each.
// CHECK-LABEL: func.func @deduplicate_dependencies
func.func @deduplicate_dependencies() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[A:.+]] = "test.work"
  // CHECK: %[[B:.+]] = "test.work2"
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[A]], %[[B]]) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %a = "test.work"() : () -> !mgp.device_graph.node
  %b = "test.work2"() : () -> !mgp.device_graph.node
  %node, %r = mgp.device_graph.dependencies builder(%bldr) dependencies(%a, %b, %a) {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.body"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  return %r : !mgp.buffer<<"gpu", 0>>
}

// -----

// Negative case: all dependency edges already distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change_dependencies
func.func @no_change_dependencies() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%{{.+}}, %{{.+}}) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %a = "test.work"() : () -> !mgp.device_graph.node
  %b = "test.work2"() : () -> !mgp.device_graph.node
  %node, %r = mgp.device_graph.dependencies builder(%bldr) dependencies(%a, %b) {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.body"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  return %r : !mgp.buffer<<"gpu", 0>>
}

// -----

// An unused `dependencies` body result is removed (along with the matching
// `dependencies.yield` operand); the synthesized `node` result is always kept.
// CHECK-LABEL: func.func @remove_unused_dependency_result
func.func @remove_unused_dependency_result() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[A:.+]] = "test.dep"
  // CHECK: %[[NODE:.+]], %[[OUT:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[A]]) {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: %[[DEAD:.+]] = "test.dead"
  // CHECK: mgp.device_graph.dependencies.yield %[[USED]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: return %[[OUT]]
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %a = "test.dep"() : () -> !mgp.device_graph.node
  %node, %used, %unused = mgp.device_graph.dependencies builder(%bldr) dependencies(%a) {
  ^bb0(%builder: !mgp.device_graph.builder):
    %x = "test.work"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    %y = "test.dead"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %x, %y : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  return %used : !mgp.buffer<<"gpu", 0>>
}

// -----

// The same value yielded twice collapses to a single body result; both uses
// point at it. The `node` result is preserved.
// CHECK-LABEL: func.func @deduplicate_dependency_results
func.func @deduplicate_dependency_results() {
  // CHECK: %[[NODE:.+]], %[[OUT:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[V:.+]] = "test.work"
  // CHECK: mgp.device_graph.dependencies.yield %[[V]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: "test.use"(%[[OUT]], %[[OUT]])
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %node, %first, %second = mgp.device_graph.dependencies builder(%bldr) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %x = "test.work"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %x, %x : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  "test.use"(%first, %second) : (!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> ()
  return
}

// -----

// Both patterns compose: a duplicated-and-unused body-result set reduces to the
// one distinct used value, keeping the `node` result.
// CHECK-LABEL: func.func @dedup_and_remove_unused_dependency_results
func.func @dedup_and_remove_unused_dependency_results() -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[NODE:.+]], %[[OUT:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[USED:.+]] = "test.work"
  // CHECK: mgp.device_graph.dependencies.yield %[[USED]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: return %[[OUT]]
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %node, %a0, %a1, %dead = mgp.device_graph.dependencies builder(%bldr) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %x = "test.work"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    %y = "test.dead"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %x, %x, %y : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  return %a0 : !mgp.buffer<<"gpu", 0>>
}

// -----

// Negative case: all body results used and distinct -- left unchanged.
// CHECK-LABEL: func.func @no_change_dependency_results
func.func @no_change_dependency_results() {
  // CHECK: %[[NODE:.+]], %[[R:.+]]:2 = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[X:.+]] = "test.work"
  // CHECK: %[[Y:.+]] = "test.work2"
  // CHECK: mgp.device_graph.dependencies.yield %[[X]], %[[Y]] : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  // CHECK: "test.use"(%[[NODE]], %[[R]]#0, %[[R]]#1)
  %bldr = "test.builder"() : () -> !mgp.device_graph.builder
  %node, %x, %y = mgp.device_graph.dependencies builder(%bldr) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %a = "test.work"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    %b = "test.work2"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %a, %b : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  "test.use"(%node, %x, %y) : (!mgp.device_graph.node, !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) -> ()
  return
}

// -----

// Forwarding: a body result whose `dependencies.yield` operand is produced
// OUTSIDE the body (the op is not IsolatedFromAbove) is a pure pass-through.
// Its uses are rewritten to the outer value directly, then the dead result is
// dropped, leaving only the synthesized `node`.
// CHECK-LABEL: func.func @forward_external_result
func.func @forward_external_result(%b: !mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>> {
  // CHECK: %[[OUTER:.+]] = "test.outer"
  // The op keeps only the `node` result; its body has no results, so the
  // operandless yield is elided on print.
  // CHECK: %[[NODE:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK-NEXT: }
  // CHECK: "test.keep"(%[[NODE]])
  // CHECK: return %[[OUTER]]
  %outer = "test.outer"() : () -> !mgp.buffer<<"gpu", 0>>
  %node, %r = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    mgp.device_graph.dependencies.yield %outer : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  "test.keep"(%node) : (!mgp.device_graph.node) -> ()
  return %r : !mgp.buffer<<"gpu", 0>>
}

// -----

// Mixed: only the externally-produced result is forwarded; the result produced
// inside the body is retained.
// CHECK-LABEL: func.func @forward_external_result_partial
func.func @forward_external_result_partial(%b: !mgp.device_graph.builder) -> (!mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>) {
  // CHECK: %[[OUTER:.+]] = "test.outer"
  // CHECK: %[[NODE:.+]], %[[R:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%{{.+}}: !mgp.device_graph.builder):
  // CHECK: %[[INNER:.+]] = "test.work"
  // CHECK: mgp.device_graph.dependencies.yield %[[INNER]] : !mgp.buffer<<"gpu", 0>>
  // CHECK: } : !mgp.buffer<<"gpu", 0>>
  // CHECK: return %[[OUTER]], %[[R]]
  %outer = "test.outer"() : () -> !mgp.buffer<<"gpu", 0>>
  %node, %ext, %inner = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %w = "test.work"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %outer, %w : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
  return %ext, %inner : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 0>>
}

// -----

// Hoisting: body ops that depend on neither the builder block arg nor any other
// in-body value are body-invariant -- they would recompute on every graph build
// -- so they are lifted out in front of the op (where the emitted `region`
// closure captures them). The lift is transitive: a consumer of an already
// hoisted value becomes hoistable too. Builder-dependent ops stay in the body.
// CHECK-LABEL: func.func @hoist_body_invariant_ops
func.func @hoist_body_invariant_ops(%b: !mgp.device_graph.builder) {
  // The invariant op and its transitive consumer are hoisted before the op...
  // CHECK: %[[INV:.+]] = "test.invariant"() : () -> !mgp.buffer<<"gpu", 0>>
  // CHECK: %[[DERIVED:.+]] = "test.derived"(%[[INV]])
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%[[BB:.+]]: !mgp.device_graph.builder):
  // ...while the builder-dependent op remains in the body.
  // CHECK-NEXT: "test.record"(%[[BB]], %[[DERIVED]])
  // CHECK-NEXT: }
  %node = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %inv = "test.invariant"() : () -> !mgp.buffer<<"gpu", 0>>
    %derived = "test.derived"(%inv) : (!mgp.buffer<<"gpu", 0>>) -> !mgp.buffer<<"gpu", 0>>
    "test.record"(%builder, %derived) : (!mgp.device_graph.builder, !mgp.buffer<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  }
  return
}

// -----

// Negative case: every body op transitively depends on the builder block arg,
// so nothing is body-invariant and the body is left unchanged.
// CHECK-LABEL: func.func @hoist_no_op
func.func @hoist_no_op(%b: !mgp.device_graph.builder) {
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%[[BB:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: %[[V:.+]] = "test.work"(%[[BB]])
  // CHECK-NEXT: "test.record"(%[[BB]], %[[V]])
  %node = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.work"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    "test.record"(%builder, %v) : (!mgp.device_graph.builder, !mgp.buffer<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  }
  return
}

// -----

// `BypassEmptyDependencies`: an edge to an empty (node-less) producer is
// replaced by that producer's own dependency edges, preserving the transitive
// ordering. Here %nc depends on the empty %nb, which depends on the
// node-producing %na; %nc is rewired straight to %na and the now-dead empty op
// is DCE'd. The empty op's body has no `test.record`, so the second matched op
// below can only be the rewired %nc (not a surviving empty op).
//
// %na's node is kept live by a second user so that `MergeChainedDependencies`
// does not go on to fuse the rewired pair, leaving the case about the bypass.
// CHECK-LABEL: func.func @bypass_empty_dependency
func.func @bypass_empty_dependency(%b: !mgp.device_graph.builder) {
  // CHECK: %[[A:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%[[BA:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: "test.record"(%[[BA]])
  // CHECK: %[[C:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[A]]) {
  // CHECK-NEXT: ^bb0(%[[BC:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: "test.record"(%[[BC]])
  // CHECK: "test.keep"(%[[A]], %[[C]])
  // CHECK-NOT: mgp.device_graph.dependencies
  %na = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  %nb = mgp.device_graph.dependencies builder(%b) dependencies(%na) {
  ^bb0(%builder: !mgp.device_graph.builder):
    mgp.device_graph.dependencies.yield
  }
  %nc = mgp.device_graph.dependencies builder(%b) dependencies(%nb) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  "test.keep"(%na, %nc) : (!mgp.device_graph.node, !mgp.device_graph.node) -> ()
  return
}

// -----

// Negative case: an edge to a node-producing (non-empty) dependency is left
// untouched -- %nc keeps its edge to %na and nothing is spliced or removed.
// The second user of %na also keeps `MergeChainedDependencies` off the pair.
// CHECK-LABEL: func.func @bypass_no_op
func.func @bypass_no_op(%b: !mgp.device_graph.builder) {
  // CHECK: %[[A:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK: mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[A]]) {
  %na = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  %nc = mgp.device_graph.dependencies builder(%b) dependencies(%na) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  "test.keep"(%na, %nc) : (!mgp.device_graph.node, !mgp.device_graph.node) -> ()
  return
}

// -----

// `MergeChainedDependencies`: a chain %n0 -> %n1 -> %n2 collapses into one op
// holding all three bodies. Body order is the order the emitted closure records
// in, so it has to be predecessor-first: record0, record1, record2.
// CHECK-LABEL: func.func @merge_linear_chain
func.func @merge_linear_chain(%b: !mgp.device_graph.builder) {
  // CHECK: %[[N:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies() {
  // CHECK-NEXT: ^bb0(%[[ARG:.+]]: !mgp.device_graph.builder):
  // CHECK-NEXT: "test.record0"(%[[ARG]])
  // CHECK-NEXT: "test.record1"(%[[ARG]])
  // CHECK-NEXT: "test.record2"(%[[ARG]])
  // CHECK-NEXT: }
  %n0 = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record0"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  %n1 = mgp.device_graph.dependencies builder(%b) dependencies(%n0) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record1"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  %n2 = mgp.device_graph.dependencies builder(%b) dependencies(%n1) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record2"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK-NOT: mgp.device_graph.dependencies
  // CHECK: "test.keep"(%[[N]])
  "test.keep"(%n2) : (!mgp.device_graph.node) -> ()
  return
}

// -----

// The merged op takes over the predecessor's edges, so a chain hanging off an
// edge from outside the chain stays ordered after it.
// CHECK-LABEL: func.func @merge_inherits_producer_edges
func.func @merge_inherits_producer_edges(%b: !mgp.device_graph.builder) {
  // An edge from something that is not a dependencies op, so it cannot itself
  // be absorbed into the chain.
  // CHECK: %[[EXT:.+]] = "test.node"()
  %ext = "test.node"() : () -> !mgp.device_graph.node
  // CHECK: %[[N:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[EXT]]) {
  // CHECK: "test.record0"
  // CHECK-NEXT: "test.record1"
  %n0 = mgp.device_graph.dependencies builder(%b) dependencies(%ext) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record0"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  %n1 = mgp.device_graph.dependencies builder(%b) dependencies(%n0) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record1"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[N]])
  "test.keep"(%n1) : (!mgp.device_graph.node) -> ()
  return
}

// -----

// Negative: the successor has a second edge, so absorbing %n0 would order
// %n0's body after %other too. Genuine fan-in must survive.
// CHECK-LABEL: func.func @merge_extra_dependency
func.func @merge_extra_dependency(%b: !mgp.device_graph.builder) {
  // CHECK: %[[N0:.+]] = mgp.device_graph.dependencies
  // CHECK: "test.record0"
  %n0 = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record0"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[OTHER:.+]] = mgp.device_graph.dependencies
  // CHECK: "test.other"
  %other = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.other"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[N1:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[N0]], %[[OTHER]]) {
  // CHECK: "test.record1"
  %n1 = mgp.device_graph.dependencies builder(%b) dependencies(%n0, %other) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record1"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[N1]])
  "test.keep"(%n1) : (!mgp.device_graph.node) -> ()
  return
}

// -----

// Negative: %n0's node has a second user, which would still need it after the
// merge erased it.
// CHECK-LABEL: func.func @merge_node_reused
func.func @merge_node_reused(%b: !mgp.device_graph.builder) {
  // CHECK: %[[N0:.+]] = mgp.device_graph.dependencies
  // CHECK: "test.record0"
  %n0 = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record0"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[N1:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[N0]]) {
  // CHECK: "test.record1"
  %n1 = mgp.device_graph.dependencies builder(%b) dependencies(%n0) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record1"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[N0]], %[[N1]])
  "test.keep"(%n0, %n1) : (!mgp.device_graph.node, !mgp.device_graph.node) -> ()
  return
}

// -----

// Negative: the predecessor yields a value. Absorbing it would move the
// definition of %r0 behind its user.
// CHECK-LABEL: func.func @merge_producer_yields
func.func @merge_producer_yields(%b: !mgp.device_graph.builder) {
  // CHECK: %[[N0:.+]], %[[R0:.+]] = mgp.device_graph.dependencies
  // CHECK: "test.alloc"
  %n0, %r0 = mgp.device_graph.dependencies builder(%b) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    %v = "test.alloc"(%builder) : (!mgp.device_graph.builder) -> !mgp.buffer<<"gpu", 0>>
    mgp.device_graph.dependencies.yield %v : !mgp.buffer<<"gpu", 0>>
  } : !mgp.buffer<<"gpu", 0>>
  // CHECK: %[[N1:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[N0]]) {
  // CHECK: "test.record1"(%{{.+}}, %[[R0]])
  %n1 = mgp.device_graph.dependencies builder(%b) dependencies(%n0) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record1"(%builder, %r0) : (!mgp.device_graph.builder, !mgp.buffer<<"gpu", 0>>) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[N1]])
  "test.keep"(%n1) : (!mgp.device_graph.node) -> ()
  return
}

// -----

// Negative: the two ops record into different builders, so neither body can
// host the other's ops.
// CHECK-LABEL: func.func @merge_different_builders
func.func @merge_different_builders(%b0: !mgp.device_graph.builder, %b1: !mgp.device_graph.builder) {
  // CHECK: %[[N0:.+]] = mgp.device_graph.dependencies
  // CHECK: "test.record0"
  %n0 = mgp.device_graph.dependencies builder(%b0) dependencies() {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record0"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: %[[N1:.+]] = mgp.device_graph.dependencies builder(%{{.+}}) dependencies(%[[N0]]) {
  // CHECK: "test.record1"
  %n1 = mgp.device_graph.dependencies builder(%b1) dependencies(%n0) {
  ^bb0(%builder: !mgp.device_graph.builder):
    "test.record1"(%builder) : (!mgp.device_graph.builder) -> ()
    mgp.device_graph.dependencies.yield
  }
  // CHECK: "test.keep"(%[[N1]])
  "test.keep"(%n1) : (!mgp.device_graph.node) -> ()
  return
}

