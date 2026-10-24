// RUN: mo-opt %s --split-input-file --allow-unregistered-dialect --canonicalize | FileCheck %s

// Constants are not hoisted out of a `mgp.device_graph.collective` body: each
// region records a separate per-device graph, so a constant stays local to the
// region that uses it instead of becoming a value captured from the enclosing
// function. Each region keeps its own copy rather than sharing one.

// CHECK-LABEL: func.func @constants_stay_in_collective_bodies
func.func @constants_stay_in_collective_bodies() {
  // CHECK-NOT:   index.constant
  // CHECK:       mgp.device_graph.collective builder(
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    %[[C0:.+]] = index.constant 32
  // CHECK-NEXT:    "test.consume"(%[[C0]]) : (index) -> ()
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }, <"gpu", 1> {
  // CHECK-NEXT:    %[[C1:.+]] = index.constant 32
  // CHECK-NEXT:    "test.consume"(%[[C1]]) : (index) -> ()
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }
  mgp.device_graph.collective builder(%b) stable() <"gpu", 0> {
    %c0 = index.constant 32
    "test.consume"(%c0) : (index) -> ()
    mgp.device_graph.collective.yield
  }, <"gpu", 1> {
    %c1 = index.constant 32
    "test.consume"(%c1) : (index) -> ()
    mgp.device_graph.collective.yield
  }
  return
}

// -----

// Control: the interface targets only the collective op. A constant inside
// another non-isolated region, here `mogg.with_trace`, is still hoisted to the
// function entry block.

// CHECK-LABEL: func.func @constant_hoisted_out_of_other_regions
func.func @constant_hoisted_out_of_other_regions() {
  // CHECK:       %[[C:.+]] = index.constant 32
  // CHECK-NEXT:  mogg.with_trace {
  // CHECK-NEXT:    "test.consume"(%[[C]]) : (index) -> ()
  mogg.with_trace {
    %c = index.constant 32
    "test.consume"(%c) : (index) -> ()
    "test.yield"() : () -> ()
  }
  return
}
