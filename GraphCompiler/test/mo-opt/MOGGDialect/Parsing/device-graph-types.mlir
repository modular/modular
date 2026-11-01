// RUN: mo-opt %s --allow-unregistered-dialect | FileCheck %s
// RUN: mo-opt %s --allow-unregistered-dialect | mo-opt --allow-unregistered-dialect

// Round-trip the opaque device-graph builder/node types used by
// mogg.call.build_graph and the mgp device-graph synthesis ops.

// CHECK-LABEL: func.func @device_graph_types
// CHECK-SAME: (%{{.*}}: !mgp.device_graph.builder) -> !mgp.device_graph.node
func.func @device_graph_types(%b: !mgp.device_graph.builder)
    -> !mgp.device_graph.node {
  // CHECK: "test.record"(%{{.*}}) : (!mgp.device_graph.builder) -> !mgp.device_graph.node
  %n = "test.record"(%b) : (!mgp.device_graph.builder) -> !mgp.device_graph.node
  return %n : !mgp.device_graph.node
}
