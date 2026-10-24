// RUN: mo-opt --allow-unregistered-dialect %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @device_graph_collective_with_stable_slots_and_results
func.func @device_graph_collective_with_stable_slots_and_results() {
  // CHECK:       %[[VAL:[a-zA-Z0-9_]+]] = "test.op"() : () -> !mgp.tensor<4xf32>
  %val = "test.op"() : () -> !mgp.tensor<4xf32>

  // CHECK:       %{{[a-zA-Z0-9_]+}}:2 = mgp.device_graph.collective builder(
  // CHECK-SAME:    %{{[a-zA-Z0-9_]+}}
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
  // CHECK-SAME:    !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    mgp.device_graph.collective.yield %[[VAL]] : !mgp.tensor<4xf32>
  // CHECK-NEXT:  }, <"gpu", 1> {
  // CHECK-NEXT:    mgp.device_graph.collective.yield %[[VAL]] : !mgp.tensor<4xf32>
  // CHECK-NEXT:  } : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>
  %0:2 = mgp.device_graph.collective builder(%b) stable(
    !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
    !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  ) <"gpu", 0> {
    mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
  }, <"gpu", 1> {
    mgp.device_graph.collective.yield %val : !mgp.tensor<4xf32>
  } : !mgp.tensor<4xf32>, !mgp.tensor<4xf32>

  return
}

// -----

// Each region reads a slot by index through its builder; slot 1 is read from
// gpu 0 as well, since slots are shared across the collective.

// CHECK-LABEL: func.func @device_graph_collective_uses_stable_slots
func.func @device_graph_collective_uses_stable_slots() {
  // CHECK:       mgp.device_graph.collective builder(
  // CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
  // CHECK-SAME:    !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    %[[A0:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0] : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  // CHECK-NEXT:    %[[A1:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[B]][1] : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-NEXT:    "test.consume"(%[[A0]], %[[A1]]) : (!mgp.stable_addr<<4xf32, <"gpu", 0>>>, !mgp.stable_addr<<4xf32, <"gpu", 1>>>) -> ()
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }, <"gpu", 1> {
  // CHECK-NEXT:    %[[A1B:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[B]][1] : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  // CHECK-NEXT:    "test.consume"(%[[A1B]]) : (!mgp.stable_addr<<4xf32, <"gpu", 1>>>) -> ()
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }
  mgp.device_graph.collective builder(%b) stable(
    !mgp.stable_addr<<4xf32, <"gpu", 0>>>,
    !mgp.stable_addr<<4xf32, <"gpu", 1>>>
  ) <"gpu", 0> {
    %a0 = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<4xf32, <"gpu", 0>>>
    %a1 = mgp.device_graph.stable.get %b[1] : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    "test.consume"(%a0, %a1) : (!mgp.stable_addr<<4xf32, <"gpu", 0>>>, !mgp.stable_addr<<4xf32, <"gpu", 1>>>) -> ()
    mgp.device_graph.collective.yield
  }, <"gpu", 1> {
    %a1 = mgp.device_graph.stable.get %b[1] : !mgp.stable_addr<<4xf32, <"gpu", 1>>>
    "test.consume"(%a1) : (!mgp.stable_addr<<4xf32, <"gpu", 1>>>) -> ()
    mgp.device_graph.collective.yield
  }

  return
}

// -----

// CHECK-LABEL: func.func @device_graph_collective_stable_place
func.func @device_graph_collective_stable_place() {
  // CHECK:       %[[VAL:[a-zA-Z0-9_]+]] = "test.op"() : () -> !mgp.tensor<4xf32, <"gpu", 0>>
  %val = "test.op"() : () -> !mgp.tensor<4xf32, <"gpu", 0>>

  // CHECK:       mgp.device_graph.collective builder(
  // CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  // CHECK-SAME:  ) <"gpu", 0> {
  // CHECK-NEXT:    %[[P0:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.place %[[B]][0], %[[VAL]] : !mgp.tensor<4xf32, <"gpu", 0>>
  // CHECK-NEXT:    "test.consume"(%[[P0]]) : (!mgp.stable_addr<<4xf32, <"gpu", 0>>>) -> ()
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }
  mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>>>
  ) <"gpu", 0> {
    %p0 = mgp.device_graph.stable.place %b[0], %val : !mgp.tensor<4xf32, <"gpu", 0>>
    "test.consume"(%p0) : (!mgp.stable_addr<<4xf32, <"gpu", 0>>>) -> ()
    mgp.device_graph.collective.yield
  }

  return
}

// -----

// CHECK-LABEL: func.func @device_graph_stable_tensor
func.func @device_graph_stable_tensor() {
  // CHECK:       mgp.device_graph.collective builder(
  // CHECK-SAME:    %[[B:[a-zA-Z0-9_]+]]
  // CHECK-SAME:  ) stable(
  // CHECK-SAME:    !mgp.stable_addr<<4xf32, <"gpu", 0>, mut>>
  // CHECK-SAME:  ) <"gpu", 1> {
  // CHECK-NEXT:    %[[A0:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.get %[[B]][0] : !mgp.stable_addr<<4xf32, <"gpu", 0>, mut>>
  // CHECK-NEXT:    %[[S:[a-zA-Z0-9_]+]] = mgp.tensor_spec.create : !mgp.tensor_spec<4xf32>
  // CHECK-NEXT:    %[[T:[a-zA-Z0-9_]+]] = mgp.device_graph.stable.tensor %[[A0]], %[[S]] : !mgp.stable_addr<<4xf32, <"gpu", 0>, mut>>
  // CHECK-NEXT:    "test.consume"(%[[T]]) : (!mgp.tensor<4xf32, <"gpu", 0>, mut, stable>) -> ()
  // CHECK-NEXT:    mgp.device_graph.collective.yield
  // CHECK-NEXT:  }
  mgp.device_graph.collective builder(%b) stable(
      !mgp.stable_addr<<4xf32, <"gpu", 0>, mut>>
  ) <"gpu", 1> {
    %a0 = mgp.device_graph.stable.get %b[0] : !mgp.stable_addr<<4xf32, <"gpu", 0>, mut>>
    %spec = mgp.tensor_spec.create : !mgp.tensor_spec<4xf32>
    %t = mgp.device_graph.stable.tensor %a0, %spec : !mgp.stable_addr<<4xf32, <"gpu", 0>, mut>>
    "test.consume"(%t) : (!mgp.tensor<4xf32, <"gpu", 0>, mut, stable>) -> ()
    mgp.device_graph.collective.yield
  }

  return
}

// -----

// CHECK-LABEL: func.func @device_graph_stable_create
func.func @device_graph_stable_create(%val: !mgp.tensor<4xf32, <"gpu", 0>, mut>) -> !mgp.tensor<4xf32, <"gpu", 0>, mut, stable> {
  // CHECK: %{{[a-zA-Z0-9_]+}} = mgp.device_graph.stable.create %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32, <"gpu", 0>, mut>
  %shared = mgp.device_graph.stable.create %val : !mgp.tensor<4xf32, <"gpu", 0>, mut>
  // CHECK: return %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32, <"gpu", 0>, mut, stable>
  return %shared : !mgp.tensor<4xf32, <"gpu", 0>, mut, stable>
}

// CHECK-LABEL: func.func @device_graph_stable_create_dynamic
func.func @device_graph_stable_create_dynamic(%val: !mgp.tensor<?x3xf32, <"gpu", 0>>, %n: index) -> !mgp.tensor<?x3xf32, <"gpu", 0>, stable> {
  // CHECK: %{{[a-zA-Z0-9_]+}} = mgp.device_graph.stable.create %{{[a-zA-Z0-9_]+}}[%{{[a-zA-Z0-9_]+}}] : !mgp.tensor<?x3xf32, <"gpu", 0>>
  %shared = mgp.device_graph.stable.create %val[%n] : !mgp.tensor<?x3xf32, <"gpu", 0>>
  return %shared : !mgp.tensor<?x3xf32, <"gpu", 0>, stable>
}

// -----

// CHECK-LABEL: func.func @device_graph_stable_create_default_device
func.func @device_graph_stable_create_default_device(%val: !mgp.tensor<4xf32>) -> !mgp.tensor<4xf32, stable> {
  // CHECK: %{{[a-zA-Z0-9_]+}} = mgp.device_graph.stable.create %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32>
  %shared = mgp.device_graph.stable.create %val : !mgp.tensor<4xf32>
  // CHECK: return %{{[a-zA-Z0-9_]+}} : !mgp.tensor<4xf32, stable>
  return %shared : !mgp.tensor<4xf32, stable>
}
