// RUN: mo-opt --allow-unregistered-dialect %s --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @collective_create_and_execute
func.func @collective_create_and_execute(
    %ctx: !mgp.context,
    %d0: !mgp.device_context<<"gpu", 0>>,
    %d1: !mgp.device_context<<"gpu", 1>>,
    %in0: !mgp.buffer<<"gpu", 0>>,
    %in1: !mgp.buffer<<"gpu", 1>>
) -> !mef.chain {
  // Collectively create two device graphs, one per device. Each region consumes
  // its own input from the flat `inputs` list.
  // CHECK: %{{[a-zA-Z0-9_]+}}:2 = mgp.device_graph.collective.create
  // CHECK-SAME: %{{[^ ]+}}
  // CHECK-SAME: graph(%{{.+}} : !mgp.device_context<<"gpu", 0>>) builder(%{{.+}}) inputs(%{{.+}} : !mgp.buffer<<"gpu", 0>>) {
  // CHECK-NEXT: mgp.device_graph.yield %{{.+}} : !mgp.buffer<<"gpu", 0>>
  // CHECK: }, graph(%{{.+}} : !mgp.device_context<<"gpu", 1>>) builder(%{{.+}}) inputs(%{{.+}} : !mgp.buffer<<"gpu", 1>>) {
  // CHECK-NEXT: mgp.device_graph.yield %{{.+}} : !mgp.buffer<<"gpu", 1>>
  // CHECK: }
  %g0, %g1 = mgp.device_graph.collective.create %ctx
      graph(%d0: !mgp.device_context<<"gpu", 0>>) builder(%b0)
          inputs(%in0: !mgp.buffer<<"gpu", 0>>) {
        mgp.device_graph.yield %in0 : !mgp.buffer<<"gpu", 0>>
      }, graph(%d1: !mgp.device_context<<"gpu", 1>>) builder(%b1)
          inputs(%in1: !mgp.buffer<<"gpu", 1>>) {
        mgp.device_graph.yield %in1 : !mgp.buffer<<"gpu", 1>>
      } : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
          !mgp.device_graph<(!mgp.buffer<<"gpu", 1>>) -> (!mgp.buffer<<"gpu", 1>>)>

  // Replay both graphs with the same flat input list.
  // CHECK: %{{[a-zA-Z0-9_]+}}, %{{[a-zA-Z0-9_]+}}:2 = mgp.device_graph.collective.execute
  // CHECK-SAME: %{{.+}}, %{{.+}} inputs(%{{.+}}, %{{.+}} : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 1>>)
  %chain, %r0, %r1 = mgp.device_graph.collective.execute
      %g0, %g1
      inputs(%in0, %in1 : !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 1>>)
      : !mgp.device_graph<(!mgp.buffer<<"gpu", 0>>) -> (!mgp.buffer<<"gpu", 0>>)>,
        !mgp.device_graph<(!mgp.buffer<<"gpu", 1>>) -> (!mgp.buffer<<"gpu", 1>>)>
        -> !mgp.buffer<<"gpu", 0>>, !mgp.buffer<<"gpu", 1>>

  return %chain : !mef.chain
}
