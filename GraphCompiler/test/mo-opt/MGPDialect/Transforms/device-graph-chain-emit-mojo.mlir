// RUN: mo-opt %s \
// RUN:     --gc-pipeline-pre-jit="kernel-library-paths=%kernel_api_test_lib dump-mojo-text=true" \
// RUN:     -o /dev/null 2>&1 | FileCheck %s

// End-to-end emission for two chained dispatches, kept in its own file because
// the previous section of `device-graph-synthesis-emit-mojo.mlir` now outlines a
// unit of its own: the checks here have to anchor on this module's output from
// the start.

// The two dispatches form a linear chain of `mgp.device_graph.dependencies` ops
// that canonicalization fuses (the `MergeChainedDependencies` pattern), and
// outlining turns the fused body into one function that records both stubs.
// The chain edge survives as the body order: `DeviceGraphBuilder.region` chains
// the nodes its body records.
// CHECK: def __device_graph_dependencies{{[_0-9]*}}({{.*}}mut {{.*}}: {{.*}}DeviceGraphBuilder{{.*}}) raises:
// CHECK: [[CHAIN_STUB:__[A-Za-z0-9_]+_build_graph]]({{.*}}, {{.*}})
// CHECK: [[CHAIN_STUB]]({{.*}}, {{.*}})
// The graph body is a function too: its dispatch closure only calls the
// outlined unit.
// CHECK: def [[WORK:work[_0-9]+]](mut builder: {{.*}}DeviceGraphBuilder) raises {imm}:
// CHECK: __device_graph_dependencies{{[_0-9]*}}({{.*}}builder{{.*}})
// CHECK: {{.*}}.region([[WORK]], dependencies =

mo.graph @dg_chain(%x: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0> isDeviceGraph {
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
      (%x) : (!mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  %1 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
      (%0) : (!mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[2, 3], f32, gpu:0>
  mo.output %1 : !mo.tensor<[2, 3], f32, gpu:0>
}
