// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%device_graph_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s

// Each kernel in the device_graph_invalid_stubs package breaks one rule the
// verifier enforces on device-graph kernels. Slot-backed arguments
// (`StableTensor` / `StableVariadicTensors`) are a recording concept, so they
// may only appear as inputs of `build_graph`. A recording stub hands both
// entry points the same values, so `build_graph` must declare the same
// arguments as `execute`, fused in the same form. The verifier reports every
// violation rather than stopping at the first.

// CHECK-DAG: error: stable tensor arguments (StableTensor / StableVariadicTensors) are only supported on build_graph functions (argument at position 1 of the execute function)
// CHECK-DAG: error: a stable tensor argument cannot be an output (argument at position 0 of the build_graph function)
// CHECK-DAG: error: the input at position 1 of build_graph reads a stable address slot, but the input at position 1 of execute is fused; a slot publishes the address of a tensor in memory, and a fused input is a functor with none
// CHECK-DAG: error: the output at position 0 is FusedOutputTensor on execute but OutputTensor on build_graph; build_graph takes the same arguments as execute, with the DeviceContext replaced by a DeviceGraphBuilder
// CHECK-DAG: error: the output at position 0 is FusedComputeOutputTensor on execute but FusedOutputTensor on build_graph; build_graph takes the same arguments as execute, with the DeviceContext replaced by a DeviceGraphBuilder

module {}
