// RUN: not mo-opt %s --mogg-import-packages="mojo-pkg-paths=%device_graph_invalid_pkg" --mogg-verify-kernels 2>&1 | FileCheck %s

// Each kernel in the device_graph_invalid_stubs package breaks one rule the
// verifier enforces on device-graph kernels. `build_graph` takes tensor trait
// arguments only. A recording stub hands both entry points the same values, so
// `build_graph` must declare the same arguments as `execute`, fused in the same
// form; `build_graph` may add `Stable` to an input. The verifier reports every
// violation rather than stopping at the first.

// A `Stable` output is rejected earlier, at import; see
// infra-import-tensor-arg-traits-invalid.mlir.

// `mts_on_build_graph`: a `ManagedTensorSlice` on `build_graph`.
// CHECK-DAG: error: build_graph functions take tensor trait arguments; argument at position 1 is a 'ManagedTensorSlice'; use a trait bound such as 'Some[Input & Tensor]', or 'TensorTuple' for variadic operands
// CHECK-DAG: error: the input at position 1 of build_graph reads a stable address slot, but the input at position 1 of execute is fused; a slot publishes the address of a tensor in memory, and a fused input is a functor with none
// `trait_stable_on_execute_only`, `trait_stable_on_execute` and
// `trait_stable_on_execute_plain_build_graph`: `Stable` is only valid on
// `build_graph`, whether the kernel has one and whatever form its argument has
// there.
// CHECK-DAG: error: Stable trait arguments are only supported on build_graph functions (argument at position 1 of the execute function)
// CHECK-DAG: error: Stable trait arguments are only supported on build_graph functions (argument at position 1 of the execute function)
// CHECK-DAG: error: Stable trait arguments are only supported on build_graph functions (argument at position 1 of the execute function)
// CHECK-DAG: error: the output at position 0 is trait #mogg.tensor_arg<{role = output, fused = true, access = store, capabilities = none}> on execute but trait #mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}> on build_graph; build_graph takes the same arguments as execute, with the DeviceContext replaced by a DeviceGraphBuilder
// CHECK-DAG: error: the output at position 0 is trait #mogg.tensor_arg<{role = output, fused = true, access = transform, capabilities = none}> on execute but trait #mogg.tensor_arg<{role = output, fused = true, access = store, capabilities = none}> on build_graph; build_graph takes the same arguments as execute, with the DeviceContext replaced by a DeviceGraphBuilder

module {}
