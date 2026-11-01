// RUN: mo-opt --split-input-file \
// RUN:   --mogg-import-packages="mojo-pkg-paths=%kernel_api_test_lib" \
// RUN:   --mo-mark-stable-tensors --cse %s | FileCheck %s
//
// The pass builds one `mo.stable.create` per stable operand, immediately
// before its consumer; the op is pure, so `--cse` is what folds the copies a
// shared value picks up into a single slot.

// Without the isDeviceGraph attribute the pass is a no-op, even for a kernel
// whose `build_graph` declares stable operands.
// CHECK-LABEL: mo.graph @no_attr
// CHECK-NOT: mo.stable.create
mo.graph @no_attr(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0> {
  %0 = mo.custom [symbol = "imposter_stable_single_build_graph", device = #M.device_ref<"gpu", 0>]
    (%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// A `build_graph` without stable arguments leaves the graph untouched.
// CHECK-LABEL: mo.graph @plain_build_graph
// CHECK-NOT: mo.stable.create
mo.graph @plain_build_graph(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    attributes {isDeviceGraph} {
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
    (%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// A single `StableInputTensor` argument: the block argument is routed through
// a create placed at the top of the body, and the consumer is retyped.
// CHECK-LABEL: mo.graph @single
// CHECK-SAME: (%[[ARG:.*]]: !mo.tensor<[4], f32, gpu:0>)
// CHECK-NEXT: %[[S:.*]] = mo.stable.create %[[ARG]] : !mo.tensor<[4], f32, gpu:0, {stable}>
// CHECK-NEXT: mo.custom {{.*}}(%[[S]]) : (!mo.tensor<[4], f32, gpu:0, {stable}>) -> !mo.tensor<[4], f32, gpu:0>
mo.graph @single(%arg0: !mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
    attributes {isDeviceGraph} {
  %0 = mo.custom [symbol = "imposter_stable_single_build_graph", device = #M.device_ref<"gpu", 0>]
    (%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// A `StableInputVariadicTensors` run: every peer of the run gets its own
// create, on its own device.
// CHECK-LABEL: mo.graph @variadic
// CHECK-SAME: (%[[A:.*]]: !mo.tensor<[4], f32, gpu:0>, %[[B:.*]]: !mo.tensor<[4], f32, gpu:1>)
// CHECK-DAG: %[[SA:.*]] = mo.stable.create %[[A]] : !mo.tensor<[4], f32, gpu:0, {stable}>
// CHECK-DAG: %[[SB:.*]] = mo.stable.create %[[B]] : !mo.tensor<[4], f32, gpu:1, {stable}>
// CHECK: mo.custom {{.*}}(%[[SA]], %[[SB]]) : (!mo.tensor<[4], f32, gpu:0, {stable}>, !mo.tensor<[4], f32, gpu:1, {stable}>) -> !mo.tensor<[4], f32, gpu:0>
mo.graph @variadic(%arg0: !mo.tensor<[4], f32, gpu:0>, %arg1: !mo.tensor<[4], f32, gpu:1>) -> !mo.tensor<[4], f32, gpu:0>
    attributes {isDeviceGraph} {
  %0 = mo.custom [symbol = "imposter_stable_build_graph", device = #M.device_ref<"gpu", 0>]
    (%arg0, %arg1) : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}

// -----

// A value feeding several stable positions ends up with a single create,
// shared by every consumer, once CSE has folded the per-consumer copies. The
// consumers differ (single vs. variadic kernel) so CSE folds only the creates.
// CHECK-LABEL: mo.graph @shared_operand
// CHECK: %[[P:.*]] = mo.custom {{.*}}symbol = "imposter_build_graph_capable"
// CHECK-NEXT: %[[S:.*]] = mo.stable.create %[[P]] : !mo.tensor<[4], f32, gpu:0, {stable}>
// CHECK-NOT: mo.stable.create
// CHECK: mo.custom {{.*}}symbol = "imposter_stable_single_build_graph"{{.*}}](%[[S]]) : (!mo.tensor<[4], f32, gpu:0, {stable}>)
// CHECK: mo.custom {{.*}}symbol = "imposter_stable_build_graph"{{.*}}](%[[S]]) : (!mo.tensor<[4], f32, gpu:0, {stable}>)
mo.graph @shared_operand(%arg0: !mo.tensor<[4], f32, gpu:0>) -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>)
    attributes {isDeviceGraph} {
  %0 = mo.custom [symbol = "imposter_build_graph_capable", device = #M.device_ref<"gpu", 0>]
    (%arg0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  %1 = mo.custom [symbol = "imposter_stable_single_build_graph", device = #M.device_ref<"gpu", 0>]
    (%0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  %2 = mo.custom [symbol = "imposter_stable_build_graph", device = #M.device_ref<"gpu", 0>]
    (%0) : (!mo.tensor<[4], f32, gpu:0>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>
}

// -----

// An operand that is already stable is left alone.
// CHECK-LABEL: mo.graph @already_stable
// CHECK-NOT: mo.stable.create
mo.graph @already_stable(%arg0: !mo.tensor<[4], f32, gpu:0, {stable}>) -> !mo.tensor<[4], f32, gpu:0>
    attributes {isDeviceGraph} {
  %0 = mo.custom [symbol = "imposter_stable_single_build_graph", device = #M.device_ref<"gpu", 0>]
    (%arg0) : (!mo.tensor<[4], f32, gpu:0, {stable}>) -> !mo.tensor<[4], f32, gpu:0>
  mo.output %0 : !mo.tensor<[4], f32, gpu:0>
}
