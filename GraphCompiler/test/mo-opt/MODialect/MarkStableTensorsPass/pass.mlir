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

// A single `Stable` argument: the block argument is routed through
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

// A `TensorTuple` of `Stable` elements: every peer of the run gets its own
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

// -----

// A `mo.distributed.allreduce.sum` becomes one `mo.bundled.allreduce.sum`
// per device. Every copy takes the distributed op's in-chain (serializing the
// launches of a collective would deadlock it), reads every peer through a
// stable slot, names its own position with `peer_rank`, and the out-chains
// merge into the chain the distributed op produced.
// CHECK-LABEL: mo.graph @distributed_allreduce
// CHECK-SAME: (%[[A:.*]]: !mo.tensor<[4], f32, gpu:0>, %[[B:.*]]: !mo.tensor<[4], f32, gpu:1>, %[[SA:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SB:.*]]: !mo.buffer<[1], ui8, gpu:1>)
// CHECK: %[[CH:.*]] = mo.chain.create()
// CHECK-DAG: %[[STA:.*]] = mo.stable.create %[[A]] : !mo.tensor<[4], f32, gpu:0, {stable}>
// CHECK-DAG: %[[STB:.*]] = mo.stable.create %[[B]] : !mo.tensor<[4], f32, gpu:1, {stable}>
// CHECK: %[[O0:.*]], %[[C0:.*]] = mo.bundled.allreduce.sum(%[[STA]], %[[STB]], %[[SA]], %[[SB]], %[[CH]]) <peer_rank = 0> : ({{.*}}) -> (!mo.tensor<[4], f32, gpu:0>, !mo.chain)
// CHECK: %[[O1:.*]], %[[C1:.*]] = mo.bundled.allreduce.sum(%[[STA]], %[[STB]], %[[SA]], %[[SB]], %[[CH]]) <peer_rank = 1> : ({{.*}}) -> (!mo.tensor<[4], f32, gpu:1>, !mo.chain)
// CHECK: %[[DONE:.*]] = mo.chain.create(%[[C0]], %[[C1]])
// CHECK-NOT: mo.distributed.allreduce.sum
// CHECK: mo.output %[[O0]], %[[O1]], %[[DONE]]
mo.graph @distributed_allreduce(
    %arg0: !mo.tensor<[4], f32, gpu:0>, %arg1: !mo.tensor<[4], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  %0, %1, %och = mo.distributed.allreduce.sum(%arg0, %arg1, %sig0, %sig1, %ch)
    : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>,
       !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.chain)
  mo.output %0, %1, %och : !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.chain
}

// -----

// `group_size` is carried onto every copy along with the whole world of
// peers and signal buffers: the kernel indexes them by `peer_rank` and picks
// its group from that rank, so sibling groups may differ in shape.
// CHECK-LABEL: mo.graph @grouped_allreduce
// CHECK-SAME: (%[[A:.*]]: !mo.tensor<[4], f32, gpu:0>, %[[B:.*]]: !mo.tensor<[4], f32, gpu:1>, %[[C:.*]]: !mo.tensor<[6], f32, gpu:2>, %[[D:.*]]: !mo.tensor<[6], f32, gpu:3>, %[[SA:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SB:.*]]: !mo.buffer<[1], ui8, gpu:1>, %[[SC:.*]]: !mo.buffer<[1], ui8, gpu:2>, %[[SD:.*]]: !mo.buffer<[1], ui8, gpu:3>)
// CHECK-DAG: %[[STA:.*]] = mo.stable.create %[[A]]
// CHECK-DAG: %[[STB:.*]] = mo.stable.create %[[B]]
// CHECK-DAG: %[[STC:.*]] = mo.stable.create %[[C]]
// CHECK-DAG: %[[STD:.*]] = mo.stable.create %[[D]]
// CHECK: mo.bundled.allreduce.sum(%[[STA]], %[[STB]], %[[STC]], %[[STD]], %[[SA]], %[[SB]], %[[SC]], %[[SD]], %{{.*}}) <group_size = 2, peer_rank = 0> : ({{.*}}) -> (!mo.tensor<[4], f32, gpu:0>, !mo.chain)
// CHECK: mo.bundled.allreduce.sum(%[[STA]], %[[STB]], %[[STC]], %[[STD]], %[[SA]], %[[SB]], %[[SC]], %[[SD]], %{{.*}}) <group_size = 2, peer_rank = 1> : ({{.*}}) -> (!mo.tensor<[4], f32, gpu:1>, !mo.chain)
// CHECK: mo.bundled.allreduce.sum(%[[STA]], %[[STB]], %[[STC]], %[[STD]], %[[SA]], %[[SB]], %[[SC]], %[[SD]], %{{.*}}) <group_size = 2, peer_rank = 2> : ({{.*}}) -> (!mo.tensor<[6], f32, gpu:2>, !mo.chain)
// CHECK: mo.bundled.allreduce.sum(%[[STA]], %[[STB]], %[[STC]], %[[STD]], %[[SA]], %[[SB]], %[[SC]], %[[SD]], %{{.*}}) <group_size = 2, peer_rank = 3> : ({{.*}}) -> (!mo.tensor<[6], f32, gpu:3>, !mo.chain)
// CHECK-NOT: mo.distributed.allreduce.sum
// CHECK: mo.output
mo.graph @grouped_allreduce(
    %arg0: !mo.tensor<[4], f32, gpu:0>, %arg1: !mo.tensor<[4], f32, gpu:1>,
    %arg2: !mo.tensor<[6], f32, gpu:2>, %arg3: !mo.tensor<[6], f32, gpu:3>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>
) -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[6], f32, gpu:2>, !mo.tensor<[6], f32, gpu:3>)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  %0, %1, %2, %3, %och = mo.distributed.allreduce.sum(%arg0, %arg1, %arg2, %arg3, %sig0, %sig1, %sig2, %sig3, %ch) [group_size = 2]
    : (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[6], f32, gpu:2>, !mo.tensor<[6], f32, gpu:3>,
       !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[6], f32, gpu:2>, !mo.tensor<[6], f32, gpu:3>, !mo.chain)
  mo.output %0, %1, %2, %3 : !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[6], f32, gpu:2>, !mo.tensor<[6], f32, gpu:3>
}

// -----

// A `mo.distributed.broadcast` becomes one `mo.bundled.broadcast` per device.
// Every copy reads the root tensor, which goes through a single stable slot.
// CHECK-LABEL: mo.graph @distributed_broadcast
// CHECK-SAME: (%[[A:.*]]: !mo.tensor<[4], f32, gpu:1>, %[[SA:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SB:.*]]: !mo.buffer<[1], ui8, gpu:1>)
// CHECK: %[[CH:.*]] = mo.chain.create()
// CHECK: %[[ST:.*]] = mo.stable.create %[[A]] : !mo.tensor<[4], f32, gpu:1, {stable}>
// CHECK: %[[O0:.*]], %[[C0:.*]] = mo.bundled.broadcast(%[[ST]], %[[SA]], %[[SB]], %[[CH]]) <root = 1, peer_rank = 0> : ({{.*}}) -> (!mo.tensor<[4], f32, gpu:0>, !mo.chain)
// CHECK: %[[O1:.*]], %[[C1:.*]] = mo.bundled.broadcast(%[[ST]], %[[SA]], %[[SB]], %[[CH]]) <root = 1, peer_rank = 1> : ({{.*}}) -> (!mo.tensor<[4], f32, gpu:1>, !mo.chain)
// CHECK: %[[DONE:.*]] = mo.chain.create(%[[C0]], %[[C1]])
// CHECK-NOT: mo.distributed.broadcast(
// CHECK: mo.output %[[O0]], %[[O1]], %[[DONE]]
mo.graph @distributed_broadcast(
    %arg0: !mo.tensor<[4], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  %0, %1, %och = mo.distributed.broadcast(%arg0, %sig0, %sig1, %ch) [root = 1]
    : (!mo.tensor<[4], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.chain)
  mo.output %0, %1, %och : !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.chain
}

// -----

// A full-world `mo.distributed.allgather` becomes one `mo.bundled.allgather`
// per device, each producing that device's piece per peer in peer order and
// reading every peer through a stable slot.
// CHECK-LABEL: mo.graph @distributed_allgather
// CHECK-SAME: (%[[A:.*]]: !mo.tensor<[2], f32, gpu:0>, %[[B:.*]]: !mo.tensor<[3], f32, gpu:1>, %[[SA:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SB:.*]]: !mo.buffer<[1], ui8, gpu:1>)
// CHECK: %[[CH:.*]] = mo.chain.create()
// CHECK-DAG: %[[STA:.*]] = mo.stable.create %[[A]] : !mo.tensor<[2], f32, gpu:0, {stable}>
// CHECK-DAG: %[[STB:.*]] = mo.stable.create %[[B]] : !mo.tensor<[3], f32, gpu:1, {stable}>
// CHECK: %[[O0:.*]]:2, %[[C0:.*]] = mo.bundled.allgather(%[[STA]], %[[STB]], %[[SA]], %[[SB]], %[[CH]]) <peer_rank = 0> : ({{.*}}) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.chain)
// CHECK: %[[O1:.*]]:2, %[[C1:.*]] = mo.bundled.allgather(%[[STA]], %[[STB]], %[[SA]], %[[SB]], %[[CH]]) <peer_rank = 1> : ({{.*}}) -> (!mo.tensor<[2], f32, gpu:1>, !mo.tensor<[3], f32, gpu:1>, !mo.chain)
// CHECK: %[[DONE:.*]] = mo.chain.create(%[[C0]], %[[C1]])
// CHECK-NOT: mo.distributed.allgather(
// CHECK: mo.output %[[O0]]#0, %[[O0]]#1, %[[O1]]#0, %[[O1]]#1, %[[DONE]]
mo.graph @distributed_allgather(
    %arg0: !mo.tensor<[2], f32, gpu:0>, %arg1: !mo.tensor<[3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>
) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[3], f32, gpu:1>, !mo.chain)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  %0, %1, %2, %3, %och = mo.distributed.allgather(%arg0, %arg1, %sig0, %sig1, %ch) [group_size = 2]
    : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[3], f32, gpu:1>, !mo.chain)
  mo.output %0, %1, %2, %3, %och : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[2], f32, gpu:1>, !mo.tensor<[3], f32, gpu:1>, !mo.chain
}
