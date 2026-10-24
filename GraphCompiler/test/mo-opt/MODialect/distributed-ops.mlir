// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @all_reduce_sum(%[[IN0:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, 3], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[2, 3], f32, gpu:2>
  mo.graph @all_reduce_sum(%in0: !mo.tensor<[2, 3], f32, gpu:0>, %in1: !mo.tensor<[2, 3], f32, gpu:1>, %in2: !mo.tensor<[2, 3], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.allreduce.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.allreduce.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.tensor<[2, 3], f32, gpu:2>
}

// CHECK: mo.graph @all_reduce_sum_dynamic(%[[IN0:.*]]: !mo.tensor<[2, ?], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, ?], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[2, ?], f32, gpu:2>
mo.graph @all_reduce_sum_dynamic(%in0: !mo.tensor<[2, ?], f32, gpu:0>, %in1: !mo.tensor<[2, ?], f32, gpu:1>, %in2: !mo.tensor<[2, ?], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.allreduce.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) : (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.allreduce.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) : (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[2, ?], f32, gpu:0>, !mo.tensor<[2, ?], f32, gpu:1>, !mo.tensor<[2, ?], f32, gpu:2>
}

// Grouped allreduce: contiguous pairs reduce independently, so the two
// groups may carry different (e.g. per-DP-replica) row counts.
// CHECK-LABEL: mo.graph @all_reduce_sum_grouped
// CHECK: mo.distributed.allreduce.sum
// CHECK-SAME: group_size = 2
mo.graph @all_reduce_sum_grouped(%in0: !mo.tensor<[5, 8], f32, gpu:0>, %in1: !mo.tensor<[5, 8], f32, gpu:1>, %in2: !mo.tensor<[3, 8], f32, gpu:2>, %in3: !mo.tensor<[3, 8], f32, gpu:3>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>) -> (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>) {
  %ch = mo.chain.create()
  %out:4, %outChain = "mo.distributed.allreduce.sum"(%in0, %in1, %in2, %in3, %sig0, %sig1, %sig2, %sig3, %ch) {group_size = 2 : i64} : (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, !mo.chain)
  mo.output %out#0, %out#1, %out#2, %out#3 : !mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>
}

// CHECK-LABEL: mo.graph @allgather_grouped
// CHECK: mo.distributed.allgather
// CHECK-SAME: group_size = 2
mo.graph @allgather_grouped(%in0: !mo.tensor<[5, 8], f32, gpu:0>, %in1: !mo.tensor<[4, 8], f32, gpu:1>, %in2: !mo.tensor<[3, 8], f32, gpu:2>, %in3: !mo.tensor<[2, 8], f32, gpu:3>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>) -> (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[4, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, !mo.tensor<[2, 8], f32, gpu:3>) {
  %ch = mo.chain.create()
  %out:8, %outChain = "mo.distributed.allgather"(%in0, %in1, %in2, %in3, %sig0, %sig1, %sig2, %sig3, %ch) {group_size = 2 : i64} : (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[2, 8], f32, gpu:3>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[4, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, !mo.tensor<[2, 8], f32, gpu:3>, !mo.chain)
  mo.output %out#0, %out#1, %out#2, %out#3, %out#4, %out#5, %out#6, %out#7 : !mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[4, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, !mo.tensor<[2, 8], f32, gpu:3>
}

// CHECK-LABEL: mo.graph @all_reduce_folds
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2, 17], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @all_reduce_folds(%in0: !mo.tensor<[2, 17], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %ch: !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain) {
  // CHECK-NOT: mo.distributed.allreduce.sum
  // CHECK: mo.output %[[IN0]], %[[CH]]
  %out, %outChain = mo.distributed.allreduce.sum(%in0, %sig0, %ch) : (!mo.tensor<[2, 17], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain)
  mo.output %out, %outChain : !mo.tensor<[2, 17], f32, gpu:0>, !mo.chain
}

// CHECK: mo.graph @reduce_scatter_sum(%[[IN0:.*]]: !mo.tensor<[3, 6], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3, 6], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[3, 6], f32, gpu:2>
mo.graph @reduce_scatter_sum(%in0: !mo.tensor<[3, 6], f32, gpu:0>, %in1: !mo.tensor<[3, 6], f32, gpu:1>, %in2: !mo.tensor<[3, 6], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) [group_size = 3] : (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>, !mo.tensor<[3, 6], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) [group_size = 3] : (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>, !mo.tensor<[3, 6], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[3, 2], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>
}

// Ragged case: 7 elements / 3 GPUs = 2 base + 1 remainder. GPU 0 gets 3, others get 2.
// CHECK: mo.graph @reduce_scatter_sum_ragged(%[[IN0:.*]]: !mo.tensor<[3, 7], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3, 7], f32, gpu:1>, %[[IN2:.*]]: !mo.tensor<[3, 7], f32, gpu:2>
mo.graph @reduce_scatter_sum_ragged(%in0: !mo.tensor<[3, 7], f32, gpu:0>, %in1: !mo.tensor<[3, 7], f32, gpu:1>, %in2: !mo.tensor<[3, 7], f32, gpu:2>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>) -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[IN0]], %[[IN1]], %[[IN2]], {{.*}}) [group_size = 3] : (!mo.tensor<[3, 7], f32, gpu:0>, !mo.tensor<[3, 7], f32, gpu:1>, !mo.tensor<[3, 7], f32, gpu:2>, {{.*}}) -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  %out:3, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %in2, %sig0, %sig1, %sig2, %ch) [group_size = 3] : (!mo.tensor<[3, 7], f32, gpu:0>, !mo.tensor<[3, 7], f32, gpu:1>, !mo.tensor<[3, 7], f32, gpu:2>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain) -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  mo.output %out#0, %out#1, %out#2 : !mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>
}

// CHECK: mo.graph @reduce_scatter_sum_dynamic(%[[IN0:.*]]: !mo.tensor<[3, ?], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3, ?], f32, gpu:1>
mo.graph @reduce_scatter_sum_dynamic(%in0: !mo.tensor<[3, ?], f32, gpu:0>, %in1: !mo.tensor<[3, ?], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[IN0]], %[[IN1]], {{.*}}) [group_size = 2] : (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, {{.*}}) -> (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, !mo.chain)
  %out:2, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %sig0, %sig1, %ch) [group_size = 2] : (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[3, ?], f32, gpu:0>, !mo.tensor<[3, ?], f32, gpu:1>
}

// Grouped ragged case: first group scatters 5 rows across 2 GPUs, second group scatters 3 rows across 2 GPUs.
// CHECK: mo.graph @reduce_scatter_sum_grouped(%[[G0:.*]]: !mo.tensor<[5, 8], f32, gpu:0>, %[[G1:.*]]: !mo.tensor<[5, 8], f32, gpu:1>, %[[G2:.*]]: !mo.tensor<[3, 8], f32, gpu:2>, %[[G3:.*]]: !mo.tensor<[3, 8], f32, gpu:3>
mo.graph @reduce_scatter_sum_grouped(%in0: !mo.tensor<[5, 8], f32, gpu:0>, %in1: !mo.tensor<[5, 8], f32, gpu:1>, %in2: !mo.tensor<[3, 8], f32, gpu:2>, %in3: !mo.tensor<[3, 8], f32, gpu:3>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>, %sig2: !mo.buffer<[1], ui8, gpu:2>, %sig3: !mo.buffer<[1], ui8, gpu:3>) -> (!mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.reducescatter.sum(%[[G0]], %[[G1]], %[[G2]], %[[G3]], {{.*}}) [axis = 0, group_size = 2] : (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, {{.*}}) -> (!mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>, !mo.chain)
  %out:4, %outChain = mo.distributed.reducescatter.sum(%in0, %in1, %in2, %in3, %sig0, %sig1, %sig2, %sig3, %ch) [axis = 0, group_size = 2] : (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain) -> (!mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>, !mo.chain)
  mo.output %out#0, %out#1, %out#2, %out#3 : !mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>
}

// CHECK-LABEL: mo.graph @broadcast
// CHECK: (%[[IN:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SIG1:.*]]: !mo.buffer<[1], ui8, gpu:1>
mo.graph @broadcast(%in: !mo.tensor<[2, 3], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.broadcast(%[[IN]], %[[SIG0]], %[[SIG1]], {{.*}}) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  %out:2, %outChain = mo.distributed.broadcast(%in, %sig0, %sig1, %ch) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// Broadcast with a non-root-0 input: the input must live on
// `signalBuffers[root]`'s device, here gpu:1 for root 1.
// CHECK-LABEL: mo.graph @broadcast_nonzero_root
// CHECK: (%[[IN:.*]]: !mo.tensor<[2, 3], f32, gpu:1>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SIG1:.*]]: !mo.buffer<[1], ui8, gpu:1>
mo.graph @broadcast_nonzero_root(%in: !mo.tensor<[2, 3], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.broadcast(%[[IN]], %[[SIG0]], %[[SIG1]], {{.*}}) [root = 1] : (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  %out:2, %outChain = mo.distributed.broadcast(%in, %sig0, %sig1, %ch) [root = 1] : (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// CHECK-LABEL: mo.graph @broadcast_folds
// CHECK: (%[[IN:.*]]: !mo.tensor<[2, 17], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @broadcast_folds(%in: !mo.tensor<[2, 17], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %ch: !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain) {
  // CHECK-NOT: mo.distributed.broadcast
  // CHECK: mo.output %[[IN]], %[[CH]]
  %out, %outChain = mo.distributed.broadcast(%in, %sig0, %ch) : (!mo.tensor<[2, 17], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain)
  mo.output %out, %outChain : !mo.tensor<[2, 17], f32, gpu:0>, !mo.chain
}

// CHECK-LABEL: mo.graph @scatter
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SIG1:.*]]: !mo.buffer<[1], ui8, gpu:1>
mo.graph @scatter(%in0: !mo.tensor<[2, 3], f32, gpu:0>, %in1: !mo.tensor<[2, 3], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  // CHECK: mo.distributed.scatter(%[[IN0]], %[[IN1]], %[[SIG0]], %[[SIG1]], {{.*}}) [root = 0] : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  %out:2, %outChain = mo.distributed.scatter(%in0, %in1, %sig0, %sig1, %ch) [root = 0] : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %out#0, %out#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// CHECK-LABEL: mo.graph @scatter_folds
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2, 17], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @scatter_folds(%in0: !mo.tensor<[2, 17], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %ch: !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain) {
  // CHECK-NOT: mo.distributed.scatter
  // CHECK: mo.output %[[IN0]], %[[CH]]
  %out, %outChain = mo.distributed.scatter(%in0, %sig0, %ch) [root = 0] : (!mo.tensor<[2, 17], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain)
  mo.output %out, %outChain : !mo.tensor<[2, 17], f32, gpu:0>, !mo.chain
}

// CHECK-LABEL: mo.graph @reduce_scatter_folds
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2, 17], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @reduce_scatter_folds(%in0: !mo.tensor<[2, 17], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %ch: !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain) {
  // CHECK-NOT: mo.distributed.reducescatter.sum
  // CHECK: mo.output %[[IN0]], %[[CH]]
  %out, %outChain = mo.distributed.reducescatter.sum(%in0, %sig0, %ch) [group_size = 1] : (!mo.tensor<[2, 17], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain) -> (!mo.tensor<[2, 17], f32, gpu:0>, !mo.chain)
  mo.output %out, %outChain : !mo.tensor<[2, 17], f32, gpu:0>, !mo.chain
}

// Single-device `mo.parallel { mo.bundled.expand -> mo.bundled.allreduce.sum }`
// folds away entirely:
//   - `mo.bundled.allreduce.sum` folds to its input via the new identity fold.
//   - `mo.bundled.expand` (single result, type matches input) folds to its
//     input and is then DCE'd as a Pure op with no users.
//   - `mo.parallel` folds because its body is `mo.yield %arg` and its buffers
//     and chain have no users in the body.
// CHECK-LABEL: mo.graph @bundled_all_reduce_folds
// CHECK: (%[[IN:.*]]: !mo.tensor<[3], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @bundled_all_reduce_folds(
    %in: !mo.tensor<[3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %ch: !mo.chain) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain) {
  %dt = mo.tensor.bundle(%in) : (!mo.tensor<[3], f32, gpu:0>) -> (!mo.bundle<[3], f32, gpu:0>)
  // CHECK-NOT: mo.parallel
  // CHECK-NOT: mo.bundled.expand
  // CHECK-NOT: mo.bundled.allreduce.sum
  %res, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[3], f32, gpu:0>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0>) {
    %peer0 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>)
    %out, %ch_out = mo.bundled.allreduce.sum(%peer0, %sig0, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %unb = mo.tensor.unbundle(%res) : (!mo.bundle<[3], f32, gpu:0>) -> (!mo.tensor<[3], f32, gpu:0>)
  // CHECK: mo.output %[[IN]], %[[CH]]
  mo.output %unb, %out_ch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// Two per-device bundled broadcasts of one root tensor.
// CHECK-LABEL: mo.graph @bundled_broadcast
// CHECK: (%[[IN:.*]]: !mo.tensor<[2, 3], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SIG1:.*]]: !mo.buffer<[1], ui8, gpu:1>
mo.graph @bundled_broadcast(%in: !mo.tensor<[2, 3], f32, gpu:0>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // CHECK: %[[B0:.*]], %[[C0:.*]] = mo.bundled.broadcast(%[[IN]], %[[SIG0]], %[[SIG1]], %{{.*}}) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  %b0, %c0 = mo.bundled.broadcast(%in, %sig0, %sig1, %ch) [root = 0]
      : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  // CHECK: %[[B1:.*]], %[[C1:.*]] = mo.bundled.broadcast(%[[IN]], %[[SIG0]], %[[SIG1]], %{{.*}}) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  %b1, %c1 = mo.bundled.broadcast(%in, %sig0, %sig1, %ch) [root = 0]
      : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  // CHECK: mo.output %[[B0]], %[[B1]] : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
  mo.output %b0, %b1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// Two per-device bundled broadcasts of a non-root-0 root tensor: the input
// must live on `signalBuffers[root]`'s device, here gpu:1 for root 1.
// CHECK-LABEL: mo.graph @bundled_broadcast_nonzero_root
// CHECK: (%[[IN:.*]]: !mo.tensor<[2, 3], f32, gpu:1>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SIG1:.*]]: !mo.buffer<[1], ui8, gpu:1>
mo.graph @bundled_broadcast_nonzero_root(%in: !mo.tensor<[2, 3], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // CHECK: %[[B0:.*]], %[[C0:.*]] = mo.bundled.broadcast(%[[IN]], %[[SIG0]], %[[SIG1]], %{{.*}}) [root = 1] : (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  %b0, %c0 = mo.bundled.broadcast(%in, %sig0, %sig1, %ch) [root = 1]
      : (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  // CHECK: %[[B1:.*]], %[[C1:.*]] = mo.bundled.broadcast(%[[IN]], %[[SIG0]], %[[SIG1]], %{{.*}}) [root = 1] : (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  %b1, %c1 = mo.bundled.broadcast(%in, %sig0, %sig1, %ch) [root = 1]
      : (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  // CHECK: mo.output %[[B0]], %[[B1]] : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
  mo.output %b0, %b1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// Two-peer bundled allgather with ragged peer shapes; each gathered piece
// keeps its peer's shape on this device's output device.
// CHECK-LABEL: mo.graph @bundled_allgather
// CHECK: (%[[IN0:.*]]: !mo.tensor<[2], f32, gpu:0>, %[[IN1:.*]]: !mo.tensor<[3], f32, gpu:1>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[SIG1:.*]]: !mo.buffer<[1], ui8, gpu:1>
mo.graph @bundled_allgather(%in0: !mo.tensor<[2], f32, gpu:0>, %in1: !mo.tensor<[3], f32, gpu:1>, %sig0: !mo.buffer<[1], ui8, gpu:0>, %sig1: !mo.buffer<[1], ui8, gpu:1>) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // CHECK: %[[O:.*]]:2, %[[OC:.*]] = mo.bundled.allgather(%[[IN0]], %[[IN1]], %[[SIG0]], %[[SIG1]], %[[CH:.*]]) : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain) -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.chain)
  %o:2, %c = mo.bundled.allgather(%in0, %in1, %sig0, %sig1, %ch)
      : (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
     -> (!mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>, !mo.chain)
  // CHECK: mo.output %[[O]]#0, %[[O]]#1 : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>
  mo.output %o#0, %o#1 : !mo.tensor<[2], f32, gpu:0>, !mo.tensor<[3], f32, gpu:0>
}

// Single-device `mo.parallel { mo.bundled.expand -> mo.bundled.broadcast }`
// folds away entirely, like the allreduce above: the broadcast folds to its
// input (single device, matching type), the expand folds to its input, and
// the parallel folds because its body is `mo.yield %arg`.
// CHECK-LABEL: mo.graph @bundled_broadcast_folds
// CHECK: (%[[IN:.*]]: !mo.tensor<[3], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @bundled_broadcast_folds(
    %in: !mo.tensor<[3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %ch: !mo.chain) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain) {
  %dt = mo.tensor.bundle(%in) : (!mo.tensor<[3], f32, gpu:0>) -> (!mo.bundle<[3], f32, gpu:0>)
  // CHECK-NOT: mo.parallel
  // CHECK-NOT: mo.bundled.expand
  // CHECK-NOT: mo.bundled.broadcast
  %res, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[3], f32, gpu:0>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0>) {
    %peer0 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>)
    %out, %ch_out = mo.bundled.broadcast(%peer0, %sig0, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %out, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %unb = mo.tensor.unbundle(%res) : (!mo.bundle<[3], f32, gpu:0>) -> (!mo.tensor<[3], f32, gpu:0>)
  // CHECK: mo.output %[[IN]], %[[CH]]
  mo.output %unb, %out_ch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}

// Single-device `mo.parallel { mo.bundled.expand -> mo.bundled.allgather }`
// folds away entirely, like the allreduce above.
// CHECK-LABEL: mo.graph @bundled_allgather_folds
// CHECK: (%[[IN:.*]]: !mo.tensor<[3], f32, gpu:0>, %[[SIG0:.*]]: !mo.buffer<[1], ui8, gpu:0>, %[[CH:.*]]: !mo.chain)
mo.graph @bundled_allgather_folds(
    %in: !mo.tensor<[3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %ch: !mo.chain) -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain) {
  %dt = mo.tensor.bundle(%in) : (!mo.tensor<[3], f32, gpu:0>) -> (!mo.bundle<[3], f32, gpu:0>)
  // CHECK-NOT: mo.parallel
  // CHECK-NOT: mo.bundled.expand
  // CHECK-NOT: mo.bundled.allgather
  %res, %out_ch = mo.parallel (%arg, %bch) in (%dt : !mo.bundle<[3], f32, gpu:0>)
      buffers(%sig0 : !mo.buffer<[1], ui8, gpu:0>)
      chain(%ch) -> (!mo.bundle<[3], f32, gpu:0>) {
    %peer0 = mo.bundled.expand(%arg) : !mo.tensor<[3], f32, gpu:0> -> (!mo.tensor<[3], f32, gpu:0>)
    %o0, %ch_out = mo.bundled.allgather(%peer0, %sig0, %bch)
        : (!mo.tensor<[3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.chain)
       -> (!mo.tensor<[3], f32, gpu:0>, !mo.chain)
    mo.yield %o0, %ch_out : !mo.tensor<[3], f32, gpu:0>, !mo.chain
  }
  %unb = mo.tensor.unbundle(%res) : (!mo.bundle<[3], f32, gpu:0>) -> (!mo.tensor<[3], f32, gpu:0>)
  // CHECK: mo.output %[[IN]], %[[CH]]
  mo.output %unb, %out_ch : !mo.tensor<[3], f32, gpu:0>, !mo.chain
}
