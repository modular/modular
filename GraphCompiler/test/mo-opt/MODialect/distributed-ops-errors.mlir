// RUN: mo-opt %s --split-input-file --verify-diagnostics

// Test reducescatter with mismatched input shapes
mo.graph @reducescatter_mismatched_shapes(
    %arg0: !mo.tensor<[6, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[4, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{all inputs in each reduce-scatter group must have the same shape}}
  %0:3 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %sig0, %sig1, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[6, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:1>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>
}

// -----

// Test reducescatter with mismatched element types
mo.graph @reducescatter_mismatched_dtypes(
    %arg0: !mo.tensor<[6, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[6, 3], f16, gpu:1>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{all inputs must have the same element type}}
  %0:3 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %sig0, %sig1, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[6, 3], f32, gpu:0>, !mo.tensor<[6, 3], f16, gpu:1>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>
}

// -----

// Test allgather with incorrect number of outputs (should be num_inputs * num_inputs)
mo.graph @allgather_wrong_output_count(
    %arg0: !mo.tensor<[2, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{number of outputs must be num_inputs * group_size for all-gather; expected 4 outputs, found 2 (num_inputs=2, group_size=2)}}
  %0:3 = "mo.distributed.allgather"(%arg0, %arg1, %sig0, %sig1, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// -----

// Test allgather with 3 inputs but only 6 outputs (should be 9)
mo.graph @allgather_three_inputs_wrong_outputs(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>,
    %arg2: !mo.tensor<[4], f32, gpu:2>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>,
    %sig2: !mo.buffer<[537919488], ui8, gpu:2>)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
        !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{number of outputs must be num_inputs * group_size for all-gather; expected 9 outputs, found 6 (num_inputs=3, group_size=3)}}
  %0:7 = "mo.distributed.allgather"(%arg0, %arg1, %arg2, %sig0, %sig1, %sig2, %chain_in) {group_size = 3 : i64} :
    (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:2>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.chain)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
        !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 :
    !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
    !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>
}

// -----

// Test grouped allgather with a group size that does not divide inputs.
mo.graph @allgather_group_size_not_divisible(
    %arg0: !mo.tensor<[4], f32, gpu:0>,
    %arg1: !mo.tensor<[4], f32, gpu:1>,
    %arg2: !mo.tensor<[4], f32, gpu:2>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>,
    %sig2: !mo.buffer<[537919488], ui8, gpu:2>)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:2>, !mo.tensor<[4], f32, gpu:2>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{group_size must evenly divide the number of inputs (3), but got 2}}
  %0:7 = "mo.distributed.allgather"(%arg0, %arg1, %arg2, %sig0, %sig1, %sig2, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:2>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.buffer<[537919488], ui8, gpu:2>, !mo.chain)
    -> (!mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:2>, !mo.tensor<[4], f32, gpu:2>, !mo.chain)
  mo.output %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 :
    !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:1>, !mo.tensor<[4], f32, gpu:2>, !mo.tensor<[4], f32, gpu:2>
}

// -----

// Test valid allgather with 1 input and 1 output (1 * 1 = 1)
mo.graph @allgather_single_device(
    %arg0: !mo.tensor<[10, 20], f32, gpu:0>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>)
    -> (!mo.tensor<[10, 20], f32, gpu:0>) {
  %chain_in = mo.chain.create()
  %0:2 = "mo.distributed.allgather"(%arg0, %sig0, %chain_in) {group_size = 1 : i64} :
    (!mo.tensor<[10, 20], f32, gpu:0>, !mo.buffer<[537919488], ui8, gpu:0>, !mo.chain)
    -> (!mo.tensor<[10, 20], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[10, 20], f32, gpu:0>
}

// -----

// Test valid allgather with 2 inputs and 4 outputs (2 * 2 = 4)
mo.graph @allgather_two_devices(
    %arg0: !mo.tensor<[5, 10], f32, gpu:0>,
    %arg1: !mo.tensor<[5, 10], f32, gpu:1>,
    %sig0: !mo.buffer<[537919488], ui8, gpu:0>,
    %sig1: !mo.buffer<[537919488], ui8, gpu:1>)
    -> (!mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:0>,
        !mo.tensor<[5, 10], f32, gpu:1>, !mo.tensor<[5, 10], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  %0:5 = "mo.distributed.allgather"(%arg0, %arg1, %sig0, %sig1, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:1>,
     !mo.buffer<[537919488], ui8, gpu:0>, !mo.buffer<[537919488], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:0>,
        !mo.tensor<[5, 10], f32, gpu:1>, !mo.tensor<[5, 10], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1, %0#2, %0#3 :
    !mo.tensor<[5, 10], f32, gpu:0>, !mo.tensor<[5, 10], f32, gpu:0>,
    !mo.tensor<[5, 10], f32, gpu:1>, !mo.tensor<[5, 10], f32, gpu:1>
}

// -----

// Test reducescatter with wrong output size on scatter axis (axis -1)
// Input [3, 6] with 3 GPUs should give [3, 2] per device, not [3, 3]
mo.graph @reducescatter_wrong_scatter_dim(
    %arg0: !mo.tensor<[3, 6], f32, gpu:0>,
    %arg1: !mo.tensor<[3, 6], f32, gpu:1>,
    %arg2: !mo.tensor<[3, 6], f32, gpu:2>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{output dimension 1 should be 2 for device 0 (input dim 6 / group_size 3), but got 3}}
  %0:4 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %arg2, %sig0, %sig1, %sig2, %chain_in) {group_size = 3 : i64} :
    (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>, !mo.tensor<[3, 6], f32, gpu:2>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  mo.output %0#0, %0#1, %0#2 : !mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 2], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>
}

// -----

// Test reducescatter with wrong non-scatter dimension
// Non-scatter dims must match exactly: input [3, 6] should give [3, x], not [4, x]
mo.graph @reducescatter_wrong_non_scatter_dim(
    %arg0: !mo.tensor<[3, 6], f32, gpu:0>,
    %arg1: !mo.tensor<[3, 6], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[4, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{output dimension 0 must match input (3 vs 4)}}
  %0:3 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %sig0, %sig1, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[4, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[4, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>
}

// -----

// Test reducescatter with wrong output rank
mo.graph @reducescatter_wrong_rank(
    %arg0: !mo.tensor<[3, 6], f32, gpu:0>,
    %arg1: !mo.tensor<[3, 6], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{output rank must match input rank}}
  %0:3 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %sig0, %sig1, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[3, 6], f32, gpu:0>, !mo.tensor<[3, 6], f32, gpu:1>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>
}

// -----

// Test reducescatter ragged case with wrong partition size for GPU 1
// Input [3, 7] with 3 GPUs: 7/3 = 2 base + 1 remainder
// GPU 0 gets 3, GPU 1 gets 2, GPU 2 gets 2
mo.graph @reducescatter_ragged_wrong_size(
    %arg0: !mo.tensor<[3, 7], f32, gpu:0>,
    %arg1: !mo.tensor<[3, 7], f32, gpu:1>,
    %arg2: !mo.tensor<[3, 7], f32, gpu:2>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{output dimension 1 should be 2 for device 1 (input dim 7 / group_size 3), but got 3}}
  %0:4 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %arg2, %sig0, %sig1, %sig2, %chain_in) {group_size = 3 : i64} :
    (!mo.tensor<[3, 7], f32, gpu:0>, !mo.tensor<[3, 7], f32, gpu:1>, !mo.tensor<[3, 7], f32, gpu:2>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain)
    -> (!mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>, !mo.chain)
  mo.output %0#0, %0#1, %0#2 : !mo.tensor<[3, 3], f32, gpu:0>, !mo.tensor<[3, 3], f32, gpu:1>, !mo.tensor<[3, 2], f32, gpu:2>
}

// -----

// Test grouped reducescatter with a group size that does not divide inputs.
mo.graph @reducescatter_group_size_not_divisible(
    %arg0: !mo.tensor<[4, 8], f32, gpu:0>,
    %arg1: !mo.tensor<[4, 8], f32, gpu:1>,
    %arg2: !mo.tensor<[4, 8], f32, gpu:2>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>)
    -> (!mo.tensor<[2, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{group_size must evenly divide the number of inputs (3), but got 2}}
  %0:4 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %arg2, %sig0, %sig1, %sig2, %chain_in) {group_size = 2 : i64} :
    (!mo.tensor<[4, 8], f32, gpu:0>, !mo.tensor<[4, 8], f32, gpu:1>, !mo.tensor<[4, 8], f32, gpu:2>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.chain)
    -> (!mo.tensor<[2, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>, !mo.chain)
  mo.output %0#0, %0#1, %0#2 : !mo.tensor<[2, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[2, 8], f32, gpu:2>
}

// -----

// Test grouped reducescatter validates partition size by local rank.
mo.graph @reducescatter_grouped_wrong_size(
    %arg0: !mo.tensor<[5, 8], f32, gpu:0>,
    %arg1: !mo.tensor<[5, 8], f32, gpu:1>,
    %arg2: !mo.tensor<[3, 8], f32, gpu:2>,
    %arg3: !mo.tensor<[3, 8], f32, gpu:3>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %sig2: !mo.buffer<[1], ui8, gpu:2>,
    %sig3: !mo.buffer<[1], ui8, gpu:3>)
    -> (!mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[1, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>) {
  %chain_in = mo.chain.create()
  // expected-error@+1 {{output dimension 0 should be 2 for device 2 (input dim 3 / group_size 2), but got 1}}
  %0:5 = "mo.distributed.reducescatter.sum"(%arg0, %arg1, %arg2, %arg3, %sig0, %sig1, %sig2, %sig3, %chain_in) {axis = 0 : i64, group_size = 2 : i64} :
    (!mo.tensor<[5, 8], f32, gpu:0>, !mo.tensor<[5, 8], f32, gpu:1>, !mo.tensor<[3, 8], f32, gpu:2>, !mo.tensor<[3, 8], f32, gpu:3>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.buffer<[1], ui8, gpu:2>, !mo.buffer<[1], ui8, gpu:3>, !mo.chain)
    -> (!mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[1, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>, !mo.chain)
  mo.output %0#0, %0#1, %0#2, %0#3 : !mo.tensor<[3, 8], f32, gpu:0>, !mo.tensor<[2, 8], f32, gpu:1>, !mo.tensor<[1, 8], f32, gpu:2>, !mo.tensor<[1, 8], f32, gpu:3>
}

// -----

// Test broadcast with root index out of range
mo.graph @broadcast_root_out_of_range(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  // expected-error@+1 {{root index 5 out of range for 2 devices}}
  %0:3 = "mo.distributed.broadcast"(%in, %sig0, %sig1, %ch) {root = 5 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// -----

// Test broadcast with output/signal buffer count mismatch
mo.graph @broadcast_count_mismatch(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>) {
  %ch = mo.chain.create()
  // expected-error@+1 {{number of outputs must match number of signal buffers. Expected 2 outputs but got 1}}
  %0:2 = "mo.distributed.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// Test broadcast with output shape mismatch
mo.graph @broadcast_shape_mismatch(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  // expected-error@+1 {{all outputs must have the same shape as the input}}
  %0:3 = "mo.distributed.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:1>
}

// -----

// Test broadcast with output element type mismatch
mo.graph @broadcast_dtype_mismatch(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:1>) {
  %ch = mo.chain.create()
  // expected-error@+1 {{all outputs must have the same element type as the input}}
  %0:3 = "mo.distributed.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:1>
}

// -----

// Test broadcast with the input on a device other than the root's: the
// input is the root's copy of the tensor, so it must live on
// `signalBuffers[root]`'s device.
mo.graph @broadcast_input_not_on_root_device(
    %in: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  %ch = mo.chain.create()
  // expected-error@+1 {{the input must be on the root's device, #M.device_ref<"gpu", 0>, but is on #M.device_ref<"gpu", 1>}}
  %0:3 = "mo.distributed.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// -----

// Test scatter with empty inputs (no tensors, no signal buffers).
// NOTE: signal buffer count mismatch (condition 3) cannot be triggered
// independently because the SameVariadicOperandSize trait enforces
// inputs.size() == signalBuffers.size() before the custom verifier runs.
mo.graph @scatter_empty_inputs(%ch: !mo.chain) -> (!mo.chain) {
  // expected-error@+1 {{scatter requires at least one input}}
  %0:1 = "mo.distributed.scatter"(%ch) {root = 0 : i64} :
    (!mo.chain) -> (!mo.chain)
  mo.output %0#0 : !mo.chain
}

// -----

// Test scatter with output count != input count.
mo.graph @scatter_output_count_mismatch(
    %arg0: !mo.tensor<[2, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %ch: !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>) {
  // expected-error@+1 {{number of outputs must match number of inputs}}
  %0:2 = "mo.distributed.scatter"(%arg0, %arg1, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// Test scatter with mismatched input element types.
mo.graph @scatter_element_type_mismatch(
    %arg0: !mo.tensor<[2, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[2, 3], f16, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %ch: !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:1>) {
  // expected-error@+1 {{all inputs must have the same element type}}
  %0:3 = "mo.distributed.scatter"(%arg0, %arg1, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:0>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:1>
}

// -----

// Test scatter with root index out of range.
mo.graph @scatter_root_out_of_range(
    %arg0: !mo.tensor<[2, 3], f32, gpu:0>,
    %arg1: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>,
    %ch: !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>) {
  // expected-error@+1 {{root must be in range [0, ngpus)}}
  %0:3 = "mo.distributed.scatter"(%arg0, %arg1, %sig0, %sig1, %ch) {root = 5 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:0>,
     !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}

// -----

// Bundled-collective op verifiers are exercised inside isDeviceGraph graphs,
// the home that mo-mark-stable-tensors produces them in.

// Test bundled broadcast with no signal buffers.
mo.graph @bundled_broadcast_no_signal_buffers(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %ch: !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
    attributes {isDeviceGraph} {
  // expected-error@+1 {{bundled broadcast requires at least one signal buffer}}
  %0:2 = "mo.bundled.broadcast"(%in, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.chain
}

// -----

// Test bundled broadcast with root index out of range.
mo.graph @bundled_broadcast_root_out_of_range(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{root index 5 out of range for 2 devices}}
  %0:2 = "mo.bundled.broadcast"(%in, %sig0, %sig1, %ch) {root = 5 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// Test bundled broadcast with output shape mismatch.
mo.graph @bundled_broadcast_shape_mismatch(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[4, 3], f32, gpu:0>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{the output must have the same shape as the input}}
  %0:2 = "mo.bundled.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[4, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[4, 3], f32, gpu:0>
}

// -----

// Test bundled broadcast with output element type mismatch.
mo.graph @bundled_broadcast_dtype_mismatch(
    %in: !mo.tensor<[2, 3], f32, gpu:0>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f16, gpu:0>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{the output must have the same element type as the input}}
  %0:2 = "mo.bundled.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f16, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[2, 3], f16, gpu:0>
}

// -----

// Test bundled broadcast with the input on a device other than the root's:
// the input is the root's copy of the tensor, so it must live on
// `signalBuffers[root]`'s device.
mo.graph @bundled_broadcast_input_not_on_root_device(
    %in: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{the input must be on the root's device, #M.device_ref<"gpu", 0>, but is on #M.device_ref<"gpu", 1>}}
  %0:2 = "mo.bundled.broadcast"(%in, %sig0, %sig1, %ch) {root = 0 : i64} :
    (!mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// NOTE: an input/signal-buffer count mismatch for bundled allgather cannot
// reach the custom verifier: the operand type map rejects the op first (the
// buffer after the last mapped input lands on the chain operand). A fully
// empty op is likewise rejected by MO_SingleDevice first, so the verifier's
// "requires at least one input" is unreachable from parsed IR.

// Test bundled allgather with an output count mismatch: one gathered piece
// per peer, so two peers need two outputs.
mo.graph @bundled_allgather_wrong_output_count(
    %in0: !mo.tensor<[2, 3], f32, gpu:0>,
    %in1: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>) attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{number of outputs must match number of inputs (one gathered piece per peer); expected 2 outputs but got 1}}
  %0:2 = "mo.bundled.allgather"(%in0, %in1, %sig0, %sig1, %ch) :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0 : !mo.tensor<[2, 3], f32, gpu:0>
}

// -----

// Test bundled allgather with output 1 shape mismatch.
mo.graph @bundled_allgather_shape_mismatch(
    %in0: !mo.tensor<[2, 3], f32, gpu:0>,
    %in1: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:0>)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{output 1 must have the same shape as input 1}}
  %0:3 = "mo.bundled.allgather"(%in0, %in1, %sig0, %sig1, %ch) :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:0>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[4, 3], f32, gpu:0>
}

// -----

// Test bundled allgather with output 1 element type mismatch.
mo.graph @bundled_allgather_dtype_mismatch(
    %in0: !mo.tensor<[2, 3], f32, gpu:0>,
    %in1: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:0>)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{output 1 must have the same element type as input 1}}
  %0:3 = "mo.bundled.allgather"(%in0, %in1, %sig0, %sig1, %ch) :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:0>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f16, gpu:0>
}

// -----

// Test bundled allgather with output 1 on the wrong device: every gathered
// piece lands on this device's output device.
mo.graph @bundled_allgather_wrong_output_device(
    %in0: !mo.tensor<[2, 3], f32, gpu:0>,
    %in1: !mo.tensor<[2, 3], f32, gpu:1>,
    %sig0: !mo.buffer<[1], ui8, gpu:0>,
    %sig1: !mo.buffer<[1], ui8, gpu:1>)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>)
    attributes {isDeviceGraph} {
  %ch = mo.chain.create()
  // expected-error@+1 {{output 1 must be on this device's output device, #M.device_ref<"gpu", 0>, but is on #M.device_ref<"gpu", 1>}}
  %0:3 = "mo.bundled.allgather"(%in0, %in1, %sig0, %sig1, %ch) :
    (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.buffer<[1], ui8, gpu:0>, !mo.buffer<[1], ui8, gpu:1>, !mo.chain)
    -> (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>, !mo.chain)
  mo.output %0#0, %0#1 : !mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2, 3], f32, gpu:1>
}
