// RUN: mo-opt %s --split-input-file --canonicalize --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BroadcastShapeOp
//===----------------------------------------------------------------------===//

mo.graph @broadcast_shape_f32_output(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[2], f32> {
  // expected-error@+1 {{'mo.broadcast_shape' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[3], f32>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], f32>
  mo.output %res : !mo.tensor<[3], f32>
}

// -----

mo.graph @broadcast_shape_vector_output(%arg0: !mo.tensor<[3], si64>, %arg1: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 1], si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2, 1], si64>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 1], si64>
  mo.output %res : !mo.tensor<[2, 1], si64>
}

// -----

mo.graph @broadcast_shape_f32_input(%arg0: !mo.tensor<[3], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op operand #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[3], f32>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// -----

mo.graph @broadcast_shape_f32_input2(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64> {
  // expected-error@+1 {{'mo.broadcast_shape' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3], si64>
  mo.output %res : !mo.tensor<[3], si64>
}

// -----

mo.graph @broadcast_shape_invalid_output_rank(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[2], si32>) -> !mo.tensor<[3], si32> {
  // expected-error @below {{output rank must be the max between 2 input ranks, if all are statically ranked}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[2], si32>) -> !mo.tensor<[4], si64>
  mo.output %arg0 : !mo.tensor<[3], si32>
}

// -----

mo.graph @broadcast_shape_input_1_partially_ranked<D0>(%arg0: !mo.tensor<[D0], si32>, %arg1: !mo.tensor<[3], si32>) -> !mo.tensor<[2], si32> {
  // expected-error @below {{result rank must be no less than second input rank if both are statically ranked}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[D0], si32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2], si32>
  mo.output %res : !mo.tensor<[2], si32>
}

// -----

mo.graph @broadcast_shape_input_2_partially_ranked<D0>(%arg0: !mo.tensor<[3], si32>, %arg1: !mo.tensor<[D0], si32>) -> !mo.tensor<[2], si32> {
  // expected-error @below {{result rank must be no less than first input rank if both are statically ranked}}
  %res = mo.broadcast_shape(%arg0, %arg1) : (!mo.tensor<[3], si32>, !mo.tensor<[D0], si32>) -> !mo.tensor<[2], si32>
  mo.output %res : !mo.tensor<[2], si32>
}

// -----

//===----------------------------------------------------------------------===//
// BroadcastToOp
//===----------------------------------------------------------------------===//

mo.graph @broadcast_to_bad_shape<D0, D1>(%from: !mo.tensor<[D0, D1], f32>, %to: !mo.tensor<[2, 3], si64>) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{'mo.broadcast_to' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[2, 3], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @broadcast_to_bad_type<D0, D1, D2>(%from: !mo.tensor<[D0, D1], f32>, %to: !mo.tensor<[D2], f32>) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{'mo.broadcast_to' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @broadcast_to_change_dtype<D0, D1, D2>(%from: !mo.tensor<[D0, D1], f32>, %to: !mo.tensor<[D2], si32>) -> !mo.tensor<[2, 3], f64> {
  // expected-error@+1 {{'mo.broadcast_to' op requires the element type of results match first operand}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2], si32>) -> !mo.tensor<[2, 3], f64>
  mo.output %res : !mo.tensor<[2, 3], f64>
}

// -----

mo.graph @broadcast_new_shape_size_res_rank<D0, D1>(
  %from: !mo.tensor<[D0, D1], f32>, %to: !mo.tensor<[3], si32>
) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{new shape size and output rank must match}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @broadcast_input_shape_rank_res_rank(
  %from: !mo.tensor<[1, 2, 3], f32>, %to: !mo.tensor<[2], si32>
) -> !mo.tensor<[2, ?], f32> {
  // expected-error@+1 {{result rank must be at least the input rank}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[2], si32>) -> !mo.tensor<[2, ?], f32>
  mo.output %res : !mo.tensor<[2, ?], f32>
}

// -----

mo.graph @broadcast_input_res_not_broadcastable(
  %from: !mo.tensor<[1, 2, 3], f32>, %to: !mo.tensor<[3], si32>
) -> !mo.tensor<[4, 5, ?], f32> {
  // expected-error@+1 {{input shape not broadcastable to result shape}}
  %res = mo.broadcast_to(%from, %to) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[4, 5, ?], f32>
  mo.output %res : !mo.tensor<[4, 5, ?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

mo.graph @concat_no_args() -> !mo.tensor<[7], f32> {
  %ax = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[], si64>
  // expected-error@+1 {{concat must have at least 1 input tensor}}
  %0 = mo.concat[0]() : () -> !mo.tensor<[7], f32>
  mo.output %0 : !mo.tensor<[7], f32>
}

// -----

mo.graph @concat_non_matching_input_ranks(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32> {
  %ax = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.concat[0](%arg0, %arg1) : (!mo.tensor<[7], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32>
  mo.output %0 : !mo.tensor<[9], f32>
}

// -----

mo.graph @concat_non_matching_output_rank(%arg0: !mo.tensor<[2, 2], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32> {
  %ax = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0 = mo.concat[0](%arg0, %arg1) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[9], f32>
  mo.output %0 : !mo.tensor<[9], f32>
}

// -----

mo.graph @concat_incorrect_element_type(%arg0: !mo.tensor<[1], si64>, %arg1: !mo.tensor<[1], si32>) -> !mo.tensor<[2], si64> {
  %axis = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>

  // expected-error @below {{input and output element types must match}}
  %res = mo.concat[0](%arg0, %arg1) : (
    !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
  ) -> !mo.tensor<[2], si64>

  mo.output %res: !mo.tensor<[2], si64>
}

// -----

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

mo.graph @split_non_shapelike_splits(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2], f32>) -> (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>) {
  // expected-error @below {{'mo.split' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[2], f32>}}
  %0, %1 = mo.split(%arg0, %arg1) [axis = 0] : (!mo.tensor<[7], f32>, !mo.tensor<[2], f32>) -> (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>)
  mo.output %0, %1 : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
}

// -----

mo.graph @split_axis_out_of_range(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>) {
  // expected-error @below {{split axis must be in range [-rank, rank)}}
  %0, %1 = mo.split(%arg0, %arg1) [axis = 7] : (!mo.tensor<[7], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>)
  mo.output %0, %1 : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
}

// -----

mo.graph @split_non_matching_output_rank(%arg0: !mo.tensor<[6, 7], f32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>) {
  // expected-error @below {{input and output ranks must be all dynamic, or all static and match}}
  %0, %1 = mo.split(%arg0, %arg1) [axis = 0] : (!mo.tensor<[6, 7], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?], f32>, !mo.tensor<[?], f32>)
  mo.output %0, %1 : !mo.tensor<[?], f32>, !mo.tensor<[?], f32>
}

// -----

mo.graph @split_no_matching_elem_types(%arg0: !mo.tensor<[7], f32>, %arg1: !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>) {
  // expected-error @below {{input and output element types must match}}
  %0, %1 = mo.split(%arg0, %arg1) [axis = 0] : (!mo.tensor<[7], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[?], si32>, !mo.tensor<[?], si32>)
  mo.output %0, %1 : !mo.tensor<[?], si32>, !mo.tensor<[?], si32>
}

// -----

mo.graph @split_no_matching_static_shapes() -> (!mo.tensor<[1, 1], si32>, !mo.tensor<[3, 3], si32>) {
  %array = mo.constant <#M.dense_array<2, 3, 5, 7> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %shape = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %input = mo.reshape(%array, %shape) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], si32>
  %splitSizes = mo.constant <#M.dense_array<1, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error @below {{input and output static shapes must match except split axis}}
  %0, %1 = mo.split(%input, %splitSizes) [axis = 1] : (
    !mo.tensor<[2, 2], si32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[1, 1], si32>, !mo.tensor<[3, 3], si32>)
  mo.output %0, %1 : !mo.tensor<[1, 1], si32>, !mo.tensor<[3, 3], si32>
}

// -----

mo.graph @split_wrong_split_dim_sum() -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>) {
  %input = mo.constant <#M.dense_array<2, 3> : tensor<2xsi32>> : !mo.tensor<[2], si32>
  %splitSizes = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error @below {{sum of output shapes along split axis must match input shape}}
  %0, %1 = mo.split(%input, %splitSizes) [axis = 0] : (
    !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>)
  mo.output %0, %1 : !mo.tensor<[1], si32>, !mo.tensor<[2], si32>
}

// -----

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

mo.graph @gather_mixed_input_output_types(%input: !mo.tensor<[4], si32>, %indices: !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2], f32> {
  // expected-error @below {{input and output must have the same element type}}
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[4], si32>, !mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_scalar_param(%input: !mo.tensor<[], f32>, %indices: !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32> {
  // expected-error@+1 {{'mo.gather' op operand #0 must be unranked or non-scalar tensor, but got '!mo.tensor<[], f32>'}}
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_float_indices(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], f32>) -> !mo.tensor<[2, 2], f32> {
  // expected-error@+1 {{'mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_non_int_axis(%input: !mo.tensor<[2, 2], f32>, %indices: !mo.tensor<[2], f32>) -> !mo.tensor<[2, 2], f32> {
  // expected-error @below {{'mo.gather' op operand #1 must be tensor with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[2, 2], f32>
  mo.output %0 : !mo.tensor<[2, 2], f32>
}

// -----

mo.graph @gather_result_rank_too_big(%input: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?, ?], f32> {
  // expected-error @below {{result rank must be 1 less than the sum of param and indices ranks if they are all static}}
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @gather_result_rank_too_small(%input: !mo.tensor<[?, ?], f32>, %indices: !mo.tensor<[?], si64>) -> !mo.tensor<[?], f32> {
  // expected-error @below {{result rank must be 1 less than the sum of param and indices ranks if they are all static}}
  %0 = mo.gather(%input, %indices) [axis = 0] : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?], f32>
  mo.output %0 : !mo.tensor<[?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

mo.graph @pad_non_scalar_constant(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[4], si64>, %constant: !mo.tensor<[2], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{'mo.pad.constant' op operand #2 must be ranked scalar tensor, but got '!mo.tensor<[2], f32>'}}
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[2], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @pad_float_paddings(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[4], f32>, %constant: !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{'mo.pad.constant' op operand #1 must be shape [N] and signed 64 or signed 32 integer elements, but got '!mo.tensor<[4], f32>}}
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[4], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @pad_invalid_static_paddings(%input: !mo.tensor<[2, 3], f32>, %paddings: !mo.tensor<[6], si64>, %constant: !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{'mo.pad.constant' op paddings shape does not match static input rank}}
  %res = mo.pad.constant(%input, %paddings, %constant) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[6], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// RangeOp
//===----------------------------------------------------------------------===//

mo.graph @range_result_rank(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>
) -> !mo.tensor<[5, 1], si32> {
  // expected-error@+1 {{'mo.range' op result #0 must be rank 1 or unranked, but got '!mo.tensor<[5, 1], si32>'}}
  %res = mo.range(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[5, 1], si32>
  mo.output %res : !mo.tensor<[5, 1], si32>
}

// -----

mo.graph @range_arg_eltype_mismatch(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], f32>
) -> !mo.tensor<[?], si32> {
  // expected-error@+1 {{'mo.range' op requires the same element type for all operands and results}}
  %res = mo.range(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], f32>) -> !mo.tensor<[?], si32>
  mo.output %res : !mo.tensor<[?], si32>
}

// -----

mo.graph @range_res_eltype_mismatch(
  %arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>, %arg2: !mo.tensor<[], si32>
) -> !mo.tensor<[?], f32> {
  // expected-error@+1 {{'mo.range' op requires the same element type for all operands and results}}
  %res = mo.range(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[?], f32>
  mo.output %res : !mo.tensor<[?], f32>
}

// -----

mo.graph @range_too_many_params<() -> K>(
  %arg0: !mo.tensor<[], f32>, %arg1: !mo.tensor<[], f32>, %arg2: !mo.tensor<[], f32>
) -> !mo.tensor<[?], f32> {
  // expected-error@+1 {{cannot declare more than one parameter}}
  %res = mo.range<() -> M, N>(%arg0, %arg1, %arg2) : (
    !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[M], f32>
  mo.output<M> %res : !mo.tensor<[M], f32>
}

// -----

//===----------------------------------------------------------------------===//
// ShapeOfOp
//===----------------------------------------------------------------------===//

mo.graph @shape_of_f32_output(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], f32> {
  // expected-error@+1 {{'mo.shape_of' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2], f32>
  mo.output %res : !mo.tensor<[2], f32>
}

// -----

mo.graph @shape_of_rank_2(%arg: !mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 1], si64> {
  // expected-error@+1 {{'mo.shape_of' op result #0 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2, 1], si64>'}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, 3], f32>) -> !mo.tensor<[2, 1], si64>
  mo.output %res : !mo.tensor<[2, 1], si64>
}

// -----

mo.graph @shape_of_rank_mismatch(%arg: !mo.tensor<[1, ?], f32>) -> !mo.tensor<[1], si64> {
  // expected-error@+1 {{input rank and output size must match}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[1, ?], f32>) -> !mo.tensor<[1], si64>
  mo.output %res : !mo.tensor<[1], si64>
}

// -----

mo.graph @shape_of_dyn_rank_static_res(%arg: !mo.tensor<[?, ?], f32>) -> !mo.tensor<[?], si64> {
  // expected-error@+1 {{input must have dynamic rank if result has unknown size}}
  %res = mo.shape_of(%arg) : (!mo.tensor<[?, ?], f32>) -> !mo.tensor<[?], si64>
  mo.output %res : !mo.tensor<[?], si64>
}

// -----

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

mo.graph @slice_non_vector_index(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[1, 2], si64>,
  %stop : !mo.tensor<[2], si32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{'mo.slice' op operand #1 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[1, 2], si64>'}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[1, 2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_f32_index(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[2], f32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{'mo.slice' op operand #2 must be rank 1 or unranked with integer or index elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], f32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_start_invalid_rank(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[1], si64>,
  %stop : !mo.tensor<[2], si32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{start shape must be [inputRank] if it's static}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[1], si64>, !mo.tensor<[2], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_end_invalid_rank(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[3], si32>,
  %step : !mo.tensor<[2], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{stop shape must be [inputRank] if it's static}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[3], si32>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

mo.graph @slice_step_invalid_rank(
  %input: !mo.tensor<[10, 10], f32>,
  %start : !mo.tensor<[2], si64>,
  %stop : !mo.tensor<[2], si32>,
  %step : !mo.tensor<[3], si64>
) -> !mo.tensor<[8, 5], f32> {
  // expected-error @below {{step shape must be [inputRank] if it's static}}
  %res = mo.slice(%input, %start, %stop, %step) : (
    !mo.tensor<[10, 10], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si32>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[8, 5], f32>
  mo.output %res : !mo.tensor<[8, 5], f32>
}

// -----

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

mo.graph @transpose_fp32_perm(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<1.0, 0.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  // expected-error@+1 {{'mo.transpose' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// -----

mo.graph @transpose_2d_perm(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<1x2xsi64>> : !mo.tensor<[1, 2], si64>
  // expected-error@+1 {{'mo.transpose' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[1, 2], si64>'}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1, 2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// -----

mo.graph @transpose_invalid_perm_size_static(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[3, 2], f32> {
  %perm = mo.constant <#M.dense_array<0, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{output shape is invalid based on constant permutation}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>
  mo.output %res : !mo.tensor<[3, 2], f32>
}

// -----

mo.graph @transpose_perm_size_dynamic(%input: !mo.tensor<[?, ?], f32>, %perm: !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{argument and result must be dynamically ranked if perm has unknown shape}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @transpose_arg_res_rank_mismatch(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{input and result ranks do not match}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @transpose_perm_rank_mismatch(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %perm = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{input and result ranks do not match perm size}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @transpose_duplicated_perm(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %perm = mo.constant <#M.dense_array<1, 1, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  // expected-error@+1 {{invalid permutation}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// -----

mo.graph @transpose_out_of_bound_perm(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %perm = mo.constant <#M.dense_array<-1, 3, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  // expected-error@+1 {{invalid permutation}}
  %res = mo.transpose(%input, %perm) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// -----

//===----------------------------------------------------------------------===//
// ArgNonzeroOp
//===----------------------------------------------------------------------===//

mo.graph @arg_nonzero_bad_result_rank(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1, 2], si64> {
  // expected-error@+1 {{mo.arg_nonzero: invalid output shape: rank must be 2}}
  %res = mo.arg_nonzero(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1, 2], si64>
  mo.output %res : !mo.tensor<[?, 1, 2], si64>
}

// -----

mo.graph @arg_nonzero_bad_result_dim(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1], si64> {
  // expected-error@+1 {{mo.arg_nonzero: invalid output shape: inner_dim != input_rank}}
  %res = mo.arg_nonzero(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 1], si64>
  mo.output %res : !mo.tensor<[?, 1], si64>
}

// -----

mo.graph @arg_nonzero_fp_result_type(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], si64> {
  // expected-error@+1 {{'mo.arg_nonzero' op result #0 must be signed 64 integer element type, but got '!mo.tensor<[?, 2], f32>'}}
  %res = mo.arg_nonzero(%arg) : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[?, 2], f32>
  mo.output %res : !mo.tensor<[?, 2], f32>
}

//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

// -----

mo.graph @arg_max_bad_rank(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[7, 1, 1], si64> {
  // expected-error@+1 {{'mo.reduce.arg_max' op input and output must either both have dynamic ranks or both have the same size static ranks}}
  %res = mo.reduce.arg_max(%arg) [axis = 0] : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[7, 1, 1], si64>
  mo.output %res : !mo.tensor<[7, 1, 1], si64>
}

// -----

mo.graph @arg_max_bad_dimensions(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[6, 1], si64> {
  // expected-error@+1 {{output dimensions must match input dimensions}}
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[6, 1], si64>
  mo.output %res : !mo.tensor<[6, 1], si64>
}

// -----

mo.graph @arg_max_bad_reduction_dim(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[7, 2], si64> {
  // expected-error@+1 {{output dimensions must be 1 at the reduction axis}}
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[7, 2], si64>
  mo.output %res : !mo.tensor<[7, 2], si64>
}

// -----

mo.graph @arg_max_too_many_reductions(%arg: !mo.tensor<[7, 7], f32>) -> !mo.tensor<[1, 1], si64> {
  // expected-error@+1 {{output dimensions must match input dimensions}}
  %res = mo.reduce.arg_max(%arg) [axis = 1] : (!mo.tensor<[7, 7], f32>) -> !mo.tensor<[1, 1], si64>
  mo.output %res : !mo.tensor<[1, 1], si64>
}

// -----

//===----------------------------------------------------------------------===//
// GuardOp
//===----------------------------------------------------------------------===//

mo.graph @guard_mismatched_num_results(%X: !mo.tensor<[D0, D1], f32>, %Y: !mo.tensor<[D2, D3], f32>, %chain: !mo.chain )
{
  //expected-error@+1 {{operation defines 2 results but was provided 3 to bind}}
  %guarded:3 = mo.guard[%chain](%X, %Y) : !mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>

  %Z = mo.matmul(%Xp, %Yp) :
   (!mo.tensor<[D0, D1], f32>, !mo.tensor<[D2, D3], f32>) -> !mo.tensor<[D0, D3], f32>

  mo.output
}

// -----

mo.graph @guard_disallows_buffer(%X: !mo.buffer<[2, 2], f32>)
{
  %ch0 = mo.chain.create()

  //expected-error@+1 {{'mo.guard' op operand #1 must be variadic of any type but !mo.buffer, but got '!mo.buffer<[2, 2], f32>'}}
  mo.guard[%ch0](%X) : !mo.buffer<[2, 2], f32>

  mo.output
}

// -----

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

mo.graph @tile_fp32_repeats(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32> {
  %repeats = mo.constant <#M.dense_array<2.0, 1.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  // expected-error@+1 {{'mo.tile' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[2], f32>'}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2], f32>) -> !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

// -----

mo.graph @tile_2d_repeats(%input: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[4, 3], f32> {
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<1x2xsi64>> : !mo.tensor<[1, 2], si64>
  // expected-error@+1 {{'mo.tile' op operand #1 must be rank 1 and signed 64 or signed 32 integer elements, but got '!mo.tensor<[1, 2], si64>'}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[1, 2], si64>) -> !mo.tensor<[4, 3], f32>
  mo.output %res : !mo.tensor<[4, 3], f32>
}

// -----

mo.graph @tile_repeats_size_dynamic(%input: !mo.tensor<[?, ?], f32>, %repeats: !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{argument and result must be dynamically ranked if repeats has unknown shape}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[?, ?], f32>, !mo.tensor<[?], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @tile_arg_res_rank_mismatch(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?], f32> {
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{input and result ranks do not match}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @tile_repeats_rank_mismatch(%input: !mo.tensor<[?, ?, ?], f32>) -> !mo.tensor<[?, ?, ?], f32> {
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{input and result ranks do not match repeats size}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?, ?], f32>
  mo.output %res : !mo.tensor<[?, ?, ?], f32>
}

// -----

// `repeats` is a host operand: a device-resident input is fine, but the
// repeats must stay on the CPU.
mo.graph @tile_repeats_on_gpu(%input: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[4, 3], f32, gpu:0> {
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64, gpu:0>
  // expected-error@+1 {{mo.tile: Operand 1 - mo.constant (using zero-based indexing) is constrained to be on device #M.device_ref<"cpu", 0> but has device #M.device_ref<"gpu", 0>.}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2], si64, gpu:0>) -> !mo.tensor<[4, 3], f32, gpu:0>
  mo.output %res : !mo.tensor<[4, 3], f32, gpu:0>
}

// -----

// All operands and the result must share one device; the host-resident
// repeats do not move the execution device.
mo.graph @tile_input_result_device_mismatch(%input: !mo.tensor<[2, 3], f32, gpu:0>) -> !mo.tensor<[4, 3], f32, gpu:1> {
  %repeats = mo.constant <#M.dense_array<2, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{mo.tile: Operand 0 (using zero-based indexing) is constrained to be on device #M.device_ref<"gpu", 1> but has device #M.device_ref<"gpu", 0>.}}
  %res = mo.tile(%input, %repeats) : (!mo.tensor<[2, 3], f32, gpu:0>, !mo.tensor<[2], si64>) -> !mo.tensor<[4, 3], f32, gpu:1>
  mo.output %res : !mo.tensor<[4, 3], f32, gpu:1>
}


//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

// -----
mo.graph @reshape_wrong_dtype(%input: !mo.tensor<[6], f32>) -> !mo.tensor<[2, 3], si32> {
  %shape = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{'mo.reshape' op requires the element type of results match first operand.}}
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], si32>
  mo.output %res : !mo.tensor<[2, 3], si32>
}

// -----
mo.graph @reshape_incompatible_shapes(%input: !mo.tensor<[6], f32>) -> !mo.tensor<[2, 3], f32> {
  %shape = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{'mo.reshape' op Expected requested shape and output shape to have the same number of elements requested 4 but expected 6}}
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @reshape_neg_one_constant_shape(%input: !mo.tensor<[6], f32>) -> !mo.tensor<[2, 3], f32> {
  %shape = mo.constant <#M.dense_array<2, -1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // expected-error@+1 {{'mo.reshape' op -1 size unsupported in requested shape}}
  %res = mo.reshape(%input, %shape) : (!mo.tensor<[6], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @scatter_nd_rank_mismatch(
  %input: !mo.tensor<[?, ?, ?], f32>,
  %updates: !mo.tensor<[9, 2, 1, 3], f32>,
  %indices: !mo.tensor<[?, ?, ?], si32>
) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{input and result ranks must match if both static: input rank = 3, result rank = 2}}
  %0 = mo.scatter_nd(%input, %updates, %indices)
  : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[9, 2, 1, 3], f32>, !mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @scatter_nd_shape_mismatch(
  %input: !mo.tensor<[10, 1, 3], f32>,
  %updates: !mo.tensor<[9, 2, 1, 3], f32>,
  %indices: !mo.tensor<[?, ?, ?], si32>
) -> !mo.tensor<[10, 2, 3], f32> {
  // expected-error@+1 {{input and result shapes must match if both static: input shape = [10, 1, 3], result shape = [10, 2, 3]}}
  %0 = mo.scatter_nd(%input, %updates, %indices)
  : (!mo.tensor<[10, 1, 3], f32>, !mo.tensor<[9, 2, 1, 3], f32>, !mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[10, 2, 3], f32>
  mo.output %0 : !mo.tensor<[10, 2, 3], f32>
}

// -----

mo.graph @scatter_nd_add_rank_mismatch(
  %input: !mo.tensor<[?, ?, ?], f32>,
  %updates: !mo.tensor<[9, 2, 1, 3], f32>,
  %indices: !mo.tensor<[?, ?, ?], si32>
) -> !mo.tensor<[?, ?], f32> {
  // expected-error@+1 {{input and result ranks must match if both static: input rank = 3, result rank = 2}}
  %0 = mo.scatter_nd.add(%input, %updates, %indices)
  : (!mo.tensor<[?, ?, ?], f32>, !mo.tensor<[9, 2, 1, 3], f32>, !mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[?, ?], f32>
  mo.output %0 : !mo.tensor<[?, ?], f32>
}

// -----

mo.graph @scatter_nd_min_shape_mismatch(
  %input: !mo.tensor<[10, 1, 3], f32>,
  %updates: !mo.tensor<[9, 2, 1, 3], f32>,
  %indices: !mo.tensor<[?, ?, ?], si32>
) -> !mo.tensor<[10, 2, 3], f32> {
  // expected-error@+1 {{input and result shapes must match if both static: input shape = [10, 1, 3], result shape = [10, 2, 3]}}
  %0 = mo.scatter_nd.min(%input, %updates, %indices)
  : (!mo.tensor<[10, 1, 3], f32>, !mo.tensor<[9, 2, 1, 3], f32>, !mo.tensor<[?, ?, ?], si32>) -> !mo.tensor<[10, 2, 3], f32>
  mo.output %0 : !mo.tensor<[10, 2, 3], f32>
}

//===----------------------------------------------------------------------===//
// SqueezeOp
//===----------------------------------------------------------------------===//

// -----

mo.graph @squeeze_shape_changed_dtypes(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[7], si64> {
  %indices = mo.constant <#M.dense_array<0, 2, 6> : tensor<3xsi64>> : !mo.tensor<[3], si64>

  // Input and output dtypes should match
  // expected-error@+1 {{'mo.squeeze_shape' op requires the element type of results match first operand.}}
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[7], si64>
  mo.output %res : !mo.tensor<[7], si64>
}

// -----

mo.graph @squeeze_same_indices(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[2], si32> {
  %indices = mo.constant <#M.dense_array<0, -4> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  // 0, and -4 map to the same dimension (index 0)
  // expected-error@+1 {{'mo.squeeze_shape' op Padding indices pad the same dimension twice}}
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2], si32>
  mo.output %res : !mo.tensor<[2], si32>
}

// -----

mo.graph @squeeze_index_out_of_range(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[3], si32> {
  %indices = mo.constant <#M.dense_array<-10> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  // -10 is outside [-4, 4); canonicalization must not wrap it onto a valid axis.
  // expected-error@+1 {{'mo.squeeze_shape' op Squeeze index not in [-in_rank, in_rank)}}
  %res = mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[3], si32>
  mo.output %res : !mo.tensor<[3], si32>
}


//===----------------------------------------------------------------------===//
// UnsqueezeOp
//===----------------------------------------------------------------------===//

// -----

mo.graph @unsqueeze_shape_changed_dtypes(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[7], si64> {
  %indices = mo.constant <#M.dense_array<0, 2, 6> : tensor<3xsi64>> : !mo.tensor<[3], si64>

  // Input and output dtypes should match
  // expected-error@+1 {{'mo.unsqueeze_shape' op requires the element type of results match first operand.}}
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[7], si64>
  mo.output %res : !mo.tensor<[7], si64>
}

// -----

mo.graph @unsqueeze_same_indices(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[7], si32> {
  %indices = mo.constant <#M.dense_array<0, 2, -7> : tensor<3xsi64>> : !mo.tensor<[3], si64>

  // 0, and -7 map to the same dimension (index 0)
  // expected-error@+1 {{'mo.unsqueeze_shape' op Padding indices pad the same dimension twice}}
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[3], si64>) -> !mo.tensor<[7], si32>
  mo.output %res : !mo.tensor<[7], si32>
}

// -----

mo.graph @unsqueeze_index_out_of_range(%input_shape: !mo.tensor<[4], si32>) -> !mo.tensor<[5], si32> {
  %indices = mo.constant <#M.dense_array<5> : tensor<1xsi32>> : !mo.tensor<[1], si32>

  // 5 is outside [-5, 5) for the rank-5 output.
  // expected-error@+1 {{'mo.unsqueeze_shape' op Unsqueeze index not in [-out_rank, out_rank)}}
  %res = mo.unsqueeze_shape(%input_shape, %indices) : (!mo.tensor<[4], si32>, !mo.tensor<[1], si32>) -> !mo.tensor<[5], si32>
  mo.output %res : !mo.tensor<[5], si32>
}

// -----

mo.graph @top_k_different_output_shapes<() -> D0, D1, D2, D3, D4, D5, D6, D7>(
   %input: !mo.tensor<[32, 10, 20, 3], f32>,
   %K: !mo.tensor<[], si64>,
   %axis: !mo.tensor<[], si64>,
   %sorted : !mo.tensor<[], bool>
) -> (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6, D7], si64>) {

  // expected-error@+1 {{'mo.top_k' op values and indices should have the same shape}}
  %res:2 = mo.top_k<() -> D8, D9, D10, D11, D12, D13, D14, D15>(%input, %K, %axis, %sorted) :
     (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
    -> (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D12, D13, D14, D15], si64>)
 mo.output<D8, D9, D10, D11, D12, D13, D14, D15> %res#0, %res#1 : !mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D12, D13, D14, D15], si64>
}

// -----

mo.graph @bottom_k_different_output_shapes<() -> D0, D1, D2, D3, D4, D5, D6, D7>(
   %input: !mo.tensor<[32, 10, 20, 3], f32>,
   %K: !mo.tensor<[], si64>,
   %axis: !mo.tensor<[], si64>,
   %sorted : !mo.tensor<[], bool>
) -> (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[D4, D5, D6, D7], si64>) {

  // expected-error@+1 {{'mo.bottom_k' op values and indices should have the same shape}}
  %res:2 = mo.bottom_k<() -> D8, D9, D10, D11, D12, D13, D14, D15>(%input, %K, %axis, %sorted) :
     (!mo.tensor<[32, 10, 20, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>)
    -> (!mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D12, D13, D14, D15], si64>)
 mo.output<D8, D9, D10, D11, D12, D13, D14, D15> %res#0, %res#1 : !mo.tensor<[D8, D9, D10, D11], f32>, !mo.tensor<[D12, D13, D14, D15], si64>
}
