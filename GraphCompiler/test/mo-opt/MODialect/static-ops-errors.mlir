// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// StaticBroadcastToOp
//===----------------------------------------------------------------------===//

mo.graph @static_broadcast_input_shape_rank_res_rank(%from: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  // expected-error@+1 {{result rank must be at least the input rank}}
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[1, 2, 3], f32> -> !mo.tensor<[2, 3], f32>
  mo.output %res : !mo.tensor<[2, 3], f32>
}

// -----

mo.graph @static_broadcast_change_dtype(%from: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[1, 2, 3], si32> {
  // expected-error@+1 {{'mo.static.broadcast_to' op requires the element type of results match first operand.}}
  %res = mo.static.broadcast_to(%from) : !mo.tensor<[2, 3], f32> -> !mo.tensor<[1, 2, 3], si32>
  mo.output %res : !mo.tensor<[1, 2, 3], si32>
}

//===----------------------------------------------------------------------===//
// Static_ReshapeOp
//===----------------------------------------------------------------------===//

// -----
mo.graph @static_reshape_non_matching_dims(%arg0: !mo.tensor<[5, 10], si64>) -> !mo.tensor<[45], si64> {
  // expected-error @below {{'mo.static.reshape' op input and output elements do not match: 50 vs 45}}
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[5, 10], si64> -> !mo.tensor<[45], si64>
  mo.output %0 : !mo.tensor<[45], si64>
}

// -----
mo.graph @static_reshape_wrong_dtype(%arg0: !mo.tensor<[5, 10], si64>) -> !mo.tensor<[50], si32> {
  // expected-error @below {{'mo.static.reshape' op requires the element type of results match first operand.}}
  %0 = mo.static.reshape(%arg0) : !mo.tensor<[5, 10], si64> -> !mo.tensor<[50], si32>
  mo.output %0 : !mo.tensor<[50], si32>
}
