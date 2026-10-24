// RUN: mo-opt %s --split-input-file --verify-diagnostics

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// BatchMatmulOp
//===----------------------------------------------------------------------===//

mo.graph @batch_matmul_too_low_rank(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
  // expected-error@+1 {{All operands and results must be of at least rank 3.}}
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// -----

mo.graph @batch_matmul_mismatched_batches(%arg0: !mo.tensor<[1, 10, 20], f32>, %arg1: !mo.tensor<[2, 20, 5], f32>) -> !mo.tensor<[2, 10, 5], f32> {
  // expected-error@+1 {{'mo.batch_matmul' op incompatible input/result batch dimensions}}
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[1, 10, 20], f32>, !mo.tensor<[2, 20, 5], f32>) -> !mo.tensor<[2, 10, 5], f32>
  mo.output %0 : !mo.tensor<[2, 10, 5], f32>
}

// -----

mo.graph @batch_matmul_mismatched_result_batches(%arg0: !mo.tensor<[2, 10, 20], f32>, %arg1: !mo.tensor<[2, 20, 5], f32>) -> !mo.tensor<[1, 10, 5], f32> {
  // expected-error@+1 {{'mo.batch_matmul' op incompatible input/result batch dimensions}}
  %0 = mo.batch_matmul(%arg0, %arg1) : (!mo.tensor<[2, 10, 20], f32>, !mo.tensor<[2, 20, 5], f32>) -> !mo.tensor<[1, 10, 5], f32>
  mo.output %0 : !mo.tensor<[1, 10, 5], f32>
}

// -----

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

mo.graph @rank_too_low(%arg0: !mo.tensor<[10], f32>, %arg1: !mo.tensor<[10], f32>) -> !mo.tensor<[10], f32> {
  // expected-error@+1 {{'mo.matmul' op operand #0 must be rank-2 tensor, but got '!mo.tensor<[10], f32>'}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10], f32>, !mo.tensor<[10], f32>) -> !mo.tensor<[10], f32>
  mo.output %0 : !mo.tensor<[10], f32>
}

// -----

mo.graph @different_input_ranks(%arg0: !mo.tensor<[10, 10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
  // expected-error @below {{'mo.matmul' op operand #0 must be rank-2 tensor, but got '!mo.tensor<[10, 10, 20], f32>'}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// -----

mo.graph @K_mismatch(%arg0: !mo.tensor<[10, 10], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32> {
  // expected-error@+1 {{'mo.matmul' op incompatible input/result matmul shapes}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 5], f32>
}

// -----

mo.graph @output_wrong_shape(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 20], f32> {
  // expected-error@+1 {{'mo.matmul' op incompatible input/result matmul shapes}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10, 20], f32>
  mo.output %0 : !mo.tensor<[10, 20], f32>
}

// -----

mo.graph @output_wrong_rank(%arg0: !mo.tensor<[10, 20], f32>, %arg1: !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10], f32> {
  // expected-error @below {{'mo.matmul' op result #0 must be rank-2 tensor, but got '!mo.tensor<[10], f32>'}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 20], f32>, !mo.tensor<[20, 5], f32>) -> !mo.tensor<[10], f32>
  mo.output %0 : !mo.tensor<[10], f32>
}

// -----

mo.graph @matmul_too_high_rank(%arg0: !mo.tensor<[10, 10, 20], f32>, %arg1: !mo.tensor<[10, 20, 5], f32>) -> !mo.tensor<[10, 10, 5], f32> {
  // expected-error@+1 {{'mo.matmul' op operand #0 must be rank-2 tensor, but got '!mo.tensor<[10, 10, 20], f32>'}}
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[10, 10, 20], f32>, !mo.tensor<[10, 20, 5], f32>) -> !mo.tensor<[10, 10, 5], f32>
  mo.output %0 : !mo.tensor<[10, 10, 5], f32>
}
