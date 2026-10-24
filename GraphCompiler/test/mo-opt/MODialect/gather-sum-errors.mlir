// RUN: mo-opt %s --split-input-file --verify-diagnostics

// Each case isolates one GatherSumOp::verify() rejection. Positive cases
// (valid gather_sum ops) live in the shape-inference and fusion suites.

// Indices dtype: the builtin kernel is compiled for int32 only
// (enforced by the confined operand type).
mo.graph @i64_indices(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 1, 3], f32> {
  // expected-error@below {{'mo.gather_sum' op operand #1 must be rank-2 tensor with signed 32-bit integer elements, but got '!mo.tensor<[2, 4], si64>'}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %0 : !mo.tensor<[2, 1, 3], f32>
}

// -----

// Indices rank: the kernel strides the indices buffer as
// [batch, multi_hot] (enforced by the confined operand type).
mo.graph @rank1_indices(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[4], si32>) -> !mo.tensor<[4, 1], f32> {
  // expected-error@below {{'mo.gather_sum' op operand #1 must be rank-2 tensor with signed 32-bit integer elements, but got '!mo.tensor<[4], si32>'}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4, 1], f32>
  mo.output %0 : !mo.tensor<[4, 1], f32>
}

// -----

// Input rank: the kernel binds the table as [num_embeddings, emb_dim]
// (enforced by the confined operand type).
mo.graph @rank1_table(%input: !mo.tensor<[5], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1], f32> {
  // expected-error@below {{'mo.gather_sum' op operand #0 must be rank-2 tensor, but got '!mo.tensor<[5], f32>'}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1], f32>
  mo.output %0 : !mo.tensor<[2, 1], f32>
}

// -----

// Result rank: the fused op produces [batch, 1, emb] (or [batch, emb]).
mo.graph @rank4_result(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3, 4], f32> {
  // expected-error@below {{gather_sum result must be rank 2 or 3}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3, 4], f32>
  mo.output %0 : !mo.tensor<[2, 1, 3, 4], f32>
}

// -----

// Result shape: with static shapes the result is pinned to the kernel's
// derived strides (batch from the indices, kept axis at 1, emb dim from
// the table).
mo.graph @bad_result_shape(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 2], f32> {
  // expected-error@below {{gather_sum result must be [batch, 1, emb_dim] (or [batch, emb_dim]) matching the indices batch and table width}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 2], f32>
  mo.output %0 : !mo.tensor<[2, 1, 2], f32>
}


// -----

// Parametric result: a matching symbolic dim verifies (static batch, the
// embedding dim is the table's parameter).
mo.graph @parametric_result<Emb>(%input: !mo.tensor<[5, Emb], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, Emb], f32> {
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, Emb], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, Emb], f32>
  mo.output %0 : !mo.tensor<[2, 1, Emb], f32>
}

// -----

// Parametric mismatch: the result's embedding dim is a different
// parameter than the table's.
mo.graph @parametric_mismatch<Emb, Other>(%input: !mo.tensor<[5, Emb], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, Other], f32> {
  // expected-error@below {{gather_sum result must be [batch, 1, emb_dim] (or [batch, emb_dim]) matching the indices batch and table width}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, Emb], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, Other], f32>
  mo.output %0 : !mo.tensor<[2, 1, Other], f32>
}
// -----

// Element type: the reduce keeps the table's dtype.
mo.graph @dtype_mismatch(%input: !mo.tensor<[5, 3], f32>, %indices: !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3], si32> {
  // expected-error@below {{'mo.gather_sum' op failed to verify that all of {input, result} have same dtype}}
  %0 = "mo.gather_sum"(%input, %indices) : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 1, 3], si32>
  mo.output %0 : !mo.tensor<[2, 1, 3], si32>
}
