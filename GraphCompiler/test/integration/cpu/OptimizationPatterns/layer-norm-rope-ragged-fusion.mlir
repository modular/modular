// COM: Test LayerNorm + ragged RoPE fusion pattern.
// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// ============================================================
// Positive tests — pattern should fire and produce
// mo.composite.layer_norm_rope_ragged
// ============================================================

// Basic case, confirmed against a real DeepSeekV3.2/GLM 5.2 Indexer graph:
// layer_norm runs in a f32 upcast/downcast sandwich, `mo.static.reshape`
// unsqueezes the *whole* normed tensor (not just the roped half), and the
// roped/passthrough halves are two independent `mo.slice`s -- one through
// the unsqueeze, one direct -- rather than a single `mo.split`.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_basic
// CHECK:       mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_basic<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 4], bf16, gpu:0>,
  %gamma:             !mo.tensor<[4], f32, gpu:0>,
  %beta:              !mo.tensor<[4], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 4], bf16, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %nope_start_2d  = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %input_f32 = mo.cast(%input) : (!mo.tensor<[total_seq_len, 4], bf16, gpu:0>) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>
  %normed_f32 = mo.reduce.layer_norm(%input_f32, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>
  %normed = mo.cast(%normed_f32) : (!mo.tensor<[total_seq_len, 4], f32, gpu:0>) -> !mo.tensor<[total_seq_len, 4], bf16, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 4], bf16, gpu:0> -> !mo.tensor<[total_seq_len, 1, 4], bf16, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 4], bf16, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 2], bf16, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 4], bf16, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 2], bf16, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = true, rope_first = false} :
    (!mo.tensor<[total_seq_len, 1, 2], bf16, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 2], bf16, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 2], bf16, gpu:0> -> !mo.tensor<[total_seq_len, 2], bf16, gpu:0>

  %result = mo.concat[-1](%roped_sq, %nope_half) :
    (!mo.tensor<[total_seq_len, 2], bf16, gpu:0>, !mo.tensor<[total_seq_len, 2], bf16, gpu:0>)
    -> !mo.tensor<[total_seq_len, 4], bf16, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 4], bf16, gpu:0>
}

// Same as above, but layer_norm runs directly in the activation dtype (no
// f32 upcast/downcast sandwich). The pattern must not require the sandwich.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_no_cast_sandwich
// CHECK:       mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_no_cast_sandwich<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 4], f32, gpu:0>,
  %gamma:             !mo.tensor<[4], f32, gpu:0>,
  %beta:              !mo.tensor<[4], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %nope_start_2d  = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 4], f32, gpu:0> -> !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = true, rope_first = false} :
    (!mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0> -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %result = mo.concat[-1](%roped_sq, %nope_half) :
    (!mo.tensor<[total_seq_len, 2], f32, gpu:0>, !mo.tensor<[total_seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 4], f32, gpu:0>
}

// ============================================================
// Negative tests — pattern should NOT fire
// ============================================================

// Wrong concat order: concat(nope, roped) instead of concat(roped, nope).
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_wrong_concat_order
// CHECK:       mo.reduce.layer_norm
// CHECK-NOT:   mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_wrong_concat_order<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 4], f32, gpu:0>,
  %gamma:             !mo.tensor<[4], f32, gpu:0>,
  %beta:              !mo.tensor<[4], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %nope_start_2d  = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 4], f32, gpu:0> -> !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = true, rope_first = false} :
    (!mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0> -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  // Wrong order: (nope, roped) instead of (roped, nope).
  %result = mo.concat[-1](%nope_half, %roped_sq) :
    (!mo.tensor<[total_seq_len, 2], f32, gpu:0>, !mo.tensor<[total_seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 4], f32, gpu:0>
}

// Roped slice does not start at 0 (starts at 1 instead): pattern should not
// fire even though the widths still happen to sum correctly.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_roped_not_prefix
// CHECK:       mo.reduce.layer_norm
// CHECK-NOT:   mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_roped_not_prefix<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 4], f32, gpu:0>,
  %gamma:             !mo.tensor<[4], f32, gpu:0>,
  %beta:              !mo.tensor<[4], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  // Roped slice starts at column 1 instead of 0.
  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 3> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %nope_start_2d  = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 4], f32, gpu:0> -> !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = true, rope_first = false} :
    (!mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0> -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %result = mo.concat[-1](%roped_sq, %nope_half) :
    (!mo.tensor<[total_seq_len, 2], f32, gpu:0>, !mo.tensor<[total_seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 4], f32, gpu:0>
}

// Passthrough slice leaves a gap (starts at column 3 instead of 2, where the
// roped slice's width ends): pattern should not fire.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_gap_between_slices
// CHECK:       mo.reduce.layer_norm
// CHECK-NOT:   mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_gap_between_slices<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 6], f32, gpu:0>,
  %gamma:             !mo.tensor<[6], f32, gpu:0>,
  %beta:              !mo.tensor<[6], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 5], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  // Gap: starts at column 3, leaving column 2 uncovered by either slice.
  %nope_start_2d  = mo.constant <#M.dense_array<0, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 6], f32, gpu:0>, !mo.tensor<[6], f32, gpu:0>,
     !mo.tensor<[6], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 6], f32, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 6], f32, gpu:0> -> !mo.tensor<[total_seq_len, 1, 6], f32, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 6], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 6], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 3], f32, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = true, rope_first = false} :
    (!mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0> -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %result = mo.concat[-1](%roped_sq, %nope_half) :
    (!mo.tensor<[total_seq_len, 2], f32, gpu:0>, !mo.tensor<[total_seq_len, 3], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 5], f32, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 5], f32, gpu:0>
}

// Neither branch feeds a rope.ragged op: pattern should not fire.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_no_rope_consumer
// CHECK:       mo.reduce.layer_norm
// CHECK-NOT:   mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_no_rope_consumer(
  %input: !mo.tensor<[4, 4], f32, gpu:0>,
  %gamma: !mo.tensor<[4], f32, gpu:0>,
  %beta:  !mo.tensor<[4], f32, gpu:0>
) -> !mo.tensor<[4, 4], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>
  %start_2d = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %mid_2d   = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %stop_2d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d  = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[4, 4], f32, gpu:0>

  // Two direct slices -- no unsqueeze, no rope.ragged consumer.
  %x1 = mo.slice(%normed, %start_2d, %mid_2d, %step_2d) :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[4, 2], f32, gpu:0>
  %x2 = mo.slice(%normed, %mid_2d, %stop_2d, %step_2d) :
    (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[4, 2], f32, gpu:0>

  %result = mo.concat[-1](%x1, %x2) :
    (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[4, 2], f32, gpu:0>)
    -> !mo.tensor<[4, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[4, 4], f32, gpu:0>
}

// rope.ragged is non-interleaved: the backing kernel only implements
// interleaved RoPE, so the pattern must leave this unfused.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_non_interleaved
// CHECK:       mo.reduce.layer_norm
// CHECK-NOT:   mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_non_interleaved<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 4], f32, gpu:0>,
  %gamma:             !mo.tensor<[4], f32, gpu:0>,
  %beta:              !mo.tensor<[4], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 2> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %nope_start_2d  = mo.constant <#M.dense_array<0, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[4], f32, gpu:0>,
     !mo.tensor<[4], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 4], f32, gpu:0> -> !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = false, rope_first = false} :
    (!mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 2], f32, gpu:0> -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %result = mo.concat[-1](%roped_sq, %nope_half) :
    (!mo.tensor<[total_seq_len, 2], f32, gpu:0>, !mo.tensor<[total_seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 4], f32, gpu:0>
}

// The RoPE carries `rope_first = true`, so it passes the tail of its own
// input through. The fused kernel derives the roped region from `freqs_cis`
// alone and always rotates the leading columns, so it cannot express this.
// CHECK-LABEL: mo.graph @layer_norm_rope_ragged_rope_first
// CHECK-NOT:   mo.composite.layer_norm_rope_ragged
mo.graph @layer_norm_rope_ragged_rope_first<total_seq_len, batch_plus_one, batch>(
  %input:             !mo.tensor<[total_seq_len, 6], f32, gpu:0>,
  %gamma:             !mo.tensor<[6], f32, gpu:0>,
  %beta:              !mo.tensor<[6], f32, gpu:0>,
  %input_row_offsets: !mo.tensor<[batch_plus_one], ui32, gpu:0>,
  %start_pos:         !mo.tensor<[batch], ui32, gpu:0>,
  %freqs_cis:         !mo.tensor<[1024, 2], f32, gpu:0>
) -> !mo.tensor<[total_seq_len, 6], f32, gpu:0> {
  %epsilon = mo.constant <#M.dense_array<9.99999974E-6> : tensor<f32>> : !mo.tensor<[], f32>

  %roped_start_3d = mo.constant <#M.dense_array<0, 0, 0> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %roped_stop_3d  = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807, 4> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %step_3d        = mo.constant <#M.dense_array<1, 1, 1> : !M.array<3xsi64>> : !mo.tensor<[3], si64>
  %nope_start_2d  = mo.constant <#M.dense_array<0, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %nope_stop_2d   = mo.constant <#M.dense_array<9223372036854775807, 9223372036854775807> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %step_2d        = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %normed = mo.reduce.layer_norm(%input, %gamma, %beta, %epsilon) :
    (!mo.tensor<[total_seq_len, 6], f32, gpu:0>, !mo.tensor<[6], f32, gpu:0>,
     !mo.tensor<[6], f32, gpu:0>, !mo.tensor<[], f32>) -> !mo.tensor<[total_seq_len, 6], f32, gpu:0>

  %normed_unsq = mo.static.reshape(%normed) :
    !mo.tensor<[total_seq_len, 6], f32, gpu:0> -> !mo.tensor<[total_seq_len, 1, 6], f32, gpu:0>
  %roped_half = mo.slice(%normed_unsq, %roped_start_3d, %roped_stop_3d, %step_3d) :
    (!mo.tensor<[total_seq_len, 1, 6], f32, gpu:0>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[3], si64>)
    -> !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>
  %nope_half = mo.slice(%normed, %nope_start_2d, %nope_stop_2d, %step_2d) :
    (!mo.tensor<[total_seq_len, 6], f32, gpu:0>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>)
    -> !mo.tensor<[total_seq_len, 2], f32, gpu:0>

  %roped_out = mo.composite.rope.ragged(%roped_half, %input_row_offsets, %start_pos, %freqs_cis)
    {interleaved = true, rope_first = true} :
    (!mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>, !mo.tensor<[batch_plus_one], ui32, gpu:0>,
     !mo.tensor<[batch], ui32, gpu:0>, !mo.tensor<[1024, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0>
  %roped_sq = mo.static.reshape(%roped_out) :
    !mo.tensor<[total_seq_len, 1, 4], f32, gpu:0> -> !mo.tensor<[total_seq_len, 4], f32, gpu:0>

  %result = mo.concat[-1](%roped_sq, %nope_half) :
    (!mo.tensor<[total_seq_len, 4], f32, gpu:0>, !mo.tensor<[total_seq_len, 2], f32, gpu:0>)
    -> !mo.tensor<[total_seq_len, 6], f32, gpu:0>
  mo.output %result : !mo.tensor<[total_seq_len, 6], f32, gpu:0>
}
