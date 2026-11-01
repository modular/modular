// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s

// A reduce.add over a small, static axis decomposes into a sum of per-index
// slices (slice0 + slice1 + ...) -- but ONLY when the result feeds an rms_norm
// (through the lossy f32->bf16->f32 cast round-trip). For any other consumer
// the decompose is speculative (slices + adds with nothing to fuse into), so
// the reduce.add is left as a dedicated kernel.

// -----------------------------------------------------------------------------
// Positive cases: result feeds rms_norm -> decompose.
// -----------------------------------------------------------------------------

// feeding rms_norm directly: 2 slices, no reduce.add. The trailing add and
// rms_norm are themselves fused into the single-norm residual-add boundary
// (mo.composite.rms_norm_residual_add), so no separate mo.add survives.
// CHECK-LABEL: mo.graph @decompose_reduce_add_into_rms_norm
// CHECK-NOT: mo.reduce.add
// CHECK: mo.slice
// CHECK: mo.slice
// CHECK: mo.composite.rms_norm_residual_add
mo.graph @decompose_reduce_add_into_rms_norm(
    %arg0: !mo.tensor<[2, 2, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}

// reduce.add -> cast(bf16) -> cast(f32) -> rms_norm. The
// gate looks through the round-trip casts, so this still decomposes.
// CHECK-LABEL: mo.graph @decompose_reduce_add_roundtrip_rms_norm
// CHECK-NOT: mo.reduce.add
// CHECK: mo.slice
// CHECK: mo.add
// CHECK: mo.reduce.rms_norm
mo.graph @decompose_reduce_add_roundtrip_rms_norm(
    %arg0: !mo.tensor<[2, 2, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %b = mo.cast(%r) : (!mo.tensor<[2, 1, 3], f32>) -> !mo.tensor<[2, 1, 3], bf16>
  %f = mo.cast(%b) : (!mo.tensor<[2, 1, 3], bf16>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%f, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}

// Inclusive upper bound (kMaxDecomposeExtent) -> still decompose. The final
// add in the slice/add tree feeds rms_norm, so it fuses into the composite.
// CHECK-LABEL: mo.graph @decompose_reduce_add_extent8_boundary
// CHECK-NOT: mo.reduce.add
// CHECK: mo.slice
// CHECK: mo.composite.rms_norm_residual_add
mo.graph @decompose_reduce_add_extent8_boundary(
    %arg0: !mo.tensor<[2, 8, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 8, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}

// Reducing axis 0 (an outer axis) into rms_norm still decomposes; the slice
// bounds track whichever axis is reduced. The final add fuses into the
// composite the same way as the extent-8 boundary case above.
// CHECK-LABEL: mo.graph @decompose_reduce_add_axis0
// CHECK-NOT: mo.reduce.add
// CHECK: mo.slice
// CHECK: mo.composite.rms_norm_residual_add
mo.graph @decompose_reduce_add_axis0(
    %arg0: !mo.tensor<[2, 4, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[1, 4, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[1, 4, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[1, 4, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[1, 4, 3], f32>
  mo.output %n : !mo.tensor<[1, 4, 3], f32>
}

// A negative axis attribute (-1) is normalized to the last axis (rank-1 == 2)
// by normalizeAxis at Fusion.cpp, so it decomposes identically to axis = 2 (the
// reduce verifier accepts axis in [-rank, rank-1], so it is expressible in IR).
// CHECK-LABEL: mo.graph @decompose_reduce_add_negative_axis
// CHECK-NOT: mo.reduce.add
// CHECK: mo.slice
// CHECK: mo.slice
// CHECK: mo.composite.rms_norm_residual_add
mo.graph @decompose_reduce_add_negative_axis(
    %arg0: !mo.tensor<[2, 3, 2], f32>,
    %gamma: !mo.tensor<[1], f32>) -> !mo.tensor<[2, 3, 1], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 3, 2], f32>) -> !mo.tensor<[2, 3, 1], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 3, 1], f32>, !mo.tensor<[1], f32>) -> !mo.tensor<[2, 3, 1], f32>
  mo.output %n : !mo.tensor<[2, 3, 1], f32>
}

// Dynamic outer dim with a static reduced axis: only the reduced axis (extent 2)
// must be static, so the decompose fires; the dynamic dim's slices span its full
// (clamped) extent. This is the common dynamic-batch production shape.
// CHECK-LABEL: mo.graph @decompose_dynamic_outer_static_axis
// CHECK-NOT: mo.reduce.add
// CHECK: mo.slice
// CHECK: mo.composite.rms_norm_residual_add
mo.graph @decompose_dynamic_outer_static_axis(
    %arg0: !mo.tensor<[batch, 2, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[batch, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[batch, 2, 3], f32>) -> !mo.tensor<[batch, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[batch, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[batch, 1, 3], f32>
  mo.output %n : !mo.tensor<[batch, 1, 3], f32>
}

// -----------------------------------------------------------------------------
// Negative cases: leave the reduce.add as a dedicated kernel (no slices).
// -----------------------------------------------------------------------------

// No rms_norm consumer (result goes straight to output) -> the new consumer
// gate declines, even though the extent is small and static.
// CHECK-LABEL: mo.graph @no_decompose_no_rms_norm_consumer
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
mo.graph @no_decompose_no_rms_norm_consumer(%arg0: !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %r : !mo.tensor<[2, 1, 3], f32>
}

// A non-rms_norm consumer (elementwise add) -> the gate declines: the casts are
// transparent in the walk, but `mo.add` is a real consumer that is not rms_norm.
// CHECK-LABEL: mo.graph @no_decompose_non_rms_norm_consumer
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
mo.graph @no_decompose_non_rms_norm_consumer(%arg0: !mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %s = mo.add(%r, %r) : !mo.tensor<[2, 1, 3], f32>
  mo.output %s : !mo.tensor<[2, 1, 3], f32>
}

// The result fans out (rms_norm AND output): not a single-use chain, so the gate
// declines rather than duplicate the slice/add work for the non-rms_norm use.
// CHECK-LABEL: mo.graph @no_decompose_multi_use
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
mo.graph @no_decompose_multi_use(
    %arg0: !mo.tensor<[2, 2, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[2, 1, 3], f32>) {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n, %r : !mo.tensor<[2, 1, 3], f32>, !mo.tensor<[2, 1, 3], f32>
}

// A large extent (> kMaxDecomposeExtent = 8) must NOT decompose even when it
// feeds rms_norm; it would bloat the IR / overflow the fusion set.
// CHECK-LABEL: mo.graph @no_decompose_large_extent
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
mo.graph @no_decompose_large_extent(
    %arg0: !mo.tensor<[2, 64, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 64, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}

// The *reduced* axis itself is dynamic, so the unroll count (number of slices)
// is unknown at compile time -> decline. (Dynamic *outer* dims with a static
// reduced axis still decompose; see @decompose_dynamic_outer_static_axis.)
// CHECK-LABEL: mo.graph @no_decompose_dynamic_reduced_axis
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
mo.graph @no_decompose_dynamic_reduced_axis(
    %arg0: !mo.tensor<[4, kb, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[4, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[4, kb, 3], f32>) -> !mo.tensor<[4, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[4, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[4, 1, 3], f32>
  mo.output %n : !mo.tensor<[4, 1, 3], f32>
}

// One past the inclusive upper bound (kMaxDecomposeExtent = 8), so
// even feeding rms_norm it must NOT decompose -- only the extent (9 vs 8) differs
// from the extent-8 positive case, isolating the upper gate.
// CHECK-LABEL: mo.graph @no_decompose_extent9_boundary
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
mo.graph @no_decompose_extent9_boundary(
    %arg0: !mo.tensor<[2, 9, 3], f32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %r = mo.reduce.add(%arg0) : (!mo.tensor<[2, 9, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}

// A gather feeding reduce.add: the decompose pattern's gather guard bails
// (gather + reduce is handled by GatherSumPattern), so no `mo.slice` is
// produced. GatherSumPattern then fuses the pair into mo.gather_sum.
// CHECK-LABEL: mo.graph @no_decompose_gather_input
// CHECK: %[[GS:.*]] = "mo.gather_sum"
// CHECK: %[[N:.*]] = mo.reduce.rms_norm(%[[GS]]
// CHECK-NOT: mo.slice
mo.graph @no_decompose_gather_input(
    %table: !mo.tensor<[5, 3], f32>,
    %indices: !mo.tensor<[2, 4], si32>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %g = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si32>) -> !mo.tensor<[2, 4, 3], f32>
  %r = mo.reduce.add(%g) : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}

// The same gather feeding reduce.add but with int64 indices: GatherSumPattern
// bails on the non-si32 indices, and the decompose pattern's own gather guard
// (not GatherSumPattern firing first) keeps the pair as gather + reduce.add --
// no slices, no gather_sum.
// CHECK-LABEL: mo.graph @no_decompose_gather_input_i64_indices
// CHECK: mo.gather
// CHECK: mo.reduce.add
// CHECK-NOT: mo.slice
// CHECK-NOT: mo.gather_sum
mo.graph @no_decompose_gather_input_i64_indices(
    %table: !mo.tensor<[5, 3], f32>,
    %indices: !mo.tensor<[2, 4], si64>,
    %gamma: !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32> {
  %g = mo.gather(%table, %indices) [axis = 0] : (!mo.tensor<[5, 3], f32>, !mo.tensor<[2, 4], si64>) -> !mo.tensor<[2, 4, 3], f32>
  %r = mo.reduce.add(%g) : (!mo.tensor<[2, 4, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %n = mo.reduce.rms_norm(%r, %gamma) {epsilon = #M.dense_array<1.0e-06> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %n : !mo.tensor<[2, 1, 3], f32>
}
