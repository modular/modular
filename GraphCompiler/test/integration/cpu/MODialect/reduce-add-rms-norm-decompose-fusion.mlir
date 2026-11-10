// RUN: mo-opt %s --gc-pipeline=constant-fold=false | FileCheck %s --check-prefix=CHECK-IR
// RUN: mt --execute --result-output-style=full %s | FileCheck %s --check-prefix=CHECK-MT

// End-to-end checks for the reduce.add -> rms_norm decompose+fusion: the lowered
// IR (CHECK-IR) shows whether the reduce folded into the rms_norm kernel, and
// `mt --execute` (CHECK-MT) verifies the numerics. Expected values were
// cross-checked against a numpy reference that applies the f32->bf16->f32
// round-trip (ml_dtypes.bfloat16), then matched to the kernel output.

// Case 1 -- the representative fused structure: reduce.add over a small static
// axis, a lossy f32->bf16->f32 round-trip, then rms_norm over the last axis. The
// DecomposeSmallReduceAddPattern turns the reduce.add into slice + slice + add,
// and rms_norm decomposes; the slices, add, and both casts fuse into the
// rowwise kernel's fold -- one kernel, no materialized reduce intermediate.
mo.graph @reduce_add_rms_norm_decompose() -> !mo.tensor<[2, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0> : tensor<2x2x3xf32>> : !mo.tensor<[2, 2, 3], f32>
  %gamma  = mo.constant <#M.dense_array<1.0, 1.0, 1.0> : tensor<3xf32>> : !mo.tensor<[3], f32>

  %reduced = mo.reduce.add(%input) : (!mo.tensor<[2, 2, 3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  %reduced_bf16 = mo.cast(%reduced) : (!mo.tensor<[2, 1, 3], f32>) -> !mo.tensor<[2, 1, 3], bf16>
  %reduced_f32 = mo.cast(%reduced_bf16) : (!mo.tensor<[2, 1, 3], bf16>) -> !mo.tensor<[2, 1, 3], f32>
  %normed = mo.composite.reduce.rms_norm(%reduced_f32, %gamma) {epsilon = #M.dense_array<0.00001> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[2, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[2, 1, 3], f32>
  mo.output %normed : !mo.tensor<[2, 1, 3], f32>
}

// The slices + add + casts fuse into the rowwise kernel -> a single kernel.
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Rowwise({{.*}}View(mo.slice, mo.slice, Elementwise(mo.add, mo.cast, mo.cast)), mo.mul, mo.reduce.add
// reduced = [[5,7,9],[17,19,21]], round-tripped through bf16, then rms_norm.
// CHECK-MT: 'reduce_add_rms_norm_decompose' returned tensor<2x1x3xf32> [6.9560{{[0-9]*}}e-01, 9.7385{{[0-9]*}}e-01, 1.2520{{[0-9]*}}e+00, 8.9145{{[0-9]*}}e-01, 9.9632{{[0-9]*}}e-01, 1.1012{{[0-9]*}}e+00]

// Case 2 -- extent 8 (the inclusive upper bound, kMaxDecomposeExtent): still
// decomposes and fuses to a single kernel (8 slices in the rowwise kernel).
mo.graph @reduce_add_extent8_boundary() -> !mo.tensor<[1, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0> : tensor<1x8x3xf32>> : !mo.tensor<[1, 8, 3], f32>
  %gamma  = mo.constant <#M.dense_array<1.0, 1.0, 1.0> : tensor<3xf32>> : !mo.tensor<[3], f32>

  %reduced = mo.reduce.add(%input) : (!mo.tensor<[1, 8, 3], f32>) -> !mo.tensor<[1, 1, 3], f32>
  %reduced_bf16 = mo.cast(%reduced) : (!mo.tensor<[1, 1, 3], f32>) -> !mo.tensor<[1, 1, 3], bf16>
  %reduced_f32 = mo.cast(%reduced_bf16) : (!mo.tensor<[1, 1, 3], bf16>) -> !mo.tensor<[1, 1, 3], f32>
  %normed = mo.composite.reduce.rms_norm(%reduced_f32, %gamma) {epsilon = #M.dense_array<0.00001> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[1, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[1, 1, 3], f32>
  mo.output %normed : !mo.tensor<[1, 1, 3], f32>
}

// 8 slices fuse into the rowwise kernel -> still a single fused kernel.
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Rowwise({{.*}}View(mo.slice, mo.slice, mo.slice, mo.slice, mo.slice, mo.slice, mo.slice, mo.slice, Elementwise({{.*}}mo.reduce.add
// reduced = [92,100,108], round-tripped through bf16, then rms_norm.
// CHECK-MT: 'reduce_add_extent8_boundary' returned tensor<1x1x3xf32> [9.1804{{[0-9]*}}e-01, 9.9787{{[0-9]*}}e-01, 1.0777{{[0-9]*}}e+00]

// Case 3 -- extent 9 (one past the bound): the decompose declines, so the
// reduce.add stays its own rowwise kernel, which takes the casts and the square
// as per-row consumers; a second rowwise kernel normalizes. Proves the gate's
// fallback lowers + runs.
mo.graph @reduce_add_extent9_fallback() -> !mo.tensor<[1, 1, 3], f32> {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0> : tensor<1x9x3xf32>> : !mo.tensor<[1, 9, 3], f32>
  %gamma  = mo.constant <#M.dense_array<1.0, 1.0, 1.0> : tensor<3xf32>> : !mo.tensor<[3], f32>

  %reduced = mo.reduce.add(%input) : (!mo.tensor<[1, 9, 3], f32>) -> !mo.tensor<[1, 1, 3], f32>
  %reduced_bf16 = mo.cast(%reduced) : (!mo.tensor<[1, 1, 3], f32>) -> !mo.tensor<[1, 1, 3], bf16>
  %reduced_f32 = mo.cast(%reduced_bf16) : (!mo.tensor<[1, 1, 3], bf16>) -> !mo.tensor<[1, 1, 3], f32>
  %normed = mo.composite.reduce.rms_norm(%reduced_f32, %gamma) {epsilon = #M.dense_array<0.00001> : tensor<f32>, weight_offset = #M.dense_array<0.0> : tensor<f32>, multiply_before_cast = false} : (!mo.tensor<[1, 1, 3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[1, 1, 3], f32>
  mo.output %normed : !mo.tensor<[1, 1, 3], f32>
}

// The reduce.add kernel survives, then a separate rms_norm kernel, no slices.
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Rowwise(mo.reduce.add, Elementwise(mo.cast, mo.cast), mo.mul)"
// CHECK-IR: "mgp.generic.execute"{{.*}}dSummary = "Rowwise({{.*}}mo.reduce.add{{.*}}Elementwise(mo.mul, mo.mul)))"
// reduced = [117,126,135], round-tripped through bf16, then rms_norm.
// CHECK-MT: 'reduce_add_extent9_fallback' returned tensor<1x1x3xf32> [9.2699{{[0-9]*}}e-01, 9.9830{{[0-9]*}}e-01, 1.0696{{[0-9]*}}e+00]
