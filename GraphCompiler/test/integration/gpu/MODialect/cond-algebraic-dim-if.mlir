// RUN: mo-opt --gc-pipeline %s -o /dev/null

// Regression: an `mo.if` whose result type carries a composite/algebraic dim
// (`mul_no_wrap(batch, seq)`) used to abort in MOGG->MGP lowering
// (getTensorDynamicDimensions) because HoistParameterExpressions skipped
// control-flow ops and never aliased the expression to a single parameter.
// The pipeline must now run to completion (exit 0).

mo.graph @CondRepro<batch, seq>(%arg0: !mo.tensor<[batch, seq, 1536], bf16, gpu:0>) -> !mo.tensor<[batch, seq, 1536], bf16, gpu:0> attributes {_kernel_library_paths = [], argument_names = ["input0"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %1 = rmo.reshape(%arg0) [newShape = #mosh<ape[mul_no_wrap(batch, seq), 6, 1, 256]> : !mosh.ape] : (!mo.tensor<[batch, seq, 1536], bf16, gpu:0>) -> !mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>
  %2 = rmo.shape_to_tensor() [shape = #mosh<ape[batch]> : !mosh.ape] : () -> !mo.tensor<[1], si64>
  %3 = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>
  %4 = rmo.greater(%2, %3) : (!mo.tensor<[1], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1], bool>
  %5 = rmo.mo.not(%4) : (!mo.tensor<[1], bool>) -> !mo.tensor<[1], bool>
  %6 = rmo.reshape(%5) [newShape = #mosh<ape[]> : !mosh.ape] : (!mo.tensor<[1], bool>) -> !mo.tensor<[], bool>
  %7:2 = mo.if(%6 : !mo.tensor<[], bool>) -> !mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>, !mo.chain {
    %10 = mo.constant <#M.dense_array<2.000000e+00> : tensor<bf16>> : !mo.tensor<[], bf16, gpu:0>
    %11 = rmo.mul(%1, %10) : (!mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>, !mo.tensor<[], bf16, gpu:0>) -> !mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>
    mo.yield %11, %0 : !mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>, !mo.chain
  } else {
    %10 = mo.constant <#M.dense_array<4.000000e+00> : tensor<bf16>> : !mo.tensor<[], bf16, gpu:0>
    %11 = rmo.mul(%1, %10) : (!mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>, !mo.tensor<[], bf16, gpu:0>) -> !mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>
    mo.yield %11, %0 : !mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>, !mo.chain
  }
  %8 = rmo.reshape(%7#0) [newShape = #mosh<ape[batch, seq, 1536]> : !mosh.ape] : (!mo.tensor<[mul_no_wrap(batch, seq), 6, 1, 256], bf16, gpu:0>) -> !mo.tensor<[batch, seq, 1536], bf16, gpu:0>
  %9 = rmo.mul(%8, %arg0) : (!mo.tensor<[batch, seq, 1536], bf16, gpu:0>, !mo.tensor<[batch, seq, 1536], bf16, gpu:0>) -> !mo.tensor<[batch, seq, 1536], bf16, gpu:0>
  mo.output %9 : !mo.tensor<[batch, seq, 1536], bf16, gpu:0>
} {counter = 32 : i64}
