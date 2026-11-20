// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t_mojo.mlir
// RUN: mt --execute --input-shapes=4 --input-data=0::ones --result-output-style=full %t_mojo.mlir | FileCheck %s || true

mo.graph @reshape_dyn_input_dyn_output<D0 -> D1, D2>(%input: !mo.tensor<[D0], f32>) -> !mo.tensor<[D1, D2], f32> {
  // TODO: Get a better way to generate a runtime dynamic output shape
  %shape = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.reshape<() -> D3, D4>(%input, %shape) : (!mo.tensor<[D0], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[D3, D4], f32>
  mo.output<D3, D4> %output : !mo.tensor<[D3, D4], f32>
}
// CHECK{LITERAL}: 'reshape_dyn_input_dyn_output' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @reshape_refinement<D0>(%input: !mo.tensor<[D0], f32>) -> (!mo.tensor<[2, 2], f32>) {
  %shape = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = mo.reshape(%input, %shape) : (!mo.tensor<[D0], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 2], f32>
  mo.output %output : !mo.tensor<[2, 2], f32>
}
// CHECK{LITERAL}: 'reshape_refinement' returned tensor<2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @reshape_illegal_refinement<D0>(%input: !mo.tensor<[D0], f32>) -> (!mo.tensor<[2, 4], f32>) {
  %shape = mo.constant <#M.dense_array<2, 4> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %output = rmo.mo.reshape(%input, %shape) : (!mo.tensor<[D0], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[2, 4], f32>
  mo.output %output : !mo.tensor<[2, 4], f32>
}
// CHECK{LITERAL}: 'reshape_illegal_refinement' returned <error: [reshape] input and output number of elements must match: #mosh<ape[D2]> : !mosh.ape != #mosh<ape[2, 4]> : !mosh.ape>
