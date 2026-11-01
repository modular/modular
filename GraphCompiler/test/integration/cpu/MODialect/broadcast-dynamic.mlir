// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=2x1 --input-data=0::ones --result-output-style=full %t.mlir | FileCheck %s

// TODO(GEX-1573): currently failing
mo.graph @broadcast_multiple_axes<D0, D1 -> D2, D3, D4>(%arg0 : !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3, D4], f32> {
  // TODO: Get a better way to generate a runtime dynamic output shape
  %to = mo.constant <#M.dense_array<2, 2, 2> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %output = mo.broadcast_to<() -> D5, D6, D7>(%arg0, %to) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D5, D6, D7], f32>
  mo.output<D5, D6, D7> %output : !mo.tensor<[D5, D6, D7], f32>
}
// CHECK{LITERAL}: 'broadcast_multiple_axes' returned tensor<2x2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]

mo.graph @broadcast_multiple_axes_int32<D0, D1 -> D2, D3, D4>(%arg0 : !mo.tensor<[D0, D1], f32>) -> !mo.tensor<[D2, D3, D4], f32> {
  // TODO: Get a better way to generate a runtime dynamic output shape
  %to = mo.constant <#M.dense_array<2, 2, 2> : tensor<3xsi32>> : !mo.tensor<[3], si32>
  %output = mo.broadcast_to<() -> D5, D6, D7>(%arg0, %to) : (!mo.tensor<[D0, D1], f32>, !mo.tensor<[3], si32>) -> !mo.tensor<[D5, D6, D7], f32>
  mo.output<D5, D6, D7> %output : !mo.tensor<[D5, D6, D7], f32>
}
// CHECK{LITERAL}: 'broadcast_multiple_axes_int32' returned tensor<2x2x2xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
