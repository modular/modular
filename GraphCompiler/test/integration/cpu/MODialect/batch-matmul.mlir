// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// TODO: need to infer exclusiveness + shape

mo.graph @batch_matmul() -> (!mo.tensor<[3, 1, 1], f32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0> : tensor<3x1x3xf32>} : () -> !mo.tensor<[3, 1, 3], f32>
  %value = "mo.constant"() {value = #M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0> : tensor<3x3x1xf32>} : () -> !mo.tensor<[3, 3, 1], f32>
  %output = mo.batch_matmul(%identity, %value) : (!mo.tensor<[3, 1, 3], f32>, !mo.tensor<[3, 3, 1], f32>) -> !mo.tensor<[3, 1, 1], f32>
  mo.output %output : !mo.tensor<[3, 1, 1], f32>
}
// CHECK{LITERAL}: 'batch_matmul' returned tensor<3x1x1xf32> [1.000000e+00, 5.000000e+00, 9.000000e+00]

mo.graph @batch_matmul_mixed_precision() -> (!mo.tensor<[3, 1, 1], si32>) {
  %identity = "mo.constant"() {value = #M.dense_array<1, 0, 0, 0, 1, 0, 0, 0, 1> : tensor<3x1x3xui8>} : () -> !mo.tensor<[3, 1, 3], ui8>
  %value = "mo.constant"() {value = #M.dense_array<1, 2, 3, 4, 5, 6, 7, 8, 9> : tensor<3x3x1xsi8>} : () -> !mo.tensor<[3, 3, 1], si8>
  %output = mo.batch_matmul(%identity, %value) : (!mo.tensor<[3, 1, 3], ui8>, !mo.tensor<[3, 3, 1], si8>) -> !mo.tensor<[3, 1, 1], si32>
  mo.output %output : !mo.tensor<[3, 1, 1], si32>
}
// CHECK{LITERAL}: 'batch_matmul_mixed_precision' returned tensor<3x1x1xsi32> [1, 5, 9]

mo.graph @batch_matmul_fallback_shape_func<() -> D0, D1, D2>(
  %lhs: !mo.tensor<[2, 3, 4], f32>,
  %rhs: !mo.tensor<[2, 4, 5], f32>
) -> !mo.tensor<[D0, D1, D2], si32> {

  mo.invoke_shape_func["mo.batch_matmul"]<() -> D3, D4, D5>(%lhs, %rhs)
    [dataDeptTensors = []] :
    (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 4, 5], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D3, D4, D5], si32>

  mo.output<D3, D4, D5> %val : !mo.tensor<[D3, D4, D5], si32>
}
// CHECK: 'batch_matmul_fallback_shape_func' returned tensor<2x3x5xsi32>
