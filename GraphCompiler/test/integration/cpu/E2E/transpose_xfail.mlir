// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n BasicTranspose -f %t.mlir | FileCheck %s

// CHECK-NOT: All MO values match numpy
// CHECK: Output at 0 does not match!

mo.graph @serving_default<D0, D1, D2, D3 -> D4, D5, D6, D7>(%arg0: !mo.tensor<[D0, D1, D2, D3], f32>, %arg1: !mo.tensor<[4], si32>) -> !mo.tensor<[D4, D5, D6, D7], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
  %posion_value = mo.constant <#M.dense_array<0.000001> : tensor<f32>> : !mo.tensor<[], f32>
  %sh = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2, D3], f32>) -> !mo.tensor<[4], si64>
  %bcast = mo.broadcast_to(%posion_value, %sh) : (!mo.tensor<[], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[D0, D1, D2, D3], f32>
  %add = mo.add(%arg0, %bcast): !mo.tensor<[D0, D1, D2, D3], f32>

  %0 = mo.transpose<() -> D12, D13, D14, D15>(%add, %arg1) : (!mo.tensor<[D0, D1, D2, D3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[D12, D13, D14, D15], f32>
  mo.output<D12, D13, D14, D15> %0 : !mo.tensor<[D12, D13, D14, D15], f32>
}
