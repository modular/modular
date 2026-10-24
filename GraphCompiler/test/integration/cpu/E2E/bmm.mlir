// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n BatchMatmul -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

mo.graph @serving_default<D0, D1, D2, D3, D4, D5 -> D6, D7, D8>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D6, D7, D8], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
  %0 = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[3], si64>
  %1 = mo.shape_of(%arg1) : (!mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[3], si64>
  %2 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %3 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %4 = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>
  %5 = mo.slice(%0, %2, %3, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %6 = mo.slice(%0, %3, %4, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %7 = mo.slice(%1, %2, %3, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %8 = mo.slice(%1, %3, %4, %3) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %9 = mo.broadcast_shape(%5, %7) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
  %10 = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
  %11 = mo.concat[0] (%9, %6) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %12 = mo.concat[0] (%9, %8) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
  %13 = mo.broadcast_to<() -> D9, D10, D11>(%arg0, %11) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D9, D10, D11], f32>
  %14 = mo.broadcast_to<() -> D12, D13, D14>(%arg1, %12) : (!mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D12, D13, D14], f32>
  %15 = mo.batch_matmul<() -> D15, D16, D17>(%13, %14) : (!mo.tensor<[D9, D10, D11], f32>, !mo.tensor<[D12, D13, D14], f32>) -> !mo.tensor<[D15, D16, D17], f32>
  mo.output<D15, D16, D17> %15 : !mo.tensor<[D15, D16, D17], f32>
}
