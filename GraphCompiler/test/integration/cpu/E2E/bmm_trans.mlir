// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: python3 %mo_vs_numpy_py -n BatchMatmulTrans -f %t.mlir | FileCheck %s

// CHECK: All MO values match numpy

module {
  mo.graph @serving_default<D0, D1, D2, D3, D4, D5 -> D6, D7, D8>(%arg0: !mo.tensor<[D0, D1, D2], f32>, %arg1: !mo.tensor<[D3, D4, D5], f32>) -> !mo.tensor<[D6, D7, D8], f32> attributes {argument_names = ["input_1", "input_2"], result_names = ["lambda"]} {
    %0 = mo.constant <#M.dense_array<0, 2, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
    %1 = mo.transpose<() -> D9, D10, D11>(%arg1, %0) : (!mo.tensor<[D3, D4, D5], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D9, D10, D11], f32>
    %2 = mo.shape_of(%arg0) : (!mo.tensor<[D0, D1, D2], f32>) -> !mo.tensor<[3], si64>
    %3 = mo.shape_of(%1) : (!mo.tensor<[D9, D10, D11], f32>) -> !mo.tensor<[3], si64>
    %4 = mo.constant <#M.dense_array<0> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %5 = mo.constant <#M.dense_array<1> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %6 = mo.constant <#M.dense_array<3> : tensor<1xsi64>> : !mo.tensor<[1], si64>
    %7 = mo.slice(%2, %4, %5, %5) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
    %8 = mo.slice(%2, %5, %6, %5) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
    %9 = mo.slice(%3, %4, %5, %5) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
    %10 = mo.slice(%3, %5, %6, %5) : (!mo.tensor<[3], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
    %11 = mo.broadcast_shape(%7, %9) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[1], si64>
    %12 = mo.constant <#M.dense_array<0> : tensor<si64>> : !mo.tensor<[], si64>
    %13 = mo.concat[0] (%11, %8) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
    %14 = mo.concat[0] (%11, %10) : (!mo.tensor<[1], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[3], si64>
    %15 = mo.broadcast_to<() -> D12, D13, D14>(%arg0, %13) : (!mo.tensor<[D0, D1, D2], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D12, D13, D14], f32>
    %16 = mo.broadcast_to<() -> D15, D16, D17>(%1, %14) : (!mo.tensor<[D9, D10, D11], f32>, !mo.tensor<[3], si64>) -> !mo.tensor<[D15, D16, D17], f32>
    %17 = mo.batch_matmul<() -> D18, D19, D20>(%15, %16) : (!mo.tensor<[D12, D13, D14], f32>, !mo.tensor<[D15, D16, D17], f32>) -> !mo.tensor<[D18, D19, D20], f32>
    mo.output<D18, D19, D20> %17 : !mo.tensor<[D18, D19, D20], f32>
  }
}
