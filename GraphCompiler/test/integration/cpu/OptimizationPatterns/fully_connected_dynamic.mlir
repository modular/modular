// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --input-shapes=4x2 --input-shapes=2x2 --input-shapes=2 -model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s


mo.graph @fullyConnected<M, K, N>(%arg0: !mo.tensor<[M, K], f32>, %arg1: !mo.tensor<[K, N], f32>, %arg2: !mo.tensor<[N], f32>) -> !mo.tensor<[M, N], f32> {
  %0 = mo.matmul(%arg0, %arg1) : (!mo.tensor<[M, K], f32>, !mo.tensor<[K, N], f32>) -> !mo.tensor<[M, N], f32>
  %1 = mo.shape_of(%0) : (!mo.tensor<[M, N], f32>) -> !mo.tensor<[2], si64>
  %2 = mo.shape_of(%arg2) : (!mo.tensor<[N], f32>) -> !mo.tensor<[1], si64>
  %3 = mo.broadcast_shape(%1, %2) : (!mo.tensor<[2], si64>, !mo.tensor<[1], si64>) -> !mo.tensor<[2], si64>
  %4 = mo.broadcast_to(%0, %3) : (!mo.tensor<[M, N], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[M, N], f32>
  %5 = mo.broadcast_to(%arg2, %3) : (!mo.tensor<[N], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[M, N], f32>
  %output = mo.add(%4, %5) : !mo.tensor<[M, N], f32>
  mo.output %output : !mo.tensor<[M, N], f32>
}
// CHECK{LITERAL}: 'fullyConnected' returned tensor<4x2xf32> [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00]
