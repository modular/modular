// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK: 'gather_nd_scalar' returned tensor<si64> [0]
mo.graph @gather_nd_scalar() -> !mo.tensor<[], si64> {
  // input
  // [
  //   [0, 1, 2, 3],
  //   [4, 5, 6, 7]
  // ],
  // [
  //   [8, 9, 10, 11],
  //   [12, 13, 14, 15]
  //  ]
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<2x2x4xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %indices = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>

  %result = mo.gather_nd(%input, %indices) [batchDims = 0] :
    (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) -> !mo.tensor<[], si64>

  mo.output %result : !mo.tensor<[], si64>
}

// CHECK: 'gather_nd_scalars' returned tensor<4xsi64> [0, 1, 6, 15]
mo.graph @gather_nd_scalars() -> !mo.tensor<[4], si64> {
  // input
  // [
  //   [0, 1, 2, 3],
  //   [4, 5, 6, 7]
  // ],
  // [
  //   [8, 9, 10, 11],
  //   [12, 13, 14, 15]
  //  ]
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<2x2x4xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %indices = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 1, 3> : tensor<4x3xsi64>> : !mo.tensor<[4, 3], si64>

  %result = mo.gather_nd(%input, %indices) [batchDims = 0] :
    (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[4, 3], si64>) -> !mo.tensor<[4], si64>

  mo.output %result : !mo.tensor<[4], si64>
}

// CHECK: 'gather_nd_batch' returned tensor<2xsi64> [0, 8]
mo.graph @gather_nd_batch() -> !mo.tensor<[2], si64> {
  // input
  // [
  //   [0, 1, 2, 3],
  //   [4, 5, 6, 7]
  // ],
  // [
  //   [8, 9, 10, 11],
  //   [12, 13, 14, 15]
  //  ]
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> : tensor<2x2x4xsi64>> : !mo.tensor<[2, 2, 4], si64>
  %indices = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<2x2xsi64>> : !mo.tensor<[2, 2], si64>

  %result = mo.gather_nd(%input, %indices) [batchDims = 1] :
    (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[2, 2], si64>) -> !mo.tensor<[2], si64>

  mo.output %result : !mo.tensor<[2], si64>
}

// CHECK: 'gather_nd_batch_row' returned tensor<2x2xsi64> [2, 3, 4, 5]
mo.graph @gather_nd_batch_row() -> !mo.tensor<[2, 2], si64> {
  // input
  // [
  //   [0, 1],
  //   [2, 3]
  // ],
  // [
  //   [4, 5],
  //   [6, 7]
  //  ]
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7> : tensor<2x2x2xsi64>> : !mo.tensor<[2, 2, 2], si64>
  %indices = mo.constant <#M.dense_array<1, 0> : tensor<2x1xsi64>> : !mo.tensor<[2, 1], si64>

  %result = mo.gather_nd(%input, %indices) [batchDims = 1] :
    (!mo.tensor<[2, 2, 2], si64>, !mo.tensor<[2, 1], si64>) -> !mo.tensor<[2, 2], si64>

  mo.output %result : !mo.tensor<[2, 2], si64>
}


mo.graph @gather_nd_fallback_shape_func<() -> D0, D1>(
  %input: !mo.tensor<[2, 3, 4], f32>,
  %indices: !mo.tensor<[2, 1], f32>
) -> !mo.tensor<[D0, D1], si32> {

  %axis = mo.constant <#M.dense_array<-2> : tensor<si32>> : !mo.tensor<[], si32>

  mo.invoke_shape_func["mo.gather_nd"]<() -> D2, D3>(%input, %indices)
    [kgenParams = {batch_dims = 1 : index}, dataDeptTensors = []] :
    (!mo.tensor<[2, 3, 4], f32>, !mo.tensor<[2, 1], f32>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D2, D3], si32>

  mo.output<D2, D3> %val : !mo.tensor<[D2, D3], si32>
}
// CHECK: 'gather_nd_fallback_shape_func' returned tensor<2x4xsi32>
