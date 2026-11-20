// RUN: mo-opt %s --mo-shape-inference --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s


// CHECK-LABEL: --- Running 'split_static_output':
// CHECK-NEXT: 'split_static_output' returned tensor<1xsi32> [2]
// CHECK-NEXT: tensor<2xsi32> [3, 5]
// CHECK-NEXT: tensor<3xsi32> [7, 11, 13]
mo.graph @split_static_output() -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>) {
  %input = mo.constant <#M.dense_array<2, 3, 5, 7, 11, 13> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = 0] : (
    !mo.tensor<[6], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>)
  mo.output %out1, %out2, %out3 : !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
}

// CHECK-LABEL: --- Running 'split_negative_axis':
// CHECK-NEXT: 'split_negative_axis' returned tensor<1xsi32> [0]
// CHECK-NEXT: tensor<2xsi32> [1, 1]
// CHECK-NEXT: tensor<3xsi32> [2, 3, 5]
mo.graph @split_negative_axis() -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>) {
  %input = mo.constant <#M.dense_array<0, 1, 1, 2, 3, 5> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = -1] : (
    !mo.tensor<[6], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>)
  mo.output %out1, %out2, %out3 : !mo.tensor<[1], si32>, !mo.tensor<[2], si32>, !mo.tensor<[3], si32>
}

// CHECK-LABEL: --- Running 'split_dyn_output':
// CHECK-NEXT: 'split_dyn_output' returned tensor<1xsi32> [3]
// CHECK-NEXT: tensor<2xsi32> [1, 4]
// CHECK-NEXT: tensor<3xsi32> [1, 5, 9]
mo.graph @split_dyn_output<() -> D0, D1, D2>() -> (!mo.tensor<[D0], si32>, !mo.tensor<[D1], si32>, !mo.tensor<[D2], si32>) {
  %input = mo.constant <#M.dense_array<3, 1, 4, 1, 5, 9> : tensor<6xsi32>> : !mo.tensor<[6], si32>
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split<() -> D3, D4, D5>(%input, %splitSizes) [axis = 0] : (
    !mo.tensor<[6], si32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[D3], si32>, !mo.tensor<[D4], si32>, !mo.tensor<[D5], si32>)
  mo.output<D3, D4, D5> %out1, %out2, %out3 : !mo.tensor<[D3], si32>, !mo.tensor<[D4], si32>, !mo.tensor<[D5], si32>
}

// CHECK-LABEL: --- Running 'split_fallback_shape_func':
// CHECK-NEXT: 'split_fallback_shape_func' returned tensor<10x5x10x15xsi32>
mo.graph @split_fallback_shape_func<() -> D0, D1, D2, D3>(
  %input : !mo.tensor<[10, 20], si32>
) -> !mo.tensor<[D0, D1, D2, D3], si32> {

  %split_sizes = mo.constant <#M.dense_array<5, 15> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %axis = mo.constant <#M.dense_array<-1> : tensor<si32>> : !mo.tensor<[], si32>

  %idx0 = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  mo.invoke_shape_func["split_ith_output_shape"]<() -> D4, D5>(%input, %split_sizes, %axis, %idx0)
    [dataDeptTensors = [1 : i32, 2 : i32]]
    : (!mo.tensor<[10, 20], si32>, !mo.tensor<[2], si64>, !mo.tensor<[], si32>, !mo.tensor<[], si32>)

  %idx1 = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  mo.invoke_shape_func["split_ith_output_shape"]<() -> D6, D7>(%input, %split_sizes, %axis, %idx1)
    [dataDeptTensors = [1 : i32, 2 : i32]]
    : (!mo.tensor<[10, 20], si32>, !mo.tensor<[2], si64>, !mo.tensor<[], si32>, !mo.tensor<[], si64>)

  %one = mo.constant <#M.dense_array<1> : tensor<si32>> : !mo.tensor<[], si32>
  %val = mo.static.broadcast_to(%one): !mo.tensor<[], si32> -> !mo.tensor<[D4, D5, D6, D7], si32>

  mo.output<D4, D5, D6, D7> %val : !mo.tensor<[D4, D5, D6, D7], si32>
}
