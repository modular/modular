// RUN: mo-opt %s --gc-pipeline=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Test variadic tensor lowering for mo.split and mo.concat operations
// This ensures mogg.as_tuple operations are properly emitted in Mojo

// TODO(GEX-2558): Add a test with a kernel that has FusedOutputVariadicTensors
// when we can used testing_kernels.mojo under the emit mojo lowering path.

// CHECK-LABEL: --- Running 'split_1':
// CHECK: 'split_1' returned tensor<1xf32> [2.000000e+00]
// CHECK-NEXT: , tensor<2xf32> [3.000000e+00, 5.000000e+00]
// CHECK-NEXT: , tensor<3xf32> [7.000000e+00, 1.100000e+01, 1.300000e+01]
mo.graph @split_1() -> (!mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>) {
  %input = mo.constant <#M.dense_array<2.0, 3.0, 5.0, 7.0, 11.0, 13.0> : tensor<6xf32>> : !mo.tensor<[6], f32>
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %out1, %out2, %out3 = mo.split(%input, %splitSizes) [axis = 0] : (
    !mo.tensor<[6], f32>, !mo.tensor<[3], si64>
  ) -> (!mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>)
  mo.output %out1, %out2, %out3 : !mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>
}

// CHECK-LABEL: --- Running 'concat_1':
// CHECK: 'concat_1' returned tensor<6xf32> [2.000000e+00, 3.000000e+00, 5.000000e+00, 7.000000e+00, 1.100000e+01, 1.300000e+01]
mo.graph @concat_1() -> (!mo.tensor<[6], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %arg1 = mo.constant <#M.dense_array<2.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %arg2 = mo.constant <#M.dense_array<3.0, 5.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %arg3 = mo.constant <#M.dense_array<7.0, 11.0, 13.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %out = mo.concat[0](%arg1, %arg2, %arg3) : (
    !mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>
  ) -> !mo.tensor<[6], f32>
  mo.output %out : !mo.tensor<[6], f32>
}

// CHECK-LABEL: --- Running 'split_2':
// CHECK: 'split_2' returned tensor<1xf32> [1.000000e+00]
// CHECK-NEXT: tensor<2xf32> [2.000000e+00, 3.000000e+00]
// CHECK-NEXT: tensor<3xf32> [4.000000e+00, 5.000000e+00, 6.000000e+00]
// CHECK-NEXT: tensor<4xf32> [7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]
mo.graph @split_2() -> (!mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[4], f32>) {
  %input = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0> : tensor<10xf32>> : !mo.tensor<[10], f32>
  %splitSizes = mo.constant <#M.dense_array<1, 2, 3, 4> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %out1, %out2, %out3, %out4 = mo.split(%input, %splitSizes) [axis = 0] : (
    !mo.tensor<[10], f32>, !mo.tensor<[4], si64>
  ) -> (!mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[4], f32>)
  mo.output %out1, %out2, %out3, %out4 : !mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[4], f32>
}

// CHECK-LABEL: --- Running 'concat_2':
// CHECK: 'concat_2' returned tensor<10xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01]
mo.graph @concat_2() -> (!mo.tensor<[10], f32>) {
  %ax = mo.constant <#M.dense_array<0> : tensor<si32>> : !mo.tensor<[], si32>
  %arg1 = mo.constant <#M.dense_array<1.0> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %arg2 = mo.constant <#M.dense_array<2.0, 3.0> : tensor<2xf32>> : !mo.tensor<[2], f32>
  %arg3 = mo.constant <#M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>> : !mo.tensor<[3], f32>
  %arg4 = mo.constant <#M.dense_array<7.0, 8.0, 9.0, 10.0> : tensor<4xf32>> : !mo.tensor<[4], f32>
  %out = mo.concat[0](%arg1, %arg2, %arg3, %arg4) : (
    !mo.tensor<[1], f32>, !mo.tensor<[2], f32>, !mo.tensor<[3], f32>, !mo.tensor<[4], f32>
  ) -> !mo.tensor<[10], f32>
  mo.output %out : !mo.tensor<[10], f32>
}
