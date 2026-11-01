// RUN: mo-opt %s --gc-pipeline="default-device-label=gpu constant-fold=false" -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// CHECK-LABEL: Running 'split_outer_axis':
// CHECK-NEXT: tensor<1x5xsi32> [0, 1, 2, 3, 4]
// CHECK-NEXT: tensor<2x5xsi32> [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// CHECK-NEXT: tensor<1x5xsi32> [15, 16, 17, 18, 19]
mo.graph @split_outer_axis() -> (!mo.tensor<[1, 5], si32>, !mo.tensor<[2, 5], si32>, !mo.tensor<[1, 5], si32>) {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19> : tensor<20xsi32>> : !mo.tensor<[4, 5], si32, gpu:0>
  %split_shapes = mo.constant <#M.dense_array<1, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>

  %res:3 = mo.split(%input, %split_shapes) [axis = 0] :
    (!mo.tensor<[4, 5], si32, gpu:0>, !mo.tensor<[3], si32>)
    -> (!mo.tensor<[1, 5], si32, gpu:0>, !mo.tensor<[2, 5], si32, gpu:0>, !mo.tensor<[1, 5], si32, gpu:0>)
  %ch0 = mo.chain.create()
  %res0, %ch1 = mo.transfer[%ch0] %res#0 : !mo.tensor<[1, 5], si32, gpu:0> to <"cpu", 0>
  %res1, %ch2 = mo.transfer[%ch1] %res#1 : !mo.tensor<[2, 5], si32, gpu:0> to <"cpu", 0>
  %res2, %ch3 = mo.transfer[%ch2] %res#2 : !mo.tensor<[1, 5], si32, gpu:0> to <"cpu", 0>

  mo.output %res0, %res1, %res2 : !mo.tensor<[1, 5], si32>, !mo.tensor<[2, 5], si32>, !mo.tensor<[1, 5], si32>
}

// CHECK-LABEL: Running 'split_inner_axis':
// CHECK-NEXT: tensor<4x2xsi32> [0, 1, 5, 6, 10, 11, 15, 16]
// CHECK-NEXT: tensor<4x2xsi32> [2, 3, 7, 8, 12, 13, 17, 18]
// CHECK-NEXT: tensor<4x1xsi32> [4, 9, 14, 19]
mo.graph @split_inner_axis() -> (!mo.tensor<[4, 2], si32>, !mo.tensor<[4, 2], si32>, !mo.tensor<[4, 1], si32>) {
  %input = mo.constant <#M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19> : tensor<20xsi32>> : !mo.tensor<[4, 5], si32, gpu:0>
  %split_shapes = mo.constant <#M.dense_array<2, 2, 1> : tensor<3xsi32>> : !mo.tensor<[3], si32>

  %res:3 = mo.split(%input, %split_shapes) [axis = 1] :
    (!mo.tensor<[4, 5], si32, gpu:0>, !mo.tensor<[3], si32>)
    -> (!mo.tensor<[4, 2], si32, gpu:0>, !mo.tensor<[4, 2], si32, gpu:0>, !mo.tensor<[4, 1], si32, gpu:0>)
  %ch0 = mo.chain.create()
  %res0, %ch1 = mo.transfer[%ch0] %res#0 : !mo.tensor<[4, 2], si32, gpu:0> to <"cpu", 0>
  %res1, %ch2 = mo.transfer[%ch1] %res#1 : !mo.tensor<[4, 2], si32, gpu:0> to <"cpu", 0>
  %res2, %ch3 = mo.transfer[%ch2] %res#2 : !mo.tensor<[4, 1], si32, gpu:0> to <"cpu", 0>

  mo.output %res0, %res1, %res2 : !mo.tensor<[4, 2], si32>, !mo.tensor<[4, 2], si32>, !mo.tensor<[4, 1], si32>
}
