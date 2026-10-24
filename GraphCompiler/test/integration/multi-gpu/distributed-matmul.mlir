// GEX-2595
// UNSUPPORTED: AMD-GPU
// RUN: mt --execute --default-device-label=gpu --model-inputs=ones --num-runs=1 --result-output-style=compact %s | FileCheck %s

// CHECK-LABEL: --- Running 'col_parallel_matmul':
// CHECK: 'col_parallel_matmul' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000],
// CHECK-NEXT: [2.0000, 2.0000, 2.0000, 2.0000],
// CHECK-NEXT: [2.0000, 2.0000, 2.0000, 2.0000],
// CHECK-NEXT: [2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4,4])
mo.graph @col_parallel_matmul(%a: !mo.tensor<[4,4], f32>) -> (!mo.tensor<[4, 4], f32>) {
  %identityWeights = "mo.constant"() {value = #M.dense_array<2.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00> : tensor<4x4xf32>} : () -> !mo.tensor<[4, 4], f32, gpu:0>
  %splitSizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %weightsSplit:2 = mo.split(%identityWeights, %splitSizes) [axis = 1] : (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[4, 2], f32, gpu:0>)
  %ch0 = mo.chain.create()
  %weights0, %ch1 = mo.transfer[%ch0] %weightsSplit#0: !mo.tensor<[4, 2], f32, gpu:0> to <"gpu", 0>
  %weights1, %ch2 = mo.transfer[%ch1] %weightsSplit#1: !mo.tensor<[4, 2], f32, gpu:0> to <"gpu", 1>
  %da0, %ch3 = mo.transfer[%ch2] %a: !mo.tensor<[4, 4], f32> to <"gpu", 0>
  %da1, %ch4 = mo.transfer[%ch3] %a: !mo.tensor<[4, 4], f32> to <"gpu", 1>
  %outputLeft = mo.matmul(%da0, %weights0) : (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[4, 2], f32, gpu:0>) -> !mo.tensor<[4, 2], f32, gpu:0>
  %outputRight = mo.matmul(%da1, %weights1) : (!mo.tensor<[4, 4], f32, gpu:1>, !mo.tensor<[4, 2], f32, gpu:1>) -> !mo.tensor<[4, 2], f32, gpu:1>
  %outputRight0, %ch5 = mo.transfer[%ch4] %outputRight: !mo.tensor<[4, 2], f32, gpu:1> to <"gpu", 0>
  %doutput = mo.concat[1](%outputLeft, %outputRight0) : (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[4, 2], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %output, %ch6 = mo.transfer[%ch5] %doutput : !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %output : !mo.tensor<[4, 4], f32>
}

// CHECK-LABEL: --- Running 'row_parallel_matmul':
// CHECK: 'row_parallel_matmul' returned tensor({{\[}}[2.0000, 2.0000, 2.0000, 2.0000],
// CHECK-NEXT: [2.0000, 2.0000, 2.0000, 2.0000],
// CHECK-NEXT: [2.0000, 2.0000, 2.0000, 2.0000],
// CHECK-NEXT: [2.0000, 2.0000, 2.0000, 2.0000]], dtype=f32, shape=[4,4])
mo.graph @row_parallel_matmul(%a: !mo.tensor<[4,4], f32>) -> (!mo.tensor<[4, 4], f32>) {
  %identityWeights = "mo.constant"() {value = #M.dense_array<2.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00> : tensor<4x4xf32>} : () -> !mo.tensor<[4, 4], f32, gpu:0>
  %splitSizes = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %weightsSplit:2 = mo.split(%identityWeights, %splitSizes) [axis = 0] : (!mo.tensor<[4, 4], f32, gpu:0>, !mo.tensor<[2], si64>) -> (!mo.tensor<[2, 4], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>)
  %ch0 = mo.chain.create()
  %weights0, %ch1 = mo.transfer[%ch0] %weightsSplit#0: !mo.tensor<[2, 4], f32, gpu:0> to <"gpu", 0>
  %weights1, %ch2 = mo.transfer[%ch1] %weightsSplit#1: !mo.tensor<[2, 4], f32, gpu:0> to <"gpu", 1>
  %aSplit:2 = mo.split(%a, %splitSizes) [axis = 1] : (!mo.tensor<[4, 4], f32>, !mo.tensor<[2], si64>) -> (!mo.tensor<[4, 2], f32>, !mo.tensor<[4, 2], f32>)
  %da0, %ch3 = mo.transfer[%ch2] %aSplit#0: !mo.tensor<[4, 2], f32> to <"gpu", 0>
  %da1, %ch4 = mo.transfer[%ch3] %aSplit#1: !mo.tensor<[4, 2], f32> to <"gpu", 1>
  %outputPartial0 = mo.matmul(%da0, %weights0) : (!mo.tensor<[4, 2], f32, gpu:0>, !mo.tensor<[2, 4], f32, gpu:0>) -> !mo.tensor<[4, 4], f32, gpu:0>
  %outputPartial1 = mo.matmul(%da1, %weights1) : (!mo.tensor<[4, 2], f32, gpu:1>, !mo.tensor<[2, 4], f32, gpu:1>) -> !mo.tensor<[4, 4], f32, gpu:1>
  %outputPartial1On0, %ch5 = mo.transfer[%ch4] %outputPartial1: !mo.tensor<[4, 4], f32, gpu:1> to <"gpu", 0>
  %doutput = mo.add(%outputPartial0, %outputPartial1On0) : !mo.tensor<[4, 4], f32, gpu:0>
  %output, %ch6 = mo.transfer[%ch5] %doutput: !mo.tensor<[4, 4], f32, gpu:0> to <"cpu", 0>
  mo.output %output : !mo.tensor<[4, 4], f32>
}
