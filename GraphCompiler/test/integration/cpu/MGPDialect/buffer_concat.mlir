// RUN: mt --execute --model-inputs=ones --result-output-style=full %s | FileCheck %s

// CHECK-LABEL:--- Running 'buffer_concat_test':
// CHECK: 'buffer_concat_test' returned tensor<15xf32> [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00]

mo.graph @buffer_concat_test(%arg0: !mo.tensor<[3], f32>) -> (!mo.tensor<[15], f32>) {
  %0 = mo.add(%arg0, %arg0): !mo.tensor<[3], f32>
  %1 = mo.add(%0, %arg0): !mo.tensor<[3], f32>
  %2 = mo.add(%1, %arg0): !mo.tensor<[3], f32>
  %3 = mo.add(%2, %arg0): !mo.tensor<[3], f32>
  %12 = mogg.concat(%arg0, %0, %1, %2, %3) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> !mo.tensor<[15], f32>
  mo.output %12 : !mo.tensor<[15], f32>
}
