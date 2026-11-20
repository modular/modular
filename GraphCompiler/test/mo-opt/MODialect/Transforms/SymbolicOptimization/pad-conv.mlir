// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s

mo.graph @pad_conv(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  %constant = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 2, 2, 0, 0, 0, 0> : tensor<8xsi64>> : !mo.tensor<[8], si64>
  %pads2 = mo.constant <#M.dense_array<1, 0, 1, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %tmp = mo.pad.constant(%input, %pads, %constant) : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[8], si64>, !mo.tensor<[], f32>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  %res = mo.conv(%tmp, %filter) [strides = %ones, dilations = %ones, paddings = %pads2, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si32>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  mo.output %res : !mo.tensor<[?, ?, ?, ?], f32>
}

// CHECK: mo.graph @pad_conv(%[[ARG0:.*]]: !mo.tensor<[32, 10, 10, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 1, 3, 9], f32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  // CHECK-DAG: %[[VAL1:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[VAL2:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  // CHECK-DAG: %[[VAL3:.*]] = mo.constant <#M.dense_array<2, 2, 0, 0> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[VAL4:.*]] = mo.constant <#M.dense_array<1, 0, 1, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  // CHECK-NEXT: %[[VAL5:.*]] = mo.add(%[[VAL4]], %[[VAL3]]) : !mo.tensor<[4], si64>
  // CHECK-NEXT: %[[VAL6:.*]] = mo.conv(%[[ARG0]], %[[ARG1]]) [strides = %[[VAL1]], dilations = %[[VAL1]], paddings = %[[VAL5]], num_groups = %[[VAL2]], filter_layout = ""] : (!mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[1, 1, 3, 9], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?, ?], f32>
  // CHECK: mo.output %[[VAL6]] : !mo.tensor<[?, ?, ?, ?], f32>
// CHECK: }

mo.graph @pad_channels_conv(
  %input: !mo.tensor<[32, 10, 10, 3], f32>,
  %filter: !mo.tensor<[1, 1, 3, 9], f32>
) -> !mo.tensor<[?, ?, ?, ?], f32> {

  %constant = mo.constant <#M.dense_array<0.0> : tensor<f32>> : !mo.tensor<[], f32>
  %ones = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pads = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0, 1, 1> : tensor<8xsi64>> : !mo.tensor<[8], si64>
  %pads2 = mo.constant <#M.dense_array<1, 0, 1, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %num_groups = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  %tmp = mo.pad.constant(%input, %pads, %constant) : (
    !mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[8], si64>, !mo.tensor<[], f32>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>
  %res = mo.conv(%tmp, %filter) [strides = %ones, dilations = %ones, paddings = %pads2, num_groups = %num_groups, filter_layout = ""] : (
    !mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[1, 1, 3, 9], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si32>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[?, ?, ?, ?], f32>

  mo.output %res : !mo.tensor<[?, ?, ?, ?], f32>
}

//CHECK: mo.graph @pad_channels_conv(%[[ARG0:.*]]: !mo.tensor<[32, 10, 10, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[1, 1, 3, 9], f32>) -> !mo.tensor<[?, ?, ?, ?], f32> {
  //CHECK:   %[[VAL0:.*]] = mo.constant <#M.dense_array<0.000000e+00> : tensor<f32>> : !mo.tensor<[], f32>
  //CHECK:   %[[VAL1:.*]] = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  //CHECK:   %[[VAL2:.*]] = mo.constant <#M.dense_array<0, 0, 0, 0, 0, 0, 1, 1> : tensor<8xsi64>> : !mo.tensor<[8], si64>
  //CHECK:   %[[VAL3:.*]] = mo.constant <#M.dense_array<1, 0, 1, 0> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  //CHECK:   %[[VAL4:.*]] = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  //CHECK:   %[[VAL5:.*]] = mo.pad.constant(%[[ARG0]], %[[VAL2]], %[[VAL0]]) : (!mo.tensor<[32, 10, 10, 3], f32>, !mo.tensor<[8], si64>, !mo.tensor<[], f32>) -> !mo.tensor<[?, ?, ?, ?], f32>
  //CHECK:   %[[VAL6:.*]] = mo.conv(%[[VAL5]], %[[ARG1]]) [strides = %[[VAL1]], dilations = %[[VAL1]], paddings = %[[VAL3]], num_groups = %[[VAL4]], filter_layout = ""] : (!mo.tensor<[?, ?, ?, ?], f32>, !mo.tensor<[1, 1, 3, 9], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, ?, ?], f32>
//CHECK:   mo.output %[[VAL6]] : !mo.tensor<[?, ?, ?, ?], f32>
//CHECK: }
