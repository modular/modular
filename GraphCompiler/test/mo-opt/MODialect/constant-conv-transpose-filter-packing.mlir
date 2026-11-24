// RUN: mo-opt %s --mo-infer-layouts --mo-pattern-fusion | FileCheck %s

// CHECK-LABEL: mo.graph @conv_transpose2d_constant_filter
mo.graph @conv_transpose2d_constant_filter(%input: !mo.tensor<[1, 4, 4, 2], f32>) -> !mo.tensor<[1, 6, 6, 2], f32> {
  // CHECK: %[[FILTER:.*]] = mo.constant
  %filter  = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0> : tensor<2x2x2x2xf32>> : !mo.tensor<[2, 2, 2, 2], f32>
  %strides = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %opa = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK: mo.layout.transform(%[[FILTER]])
  // CHECK-SAME: {layout = #mo.layout<RSFC>}
  // CHECK-SAME: {layout = #mo.layout<FRSCf>}

  %res = mo.conv_transpose(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, output_paddings = %opa, filter_layout = ""] : (
    !mo.tensor<[1, 4, 4, 2], f32>, !mo.tensor<[2, 2, 2, 2], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 6, 6, 2], f32>

  mo.output %res : !mo.tensor<[1, 6, 6, 2], f32>
}

// CHECK-LABEL: mo.graph @conv_transpose3d_constant_filter
mo.graph @conv_transpose3d_constant_filter(%input: !mo.tensor<[1, 4, 4, 4, 2], f32>) -> !mo.tensor<[1, 6, 6, 6, 2], f32> {
  // CHECK: %[[FILTER:.*]] = mo.constant
  %filter  = mo.constant <#M.dense_array<1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0> : tensor<2x2x2x2x2xf32>> : !mo.tensor<[2, 2, 2, 2, 2], f32>
  %strides = mo.constant <#M.dense_array<2, 2, 2> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %dilations = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %pa = mo.constant <#M.dense_array<1, 1, 1, 1, 1, 1> : tensor<6xsi64>> : !mo.tensor<[6], si64>
  %opa = mo.constant <#M.dense_array<0, 0, 0> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK: mo.layout.transform(%[[FILTER]])
  // CHECK-SAME: {layout = #mo.layout<QRSFC>}
  // CHECK-SAME: {layout = #mo.layout<FQRSCf>}

  %res = mo.conv_transpose(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, output_paddings = %opa, filter_layout = ""] : (
    !mo.tensor<[1, 4, 4, 4, 2], f32>, !mo.tensor<[2, 2, 2, 2, 2], f32>,
    !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[3], si64>
  ) -> !mo.tensor<[1, 6, 6, 6, 2], f32>

  mo.output %res : !mo.tensor<[1, 6, 6, 6, 2], f32>
}

// CHECK-LABEL: mo.graph @conv_transpose2d_dynamic_filter
mo.graph @conv_transpose2d_dynamic_filter(%input: !mo.tensor<[1, 4, 4, 2], f32>, %filter: !mo.tensor<[2, 2, 2, 2], f32>) -> !mo.tensor<[1, 6, 6, 2], f32> {
  // CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[2, 2, 2, 2], f32, {layout = #mo.layout<RSFC>}>
  %strides = mo.constant <#M.dense_array<2, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<1, 1, 1, 1> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %opa = mo.constant <#M.dense_array<0, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK-NOT: mo.layout.transform(%[[FILTER]])

  %res = mo.conv_transpose(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, output_paddings = %opa, filter_layout = ""] : (
    !mo.tensor<[1, 4, 4, 2], f32>, !mo.tensor<[2, 2, 2, 2], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[1, 6, 6, 2], f32>

  mo.output %res : !mo.tensor<[1, 6, 6, 2], f32>
}
