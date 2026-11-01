// RUN: mo-opt %s --mo-infer-layouts --mo-pattern-fusion | FileCheck %s

// CHECK-LABEL: mo.graph @conv2d_constant_filter
mo.graph @conv2d_constant_filter(
  %input: !mo.tensor<[4, 3, 3, 1], f32>
) -> !mo.tensor<[4, 2, 2, 1], f32> {
  // CHECK-NEXT: %[[FILTER:.*]] = mo.constant
  %filter = mo.constant <#M.dense_array<2.0, 1.0, 3.0, 5.0> : tensor<2x2x1x1xf32>> : !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>

  %strides = mo.constant <#M.dense_array<42, 23> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<-24141, 4352> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %pa = mo.constant <#M.dense_array<1, 2, 3, -16> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK: mo.layout.transform(%[[FILTER]])
  // CHECK-SAME: {layout = #mo.layout<RSCF>}
  // CHECK-SAME: {layout = #mo.layout<FRSCf>}

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 1], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 1], f32>
}

// CHECK-LABEL: mo.graph @conv3d_constant_filter
mo.graph @conv3d_constant_filter(%input: !mo.tensor<[1, 8, 8, 8, 3], f32>) -> !mo.tensor<[1, 9, 9, 9, 4], f32> {
  // CHECK-NEXT: %[[FILTER:.*]] = mo.constant
  %filter  = mo.constant <#M.dense_array<-0.0971, 0.0743, 0.0571, -0.0177, 0.0286, 0.0670, 0.1940, 0.0659, 0.0296, 0.0854, -0.0657, -0.0050, 0.1997, 0.1996, -0.1203, -0.0020, 0.1416, -0.1847, -0.0009, -0.0850, 0.1811, 0.0955, -0.0745, -0.1325, -0.1361, -0.0344, -0.1309, -0.0923, 0.1951, 0.1099, -0.0954, -0.2011, 0.0918, -0.0025, 0.0513, -0.1473, -0.0896, 0.0627, -0.1697, -0.1049, -0.0022, 0.1964, 0.1584, -0.1878, -0.0639, 0.0529, -0.0872, -0.1812, 0.0037, 0.1288, 0.0819, 0.0929, -0.1218, 0.0385, 0.1567, 0.1652, -0.1691, -0.1471, -0.1245, -0.1520, -0.0539, -0.0419, -0.1415, -0.1816, -0.0887, 0.1642, -0.1002, -0.1354, -0.0548, 0.0583, 0.0329, 0.0435, 0.1973, -0.1223, -0.0378, -0.1172, 0.0487, -0.1580, 0.0708, 0.2036, -0.1260, -0.1114, 0.0741, -0.0028, 0.1267, -0.1357, -0.0341, -0.0342, -0.1606, -0.1281, -0.0569, 0.0097, 0.0790, -0.1266, 0.1697, -0.1602 > : tensor<2x2x2x3x4xf32>> : !mo.tensor<[2, 2, 2, 3, 4], f32, {layout = #mo.layout<QRSCF>}>
  %strides = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %dilations = mo.constant <#M.dense_array<1, 1, 1> : tensor<3xsi64>> : !mo.tensor<[3], si64>
  %pa = mo.constant <#M.dense_array<0, 1, 0, 1, 0, 1> : tensor<6xsi64>> : !mo.tensor<[6], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK: mo.layout.transform(%[[FILTER]])
  // CHECK-SAME: {layout = #mo.layout<QRSCF>}
  // CHECK-SAME: {layout = #mo.layout<FQRSCf>}

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 8, 8, 8, 3], f32>, !mo.tensor<[2, 2, 2, 3, 4], f32, {layout = #mo.layout<QRSCF>}>,
    !mo.tensor<[3], si64>, !mo.tensor<[3], si64>, !mo.tensor<[6], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 9, 9, 9, 4], f32>

  mo.output %res : !mo.tensor<[1, 9, 9, 9, 4], f32>
}

// The convolutional filter isn't constant so the pattern shouldn't fire in this case
// CHECK-LABEL: mo.graph @conv2d_dynamic_filter
mo.graph @conv2d_dynamic_filter(
  %input: !mo.tensor<[4, 3, 3, 1], f32>,
  // CHECK: %[[FILTER:.*]]: !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>
  %filter: !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>
) -> !mo.tensor<[4, 2, 2, 1], f32> {
  %strides = mo.constant <#M.dense_array<42, 23> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<-24141, 4352> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %pa = mo.constant <#M.dense_array<1, 2, 3, -16> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK-NOT: mo.layout.transform(%[[FILTER]])

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[4, 3, 3, 1], f32>, !mo.tensor<[2, 2, 1, 1], f32, {layout = #mo.layout<RSCF>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 1], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 1], f32>
}

// The filter isn't constant, but packing is required for num_groups > 1, so
// the pattern should fire despite the filter being dynamic.
// CHECK-LABEL: mo.graph @conv2d_dynamic_filter_grouped
mo.graph @conv2d_dynamic_filter_grouped(
  %input: !mo.tensor<[4, 3, 3, 2], f32>,
  // CHECK: %[[FILTER:[a-zA-Z0-9_]+]]: !mo.tensor<[2, 2, 1, 2], f32, {layout = #mo.layout<RSCF>}>
  %filter: !mo.tensor<[2, 2, 1, 2], f32, {layout = #mo.layout<RSCF>}>
) -> !mo.tensor<[4, 2, 2, 2], f32> {
  %strides = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %dilations = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>

  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<2> : tensor<si64>> : !mo.tensor<[], si64>

  // CHECK: mo.layout.transform(%[[FILTER]])
  // CHECK-SAME: {layout = #mo.layout<RSCF>}
  // CHECK-SAME: {layout = #mo.layout<FRSCf>}

  %res = mo.conv(%input, %filter) [strides = %strides, dilations = %dilations, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[4, 3, 3, 2], f32>, !mo.tensor<[2, 2, 1, 2], f32, {layout = #mo.layout<RSCF>}>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[4, 2, 2, 2], f32>

  mo.output %res : !mo.tensor<[4, 2, 2, 2], f32>
}
