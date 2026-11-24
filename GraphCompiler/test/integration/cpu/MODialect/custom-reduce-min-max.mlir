// RUN: mo-opt %s --mo-shape-inference --gc-pipeline -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

mo.graph @custom_reduce_inner_min_and_max_1d<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %x = mo.constant <#M.dense_array< -1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 30.0, -20.0, 4.0, 3., 2., -40., 80., 0., 0., 0.> : tensor<2x10xf32>> : !mo.tensor<[2, 10], f32>
  %minmax= mo.reduce.reduce_min_and_max<() -> D2, D3>(%x) [axis = 1] : (!mo.tensor<[2, 10], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %minmax : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: --- Running 'custom_reduce_inner_min_and_max_1d':
// CHECK-NEXT: 'custom_reduce_inner_min_and_max_1d' returned tensor<2x2xf32> [-1.000000e+00, 9.000000e+00, -4.000000e+01, 8.000000e+01]

mo.graph @custom_reduce_inner_min_and_max_2d_variant1_axis1<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %x = mo.constant <#M.dense_array< -1., 1., 2., 3., 4., 5., 6., 7., 8., 9.> : tensor<5x2xf32>> : !mo.tensor<[5, 2], f32>
  %minmax= mo.reduce.reduce_min_and_max<() -> D2, D3>(%x) [axis = 1] : (!mo.tensor<[5, 2], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %minmax : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: --- Running 'custom_reduce_inner_min_and_max_2d_variant1_axis1':
// CHECK-NEXT: 'custom_reduce_inner_min_and_max_2d_variant1_axis1' returned tensor<5x2xf32> [-1.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]


mo.graph @custom_reduce_inner_min_and_max_2d_variant1_axis0<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %x = mo.constant <#M.dense_array< -1., 1., 2., 3., 4., 5., 6., 7., 8., 9.> : tensor<5x2xf32>> : !mo.tensor<[5, 2], f32>
  %minmax= mo.reduce.reduce_min_and_max<() -> D2, D3>(%x) [axis = 0] : (!mo.tensor<[5, 2], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %minmax : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: --- Running 'custom_reduce_inner_min_and_max_2d_variant1_axis0':
// CHECK-NEXT: 'custom_reduce_inner_min_and_max_2d_variant1_axis0' returned tensor<2x2xf32> [-1.000000e+00, 1.000000e+00, 8.000000e+00, 9.000000e+00]


mo.graph @custom_reduce_inner_min_and_max_2d_variant2_axis0<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %x = mo.constant <#M.dense_array< 1., 2., 3., 4., 5., 6., 7., 8., 9.> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %minmax= mo.reduce.reduce_min_and_max<() -> D2, D3>(%x) [axis = 0] : (!mo.tensor<[3, 3], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %minmax : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: --- Running 'custom_reduce_inner_min_and_max_2d_variant2_axis0':
// CHECK-NEXT: 'custom_reduce_inner_min_and_max_2d_variant2_axis0' returned tensor<2x3xf32> [1.000000e+00, 2.000000e+00, 3.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00]


mo.graph @custom_reduce_inner_min_and_max_2d_variant2_axis1<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %x = mo.constant <#M.dense_array< 1., 2., 3., 4., 5., 6., 7., 8., 9.> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %minmax= mo.reduce.reduce_min_and_max<() -> D2, D3>(%x) [axis = 1] : (!mo.tensor<[3, 3], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %minmax : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: --- Running 'custom_reduce_inner_min_and_max_2d_variant2_axis1':
// CHECK-NEXT: 'custom_reduce_inner_min_and_max_2d_variant2_axis1' returned tensor<3x2xf32> [1.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, 7.000000e+00, 9.000000e+00]

mo.graph @custom_reduce_inner_min_and_max_2d_variant2_axis_neg1<() -> D0, D1>() -> !mo.tensor<[D0, D1], f32> {
  %x = mo.constant <#M.dense_array< 1., 2., 3., 4., 5., 6., 7., 8., 9.> : tensor<3x3xf32>> : !mo.tensor<[3, 3], f32>
  %minmax = mo.reduce.reduce_min_and_max<() -> D2, D3>(%x) [axis = -1] : (!mo.tensor<[3, 3], f32>) -> !mo.tensor<[D2, D3], f32>
  mo.output<D2, D3> %minmax : !mo.tensor<[D2, D3], f32>
}
// CHECK-LABEL: --- Running 'custom_reduce_inner_min_and_max_2d_variant2_axis_neg1':
// CHECK-NEXT: 'custom_reduce_inner_min_and_max_2d_variant2_axis_neg1' returned tensor<3x2xf32> [1.000000e+00, 3.000000e+00, 4.000000e+00, 6.000000e+00, 7.000000e+00, 9.000000e+00]
