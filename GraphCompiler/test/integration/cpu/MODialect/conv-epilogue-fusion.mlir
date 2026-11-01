// RUN: mo-opt %s --gc-pipeline="constant-fold=false" -o %t.mlir
// RUN: mt --execute --model-inputs=ones --result-output-style=full %t.mlir | FileCheck %s

mo.graph @conv_test0(%input : !mo.tensor<[1, 5, 5, 2], f32>, %filter : !mo.tensor<[3,3,2,19], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[1, 3 ,3, 19], f32>) {
  %st = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %output = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 5, 5, 2], f32>, !mo.tensor<[3, 3, 2, 19], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 3, 19], f32>
  %cst0 = mo.constant <#M.dense_array<-19.0,-18.0,-17.0,-16.0,-15.0,-14.0,-13.0,-12.0,-11.0,-10.0,-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0> : tensor<19xf32>> : !mo.tensor<[19], f32>
  %bcast = mo.static.broadcast_to(%cst0) : !mo.tensor<[19], f32> -> !mo.tensor<[1, 3, 3, 19], f32>
  %add = mo.add(%output, %bcast) : !mo.tensor<[1, 3, 3, 19], f32>
  %relu = mo.relu(%add) : !mo.tensor<[1, 3, 3, 19], f32>
  mo.output %relu : !mo.tensor<[1, 3, 3, 19], f32>
}

// CHECK: 'conv_test0' returned tensor<1x3x3x19xf32> [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01]

mo.graph @conv_test1(%input : !mo.tensor<[1, 8, 7, 2], f32>, %filter : !mo.tensor<[4,3,2,5], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[1, 3 ,2, 5], f32>) {
  %st = mo.constant <#M.dense_array<2, 3> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %output = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[1, 8, 7, 2], f32>, !mo.tensor<[4, 3, 2, 5], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[1, 3, 2, 5], f32>
  %cst0 = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %bcast0 = mo.static.broadcast_to(%cst0) : !mo.tensor<[1], f32> -> !mo.tensor<[1, 3, 2, 5], f32>
  %mul = mo.mul(%output, %bcast0) : !mo.tensor<[1, 3, 2, 5], f32>
  %cst1 = mo.constant <#M.dense_array<-15.0,-14.0,-13.0,-12.0,-11.0,-10.0,-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0> : tensor<1x3x5x2xf32>> : !mo.tensor<[1, 3, 5, 2], f32>
  %perm = mo.constant <#M.dense_array<0,1,3,2> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %trans = mo.transpose(%cst1, %perm) : (!mo.tensor<[1, 3, 5, 2], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[1, 3, 2, 5], f32>
  %add = mo.add(%mul, %trans) : !mo.tensor<[1, 3, 2, 5], f32>
  mo.output %add : !mo.tensor<[1, 3, 2, 5], f32>
}

// CHECK: 'conv_test1' returned tensor<1x3x2x5xf32> [-3.000000e+00, -1.000000e+00, 1.000000e+00, 3.000000e+00, 5.000000e+00, -2.000000e+00, 0.000000e+00, 2.000000e+00, 4.000000e+00, 6.000000e+00, 7.000000e+00, 9.000000e+00, 1.100000e+01, 1.300000e+01, 1.500000e+01, 8.000000e+00, 1.000000e+01, 1.200000e+01, 1.400000e+01, 1.600000e+01, 1.700000e+01, 1.900000e+01, 2.100000e+01, 2.300000e+01, 2.500000e+01, 1.800000e+01, 2.000000e+01, 2.200000e+01, 2.400000e+01, 2.600000e+01]

mo.graph @conv_test2(%input : !mo.tensor<[3, 5, 8, 2], f32>, %filter : !mo.tensor<[2,5,2,3], f32, {layout = #mo.layout<RSCF>}>) -> (!mo.tensor<[3, 2 ,2, 3], f32>) {
  %st = mo.constant <#M.dense_array<3, 2> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %di = mo.constant <#M.dense_array<1, 1> : tensor<2xsi64>> : !mo.tensor<[2], si64>
  %pa = mo.constant <#M.dense_array<0, 0, 0, 0> : tensor<4xsi64>> : !mo.tensor<[4], si64>
  %ng = mo.constant <#M.dense_array<1> : tensor<si64>> : !mo.tensor<[], si64>
  %output = mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng, filter_layout = ""] : (
    !mo.tensor<[3, 5, 8, 2], f32>, !mo.tensor<[2, 5, 2, 3], f32, {layout = #mo.layout<RSCF>}>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[3, 2, 2, 3], f32>
  %cst0 = mo.constant <#M.dense_array<0.5> : tensor<1xf32>> : !mo.tensor<[1], f32>
  %bcast0 = mo.static.broadcast_to(%cst0) : !mo.tensor<[1], f32> -> !mo.tensor<[3, 2, 2, 3], f32>
  %mul = mo.mul(%output, %bcast0) : !mo.tensor<[3, 2, 2, 3], f32>
  %cst1 = mo.constant <#M.dense_array<-18.0, -17.0, -16.0, -15.0,-14.0,-13.0,-12.0,-11.0,-10.0,-9.0,-8.0,-7.0,-6.0,-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0> : tensor<2x3x2x3xf32>> : !mo.tensor<[2, 3, 2, 3], f32>
  %perm = mo.constant <#M.dense_array<1,0,2,3> : tensor<4xsi32>> : !mo.tensor<[4], si32>
  %trans = mo.transpose(%cst1, %perm) : (!mo.tensor<[2, 3, 2, 3], f32>, !mo.tensor<[4], si32>) -> !mo.tensor<[3, 2, 2, 3], f32>
  %add = mo.add(%mul, %trans) : !mo.tensor<[3, 2, 2, 3], f32>
  %relu = mo.relu(%add) : !mo.tensor<[3, 2, 2, 3], f32>
  mo.output %relu : !mo.tensor<[3, 2, 2, 3], f32>
}

// CHECK: 'conv_test2' returned tensor<3x2x2x3xf32> [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01]
