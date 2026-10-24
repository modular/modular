// RUN: mo-opt %s --mo-pattern-fusion | FileCheck %s


// CHECK-LABEL: mo.graph @pack_matmul_no_cache
// CHECK-SAME: (
// CHECK-SAME:   %[[INPUT1:.*]]: !mo.tensor<[40000, 2], f32>,
// CHECK-SAME:   %[[INPUT2:.*]]: !mo.tensor<[80000, 2], f32>
// CHECK-SAME: ) -> (!mo.tensor<[40000, 3], f32>, !mo.tensor<[80000, 3], f32>) {
mo.graph @pack_matmul_no_cache(
  %input1: !mo.tensor<[40000, 2], f32>,
  %input2: !mo.tensor<[80000, 2], f32>
) -> (!mo.tensor<[40000, 3], f32>, !mo.tensor<[80000, 3], f32>) {

  // CHECK-NEXT: %[[WEIGHTS:.*]] = mo.constant <#M.dense_array<0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>
  %weights = mo.constant <#M.dense_array<0., 1., 2., 3., 4., 5.> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>

  // CHECK-NEXT: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[D0:.*]], [[D1:.*]]>(%0)
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[40000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[40000, 3]> : vector<2xi64>, c_type = f32, transpose_in_0 = false}]
  // CHECK-SAME:   : (!mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>)
  //
  // CHECK-NEXT: %[[PACKED_WEIGHTS_1:.*]] = mo.layout.transform(%[[WEIGHTS]])
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[40000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[40000, 3]> : vector<2xi64>, c_type = f32}]
  // CHECK-SAME:   : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[D0]], [[D1]]], f32, {layout = #mo.layout<KNkni>}>
  //
  // CHECK-NEXT: %[[MM1:.*]] = mo.matmul(%[[INPUT1]], %[[PACKED_WEIGHTS_1]])
  // CHECK-SAME:   : (!mo.tensor<[40000, 2], f32>, !mo.tensor<[[[D0]], [[D1]]], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[40000, 3], f32>
  %mm1 = mo.matmul(%input1, %weights) : (!mo.tensor<[40000, 2], f32>, !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[40000, 3], f32>

  // CHECK-NEXT: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[D2:.*]], [[D3:.*]]>(%0)
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[80000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[80000, 3]> : vector<2xi64>, c_type = f32, transpose_in_0 = false}]
  // CHECK-SAME:   : (!mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>)
  //
  // CHECK-NEXT: %[[PACKED_WEIGHTS_2:.*]] = mo.layout.transform(%[[WEIGHTS]])
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[80000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[80000, 3]> : vector<2xi64>, c_type = f32}]
  // CHECK-SAME:   : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[D2]], [[D3]]], f32, {layout = #mo.layout<KNkni>}>
  //
  // CHECK-NEXT: %[[MM2:.*]] = mo.matmul(%[[INPUT2]], %[[PACKED_WEIGHTS_2]])
  // CHECK-SAME:   : (!mo.tensor<[80000, 2], f32>, !mo.tensor<[[[D2]], [[D3]]], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[80000, 3], f32>
  %mm2 = mo.matmul(%input2, %weights) : (!mo.tensor<[80000, 2], f32>, !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[80000, 3], f32>

  // CHECK-NEXT: mo.output %[[MM1]], %[[MM2]] : !mo.tensor<[40000, 3], f32>, !mo.tensor<[80000, 3], f32>
  mo.output %mm1, %mm2 : !mo.tensor<[40000, 3], f32>, !mo.tensor<[80000, 3], f32>
}


// CHECK-LABEL: mo.graph @pack_matmul_cached
// CHECK-SAME: (
// CHECK-SAME:   %[[INPUT1:.*]]: !mo.tensor<[40000, 2], f32>,
// CHECK-SAME:   %[[INPUT2:.*]]: !mo.tensor<[40000, 2], f32>
// CHECK-SAME: ) -> (!mo.tensor<[40000, 3], f32>, !mo.tensor<[40000, 3], f32>) {
mo.graph @pack_matmul_cached(
  %input1: !mo.tensor<[40000, 2], f32>,
  %input2: !mo.tensor<[40000, 2], f32>
) -> (!mo.tensor<[40000, 3], f32>, !mo.tensor<[40000, 3], f32>) {

  // CHECK-NEXT: %[[WEIGHTS:.*]] = mo.constant <#M.dense_array<0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>
  %weights = mo.constant <#M.dense_array<0., 1., 2., 3., 4., 5.> : tensor<2x3xf32>> : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>

  // CHECK-NEXT: %[[PACKED_WEIGHTS:.*]] = mo.layout.transform(%[[WEIGHTS]])
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[40000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[40000, 3]> : vector<2xi64>, c_type = f32}]
  // CHECK-SAME:   : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[D0:.*]], [[D1:.*]]], f32, {layout = #mo.layout<KNkni>}>
  //
  // CHECK-NEXT: %[[MM1:.*]] = mo.matmul(%[[INPUT1]], %[[PACKED_WEIGHTS]])
  // CHECK-SAME:   : (!mo.tensor<[40000, 2], f32>, !mo.tensor<[[[D0]], [[D1]]], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[40000, 3], f32>
  %mm1 = mo.matmul(%input1, %weights) : (!mo.tensor<[40000, 2], f32>, !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[40000, 3], f32>

  // CHECK-NEXT: mo.invoke_shape_func["pack_matmul_b_shape_func"]<() -> [[D0]], [[D1]]>(%0)
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[40000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[40000, 3]> : vector<2xi64>, c_type = f32, transpose_in_0 = false}]
  // CHECK-SAME:   : (!mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>)
  //
  // CHECK-NEXT: %[[PACKED_WEIGHTS:.*]] = mo.layout.transform(%[[WEIGHTS]])
  // CHECK-SAME:   [kgenParams = {a_shape = dense<[40000, 2]> : vector<2xi64>, a_type = f32, b_shape = dense<[2, 3]> : vector<2xi64>, b_type = f32, c_shape = dense<[40000, 3]> : vector<2xi64>, c_type = f32}]
  // CHECK-SAME:   : !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}> -> !mo.tensor<[[[D0]], [[D1]]], f32, {layout = #mo.layout<KNkni>}>
  //
  // CHECK-NEXT: %[[MM2:.*]] = mo.matmul(%[[INPUT2]], %[[PACKED_WEIGHTS]])
  // CHECK-SAME:   : (!mo.tensor<[40000, 2], f32>, !mo.tensor<[[[D0]], [[D1]]], f32, {layout = #mo.layout<KNkni>}>) -> !mo.tensor<[40000, 3], f32>
  %mm2 = mo.matmul(%input2, %weights) : (!mo.tensor<[40000, 2], f32>, !mo.tensor<[2, 3], f32, {layout = #mo.layout<KN>}>) -> !mo.tensor<[40000, 3], f32>

  // CHECK-NEXT: mo.output %[[MM1]], %[[MM2]] : !mo.tensor<[40000, 3], f32>, !mo.tensor<[40000, 3], f32>
  mo.output %mm1, %mm2 : !mo.tensor<[40000, 3], f32>, !mo.tensor<[40000, 3], f32>
}
