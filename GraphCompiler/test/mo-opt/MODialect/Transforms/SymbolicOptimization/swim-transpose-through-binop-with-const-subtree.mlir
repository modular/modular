// RUN: mo-opt --mo-symbolic-optimize %s | FileCheck %s

// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_constant_rhs([[VARG0:[^ ]*]]: !mo.tensor<[16, 128, 128, 64], f32>) -> !mo.tensor<[16, 64, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.transpose([[V3]], [[V0]]) : (!mo.tensor<[16, 64, 128, 128], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.sub([[VARG0]], [[V4]]) : !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.transpose([[V5]], [[V1]]) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: mo.output [[V6]] : !mo.tensor<[16, 64, 128, 128], f32>
mo.graph @success_constant_rhs(
  %arg0: !mo.tensor<[16, 128, 128, 64], f32>
) ->  !mo.tensor<[16, 64, 128, 128], f32> {
  %perm = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %cst = mo.constant <#M.dense_array<1.0> : !M.array<1xf32>> : !mo.tensor<[], f32>

  %lhs = mo.transpose(%arg0, %perm) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
  %rhs = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>

  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[16, 64, 128, 128], f32>

  mo.output %out : !mo.tensor<[16, 64, 128, 128], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_constant_lhs([[VARG0:[^ ]*]]: !mo.tensor<[16, 128, 128, 64], f32>) -> !mo.tensor<[16, 64, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<0, 2, 3, 1> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.static.broadcast_to([[V2]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.transpose([[V3]], [[V0]]) : (!mo.tensor<[16, 64, 128, 128], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.sub([[V4]], [[VARG0]]) : !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.transpose([[V5]], [[V1]]) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: mo.output [[V6]] : !mo.tensor<[16, 64, 128, 128], f32>
mo.graph @success_constant_lhs(
  %arg0: !mo.tensor<[16, 128, 128, 64], f32>
) ->  !mo.tensor<[16, 64, 128, 128], f32> {
  %perm = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %cst = mo.constant <#M.dense_array<1.0> : !M.array<1xf32>> : !mo.tensor<[], f32>

  %lhs = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
  %rhs = mo.transpose(%arg0, %perm) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>

  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[16, 64, 128, 128], f32>

  mo.output %out : !mo.tensor<[16, 64, 128, 128], f32>
}

// -----

// CHECK-LABEL: mo.graph
// CHECK-SAME: @failure_non_const_subtree([[VARG0:[^ ]*]]: !mo.tensor<[16, 128, 128, 8], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[7, 7, 8, 64], f32>, [[VARG2:[^ ]*]]: !mo.tensor<[1, 1024, 1, 1], f32>, [[VARG3:[^ ]*]]: !mo.tensor<[1, 1024, 1, 1], f32>) -> !mo.tensor<[16, 64, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
// CHECK-NEXT: [[V2:[^ ]*]] = mo.constant <#M.dense_array<3, 3, 3, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.conv([[VARG0]], [[VARG1]]) [strides = [[V1]], dilations = [[V1]], paddings = [[V2]], num_groups = [[V0]], filter_layout = ""] : (!mo.tensor<[16, 128, 128, 8], f32>, !mo.tensor<[7, 7, 8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V5:[^ ]*]] = mo.transpose([[V4]], [[V3]]) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.static.reshape([[VARG3]]) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[16, 64, 1, 1], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = mo.static.broadcast_to([[V6]]) : !mo.tensor<[16, 64, 1, 1], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.sub([[V5]], [[V7]]) : !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: mo.output [[V8]] : !mo.tensor<[16, 64, 128, 128], f32>
mo.graph @failure_non_const_subtree(
  %arg0: !mo.tensor<[16, 128, 128, 8], f32>,
  %arg1: !mo.tensor<[7, 7, 8, 64], f32>, %arg2: !mo.tensor<[1, 1024, 1, 1], f32>,
  %arg3: !mo.tensor<[1, 1024, 1, 1], f32>
) -> !mo.tensor<[16, 64, 128, 128], f32> {
  %0 = mo.constant <#M.dense_array<1> : !M.array<1xsi64>> : !mo.tensor<[], si64>
  %1 = mo.constant <#M.dense_array<1, 1> : !M.array<2xsi64>> : !mo.tensor<[2], si64>
  %2 = mo.constant <#M.dense_array<3, 3, 3, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %3 = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %4 = mo.conv(%arg0, %arg1) [strides = %1, dilations = %1, paddings = %2, num_groups = %0, filter_layout = ""] : (!mo.tensor<[16, 128, 128, 8], f32>, !mo.tensor<[7, 7, 8, 64], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
  %5 = mo.transpose(%4, %3) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
  %6 = mo.static.reshape(%arg3) : !mo.tensor<[1, 1024, 1, 1], f32> -> !mo.tensor<[16, 64, 1, 1], f32>
  %7 = mo.static.broadcast_to(%6) : !mo.tensor<[16, 64, 1, 1], f32> -> !mo.tensor<[16, 64, 128, 128], f32>
  %8 = mo.sub(%5, %7) : !mo.tensor<[16, 64, 128, 128], f32>
  mo.output %8 : !mo.tensor<[16, 64, 128, 128], f32>
}

// -----
// CHECK-LABEL: mo.graph
// CHECK-SAME: @success_transpose_both_sides([[VARG0:[^ ]*]]: !mo.tensor<[16, 128, 128, 64], f32>) -> !mo.tensor<[16, 64, 128, 128], f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = mo.constant <#M.dense_array<0, 1, 3, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V1:[^ ]*]] = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
// CHECK-NEXT: [[V3:[^ ]*]] = mo.constant <#M.dense_array<1.000000e+00> : !M.array<1xf32>> : !mo.tensor<[], f32>
// CHECK-NEXT: [[V4:[^ ]*]] = mo.static.broadcast_to([[V3]]) : !mo.tensor<[], f32> -> !mo.tensor<[16, 128, 64, 128], f32>
// CHECK-NEXT: [[V6:[^ ]*]] = mo.transpose([[V4]], [[V0]]) : (!mo.tensor<[16, 128, 64, 128], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V7:[^ ]*]] = mo.sub([[VARG0]], [[V6]]) : !mo.tensor<[16, 128, 128, 64], f32>
// CHECK-NEXT: [[V8:[^ ]*]] = mo.transpose([[V7]], [[V1]]) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
// CHECK-NEXT: mo.output [[V8]] : !mo.tensor<[16, 64, 128, 128], f32>
mo.graph @success_transpose_both_sides(
  %arg0: !mo.tensor<[16, 128, 128, 64], f32>
) ->  !mo.tensor<[16, 64, 128, 128], f32> {
  %perm = mo.constant <#M.dense_array<0, 3, 1, 2> : !M.array<4xsi64>> : !mo.tensor<[4], si64>
  %perm_const = mo.constant <#M.dense_array<0, 2, 1, 3> : !M.array<4xsi64>> : !mo.tensor<[4], si64>

  %cst = mo.constant <#M.dense_array<1.0> : !M.array<1xf32>> : !mo.tensor<[], f32>

  %lhs = mo.transpose(%arg0, %perm) : (!mo.tensor<[16, 128, 128, 64], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>
  %pr = mo.static.broadcast_to(%cst) : !mo.tensor<[], f32> -> !mo.tensor<[16, 128, 64, 128], f32>
  %rhs = mo.transpose(%pr, %perm_const) : (!mo.tensor<[16, 128, 64, 128], f32>, !mo.tensor<[4], si64>) -> !mo.tensor<[16, 64, 128, 128], f32>

  %out = mo.sub(%lhs, %rhs) : !mo.tensor<[16, 64, 128, 128], f32>

  mo.output %out : !mo.tensor<[16, 64, 128, 128], f32>
}
